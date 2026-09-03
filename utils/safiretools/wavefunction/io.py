# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
The native SAFIRE wavefunction HDF5 schema, read and written here and nowhere
else.

Both representations share a header — ``dims``, ``ci_coeffs`` and the initial
Slater determinant ``Psi0_alpha``/``Psi0_beta`` — and differ only in the payload
that follows it: a `NOMSDWavefunction` writes one CSR ``PsiT_k`` group per
determinant and spin, while a `PHMSDWavefunction` writes flat occupation numbers
plus an optional orbital reference. `Wavefunction.to_hdf5` and
`Wavefunction.from_hdf5` drive both from the shared header, so this module is
the only place the layout is spelled out.

The layout is fixed by the AFQMC executable's readers (``readWfn.cpp``'s
``getCommonInput`` / ``read_nomsd_wavefunction`` / ``read_ph_wavefunction_hdf``
and ``WavefunctionFactory``'s ``getInitialGuess``), so none of it is
configurable.
"""

import numpy as np
import scipy.sparse as sps

from safiretools.hdf5 import from_complex, to_complex
from safiretools.types import SpinSymm

DEFAULT_THRESHOLD = 1e-8
"""Orbital coefficients smaller than this are dropped before sparsifying."""


# ----------------------------------------------------------------------
# the shared header
# ----------------------------------------------------------------------

def write_header(group, nmo: int, nelec, spin_symm: SpinSymm, coeffs, psi0) -> None:
    """
    Write the header both representations share.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/NOMSD`` or ``Wavefunction/PHMSD`` group.
    nmo : int
        Number of orbitals — *spatial* orbitals, even when noncollinear.
    nelec : tuple(int, int)
        Electron counts as they go on disk: ``(nup, ndown)``, or
        ``(nup + ndown, 0)`` for a noncollinear wavefunction. See
        `Wavefunction.nelec_on_disk`.
    spin_symm : SpinSymm
        Spin symmetry; its integer value is what ``dims[3]`` records.
    coeffs : array_like
        Determinant coefficients, ``(ndets,)``.
    psi0 : sequence of numpy.ndarray
        Initial Slater determinant, one ``(npol*nmo, nelec_of_that_spin)`` block
        per spin channel — one block for closed/noncollinear, two for collinear.
    """
    coeffs = np.asarray(coeffs)

    group.create_dataset(
        'dims',
        data=np.array([nmo, nelec[0], nelec[1], int(spin_symm), coeffs.size],
                      dtype=np.int32)
    )
    group.create_dataset('ci_coeffs', data=to_complex(coeffs))

    for name, block in zip(('Psi0_alpha', 'Psi0_beta'), psi0):
        group.create_dataset(name, data=to_complex(block))


def read_header(group) -> dict:
    """
    Read the header `write_header` wrote.

    Returns
    -------
    dict
        Keys ``nmo``, ``nelec``, ``spin_symm``, ``ndets``, ``coeffs`` and
        ``psi0`` (a tuple with one block per spin channel).

    Notes
    -----
    ``nelec`` comes back exactly as the file records it, so a noncollinear
    wavefunction reports ``(nup + ndown, 0)`` — the split into spin channels is
    not recoverable, and the AFQMC executable does not use it either.
    """
    dims = group['dims'][...]
    nmo = int(dims[0])
    nelec = (int(dims[1]), int(dims[2]))
    spin_symm = SpinSymm.from_input(int(dims[3]))
    ndets = int(dims[4])

    coeffs = from_complex(group['ci_coeffs'][...], real_ndim=1)[:ndets]

    names = ('Psi0_alpha', 'Psi0_beta') if spin_symm is SpinSymm.COLLINEAR \
        else ('Psi0_alpha',)
    psi0 = tuple(from_complex(group[name][...], real_ndim=2) for name in names)

    return {
        'nmo': nmo,
        'nelec': nelec,
        'spin_symm': spin_symm,
        'ndets': ndets,
        'coeffs': coeffs,
        'psi0': psi0,
    }


# ----------------------------------------------------------------------
# sparse orbital matrices
# ----------------------------------------------------------------------

def write_orbitals(group, name: str, orbitals) -> None:
    r"""
    Write one orbital matrix as the sparse :math:`\Psi^\dagger` the executable
    reads.

    Parameters
    ----------
    group : h5py.Group
        Group to write the ``name`` subgroup into.
    name : str
        Subgroup name, e.g. ``'PsiT_0'``.
    orbitals : numpy.ndarray
        Orbital matrix :math:`\Psi`, ``(npol*nmo, nelec)``. Stored conjugate
        transposed, so the subgroup holds an ``(nelec, npol*nmo)`` CSR matrix.
    """
    matrix = sps.csr_array(np.asarray(orbitals).conj().T)

    group.create_dataset(
        f'{name}/dims',
        data=np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int32)
    )
    group.create_dataset(f'{name}/data_', data=to_complex(matrix.data))
    group.create_dataset(f'{name}/jdata_',
                         data=matrix.indices.astype(np.int32, copy=False))
    group.create_dataset(f'{name}/pointers_begin_',
                         data=matrix.indptr[:-1].astype(np.int32, copy=False))
    group.create_dataset(f'{name}/pointers_end_',
                         data=matrix.indptr[1:].astype(np.int32, copy=False))


def read_orbitals(group, name: str):
    r"""
    Read back an orbital matrix written by `write_orbitals`.

    Returns :math:`\Psi`, i.e. undoes the conjugate transpose the on-disk
    :math:`\Psi^\dagger` applied.

    Returns
    -------
    numpy.ndarray
        Dense orbital matrix, ``(npol*nmo, nelec)`` complex.
    """
    subgroup = group[name]

    nrows, ncols, nnz = (int(value) for value in subgroup['dims'][...])
    # a CSR `data_` array is flat when stored real, so its interleaved rank is 2
    data = from_complex(subgroup['data_'][...], real_ndim=1)
    indices = subgroup['jdata_'][...]
    pointers_begin = subgroup['pointers_begin_'][...]
    pointers_end = subgroup['pointers_end_'][...]

    indptr = np.zeros(nrows + 1, dtype=np.int64)
    indptr[:-1] = pointers_begin
    if nrows:
        indptr[-1] = pointers_end[-1]

    matrix = sps.csr_array((data[:nnz], indices[:nnz], indptr),
                           shape=(nrows, ncols))
    return matrix.toarray().conj().T


# ----------------------------------------------------------------------
# the NOMSD payload
# ----------------------------------------------------------------------

def nomsd_orbital_index(idet: int, ispin: int, nspin: int) -> int:
    """
    The ``PsiT_k`` index holding determinant `idet`'s spin-`ispin` block.

    Collinear wavefunctions interleave the two spin channels
    (``PsiT_0``/``PsiT_1`` are determinant 0's alpha/beta); the single-channel
    symmetries number their determinants directly.
    """
    return nspin * idet + ispin


def write_nomsd(group, dets, nelec_per_spin, threshold=DEFAULT_THRESHOLD) -> None:
    """
    Write the per-determinant orbital matrices of a NOMSD wavefunction.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/NOMSD`` group.
    dets : numpy.ndarray
        Orbital matrices, ``(ndets, npol*nmo, sum(nelec_per_spin))``. The spin
        channels occupy consecutive column blocks.
    nelec_per_spin : sequence of int
        Electron count in each spin channel; its length is the number of
        channels (1 or 2).
    threshold : float, optional
        Coefficients with magnitude below this are zeroed before sparsifying.
        Default `DEFAULT_THRESHOLD`.
    """
    dets = np.asarray(dets)
    nspin = len(nelec_per_spin)

    for idet, det in enumerate(dets):
        for ispin, block in enumerate(spin_blocks(det, nelec_per_spin)):
            block = block.copy()
            block[abs(block) < threshold] = 0.0
            write_orbitals(group, f'PsiT_{nomsd_orbital_index(idet, ispin, nspin)}',
                           block)


def read_nomsd(group, ndets: int, nelec_per_spin):
    """
    Read the per-determinant orbital matrices back.

    Returns
    -------
    numpy.ndarray
        Orbital matrices, ``(ndets, npol*nmo, sum(nelec_per_spin))``.

    Raises
    ------
    ValueError
        If the file holds no ``PsiT_0`` group — which is what a finite-
        temperature NOMSD (``UL_``/``DL_``/``VL_`` blocks) looks like from here.
    """
    nspin = len(nelec_per_spin)

    if 'PsiT_0' not in group:
        raise ValueError(
            f"'{group.name}' holds no PsiT_0 group; safiretools reads only "
            "zero-temperature NOMSD wavefunctions"
        )

    blocks = [
        [read_orbitals(group, f'PsiT_{nomsd_orbital_index(idet, ispin, nspin)}')
         for ispin in range(nspin)]
        for idet in range(ndets)
    ]

    return np.array([np.concatenate(det, axis=1) for det in blocks],
                    dtype=np.complex128)


# ----------------------------------------------------------------------
# the PHMSD payload
# ----------------------------------------------------------------------

def write_phmsd(group, occa, occb, nmo: int, orbitals=None) -> None:
    """
    Write the occupation numbers, and the optional orbital reference, of a PHMSD
    wavefunction.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/PHMSD`` group.
    occa, occb : array_like
        Occupied-orbital indices per determinant, ``(ndets, nup)`` and
        ``(ndets, ndown)``. Beta indices are offset by `nmo` on disk, which is
        how the executable tells the spin channels apart.
    nmo : int
        Number of orbitals.
    orbitals : sequence of numpy.ndarray, optional
        Orbital matrices the occupation numbers refer to, one per reference
        (one for a closed-shell-like reference, two for a spin-resolved one).
        Omitted, the occupation numbers refer to the orbitals themselves and
        ``type`` is 0.

    Notes
    -----
    ``type`` records how many references follow: 0 for none, 1 for one, 2 for
    two. afqmctools wrote 1 whenever *any* orbital matrix was given, even when
    it went on to write two, so the executable read only the alpha reference and
    silently ignored the beta one.
    """
    occa = np.atleast_2d(np.asarray(occa, dtype=np.int32))
    occb = np.atleast_2d(np.asarray(occb, dtype=np.int32))

    if occa.shape[0] != occb.shape[0]:
        raise ValueError(
            f"occa and occb describe different numbers of determinants "
            f"({occa.shape[0]} and {occb.shape[0]})"
        )

    references = [] if orbitals is None else [
        matrix for matrix in orbitals if matrix is not None]

    group.create_dataset('type', data=len(references))
    for index, matrix in enumerate(references):
        write_orbitals(group, f'PsiT_{index}', matrix)

    occs = np.concatenate([occa, occb + nmo], axis=1)
    group.create_dataset('occs', data=occs.ravel().astype(np.int32, copy=False))


def read_phmsd(group, ndets: int, nelec, nmo: int):
    """
    Read the occupation numbers and orbital references back.

    Returns
    -------
    occa, occb : numpy.ndarray
        Occupied-orbital indices, ``(ndets, nup)`` and ``(ndets, ndown)``, with
        the beta offset removed.
    orbitals : tuple of numpy.ndarray or None
        The orbital references, or None when ``type`` is 0.
    """
    nup, ndown = nelec

    occs = np.asarray(group['occs'][...]).reshape((-1, nup + ndown))[:ndets]
    occa = occs[:, :nup].copy()
    occb = occs[:, nup:] - nmo

    ntype = int(group['type'][()])
    orbitals = None
    if ntype:
        orbitals = tuple(read_orbitals(group, f'PsiT_{index}')
                         for index in range(ntype))

    return occa, occb, orbitals


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------

def spin_blocks(orbitals, nelec_per_spin):
    """
    Split an orbital matrix into its per-spin column blocks.

    Parameters
    ----------
    orbitals : numpy.ndarray
        Orbital matrix, ``(npol*nmo, sum(nelec_per_spin))``.
    nelec_per_spin : sequence of int
        Electron count in each spin channel.

    Yields
    ------
    numpy.ndarray
        One view per spin channel, in order.

    Raises
    ------
    ValueError
        If the matrix has the wrong number of columns.
    """
    orbitals = np.asarray(orbitals)
    expected = sum(nelec_per_spin)

    if orbitals.shape[-1] != expected:
        raise ValueError(
            f"orbital matrix has {orbitals.shape[-1]} columns; expected "
            f"{expected} for electron counts {tuple(nelec_per_spin)}"
        )

    start = 0
    for nelec in nelec_per_spin:
        yield orbitals[..., start:start + nelec]
        start += nelec
