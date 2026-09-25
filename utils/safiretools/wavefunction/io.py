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

from warnings import warn

import numpy as np

from safiretools.hdf5 import read_complex, read_csr, write_csr
from safiretools.types import SpinSymm
from safiretools.wavefunction.slater import CONDITION_MAX, overlap_condition_number

DEFAULT_THRESHOLD = 1e-8
"""Orbital coefficients smaller than this are dropped before sparsifying."""


def warn_if_ill_conditioned(named_matrices, condition_max=CONDITION_MAX) -> None:
    r"""
    Warn about each Slater matrix whose overlap is too ill conditioned to be
    usable.

    Every matrix that reaches disk passes through here, because this is the last
    point at which a trial wavefunction AFQMC cannot work with can be flagged —
    it needs :math:`(M^\dagger M)^{-1}` and its determinant, so an
    ill-conditioned overlap makes the walker overlaps meaningless. Nothing is
    repaired; the matrix is written as it stands.

    Parameters
    ----------
    named_matrices : iterable of (str, numpy.ndarray)
        The matrices to check, each with the dataset name to report it under.
    condition_max : float, optional
        Largest condition number accepted. Default `CONDITION_MAX`.
    """
    offenders = {name: overlap_condition_number(matrix)
                 for name, matrix in named_matrices}
    offenders = {name: condition for name, condition in offenders.items()
                 if condition > condition_max}

    if offenders:
        reported = ', '.join(f'{name} (cond {condition:.2e})'
                             for name, condition in offenders.items())
        warn(
            f"Written with an ill-conditioned overlap matrix: {reported}. "
            f"The limit is {condition_max:.2e}. AFQMC inverts this overlap, so "
            "the trial wavefunction is probably unusable as written: check for "
            "linearly dependent orbitals, and call orthonormalize() if the "
            "columns were never orthonormalized."
        )


# ----------------------------------------------------------------------
# the shared header
# ----------------------------------------------------------------------

def write_header(group, spin_symm: SpinSymm, nmo: int, nelec, coeffs, psi0) -> None:
    """
    Write the header both representations share.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/NOMSD`` or ``Wavefunction/PHMSD`` group.
    spin_symm : SpinSymm
        Spin symmetry; its integer value is what ``dims[3]`` records.
    nmo : int
        Number of *spatial* orbitals, even when noncollinear.
    nelec : tuple(int, int)
        Electron counts as ``dims[1:3]`` records them; see
        `Wavefunction.nelec_on_disk`.
    coeffs : array_like
        Determinant coefficients, ``(ndets,)``.
    psi0 : sequence of numpy.ndarray
        Initial Slater determinant, one ``(npol*nmo, nelec_of_that_spin)`` block
        per spin channel — one block for closed/noncollinear, two for collinear.
    """
    coeffs = np.asarray(coeffs)

    warn_if_ill_conditioned(
        (name, block) for name, block
        in zip(('Psi0_alpha', 'Psi0_beta'), psi0))

    group.create_dataset(
        'dims',
        data=np.array([nmo, nelec[0], nelec[1], int(spin_symm), coeffs.size],
                      dtype=np.int32)
    )
    group.create_dataset('ci_coeffs', data=coeffs)

    for name, block in zip(('Psi0_alpha', 'Psi0_beta'), psi0):
        group.create_dataset(name, data=block)


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

    coeffs = read_complex(group['ci_coeffs'])[:ndets]

    names = ('Psi0_alpha', 'Psi0_beta') if spin_symm is SpinSymm.COLLINEAR \
        else ('Psi0_alpha',)
    psi0 = tuple(read_complex(group[name]) for name in names)

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
    write_csr(group, name, np.asarray(orbitals).conj().T)


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
    return read_csr(group[name]).toarray().conj().T


# ----------------------------------------------------------------------
# the NOMSD payload
# ----------------------------------------------------------------------

def nomsd_orbital_index(idet: int, ispin: int, nspin: int) -> int:
    """The ``PsiT_k`` index of determinant `idet`'s spin-`ispin` block; spin is interleaved."""
    return nspin * idet + ispin


def write_nomsd(group, dets) -> None:
    """
    Write the per-determinant orbital matrices of a NOMSD wavefunction.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/NOMSD`` group.
    dets : sequence of numpy.ndarray
        One ``(ndets, npol*nmo, nelec_of_that_spin)`` stack per spin channel
        (1 or 2).

    Notes
    -----
    Coefficients with magnitude below `DEFAULT_THRESHOLD` are zeroed before
    sparsifying, and each block's overlap is checked *after* that screening. A
    block that fails is warned about and written as it stands.
    """
    nspin = len(dets)
    written = []

    for idet in range(len(dets[0])):
        for ispin, channel in enumerate(dets):
            block = np.array(channel[idet])
            block[abs(block) < DEFAULT_THRESHOLD] = 0.0
            name = f'PsiT_{nomsd_orbital_index(idet, ispin, nspin)}'
            write_orbitals(group, name, block)
            written.append((name, block))

    warn_if_ill_conditioned(written)


def read_nomsd(group, ndets: int, nspin: int):
    """
    Read the per-determinant orbital matrices back.

    Returns
    -------
    tuple of numpy.ndarray
        One ``(ndets, npol*nmo, nelec_of_that_spin)`` stack per spin channel.

    Raises
    ------
    ValueError
        If the file holds no ``PsiT_0`` group — which is what a finite-
        temperature NOMSD (``UL_``/``DL_``/``VL_`` blocks) looks like from here.
    """
    if 'PsiT_0' not in group:
        raise ValueError(
            f"'{group.name}' holds no PsiT_0 group; safiretools reads only "
            "zero-temperature NOMSD wavefunctions"
        )

    return tuple(
        np.array([read_orbitals(group,
                                f'PsiT_{nomsd_orbital_index(idet, ispin, nspin)}')
                  for idet in range(ndets)], dtype=np.complex128)
        for ispin in range(nspin)
    )


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
    occa, occb : numpy.ndarray
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
    two. The arguments are taken as `PHMSDWavefunction` validated them.
    """
    references = orbitals or ()

    group.create_dataset('type', data=len(references))
    for index, matrix in enumerate(references):
        write_orbitals(group, f'PsiT_{index}', matrix)

    warn_if_ill_conditioned(
        (f'PsiT_{index}', matrix) for index, matrix in enumerate(references))

    occs = np.concatenate([occa, occb + nmo], axis=1)
    group.create_dataset('occs', data=occs.ravel().astype(np.int32, copy=False))


def read_phmsd(group, ndets: int, nelec, nmo: int,
               spin_symm=SpinSymm.COLLINEAR):
    """
    Read the occupation numbers and orbital references back.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/PHMSD`` group.
    ndets : int
        Number of determinants, as ``dims`` records it.
    nelec : tuple(int, int)
        The on-disk electron counts, ``dims[1:3]``.
    nmo : int
        Number of orbitals, used to remove the beta offset.
    spin_symm : SpinSymm, optional
        Spin symmetry, which decides how wide each channel is. Only a collinear
        wavefunction stores a beta channel: a closed-shell one repeats alpha
        rather than storing it twice, and a noncollinear one holds both
        polarizations in the alpha channel.

    Returns
    -------
    occa, occb : numpy.ndarray
        Occupied-orbital indices, one array per independent spin channel, with
        the beta offset removed.
    orbitals : tuple of numpy.ndarray or None
        The orbital references, or None when ``type`` is 0.
    """
    nup, ndown = nelec if spin_symm is SpinSymm.COLLINEAR else (nelec[0], 0)

    occs = np.asarray(group['occs'][...]).reshape((-1, nup + ndown))[:ndets]
    occa = occs[:, :nup].copy()
    occb = occs[:, nup:] - nmo

    ntype = int(group['type'][()])
    orbitals = None
    if ntype:
        orbitals = tuple(read_orbitals(group, f'PsiT_{index}')
                         for index in range(ntype))

    return occa, occb, orbitals
