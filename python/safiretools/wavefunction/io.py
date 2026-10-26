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

Both representations share a header — the ``format_version`` and ``spin_type``
attributes and ``ci_coeffs`` — and differ only in the payload that follows it: a
`NOMSDWavefunction` writes one CSR ``PsiT_k`` group per determinant and spin,
while a `PHMSDWavefunction` writes ``occa``/``occb`` occupation numbers plus an
optional orbital reference. Sizes are read off the shapes of these arrays, with
one exception: occupation numbers alone do not span the orbitals, so a PHMSD
records their count in a ``number_of_orbitals`` attribute. `Wavefunction.to_hdf5`
and `Wavefunction.from_hdf5` drive both from the shared header, so this module
is the only place the layout is spelled out.

No initial walker is stored. The executable derives it from whichever
wavefunction the input initializes a walker set from.

The layout is fixed by the AFQMC executable's readers (``readWfn.cpp``'s
``getCommonInput`` / ``read_nomsd_wavefunction`` / ``read_ph_wavefunction_hdf``),
so none of it is configurable.
"""

import itertools
from warnings import warn

import numpy as np

from safiretools.hdf5 import (
    check_format_version,
    read_complex,
    read_csr,
    write_csr,
    write_format_version,
)
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

def write_header(group, spin_symm: SpinSymm, coeffs) -> None:
    """
    Write the header both representations share.

    No sizes are recorded: the number of determinants is read off the shape of
    ``ci_coeffs``.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/NOMSD`` or ``Wavefunction/PHMSD`` group.
    spin_symm : SpinSymm
        Spin symmetry, recorded as the ``spin_type`` attribute.
    coeffs : array_like
        Determinant coefficients, ``(ndets,)``.
    """
    write_format_version(group)
    group.attrs['spin_type'] = spin_symm.label
    group.create_dataset('ci_coeffs', data=np.asarray(coeffs))


def read_header(group) -> dict:
    """
    Read the header `write_header` wrote.

    Returns
    -------
    dict
        Keys ``spin_symm``, ``ndets`` and ``coeffs``. The orbital and electron
        counts are not part of the header; each payload reader takes them from
        its own data.
    """
    if 'format_version' not in group.attrs and 'dims' in group:
        # CoQuí writes neither a format_version nor a spin_type attribute, only
        #   the walker type in slot 3 of a 'dims' array
        spin_symm = SpinSymm.from_input(int(group['dims'][3]))
    else:
        check_format_version(group)
        spin_symm = SpinSymm.from_input(group.attrs['spin_type'])
    coeffs = read_complex(group['ci_coeffs'])

    return {
        'spin_symm': spin_symm,
        'ndets': coeffs.size,
        'coeffs': coeffs,
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

def write_phmsd(group, nmo: int, occa, occb, orbitals=None) -> None:
    """
    Write the occupation numbers, and the optional orbital reference, of a PHMSD
    wavefunction.

    Parameters
    ----------
    group : h5py.Group
        The ``Wavefunction/PHMSD`` group.
    nmo : int
        Number of spatial orbitals, written as the ``number_of_orbitals``
        attribute.
    occa, occb : numpy.ndarray
        Occupied-orbital indices per determinant, ``(ndets, nup)`` and
        ``(ndets, ndown)``, written as the ``occa`` and ``occb`` datasets. Their
        widths are how the file records the electron count.
    orbitals : sequence of numpy.ndarray, optional
        Orbital matrices the occupation numbers refer to, one per reference
        (one for a closed-shell-like reference, two for a spin-resolved one),
        written as ``PsiT_0``, ``PsiT_1``. Omitted, the occupation numbers refer
        to the orbitals themselves.

    Notes
    -----
    How many references there are is the number of ``PsiT_<n>`` groups; nothing
    records it separately. The arguments are taken as `PHMSDWavefunction`
    validated them.
    """
    references = orbitals or ()

    # int32, as TRIQS/h5 writes an int: the executable reads attributes by exact type
    group.attrs['number_of_orbitals'] = np.int32(nmo)
    for index, matrix in enumerate(references):
        write_orbitals(group, f'PsiT_{index}', matrix)

    warn_if_ill_conditioned(
        (f'PsiT_{index}', matrix) for index, matrix in enumerate(references))

    for name, occ in (('occa', occa), ('occb', occb)):
        group.create_dataset(name, data=np.asarray(occ, dtype=np.int32))


def read_phmsd(group):
    """
    Read the orbital count, occupation numbers and orbital references back.

    Returns
    -------
    nmo : int
        Number of spatial orbitals.
    occa, occb : numpy.ndarray
        Occupied-orbital indices, one array per independent spin channel.
    orbitals : tuple of numpy.ndarray or None
        The orbital references, or None when there is no ``PsiT_0``.
    """
    nmo = int(group.attrs['number_of_orbitals'])
    occa = np.asarray(group['occa'][...])
    occb = np.asarray(group['occb'][...])

    # numbered without gaps, so the count is where they stop
    names = list(itertools.takewhile(lambda name: name in group,
                                     (f'PsiT_{index}' for index in itertools.count())))
    orbitals = tuple(read_orbitals(group, name) for name in names) or None

    return nmo, occa, occb, orbitals
