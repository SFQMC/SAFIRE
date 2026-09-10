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
Operations on Slater matrices, independent of the domain that produced the
orbitals.

Every trial wavefunction here reduces to Slater matrices in the column layout
`safiretools.NOMSDWavefunction` takes — the spin channels as consecutive column
blocks. This module holds the operations on that layout: building one from
orbital coefficients and occupied indices, expressing it in another basis,
splitting it back into its spin blocks, and checking or restoring the
orthonormality of its columns.

Nothing here reads an external convention, so a domain module (`pyscf.py`,
`pbc.py`, `free_electron.py`) keeps only the code that decodes its own — which
orbitals a PySCF ``mo_occ`` vector calls occupied, say.
"""

from collections.abc import Iterable

import numpy as np

from safiretools.types import SpinSymm

ORTHONORMAL_TOL = 1e-10
"""How far an overlap matrix may stray from the identity and still count as
orthonormal."""


# ----------------------------------------------------------------------
# building a Slater matrix
# ----------------------------------------------------------------------

def make_slater_closed(mo_coeffs, nocc: Iterable, nelec: int):
    """
    One spin channel's Slater matrix: `nelec` columns selecting the orbitals
    `nocc` out of `mo_coeffs`.
    """
    selection = np.zeros((mo_coeffs.shape[1], nelec))
    selection[nocc, np.arange(nelec)] = 1

    return mo_coeffs @ selection + 0j


def make_slater_collinear(mo_coeffs, nocc, nelec):
    """
    Both spin channels' Slater matrices, concatenated column-wise. `mo_coeffs`
    may be one matrix (ROHF) or one per spin (UHF).
    """
    if len(nocc) != len(nelec):
        raise ValueError(
            f"nocc describes {len(nocc)} spin channels and nelec {len(nelec)}"
        )

    if mo_coeffs.ndim == 3:
        blocks = [make_slater_closed(spin_coeffs, spin_nocc, spin_nelec)
                  for spin_coeffs, spin_nocc, spin_nelec
                  in zip(mo_coeffs, nocc, nelec)]
    else:
        blocks = [make_slater_closed(mo_coeffs, spin_nocc, spin_nelec)
                  for spin_nocc, spin_nelec in zip(nocc, nelec)]

    return np.concatenate(blocks, axis=1)


def make_slater(spin_symm: SpinSymm, mo_coeffs, nocc, nelec):
    """
    The Slater matrix for a reference of the given spin symmetry, in the column
    layout `safiretools.NOMSDWavefunction` takes.

    Parameters
    ----------
    spin_symm : SpinSymm
        Spin symmetry of the reference, which decides how many column blocks
        the result has.
    mo_coeffs : array_like
        Orbital coefficients: one matrix, or — for a collinear reference built
        from spin-resolved orbitals (UHF) — one per spin channel.
    nocc : sequence
        Occupied orbital indices per spin channel. Only the alpha entry is read
        for the single-channel symmetries.
    nelec : tuple(int, int)
        Electron counts ``(nup, ndown)``. A noncollinear reference takes their
        sum, since both polarizations share one channel.

    Returns
    -------
    numpy.ndarray
        ``complex128`` Slater matrix, ``(npol*nmo, sum(nelec_per_spin))``.
    """
    mo_coeffs = np.asarray(mo_coeffs)

    if spin_symm is SpinSymm.CLOSED:
        return make_slater_closed(mo_coeffs, nocc[0], nelec[0])
    if spin_symm is SpinSymm.COLLINEAR:
        return make_slater_collinear(mo_coeffs, nocc, nelec)
    return make_slater_closed(mo_coeffs, nocc[0], sum(nelec))


def transform_slater(orbitals, transform):
    """
    Express `orbitals` in the basis `transform` maps into, promoting the
    transformation to the spinor basis when the orbitals are noncollinear.

    Parameters
    ----------
    orbitals : numpy.ndarray
        Slater matrix, ``(npol*nmo, ncols)``.
    transform : numpy.ndarray
        Transformation into the working basis, over *spatial* orbitals. It is
        promoted with ``kron(eye(2), transform)`` when `orbitals` has twice as
        many rows.

    Returns
    -------
    numpy.ndarray
        The transformed Slater matrix.
    """
    if transform.shape[0] != orbitals.shape[0]:
        transform = np.kron(np.eye(2), transform)
    return transform.conj().T @ orbitals


# ----------------------------------------------------------------------
# spin blocks
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


# ----------------------------------------------------------------------
# orthonormality
# ----------------------------------------------------------------------

def modified_gram_schmidt(matrix, tol=1e-12):
    """
    Orthonormalize the columns of `matrix` by modified Gram-Schmidt.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix ``(n, m)`` whose columns are orthonormalized in place order.
    tol : float, optional
        Smallest norm accepted before columns count as linearly dependent.
        Default 1e-12.

    Returns
    -------
    numpy.ndarray
        A new ``complex128`` matrix with orthonormal columns spanning the same
        column space.

    Raises
    ------
    ValueError
        If `matrix` is not two-dimensional, or a column is (near) linearly
        dependent on the previous ones.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-dimensional array, got {matrix.ndim}D")

    orthonormal = np.zeros_like(matrix, dtype=np.complex128)

    for column in range(matrix.shape[1]):
        vector = np.array(matrix[:, column], dtype=np.complex128, copy=True)
        for previous in range(column):
            vector -= np.vdot(orthonormal[:, previous], vector) \
                * orthonormal[:, previous]

        norm = np.linalg.norm(vector)
        if norm < tol:
            raise ValueError(
                "linearly dependent vectors encountered during Gram-Schmidt "
                f"orthogonalization of column {column}"
            )

        orthonormal[:, column] = vector / norm

    return orthonormal


def is_orthonormal(matrix, tol=ORTHONORMAL_TOL) -> bool:
    """
    True when `matrix`'s columns are orthonormal, i.e. when
    :math:`M^\\dagger M` is the identity to within `tol`.

    A matrix with no columns is orthonormal: its overlap is the empty identity.
    """
    matrix = np.asarray(matrix)
    if matrix.shape[-1] == 0:
        return True

    overlap = matrix.conj().T @ matrix
    identity = np.eye(matrix.shape[-1], dtype=overlap.dtype)
    return bool(np.max(np.abs(overlap - identity)) < tol)
