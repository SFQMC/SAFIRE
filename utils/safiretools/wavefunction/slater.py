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

Every trial wavefunction here reduces to Slater matrices, held as one
``(npol*nmo, n)`` matrix per spin channel. This module holds the operations on
them: building one from orbital coefficients and occupied indices, expressing it
in another basis, converting to and from the public spin layout the
wavefunction classes take, and checking or restoring the orthonormality of its
columns.

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

LINDEP_TOL = 1e-12
"""Smallest diagonal element of ``R`` that `orthonormalize` accepts before it
calls the columns linearly dependent."""

#Largest overlap-matrix condition number a Slater matrix may have before it
#  is reported as ill conditioned
CONDITION_MAX = 1.0 / np.sqrt(np.finfo(np.float64).eps)



# ----------------------------------------------------------------------
# building a Slater matrix
# ----------------------------------------------------------------------

def _select_orbitals(mo_coeffs, nocc: Iterable, nelec: int):
    """One spin channel's Slater matrix: the orbitals `nocc` out of `mo_coeffs`."""
    selection = np.zeros((mo_coeffs.shape[1], nelec))
    selection[nocc, np.arange(nelec)] = 1

    return mo_coeffs @ selection + 0j


def make_slater(spin_symm: SpinSymm, mo_coeffs, nocc, nelec):
    """
    The Slater matrices for a reference of the given spin symmetry, one per
    spin channel.

    Parameters
    ----------
    spin_symm : SpinSymm
        Spin symmetry of the reference, which decides how many channels the
        result has.
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
    tuple of numpy.ndarray
        One ``complex128`` ``(npol*nmo, nelec_of_that_spin)`` Slater matrix per
        spin channel.
    """
    mo_coeffs = np.asarray(mo_coeffs)

    if spin_symm is SpinSymm.CLOSED:
        return (_select_orbitals(mo_coeffs, nocc[0], nelec[0]),)
    if spin_symm is SpinSymm.NONCOLLINEAR:
        return (_select_orbitals(mo_coeffs, nocc[0], sum(nelec)),)

    if len(nocc) != len(nelec):
        raise ValueError(
            f"nocc describes {len(nocc)} spin channels and nelec {len(nelec)}"
        )

    # one matrix (ROHF) or one per spin (UHF)
    per_spin = mo_coeffs if mo_coeffs.ndim == 3 else [mo_coeffs] * len(nocc)
    return tuple(_select_orbitals(coeffs, spin_nocc, spin_nelec)
                 for coeffs, spin_nocc, spin_nelec in zip(per_spin, nocc, nelec))


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
# the public spin layout
# ----------------------------------------------------------------------

def parse_spin_layout(value, spin_symm=None, stacked: bool = False,
                      name: str = 'value'):
    r"""
    Read a value in the public spin layout into one matrix per spin channel.

    The layout's type and rank carry the spin symmetry:

    ============  ======================================================
    closed        `numpy.ndarray`, ``(nmo, nup)``
    collinear     `tuple` of two arrays, ``(nmo, nup)`` and ``(nmo, ndown)``
    noncollinear  `numpy.ndarray`, ``(2, nmo, nelec)``
    ============  ======================================================

    A collinear value must be a `tuple`: a list, or an array holding both
    channels, is not accepted, since with ``nup == ndown`` it would be
    indistinguishable from a noncollinear one.

    Parameters
    ----------
    value : numpy.ndarray or tuple of numpy.ndarray
        The value to read.
    spin_symm : SpinSymm or str or int, optional
        The spin symmetry `value` must have. Inferred from the layout when
        omitted.
    stacked : bool, optional
        Every array carries one extra leading axis, such as ``ndets``.
    name : str, optional
        What `value` is called in error messages.

    Returns
    -------
    spin_symm : SpinSymm
        The spin symmetry of the layout.
    blocks : tuple of numpy.ndarray
        One ``complex128`` ``(..., npol*nmo, n)`` matrix per spin channel. A
        noncollinear value has its polarization axis folded into the rows,
        spin up first.

    Raises
    ------
    TypeError
        If `value` is none of the three layouts.
    ValueError
        If the two collinear channels span different orbitals, or the layout
        contradicts `spin_symm`.
    """
    lead = int(stacked)

    if isinstance(value, tuple) and len(value) == 2:
        blocks = tuple(np.asarray(block, dtype=np.complex128) for block in value)
        if any(block.ndim != 2 + lead for block in blocks):
            raise _layout_error(value, stacked, name)
        if blocks[0].shape[:-1] != blocks[1].shape[:-1]:
            raise ValueError(
                f"the collinear channels of {name} have shapes {blocks[0].shape} "
                f"and {blocks[1].shape}, which differ in more than their "
                "electron count"
            )
        found = SpinSymm.COLLINEAR
    elif isinstance(value, np.ndarray) and value.ndim == 2 + lead:
        blocks = (value.astype(np.complex128, copy=False),)
        found = SpinSymm.CLOSED
    elif isinstance(value, np.ndarray) and value.ndim == 3 + lead \
            and value.shape[lead] == 2:
        rows = 2 * value.shape[lead + 1]
        blocks = (value.astype(np.complex128, copy=False)
                  .reshape(value.shape[:lead] + (rows, value.shape[-1])),)
        found = SpinSymm.NONCOLLINEAR
    else:
        raise _layout_error(value, stacked, name)

    if spin_symm is not None and SpinSymm.from_input(spin_symm) is not found:
        raise ValueError(
            f"{name} must be in the {SpinSymm.from_input(spin_symm).label} "
            f"layout, got the {found.label} one"
        )

    return found, blocks


def format_spin_layout(blocks, spin_symm):
    """
    The inverse of `parse_spin_layout`: per-channel ``(..., npol*nmo, n)``
    matrices in the public spin layout of `spin_symm`.
    """
    spin_symm = SpinSymm.from_input(spin_symm)
    if spin_symm is SpinSymm.COLLINEAR:
        return tuple(blocks)

    (block,) = blocks
    if spin_symm is SpinSymm.NONCOLLINEAR:
        return block.reshape(block.shape[:-2]
                             + (2, block.shape[-2] // 2, block.shape[-1]))
    return block


def spin_layout_shape(value):
    """The shape of a value in the public spin layout; a pair of shapes when collinear."""
    if isinstance(value, tuple):
        return tuple(np.shape(block) for block in value)
    return np.shape(value)


def _layout_error(value, stacked: bool, name: str) -> TypeError:
    lead = "(ndets, " if stacked else "("
    if isinstance(value, np.ndarray):
        got = f"an array of shape {value.shape}"
    elif isinstance(value, tuple):
        got = f"a tuple of {len(value)}"
    else:
        got = f"a {type(value).__name__}"

    return TypeError(
        f"{name} must be a closed-shell array {lead}nmo, nup), a collinear "
        f"tuple of arrays {lead}nmo, nup) and {lead}nmo, ndown), or a "
        f"noncollinear array {lead}2, nmo, nelec); got {got}"
    )


# ----------------------------------------------------------------------
# orthonormality
# ----------------------------------------------------------------------

def orthonormalize(matrix, tol=ORTHONORMAL_TOL):
    """
    `matrix` with orthonormal columns spanning the same space.

    A matrix that `is_orthonormal` to within `tol` is returned as it is.
    Otherwise it is replaced by the ``Q`` of its reduced QR decomposition, as a
    new ``complex128`` array. The sign convention is pinned so that ``R`` has a
    non-negative real diagonal, which makes ``Q`` unique.

    Raises
    ------
    ValueError
        If `matrix` is not two-dimensional, or its columns are (near) linearly
        dependent: a diagonal element of ``R`` below `LINDEP_TOL`.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-dimensional array, got {matrix.ndim}D")

    if is_orthonormal(matrix, tol=tol):
        return matrix

    Q, R = np.linalg.qr(matrix.astype(np.complex128, copy=False), mode='reduced')

    diagonal = np.diagonal(R)
    dependent = np.flatnonzero(np.abs(diagonal) < LINDEP_TOL)
    if dependent.size:
        raise ValueError(
            "linearly dependent vectors encountered while orthogonalizing "
            f"column {dependent[0]}"
        )

    return Q * (np.abs(diagonal) / diagonal)


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


def overlap_condition_number(matrix) -> float:
    r"""
    The condition number of a Slater matrix's overlap :math:`S = M^\dagger M`.

    This is what decides whether a trial wavefunction is usable: AFQMC needs
    :math:`S^{-1}` and :math:`\det S`, so a large condition number means the
    walker overlaps are numerically meaningless however well the columns were
    normalized.

    Parameters
    ----------
    matrix : numpy.ndarray
        Slater matrix, ``(nrows, ncols)``.

    Returns
    -------
    float
        The 2-norm condition number of :math:`S`, computed from `matrix`'s
        singular values as :math:`(\sigma_{max}/\sigma_{min})^2`. ``inf`` when
        `matrix` is rank deficient — an all-zero matrix included — and ``1.0``
        for a matrix with no columns, whose overlap is the empty identity.

    Examples
    --------
    >>> overlap_condition_number(np.eye(4)[:, :2])
    1.0
    >>> overlap_condition_number(np.zeros((4, 2)))
    inf
    """
    matrix = np.asarray(matrix)

    if matrix.shape[-1] == 0:
        return 1.0

    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] == 0.0:
        return np.inf

    return float((singular[0] / singular[-1]) ** 2)
