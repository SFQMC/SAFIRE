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
Free-electron trial wavefunctions for lattice models: diagonalize the model's
one-body term and occupy the lowest orbitals.

Reached through `NOMSDWavefunction.from_free_electron`.

Degeneracy is handled explicitly. Orbitals are grouped into shells by
eigenvalue, and a partially filled shell is filled according to a named
strategy, because which degenerate orbitals you occupy fixes properties the
AFQMC run cares about — total momentum above all.
"""

import logging
from pathlib import Path
from warnings import warn

import numpy as np
import scipy.linalg as spl
import toml

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

# TODO: validate a "good" default small irrational twist angle
THETA_X = 1 / np.sqrt(592560607)
"""Small irrational twist along axis 1; 592560607 is prime."""

THETA_Y = 1 / np.sqrt(47603)
"""Small irrational twist along axis 2; 47603 is prime."""

DEFAULT_TWIST = 0.1 * np.array((THETA_X, THETA_Y))
"""A small *irrational* twist, which lifts the degeneracies that make a
free-electron determinant ill-defined on a finite lattice."""

FILLING_STRATEGIES = ('aufbau', 'balanced', 'hund', 'alternating')
"""Recognized ways to fill a partially occupied degenerate shell."""


def from_free_electron(hamiltonian, nelec, twist=None, spin_symm=None, lattice=None,
                       filling_strategy='aufbau', shell_tol=1e-6):
    """
    Build a free-electron trial wavefunction for a lattice model.

    Parameters
    ----------
    hamiltonian : safiretools.LatticeHamiltonian representing the model to build the 
        wavefunction for.
    nelec : tuple(int, int)
        Number of spin-up and spin-down electrons.
    twist : array-like, optional
        Twist angle per axis for the lattice. Defaults to `DEFAULT_TWIST`.
        Ignored when `source` is an already-built Hamiltonian or `lattice` is
        given, both of which carry their own twist.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry to build in. Taken from the Hamiltonian when omitted.
    lattice : ~safiretools.hamiltonian.model.lattice.Lattice, optional
        Lattice to build `source` on, when `source` is a parameter dict or TOML
        file. Built from ``source['lattice']`` if omitted.
    filling_strategy : {'aufbau', 'balanced', 'hund', 'alternating'}, optional
        How to fill a partially occupied degenerate shell. Default ``'aufbau'``.

        - ``'aufbau'``: take the shell's orbitals in order.
        - ``'balanced'``: take evenly spaced orbitals through the shell, which
          balances properties such as momentum.
        - ``'hund'``: maximize spin. Same as ``'aufbau'`` within one spin
          channel.
        - ``'alternating'``: fill from the shell's edges inward
          (0, -1, 1, -2, ...), which cancels momentum in k-space.

    shell_tol : float, optional
        Eigenvalues within this of each other belong to the same shell. Default
        1e-6.

    Returns
    -------
    NOMSDWavefunction
        A single-determinant wavefunction, with `spin_symm` set.

    Raises
    ------
    NotImplementedError
        For `SpinSymm.CLOSED`, which has no free-electron construction here.
    ValueError
        If `source` is of no supported type, or the one-body term's shape does
        not match `spin_symm`.

    Notes
    -----
    The occupied orbitals are eigenvectors of a Hermitian one-body matrix, so
    they are orthonormal by construction and nothing here orthonormalizes them.
    """
    from safiretools.wavefunction.nomsd import NOMSDWavefunction
    from safiretools import LatticeHamiltonian, SpinSymm

    if isinstance(hamiltonian, LatticeHamiltonian):
        if twist is not None and np.any(np.asarray(twist) != hamiltonian.twist):
            warn(
                "the requested twist angle differs from the Hamiltonian's "
                f"({twist} vs {hamiltonian.twist}); using the Hamiltonian's"
            )
    else:
        raise ValueError(
            f"cannot build a free-electron wavefunction from {hamiltonian!r}. "
            "Please provide a LatticeHamiltonian instance."
        )

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm
    spin_symm = SpinSymm.from_input(spin_symm)

    nmo = hamiltonian.nbands * hamiltonian.nsites
    one_body = hamiltonian.get_one_body()

    if spin_symm is SpinSymm.CLOSED:
        raise NotImplementedError(
            "there is no free-electron wavefunction for closed spin symmetry; "
            "build a collinear one instead"
        )

    if spin_symm is SpinSymm.COLLINEAR:
        orbitals = _collinear_orbitals(
            one_body, nelec=nelec, nmo=nmo,
            filling_strategy=filling_strategy, shell_tol=shell_tol)
    else:
        orbitals = _noncollinear_orbitals(
            one_body, nelec=nelec, nmo=nmo,
            filling_strategy=filling_strategy, shell_tol=shell_tol)

    return NOMSDWavefunction(
        coeffs=np.array([1.0 + 0j]),
        dets=orbitals[np.newaxis, ...],
        nelec=nelec,
        spin_symm=spin_symm,
        nmo=nmo,
    )


# ----------------------------------------------------------------------
# diagonalization and shell filling
# ----------------------------------------------------------------------

def to_dense(matrix):
    """
    `matrix` as a dense array.

    A lattice model's one-body term is sparse, but for simplicity, we 
    convert to dense before passing to ``scipy.linalg.eigh``. 
    A sparse eigensolver only pays off when a few
    eigenpairs are needed, and ``eigsh`` does not handle complex matrices
    correctly, which a twisted lattice always produces.
    """
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)


def group_by_shell(eigenvalues, orbitals, tol=1e-6):
    """
    Group eigenvalues and their orbitals into shells of degenerate states.

    Orbitals belong to the same shell when their eigenvalues lie within `tol` of
    the first eigenvalue in that shell.

    Parameters
    ----------
    eigenvalues : array_like
        Ascending eigenvalues.
    orbitals : numpy.ndarray
        Corresponding eigenvectors, as columns.
    tol : float, optional
        Degeneracy tolerance. Default 1e-6.

    Returns
    -------
    list of dict
        One dict per shell, with keys ``energy`` (the shell's mean eigenvalue),
        ``eigenvalues``, ``orbitals``, ``indices`` (into the input order) and
        ``degeneracy``.
    """
    eigenvalues = np.asarray(eigenvalues)
    shells = []
    start = 0

    while start < len(eigenvalues):
        end = start + 1
        while end < len(eigenvalues) \
                and abs(eigenvalues[end] - eigenvalues[start]) < tol:
            end += 1

        shells.append({
            'energy': np.mean(eigenvalues[start:end]),
            'eigenvalues': eigenvalues[start:end],
            'orbitals': orbitals[:, start:end],
            'indices': list(range(start, end)),
            'degeneracy': end - start,
        })
        start = end

    return shells


def _shell_selection(degeneracy: int, nelec: int, strategy: str):
    """
    Which of a shell's `degeneracy` orbitals to occupy with `nelec` electrons.

    A completely filled shell takes all of them whatever the strategy says; the
    strategies differ only in which orbitals a partial fill picks.
    """
    if strategy not in FILLING_STRATEGIES:
        raise ValueError(
            f"unknown filling strategy '{strategy}': supported strategies are "
            f"{list(FILLING_STRATEGIES)}"
        )

    if nelec == degeneracy or strategy in ('aufbau', 'hund'):
        return list(range(nelec))

    if strategy == 'balanced':
        # evenly spaced through the shell, which balances momentum in k-space
        spacing = degeneracy / nelec
        return [int(i * spacing) for i in range(nelec)]

    # 'alternating': from the edges inward, 0, -1, 1, -2, ..., which cancels
    #   momentum in k-space
    selected = []
    left, right = 0, degeneracy - 1
    for step in range(nelec):
        if step % 2 == 0:
            selected.append(left)
            left += 1
        else:
            selected.append(right)
            right -= 1
    return selected


def fill_shells(shells, nelec: int, strategy='aufbau'):
    """
    Fill `shells` with `nelec` electrons, lowest shell first.

    Parameters
    ----------
    shells : list of dict
        Shells as `group_by_shell` returns them.
    nelec : int
        Number of electrons to place.
    strategy : {'aufbau', 'balanced', 'hund', 'alternating'}, optional
        How to fill a partially occupied shell. Default ``'aufbau'``. See
        `from_free_electron` for what each one does.

    Returns
    -------
    orbitals : numpy.ndarray
        Occupied orbitals, ``(nrows, nelec)``.
    indices : list of int
        Which of the input orbitals were occupied.

    Raises
    ------
    ValueError
        If `strategy` is unknown, or the shells cannot hold `nelec` electrons.
    """
    if nelec == 0:
        nrows = shells[0]['orbitals'].shape[0] if shells else 0
        return np.zeros((nrows, 0), dtype=complex), []

    occupied = []
    indices = []
    remaining = nelec

    logger.info("filling %d electrons over %d shell(s) using the '%s' strategy",
                nelec, len(shells), strategy)

    for ishell, shell in enumerate(shells):
        if remaining == 0:
            break

        degeneracy = shell['degeneracy']
        in_shell = min(remaining, degeneracy)
        selected = _shell_selection(degeneracy, in_shell, strategy)

        logger.debug("  shell %d: energy=%.6f, degeneracy=%d, filling %d of %d "
                     "at %s", ishell, shell['energy'], degeneracy, in_shell,
                     degeneracy, selected)

        occupied.append(shell['orbitals'][:, selected])
        indices.extend(shell['indices'][i] for i in selected)
        remaining -= in_shell

    if remaining > 0:
        raise ValueError(
            f"Not enough orbitals to accommodate {nelec} electrons: "
            f"{remaining} could not be placed"
        )

    return np.hstack(occupied), indices


def _occupy(one_body, nelec: int, filling_strategy: str,
            shell_tol: float):
    """Diagonalize `one_body` and return its `nelec` occupied orbitals."""
    eigenvalues, orbitals = spl.eigh(a=to_dense(one_body))
    logger.info("eigenvalues of the non-interacting Hamiltonian: %s", eigenvalues)

    shells = group_by_shell(eigenvalues, orbitals, tol=shell_tol)
    occupied, _ = fill_shells(shells, nelec, strategy=filling_strategy)
    return occupied


def _collinear_orbitals(one_body, nelec, nmo: int, filling_strategy='aufbau',
                        shell_tol=1e-6):
    """
    Occupied orbitals for a collinear free-electron determinant: the two spin
    channels are diagonalized independently and their columns concatenated.

    Returns
    -------
    numpy.ndarray
        ``(nmo, nup + ndown)``.
    """
    occupied = []
    for label, block, nelec_spin in zip(('up', 'down'),
                                        _collinear_blocks(one_body, nmo),
                                        nelec):
        logger.info("processing the spin-%s channel (%d electrons)",
                    label, nelec_spin)
        occupied.append(_occupy(block, nelec_spin, filling_strategy,
                                shell_tol))

    logger.info("built a collinear free-electron determinant: %d electrons "
                "(up: %d, down: %d)", sum(nelec), *nelec)

    return np.hstack(occupied)


def _collinear_blocks(one_body, nmo: int):
    """
    The per-spin blocks of a collinear one-body term.

    A ``(2*nmo, nmo)`` term carries a block per spin; an ``(nmo, nmo)`` one is
    spin-independent and is used for both.
    """
    if one_body.shape == (2 * nmo, nmo):
        return one_body[:nmo, :], one_body[nmo:, :]
    if one_body.shape == (nmo, nmo):
        return one_body, one_body

    raise ValueError(
        f"the one-body term has shape {one_body.shape}, which is neither "
        f"({nmo}, {nmo}) — spin independent — nor ({2 * nmo}, {nmo}) — one "
        "block per spin"
    )


def _noncollinear_orbitals(one_body, nelec, nmo: int,
                           filling_strategy='aufbau', shell_tol=1e-6):
    """
    Occupied orbitals for a noncollinear free-electron determinant: one
    diagonalization over the full ``2*nmo`` spinor basis.

    Returns
    -------
    numpy.ndarray
        ``(2*nmo, nup + ndown)``.
    """
    nelec_total = sum(nelec)

    if one_body.shape == (2 * nmo, nmo):
        one_body = spl.block_diag(to_dense(one_body[:nmo, :]),
                                  to_dense(one_body[nmo:, :]))
    elif one_body.shape == (nmo, nmo):
        one_body = spl.block_diag(to_dense(one_body), to_dense(one_body))
    elif one_body.shape != (2 * nmo, 2 * nmo):
        raise ValueError(
            f"the one-body term has shape {one_body.shape}, which is none of "
            f"({nmo}, {nmo}) — closed shell — ({2 * nmo}, {nmo}) — collinear — "
            f"or ({2 * nmo}, {2 * nmo}) — noncollinear"
        )

    logger.info("processing a noncollinear system (%d electrons)", nelec_total)
    occupied = _occupy(one_body, nelec_total, filling_strategy, shell_tol)

    logger.info("built a noncollinear free-electron determinant: %d electrons "
                "in %d spinor orbitals (originally up: %d, down: %d)",
                nelec_total, 2 * nmo, *nelec)

    return occupied
