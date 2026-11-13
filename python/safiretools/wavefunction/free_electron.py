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
from warnings import warn

import numpy as np
import scipy.linalg as spl

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

SHELL_TOL = 1e-10
"""Default degeneracy tolerance for grouping eigenvalues into shells.

Sized against `DEFAULT_TWIST`, which splits shells by ``4e-9`` to ``3e-8`` on
the lattices we build (the splitting goes as the square of the twist, and
shrinks with lattice size), and against the eigensolver's noise floor of
roughly ``1e-13`` for a one-body term of bandwidth ``8t``. ``1e-10`` sits
between the two."""

# TODO: validate a "good" default small irrational twist angle
THETA_X = 1 / np.sqrt(592560607)
"""Small irrational twist along axis 1; 592560607 is prime."""

THETA_Y = 1 / np.sqrt(47603)
"""Small irrational twist along axis 2; 47603 is prime."""

DEFAULT_TWIST = 0.1 * np.array((THETA_X, THETA_Y))
"""A small *irrational* twist, which lifts the degeneracies in open shell cases.

Give it to the `~safiretools.hamiltonian.model.lattice.Lattice` the trial
wavefunction's Hamiltonian is built on — it is a property of the lattice, not
an argument to `from_free_electron`. The AFQMC run itself normally wants the
*untwisted* Hamiltonian, so the two are built separately.

It is small enough that the shells it splits stay within ``3e-8``, so it only
reads as non-degenerate against `SHELL_TOL`; the two are sized together."""

FILLING_STRATEGIES = ('aufbau', 'alternating')
"""Recognized ways to fill a partially occupied degenerate shell."""


def from_free_electron(hamiltonian, nelec, spin_symm=None,
                       filling_strategy='aufbau', shell_tol=SHELL_TOL):
    """
    Build a free-electron trial wavefunction for a lattice model.

    Parameters
    ----------
    hamiltonian : safiretools.LatticeHamiltonian
        The lattice model to build the wavefunction for.
    nelec : tuple(int, int)
        Number of spin-up and spin-down electrons.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry to build in. Taken from the Hamiltonian when omitted.
    filling_strategy : {'aufbau', 'alternating'}, optional
        How to fill a partially occupied degenerate shell. Default ``'aufbau'``.

        - ``'aufbau'``: take the shell's orbitals in order.
        - ``'alternating'``: fill from the shell's edges inward
          (0, -1, 1, -2, ...), which cancels momentum in k-space.

    shell_tol : float, optional
        Eigenvalues within this of each other belong to the same shell. Default
        `SHELL_TOL = 1e-10`.

    Returns
    -------
    NOMSDWavefunction
        A single-determinant wavefunction, with `spin_symm` set.

    Raises
    ------
    NotImplementedError
        For `SpinSymm.CLOSED`, which has no free-electron construction here.
    ValueError
        If `hamiltonian` is not a `safiretools.LatticeHamiltonian`,
        `filling_strategy` is unknown, the one-body term's shape does not match
        `spin_symm`, or it has too few orbitals for `nelec`.

    Warns
    -----
    UserWarning
        If the highest occupied shell is degenerate and only partly filled, so
        the determinant is not uniquely defined. Build the trial wavefunction
        from a Hamiltonian on a lattice carrying a small irrational twist, which
        lifts the degeneracy.
    """
    from safiretools.wavefunction.nomsd import NOMSDWavefunction
    from safiretools import LatticeHamiltonian

    if not isinstance(hamiltonian, LatticeHamiltonian):
        raise ValueError(
            f"cannot build a free-electron wavefunction from {hamiltonian!r}. "
            "Please provide a LatticeHamiltonian instance."
        )

    if filling_strategy not in FILLING_STRATEGIES:
        raise ValueError(
            f"unknown filling strategy '{filling_strategy}': supported "
            f"strategies are {list(FILLING_STRATEGIES)}"
        )

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm
    spin_symm = SpinSymm.from_input(spin_symm)

    if spin_symm is SpinSymm.CLOSED:
        raise NotImplementedError(
            "there is no free-electron wavefunction for closed spin symmetry; "
            "build a collinear one instead"
        )

    nmo = hamiltonian.nbands * hamiltonian.nsites
    one_body = hamiltonian.get_one_body().toarray()

    if one_body.shape == (2 * nmo, 2 * nmo):
        if spin_symm is not SpinSymm.NONCOLLINEAR:
            raise ValueError(
                "the one-body term mixes spins, so it has no collinear "
                "free-electron determinant"
            )
        blocks, nelec_blocks = [one_body], [sum(nelec)]
    elif one_body.shape == (2 * nmo, nmo):
        blocks, nelec_blocks = [one_body[:nmo], one_body[nmo:]], nelec
    elif one_body.shape == (nmo, nmo):
        blocks, nelec_blocks = [one_body, one_body], nelec
    else:
        raise ValueError(
            f"the one-body term has shape {one_body.shape}, which is none of "
            f"({nmo}, {nmo}) — spin independent — ({2 * nmo}, {nmo}) — one "
            f"block per spin — or ({2 * nmo}, {2 * nmo}) — noncollinear"
        )

    orbitals = tuple(_occupy(block, n, filling_strategy, shell_tol)
                     for block, n in zip(blocks, nelec_blocks))

    if spin_symm is SpinSymm.NONCOLLINEAR:
        # spinor orbitals, spin-up components first; per-spin blocks land on
        # the diagonal
        orbitals = spl.block_diag(*orbitals).reshape(2, nmo, sum(nelec))

    logger.info("built a %s free-electron determinant: %d electrons "
                "(up: %d, down: %d)", spin_symm.name.lower(), sum(nelec), *nelec)

    return NOMSDWavefunction.from_single_determinant(orbitals)


def _occupy(one_body, nelec: int, filling_strategy: str, shell_tol: float):
    """
    Diagonalize `one_body` and return its `nelec` lowest orbitals, filling a
    partly occupied degenerate shell by `filling_strategy`.
    """
    eigenvalues, orbitals = spl.eigh(one_body)
    logger.info("eigenvalues of the non-interacting Hamiltonian: %s", eigenvalues)

    if nelec > len(eigenvalues):
        raise ValueError(
            f"not enough orbitals to accommodate {nelec} electrons: there are "
            f"only {len(eigenvalues)}"
        )

    # shells of degenerate orbitals, as runs of indices into `eigenvalues`
    shells = np.split(np.arange(len(eigenvalues)),
                      np.flatnonzero(np.diff(eigenvalues) >= shell_tol) + 1)

    occupied = []
    for shell in shells:
        remaining = nelec - len(occupied)
        if remaining >= len(shell):
            occupied.extend(shell)
            continue

        if remaining > 0:
            if filling_strategy == 'alternating':
                # from the edges inward, 0, -1, 1, -2, ..., which cancels
                # momentum in k-space
                picks = [i // 2 if i % 2 == 0 else -(i + 1) // 2
                         for i in range(remaining)]
            else:
                picks = list(range(remaining))
            selected = shell[picks]

            warn(
                f"the highest occupied shell (energy "
                f"{eigenvalues[shell[0]]:.6f}) is {len(shell)}-fold degenerate "
                f"but holds only {remaining} electron(s), so which of its "
                "orbitals the determinant occupies is arbitrary. The "
                f"'{filling_strategy}' strategy picked {sorted(selected.tolist())}. "
                "Build the trial wavefunction's Hamiltonian on a lattice with "
                "a small irrational twist to break the degeneracy due to "
                "translational symmetry. A degeneracy between bands or spin "
                "sectors requires a one-body term that explicitly breaks those "
                "symmetries."
            )
            occupied.extend(selected)
        break

    return orbitals[:, occupied]
