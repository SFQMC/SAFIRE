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
Recipes for the lattice-model systems.

These are the cheap ones - safiretools and afqmctools only, no external
codes, seconds rather than minutes - and they cover the Hubbard,
Hubbard-Kanamori and Rashba spin-orbit models.

The ``hst_type`` variants under ``square_4x4_hubbard_nup5_ndn5`` are read only by
the C++ unit tests. They are regenerated here anyway, because the point of this
tool is that the inputs tree can be rebuilt in full. The one exception is
``square_2x2_hubbard_Beta3_nt100``, also read only by the unit tests: its
finite-temperature trial wavefunction is something safiretools cannot write, so
that directory is committed by hand.
"""

from typing import Dict, List

import numpy as np

from . import BuildContext, Recipe

# chosen for compatibility with old input files
FREE_ELECTRON_TWIST = 0.1 * np.array((1 / np.sqrt(592560607), 1 / np.sqrt(47603)))


# ============================================================================
# Helpers
# ============================================================================

def _lattices(model: Dict):
    """The lattice a model is defined on, and a twisted copy of it.

    A free-electron trial is diagonalized straight out of the one-body term, so
    at a closed shell its highest occupied shell is degenerate and the
    determinant is not uniquely defined. The twist lifts that, and it belongs to
    the lattice rather than to the wavefunction builder - hence two lattices,
    the untwisted one for the Hamiltonian the run reads and the twisted one for
    the trial built from it.
    """
    from safiretools import Lattice

    return (Lattice.from_dict(model["lattice"]),
            Lattice.from_dict({**model["lattice"], "twist": FREE_ELECTRON_TWIST}))


def _hamiltonian(model: Dict, spin_symm, lattice):
    """`model` built on `lattice` in `spin_symm`."""
    from safiretools import LatticeHamiltonian

    return LatticeHamiltonian.from_dict(
        {**model, "hamiltonian": {**model["hamiltonian"], "spin_symm": spin_symm}},
        lattice=lattice)


# ============================================================================
# 4x4 Hubbard, nup = ndn = 5
# ============================================================================

HUBBARD_4X4 = {
    "hamiltonian": {"t": 1.0, "U": 6.0},
    "lattice": {"L1": 4, "L2": 4, "boundary1": "PBC", "boundary2": "PBC"},
    "misc_params": {"nelec": (5, 5)},
}


def build_hubbard_4x4(ctx: BuildContext) -> None:
    """4x4 Hubbard at U/t = 6, five electrons per spin.

    Writes the three spin symmetries of the U = 6 hamiltonian with matching
    free-electron trials, plus four files that exist to exercise the
    Hubbard-Stratonovich decompositions in the C++ unit tests.
    """
    from safiretools import SpinSymm, Wavefunction

    out = ctx.out_dir
    nelec = HUBBARD_4X4["misc_params"]["nelec"]
    lattice, twisted = _lattices(HUBBARD_4X4)

    name = {SpinSymm.CLOSED: "closed",
            SpinSymm.COLLINEAR: "collinear",
            SpinSymm.NONCOLLINEAR: "noncollinear"}

    for spin_symm in (SpinSymm.CLOSED, SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR):
        _hamiltonian(HUBBARD_4X4, spin_symm, lattice).to_hdf5(
            out / f"ham_{name[spin_symm]}.h5")

    for spin_symm in (SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR):
        Wavefunction.from_free_electron(
            _hamiltonian(HUBBARD_4X4, spin_symm, twisted),
            nelec=nelec, spin_symm=spin_symm,
        ).to_hdf5(out / f"wfn_fe_{name[spin_symm]}.h5")

    lattice.get_directed_pairs(
        directions=["s", "+x", "+y"]).to_hdf5(out / "pair_correlators.h5")

    # --- Hubbard-Stratonovich variants (C++ unit tests only) ---------------
    # For U > 0 the builder infers a discrete *spin* decomposition and for
    # U < 0 a discrete *charge* one; `hst_types` overrides that inference.
    # Together with ham_collinear.h5's inferred discrete_spin, the three
    # overrides below cover all four decompositions the C++ reader accepts
    # (`ModelHamOpsGenerator.cpp`).
    #
    # The committed ham_collinear_cont_spin.h5 predates this override and
    # stores discrete_spin despite its name, so it duplicated ham_collinear.h5
    # and left continuous_spin exercised by nothing.
    variants = {
        "ham_collinear_cont_spin.h5":
            {"t": 1.0, "U": 6.0, "hst_types": {"U": "continuous_spin"}},
        "ham_collinear_Um4_disc_charge.h5":
            {"t": 1.0, "U": -4.0},
        "ham_collinear_Um4_cont_charge.h5":
            {"t": 1.0, "U": -4.0, "hst_types": {"U": "continuous_charge"}},
    }
    for name, terms in variants.items():
        _hamiltonian({**HUBBARD_4X4, "hamiltonian": terms},
                     SpinSymm.COLLINEAR, lattice).to_hdf5(out / name)

    _build_uhf_trial(ctx, out / "uhf_U0.1_wfn_nup5_ndn5.h5")


def _build_uhf_trial(ctx: BuildContext, filename) -> None:
    """Variational UHF trial for the 4x4 lattice at a weak U = 0.1.

    Deliberately a much weaker interaction than the hamiltonians this trial is
    paired with - it is the deliberately-imperfect trial the attractive-U unit
    tests run against.

    AutoHF is a stochastic optimiser, so this is set up to be reproducible
    rather than merely seeded: the free-electron determinant of the same
    hamiltonian is used as the starting point and the random perturbation of it
    is switched off (``state0_scale = 0``). With those settings the solve is
    seed-independent and lands on E = -23.84375. Left to randomise its own start
    it lands on a different solution roughly one run in three, and occasionally
    diverges outright, which is not something a regeneration tool should do.

    The result is checked against the free-electron determinant before it is
    written: at U = 0.1 the Hartree-Fock solution is the non-interacting one to
    within the optimiser's tolerance, which is a cheap way of catching a solve
    that has wandered off.
    """
    from safiretools import (LatticeHamiltonian, NOMSDWavefunction, SpinSymm,
                             Wavefunction)

    try:
        from autohf.hamiltonian import AutoHFHamiltonian
        from autohf.solver import solve_hf
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "the U=0.1 UHF trial needs AutoHF (install utils/AutoHF)"
        ) from exc

    weak = {
        "hamiltonian": {"t": 1.0, "U": 0.1},
        "lattice": HUBBARD_4X4["lattice"],
        "misc_params": {"nelec": (5, 5)},
    }
    nelec = weak["misc_params"]["nelec"]
    norb = 16

    hamiltonian = LatticeHamiltonian.from_dict(weak)

    # Untwisted, so the starting determinant - and the solution - stays real.
    initial_alpha, initial_beta = Wavefunction.from_free_electron(
        hamiltonian, nelec=nelec, spin_symm=SpinSymm.COLLINEAR).determinant(0)

    # AutoHFHamiltonian(source=...) only reads afqmctools hamiltonians, so the
    # terms are handed over explicitly, as its afqmctools interface would.
    one_body = hamiltonian.get_one_body().toarray()
    autohf_hamiltonian = AutoHFHamiltonian(
        T=[one_body[:norb], one_body[norb:]],
        U=np.diag(hamiltonian.get_U().toarray().diagonal()),
        N=norb,
    )

    results = solve_hf(
        autohf_hamiltonian,
        settings=dict(
            ansatz="SD_ROT",
            steps=2000,
            batch_size=1,
            nelec=nelec,
            seed=20250612,
            state0_scale=0.0,   # no random perturbation of the initial state
            verbose=ctx.verbose,
            measure_spin=False,
        ),
        # AutoHF takes the afqmctools layout, (norb, nup + ndn)
        initial_guess=np.hstack([initial_alpha, initial_beta]).real,
        suppress_logo=True,
    )
    data = results[0] if isinstance(results, tuple) else results

    # AutoHF hands back (spin, norb, nelec_per_spin)
    alpha, beta = np.asarray(data["orbitals"])
    _check_spans_free_electron(alpha, initial_alpha, "alpha")
    _check_spans_free_electron(beta, initial_beta, "beta")

    NOMSDWavefunction(
        coeffs=np.array([1.0], dtype=np.complex128),
        dets=(alpha[None], beta[None]),
    ).to_hdf5(filename)


def _check_spans_free_electron(orbitals, reference, label: str,
                               tol: float = 1e-6) -> None:
    """Fail if ``orbitals`` does not span the same space as ``reference``.

    Two determinants that span the same occupied space are the same state - the
    orbitals themselves are only defined up to a rotation among them - so this
    compares the principal angles between the two subspaces rather than the
    orbitals element by element.
    """
    qa, _ = np.linalg.qr(np.asarray(orbitals))
    qb, _ = np.linalg.qr(np.asarray(reference))
    overlaps = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    if not np.allclose(overlaps, 1.0, atol=tol):
        raise RuntimeError(
            f"the AutoHF {label} orbitals do not span the free-electron space "
            f"(smallest principal-angle overlap {overlaps.min():.6f}); the "
            "solve has converged somewhere unexpected"
        )


# ============================================================================
# 6x1 Hubbard-Kanamori, two bands
# ============================================================================

HUBBARD_KANAMORI_6X1 = {
    "hamiltonian": {"nbands": 2, "t": 1.0, "U": 2.0,
                    "U1": 1.5, "U2": 1.0, "J": 0.5},
    "lattice": {"L1": 6, "L2": 1, "boundary1": "pbc", "boundary2": "open"},
    "misc_params": {"nelec": (6, 6)},
}


def build_hubbard_kanamori(ctx: BuildContext) -> None:
    """Two-band Hubbard-Kanamori chain: the multi-band interaction case.

    Only collinear and noncollinear are built - the closed symmetry cannot
    represent the Hund's coupling term.
    """
    from safiretools import SpinSymm, Wavefunction

    out = ctx.out_dir
    nelec = HUBBARD_KANAMORI_6X1["misc_params"]["nelec"]
    lattice, twisted = _lattices(HUBBARD_KANAMORI_6X1)

    for spin_symm, name in ((SpinSymm.COLLINEAR, "collinear"),
                            (SpinSymm.NONCOLLINEAR, "noncollinear")):
        hamiltonian = _hamiltonian(HUBBARD_KANAMORI_6X1, spin_symm, lattice)
        hamiltonian.to_hdf5(out / f"ham_{name}.h5")

        Wavefunction.from_free_electron(
            _hamiltonian(HUBBARD_KANAMORI_6X1, spin_symm, twisted),
            nelec=nelec, spin_symm=spin_symm,
        ).to_hdf5(out / f"wfn_fe_{name}.h5")

    # nbands is the Hamiltonian's, not the lattice's: the lattice knows how many
    #   sites a unit cell has, not how many orbitals sit on each site.
    lattice.get_directed_pairs(
        directions=["s", "+x"], nbands=hamiltonian.nbands
    ).to_hdf5(out / "pair_correlators.h5")


# ============================================================================
# Rashba spin-orbit model
# ============================================================================

def build_rashba_soc(ctx: BuildContext) -> None:
    """3x3 honeycomb Hubbard with Rashba spin-orbit coupling.

    The only model system whose one-body term is complex, and the reason the
    noncollinear code path has a model test at all. Hamiltonian and trial share
    a single file, which is what ``functional_cases.py`` expects.

    Rashba is not an input key of ``LatticeHamiltonian.from_dict``, so the
    builder is driven directly here.
    """
    from safiretools import HamiltonianBuilder, Lattice, SpinSymm, Wavefunction

    t = 1.0
    U = 1.0
    rashba_lambda = 0.1 * np.sqrt(3.0)
    nelec = (4, 4)

    filename = ctx.out_dir / "afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5"

    lattice = Lattice.from_dict({
        "type": "honeycomb",
        "L1": 3, "L2": 3,
        "boundary1": "PBC", "boundary2": "PBC",
    })

    builder = HamiltonianBuilder(lattice=lattice, spin_symm=SpinSymm.NONCOLLINEAR)
    builder.nth_neighbor_hopping(t)
    # rashba_lambda goes positionally: the @skip_empty_params decorator claims
    # the first positional argument.
    builder.rashba_soc(rashba_lambda, t=t)
    builder.onsite_hubbard(U)
    builder.finalize()
    hamiltonian = builder.get_hamiltonian()

    hamiltonian.to_hdf5(filename)

    # No twist here, unlike _write_free_electron: the spin-orbit term already
    #   splits the shells, so the determinant is well defined without one.
    Wavefunction.from_free_electron(
        hamiltonian, nelec=nelec,
        spin_symm=SpinSymm.NONCOLLINEAR).to_hdf5(filename)


# ============================================================================
# Registry
# ============================================================================

def recipes() -> List[Recipe]:
    return [
        Recipe(key="hubbard", data_dir="square_4x4_hubbard_nup5_ndn5",
               build=build_hubbard_4x4),
        Recipe(key="hubbard_kanamori",
               data_dir="square_6x1_hubbard_kanamori_nup6_ndn6",
               build=build_hubbard_kanamori),
        Recipe(key="rashba_soc", data_dir="rashba_soc",
               build=build_rashba_soc),
    ]
