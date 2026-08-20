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

These are the cheap ones - pure afqmctools, no external codes, seconds rather
than minutes - and they cover the Hubbard, Hubbard-Kanamori and Rashba
spin-orbit models.

Two of the four directories carry files that only the C++ unit tests read:
the ``hst_type`` variants under ``square_4x4_hubbard_nup5_ndn5`` and the whole
``square_2x2_hubbard_Beta3_nt100`` directory. They are regenerated here anyway,
because the point of this tool is that the inputs tree can be rebuilt in full.
"""

import shutil
from typing import Dict, List

import numpy as np

from . import BuildContext, Recipe
from ._common import ASSETS, copy_groups

FINITE_T_ASSETS = ASSETS / "finiteT"

# The free-electron trial is written straight from the one-body term, so the
# twist is what keeps it from being degenerate at a closed shell. afqmctools
# picks a small irrational twist by default and that is what the committed
# wavefunctions were built with; nothing here overrides it.


# ============================================================================
# Helpers
# ============================================================================

def _write_model_hamiltonian(model: Dict, filename, *, spin_symm, nelec=None,
                             verbose: bool = False) -> None:
    """Build one lattice-model hamiltonian in ``spin_symm`` and write it.

    ``nelec=None`` writes ``(0, 0)`` into ``Hamiltonian/dims``. That is not an
    oversight: the hamiltonian-only files consumed by the C++ unit tests were
    written that way, and the electron count for those runs comes from the
    wavefunction instead.
    """
    from copy import deepcopy

    from afqmctools.hamiltonian.model.director import HamiltonianDirector
    from afqmctools.utils.io import write_model_hamiltonian

    local = deepcopy(model)
    local["hamiltonian"]["spin_symm"] = spin_symm

    hamiltonian = HamiltonianDirector(local).build()
    write_model_hamiltonian(hamiltonian=hamiltonian, fname=filename, nelec=nelec)
    return hamiltonian


def _write_free_electron(model: Dict, filename, *, spin_symm, nelec,
                         norb: int, verbose: bool = False) -> None:
    """Write a free-electron NOMSD trial for ``model``.

    The model dict is copied first: ``free_electron`` writes its twist back into
    ``source["lattice"]``, so passing the shared dict would leave a twist behind
    for whichever recipe step ran next.
    """
    from copy import deepcopy

    from afqmctools.wavefunction.common import write_wfn
    from afqmctools.wavefunction.free_electron import free_electron

    wfn, _ = free_electron(
        source=deepcopy(model),
        nelec=nelec,
        spin_symm=spin_symm,
        # The variational-energy measurement is a diagnostic that pulls in jax
        # and does not touch the wavefunction being written.
        measure_evar=verbose,
    )
    write_wfn(wfn=wfn, filename=filename, walker_type=spin_symm,
              nelec=nelec, norb=norb)


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
    from afqmctools.utils.types import SpinSymm

    out = ctx.out_dir
    nelec = HUBBARD_4X4["misc_params"]["nelec"]
    norb = 16

    name = {SpinSymm.CLOSED: "closed",
            SpinSymm.COLLINEAR: "collinear",
            SpinSymm.NONCOLLINEAR: "noncollinear"}

    for spin_symm in (SpinSymm.CLOSED, SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR):
        _write_model_hamiltonian(HUBBARD_4X4, out / f"ham_{name[spin_symm]}.h5",
                                 spin_symm=spin_symm, nelec=nelec,
                                 verbose=ctx.verbose)

    for spin_symm in (SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR):
        _write_free_electron(HUBBARD_4X4, out / f"wfn_fe_{name[spin_symm]}.h5",
                             spin_symm=spin_symm, nelec=nelec, norb=norb,
                             verbose=ctx.verbose)

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
    continuous_spin = {
        "hamiltonian": {"t": 1.0, "U": 6.0,
                        "hst_types": {"U": "continuous_spin"}},
        "lattice": HUBBARD_4X4["lattice"],
        "misc_params": HUBBARD_4X4["misc_params"],
    }
    _write_model_hamiltonian(continuous_spin, out / "ham_collinear_cont_spin.h5",
                             spin_symm=SpinSymm.COLLINEAR, nelec=None,
                             verbose=ctx.verbose)

    attractive = {
        "hamiltonian": {"t": 1.0, "U": -4.0},
        "lattice": HUBBARD_4X4["lattice"],
        "misc_params": HUBBARD_4X4["misc_params"],
    }
    _write_model_hamiltonian(attractive,
                             out / "ham_collinear_Um4_disc_charge.h5",
                             spin_symm=SpinSymm.COLLINEAR, nelec=None,
                             verbose=ctx.verbose)

    attractive_continuous = {
        "hamiltonian": {"t": 1.0, "U": -4.0,
                        "hst_types": {"U": "continuous_charge"}},
        "lattice": HUBBARD_4X4["lattice"],
        "misc_params": HUBBARD_4X4["misc_params"],
    }
    _write_model_hamiltonian(attractive_continuous,
                             out / "ham_collinear_Um4_cont_charge.h5",
                             spin_symm=SpinSymm.COLLINEAR, nelec=None,
                             verbose=ctx.verbose)

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
    from afqmctools.hamiltonian.model.director import HamiltonianDirector
    from afqmctools.utils.types import SpinSymm
    from afqmctools.wavefunction.common import write_wfn
    from afqmctools.wavefunction.free_electron import free_electron

    try:
        from autohf.hamiltonian import AutoHFHamiltonian
        from autohf.solver import lattice_hf
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

    hamiltonian = HamiltonianDirector(weak).build()

    # Untwisted, so the starting determinant - and the solution - stays real.
    free_wfn, _ = free_electron(source=hamiltonian, nelec=nelec,
                                spin_symm=SpinSymm.COLLINEAR, measure_evar=False)
    initial = free_wfn[1][0]

    results = lattice_hf(
        AutoHFHamiltonian(source=hamiltonian),
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
        initial_guess=initial,
        suppress_logo=True,
    )
    data = results[0] if isinstance(results, tuple) else results

    # AutoHF hands back (spin, norb, nelec_per_spin); write_wfn wants a single
    # determinant laid out as (ndet, norb, nup + ndn) with beta after alpha.
    alpha, beta = np.asarray(data["orbitals"])
    _check_spans_free_electron(alpha, initial[:, :nelec[0]], "alpha")
    _check_spans_free_electron(beta, initial[:, nelec[0]:], "beta")

    phi = np.zeros((1, norb, sum(nelec)), dtype=np.complex128)
    phi[0, :, :nelec[0]] = alpha
    phi[0, :, nelec[0]:] = beta

    write_wfn(
        filename=filename,
        wfn=(np.array([1.0], dtype=np.complex128), phi),
        walker_type=SpinSymm.COLLINEAR,
        nelec=nelec,
        norb=norb,
    )


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
    from afqmctools.utils.types import SpinSymm

    out = ctx.out_dir
    nelec = HUBBARD_KANAMORI_6X1["misc_params"]["nelec"]
    norb = 12  # 6 sites x 2 bands

    for spin_symm, name in ((SpinSymm.COLLINEAR, "collinear"),
                            (SpinSymm.NONCOLLINEAR, "noncollinear")):
        _write_model_hamiltonian(HUBBARD_KANAMORI_6X1, out / f"ham_{name}.h5",
                                 spin_symm=spin_symm, nelec=nelec,
                                 verbose=ctx.verbose)
        _write_free_electron(HUBBARD_KANAMORI_6X1, out / f"wfn_fe_{name}.h5",
                             spin_symm=spin_symm, nelec=nelec, norb=norb,
                             verbose=ctx.verbose)


# ============================================================================
# Rashba spin-orbit model
# ============================================================================

def build_rashba_soc(ctx: BuildContext) -> None:
    """3x3 honeycomb Hubbard with Rashba spin-orbit coupling.

    The only model system whose one-body term is complex, and the reason the
    noncollinear code path has a model test at all. Hamiltonian and trial share
    a single file, which is what ``functional_cases.py`` expects.

    Rashba is not one of the director's build steps, so the builder is driven
    directly here.
    """
    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    from afqmctools.systems.lattice import get_lattice
    from afqmctools.utils.io import write_model_hamiltonian
    from afqmctools.utils.types import SpinSymm
    from afqmctools.wavefunction.common import write_wfn
    from afqmctools.wavefunction.free_electron import free_electron

    t = 1.0
    U = 1.0
    rashba_lambda = 0.1 * np.sqrt(3.0)
    nelec = (4, 4)

    filename = ctx.out_dir / "afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5"

    lattice = get_lattice(params={
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
    hamiltonian = builder.hamiltonian

    write_model_hamiltonian(hamiltonian=hamiltonian, fname=filename, nelec=nelec)

    norb = hamiltonian.nbands * hamiltonian.nsites
    wfn, _ = free_electron(
        source=hamiltonian,
        nelec=nelec,
        spin_symm=SpinSymm.NONCOLLINEAR,
        measure_evar=ctx.verbose,
    )
    write_wfn(wfn=wfn, filename=filename, walker_type=SpinSymm.NONCOLLINEAR,
              nelec=nelec, norb=norb)


# ============================================================================
# 2x2 Hubbard at finite temperature (C++ unit tests only)
# ============================================================================

# On a 2x2 periodic lattice each pair of sites is connected twice (both
# directions wrap), so the builder sums the two bonds and t = 1 gives the
# hopping amplitude of -2 stored in the committed file.
HUBBARD_2X2 = {
    "hamiltonian": {"t": 1.0, "U": 4.0},
    "lattice": {"L1": 2, "L2": 2, "boundary1": "PBC", "boundary2": "PBC"},
    "misc_params": {"nelec": (2, 2)},
}


def build_hubbard_2x2_finite_t(ctx: BuildContext) -> None:
    """Hamiltonian for the finite-temperature C++ unit test fixture.

    Only the hamiltonian is computed. Two pieces of this directory have no
    generator anywhere in this repository, so they are checked in under
    ``assets/finiteT/`` and grafted on here:

    - ``wfn_collinear.h5``, a thermal propagator factorisation
      (UL/UR, VL/VR, DL/DR blocks) in a format afqmctools does not write.
    - the ``TEST_RESULTS`` group inside ``ham_collinear.h5``, holding the
      expected E1/EJ/EXX/VHS/vbias arrays the unit test asserts against.

    Copying ``TEST_RESULTS`` onto a freshly computed hamiltonian is only sound
    while that hamiltonian still matches the one the numbers were computed
    from, so run this recipe and read its diff before trusting a rebuild.
    """
    from afqmctools.utils.types import SpinSymm

    hamiltonian = ctx.out_dir / "ham_collinear.h5"
    _write_model_hamiltonian(HUBBARD_2X2, hamiltonian,
                             spin_symm=SpinSymm.COLLINEAR,
                             nelec=HUBBARD_2X2["misc_params"]["nelec"],
                             verbose=ctx.verbose)

    copy_groups(FINITE_T_ASSETS / "test_results.h5", hamiltonian, ["TEST_RESULTS"])
    shutil.copy(FINITE_T_ASSETS / "wfn_collinear.h5",
                ctx.out_dir / "wfn_collinear.h5")


# ============================================================================
# Registry
# ============================================================================

def recipes() -> List[Recipe]:
    return [
        Recipe(
            key="hubbard",
            data_dir="square_4x4_hubbard_nup5_ndn5",
            description="4x4 Hubbard at U/t = 6 with five electrons per spin, "
                        "plus the Hubbard-Stratonovich variants",
            produces=[
                "ham_closed.h5",
                "ham_collinear.h5",
                "ham_noncollinear.h5",
                "wfn_fe_collinear.h5",
                "wfn_fe_noncollinear.h5",
                "ham_collinear_cont_spin.h5",
                "ham_collinear_Um4_disc_charge.h5",
                "ham_collinear_Um4_cont_charge.h5",
                "uhf_U0.1_wfn_nup5_ndn5.h5",
            ],
            build=build_hubbard_4x4,
            notes="uhf_U0.1_wfn_nup5_ndn5.h5 comes from a variational solve; it "
                  "lands on the same determinant as the committed file but in a "
                  "different orbital gauge and scaling.",
        ),
        Recipe(
            key="hubbard_kanamori",
            data_dir="square_6x1_hubbard_kanamori_nup6_ndn6",
            description="two-band Hubbard-Kanamori chain, six electrons per spin",
            produces=[
                "ham_collinear.h5",
                "ham_noncollinear.h5",
                "wfn_fe_collinear.h5",
                "wfn_fe_noncollinear.h5",
            ],
            build=build_hubbard_kanamori,
            notes="reproduces the committed files apart from "
                  "maximum_connectivity, which write_model_hamiltonian now "
                  "floors at 12; the committed files predate that.",
        ),
        Recipe(
            key="rashba_soc",
            data_dir="rashba_soc",
            description="3x3 honeycomb Hubbard with Rashba spin-orbit coupling; "
                        "hamiltonian and trial share one file",
            produces=["afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5"],
            build=build_rashba_soc,
            notes="the trial's occupied shell is degenerate, so the orbitals "
                  "come out rotated relative to the committed file while "
                  "spanning exactly the same space.",
        ),
        Recipe(
            key="hubbard_2x2_finite_t",
            data_dir="square_2x2_hubbard_Beta3_nt100",
            description="2x2 Hubbard hamiltonian for the finite-temperature "
                        "C++ unit tests (not used by any functional case)",
            produces=["ham_collinear.h5", "wfn_collinear.h5"],
            build=build_hubbard_2x2_finite_t,
            notes="wfn_collinear.h5 and the TEST_RESULTS group have no "
                  "generator; they are copied from assets/finiteT/ rather than "
                  "recomputed, so only the hamiltonian is really rebuilt.",
        ),
    ]
