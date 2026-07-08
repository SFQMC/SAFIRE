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
Registry of functional-test cases for ``run_functional.py``.

This module holds only the data model and the per-system definitions (the list
of Hamiltonian / wavefunction / walker inputs and reference paths). The runner
imports :func:`build_systems` and :data:`WALKERS` from here. Keeping the case
list separate keeps the runner focused on scheduling and checking logic.
"""

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from afqmctools.utils.types import SpinSymm

ROOT = Path(__file__).resolve().parent


# ============================================================================
# Types (enums mirror dev_tools.test_infrastructure so the registry reads alike)
# ============================================================================

class HamiltonianClass(enum.Enum):
    GENERIC_DENSE = enum.auto()
    MODEL = enum.auto()
    KPFAC_CHOL = enum.auto()
    THC = enum.auto()


class WavefunctionClass(enum.Enum):
    NOMSD = enum.auto()
    PHMSD = enum.auto()


@dataclass
class Hamiltonian:
    file: str            # filename inside the system's afqmc_inputs dir
    spin: SpinSymm
    type: HamiltonianClass
    runparams: dict = field(default_factory=dict)


@dataclass
class Wavefunction:
    file: str
    spin: SpinSymm
    type: WavefunctionClass
    runparams: dict = field(default_factory=dict)


@dataclass
class ExplicitCase:
    """A non-combinatorial case with its own reference path."""
    hamiltonian: Hamiltonian
    wavefunction: Wavefunction
    walker: str
    reference: str       # path relative to the system's ref dir


@dataclass
class System:
    subdir: str          # location under utils/tests/functional
    hamiltonians: Dict[str, Hamiltonian]
    wavefunctions: Dict[str, Wavefunction]
    walkers: List[str]
    ref_override: Optional[Path] = None   # absolute ref dir (e.g. external ceph)
    extra: List[ExplicitCase] = field(default_factory=list)
    bp: bool = False     # whether this system has back-propagation tests

    @property
    def inputs_dir(self) -> Path:
        return ROOT / self.subdir / "afqmc_inputs"

    @property
    def ref_dir(self) -> Path:
        return self.ref_override or (ROOT / self.subdir / "afqmc_ref_runs")


# The shared walker map: defined ONCE, referenced by name from each system.
WALKERS: Dict[str, SpinSymm] = {
    "CLOSED": SpinSymm.CLOSED,
    "COLLINEAR": SpinSymm.COLLINEAR,
    "NONCOLLINEAR": SpinSymm.NONCOLLINEAR,
    "FULLYPOLARIZED": SpinSymm.FULLYPOLARIZED,
}


# ============================================================================
# Registry: every functional system, collected in one place
# ============================================================================

def build_systems() -> Dict[str, System]:
    HC, WC = HamiltonianClass, WavefunctionClass
    CL, CO, NC, FP = (SpinSymm.CLOSED, SpinSymm.COLLINEAR,
                      SpinSymm.NONCOLLINEAR, SpinSymm.FULLYPOLARIZED)
    return {
        "N2": System(
            subdir="molecules/N2",
            hamiltonians={"cas_basis_hamil_closed": Hamiltonian("cas_basis_hamil.h5", CL, HC.GENERIC_DENSE)},
            wavefunctions={"cas_wfn_collinear": Wavefunction("cas_wfn.h5", CO, WC.PHMSD)},
            walkers=["COLLINEAR"],
        ),
        "BH": System(
            subdir="molecules/BH",
            hamiltonians={
                "dense_rhf_basis_closed": Hamiltonian("afqmc_H_rhf_closed.h5", CL, HC.GENERIC_DENSE),
                "dense_rhf_basis_collinear": Hamiltonian("afqmc_H_rhf_collinear.h5", CO, HC.GENERIC_DENSE),
                "dense_rhf_basis_noncollinear": Hamiltonian("afqmc_H_rhf_noncollinear.h5", NC, HC.GENERIC_DENSE),
            },
            wavefunctions={
                "rhf_nomsd": Wavefunction("afqmc_rhf_nomsd.h5", CL, WC.NOMSD),
                "uhf_nomsd": Wavefunction("afqmc_uhf_nomsd.h5", CO, WC.NOMSD),
                "ghf_nomsd": Wavefunction("afqmc_ghf_nomsd.h5", NC, WC.NOMSD),
                "rcasci_rhf_phmsd": Wavefunction("afqmc_casci_rhf_phmsd.h5", CL, WC.PHMSD),
                "rcasci_uhf_phmsd": Wavefunction("afqmc_casci_uhf_phmsd.h5", CO, WC.PHMSD),
                "rcasci_ghf_phmsd": Wavefunction("afqmc_casci_ghf_phmsd.h5", NC, WC.PHMSD),
                "rcasci_rhf_nomsd": Wavefunction("afqmc_casci_rhf_nomsd.h5", CL, WC.NOMSD),
                "rcasci_uhf_nomsd": Wavefunction("afqmc_casci_uhf_nomsd.h5", CO, WC.NOMSD),
                "rcasci_ghf_nomsd": Wavefunction("afqmc_casci_ghf_nomsd.h5", NC, WC.NOMSD),
                "rcasci_rhf_1phmsd": Wavefunction("afqmc_casci_rhf_1phmsd.h5", CL, WC.PHMSD),
                "rcasci_uhf_1phmsd": Wavefunction("afqmc_casci_uhf_1phmsd.h5", CO, WC.PHMSD),
                "rcasci_ghf_1phmsd": Wavefunction("afqmc_casci_ghf_1phmsd.h5", NC, WC.PHMSD),
            },
            walkers=["CLOSED", "COLLINEAR", "NONCOLLINEAR"],
            bp=True,
        ),
        "Li": System(
            subdir="molecules/Li",
            hamiltonians={"hamil_closed": Hamiltonian("hamil_closed.h5", CL, HC.GENERIC_DENSE)},
            wavefunctions={"rohf_wfn_fullypolarized": Wavefunction("rohf_nomsd_fullypolarized.h5", FP, WC.NOMSD)},
            walkers=["FULLYPOLARIZED"],
            bp=True,
        ),
        "Pb": System(
            subdir="molecules/Pb",
            hamiltonians={
                "dense_rhf_basis_noncollinear_sf": Hamiltonian("afqmc_H_rhf_basis_noncollinear_sf.h5", NC, HC.GENERIC_DENSE),
                "dense_rhf_basis_noncollinear_soc": Hamiltonian("afqmc_H_rhf_basis_noncollinear_soc.h5", NC, HC.GENERIC_DENSE),
            },
            wavefunctions={
                "uhf_nomsd": Wavefunction("afqmc_uhf_nomsd.h5", CO, WC.NOMSD),
                "ghf_sf_nomsd": Wavefunction("afqmc_ghf_sf_nomsd.h5", NC, WC.NOMSD),
                "ghf_soc_nomsd": Wavefunction("afqmc_ghf_soc_nomsd.h5", NC, WC.NOMSD),
            },
            walkers=["CLOSED", "COLLINEAR", "NONCOLLINEAR"],
        ),
        "square_4x4": System(
            subdir="models/square_4x4_hubbard_nup5_ndn5",
            hamiltonians={
                "ham_closed": Hamiltonian("ham_closed.h5", CL, HC.MODEL),
                "ham_collinear": Hamiltonian("ham_collinear.h5", CO, HC.MODEL),
                "ham_collinear_cont_spin": Hamiltonian("ham_collinear_cont_spin.h5", CO, HC.MODEL),
                "ham_noncollinear": Hamiltonian("ham_noncollinear.h5", NC, HC.MODEL),
            },
            wavefunctions={
                "fe_collinear": Wavefunction("wfn_fe_collinear.h5", CO, WC.NOMSD),
                "fe_noncollinear": Wavefunction("wfn_fe_noncollinear.h5", NC, WC.NOMSD),
            },
            walkers=["CLOSED", "COLLINEAR", "NONCOLLINEAR"],
            extra=[
                ExplicitCase(
                    Hamiltonian("ham_collinear_Um4_disc_charge.h5", CO, HC.MODEL),
                    Wavefunction("uhf_U0.1_wfn_nup5_ndn5.h5", CO, WC.NOMSD),
                    "COLLINEAR",
                    "ham_collinear_disc_charge/hf_U0.1_collinear/collinear/results.h5",
                ),
                ExplicitCase(
                    Hamiltonian("ham_collinear_Um4_cont_charge.h5", CO, HC.MODEL),
                    Wavefunction("uhf_U0.1_wfn_nup5_ndn5.h5", CO, WC.NOMSD),
                    "COLLINEAR",
                    "ham_collinear_cont_charge/hf_U0.1_collinear/collinear/results.h5",
                ),
            ],
            bp=True,
        ),
        "square_6x1": System(
            subdir="models/square_6x1_hubbard_kanamori_nup6_ndn6",
            hamiltonians={
                "ham_collinear": Hamiltonian("ham_collinear.h5", CO, HC.MODEL),
                "ham_noncollinear": Hamiltonian("ham_noncollinear.h5", NC, HC.MODEL),
            },
            wavefunctions={
                "fe_collinear": Wavefunction("wfn_fe_collinear.h5", CO, WC.NOMSD),
                "fe_noncollinear": Wavefunction("wfn_fe_noncollinear.h5", NC, WC.NOMSD),
            },
            walkers=["COLLINEAR", "NONCOLLINEAR"],
            bp=True,
        ),
        "rashba_soc": System(
            subdir="models/rashba_soc",
            hamiltonians={},
            wavefunctions={},
            walkers=[],
            extra=[
                ExplicitCase(
                    Hamiltonian("afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5", NC, HC.MODEL),
                    Wavefunction("afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5", NC, WC.NOMSD),
                    "NONCOLLINEAR",
                    "lambda0.1sqrt3_noncollinear/ghf_nomsd/noncollinear/results_free_elec_trial.h5",
                ),
            ],
        ),
        "diamond_coqui": System(
            subdir="solids/C_diamond_coqui",
            hamiltonians={
                "ham_chol_closed": Hamiltonian("ham_chol_1e-5.h5", CL, HC.KPFAC_CHOL,
                                               runparams={"total_walkers": 3200}),
                "ham_thc_closed": Hamiltonian("ham_thc_1e-6.h5", CL, HC.THC,
                                              runparams={"max_num_mpi_ranks": 16, "total_walkers": 3200}),
            },
            wavefunctions={
                "pbe_closed_nomsd": Wavefunction("wfn_mf_pbe_closed.h5", CL, WC.NOMSD),
                "pbe_collinear_nomsd": Wavefunction("wfn_mf_pbe.h5", CO, WC.NOMSD),
                "pbe_collinear_nomsd_noncollinear": Wavefunction("wfn_mf_pbe_noncollinear.h5", NC, WC.NOMSD),
            },
            walkers=["CLOSED", "COLLINEAR", "NONCOLLINEAR"],
            bp=True,
        ),
        "diamond_2x2x2": System(
            subdir="solids/C_diamond_coqui",
            hamiltonians={
                "ham_2x2x2_chol_closed": Hamiltonian("ham_2x2x2_chol_1e-5.h5", CL, HC.KPFAC_CHOL,
                                                     runparams={"steps": 6000, "total_walkers": 3200}),
            },
            wavefunctions={"pbe_wfn_2x2x2_collinear": Wavefunction("wfn_mf_2x2x2_pbe.h5", CO, WC.NOMSD)},
            walkers=["COLLINEAR"],
            bp=True,
        ),
    }
