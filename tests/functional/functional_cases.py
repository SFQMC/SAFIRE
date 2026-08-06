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
imports :func:`build_systems` from here. Keeping the case list separate keeps
the runner focused on scheduling and checking logic.
"""

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from afqmctools.utils.types import SpinSymm

ROOT = Path(__file__).resolve().parent
INPUTS_ROOT = ROOT / "afqmc_inputs"
REFERENCES_ROOT = ROOT / "statistical_references"
SNAPSHOT_REFERENCES_ROOT = ROOT / "snapshot_references"


# ============================================================================
# Types
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
class System:
    """One set of cases: every hamiltonian x wavefunction x walker combination.

    The dict keys name the reference/output path components, so a group of runs
    that is not a full cross product belongs in its own ``System``.
    """

    data_dir: str        # per-system dir under afqmc_inputs/ and the reference roots
    hamiltonians: Dict[str, Hamiltonian]
    wavefunctions: Dict[str, Wavefunction]
    walkers: List[SpinSymm]  # the member name also names the reference path component
                             # and the walker_type of the generated input
    bp: bool = False     # whether this system has back-propagation tests


# ============================================================================
# Registry: every functional system, collected in one place
# ============================================================================

def build_systems() -> Dict[str, System]:
    HC, WC = HamiltonianClass, WavefunctionClass
    S = SpinSymm
    return {
        "N2": System(
            data_dir="N2",
            hamiltonians={"cas_basis_hamil_closed": Hamiltonian("cas_basis_hamil.h5", S.CLOSED, HC.GENERIC_DENSE)},
            wavefunctions={"cas_wfn_collinear": Wavefunction("cas_wfn.h5", S.COLLINEAR, WC.PHMSD)},
            walkers=[S.COLLINEAR],
        ),
        "BH": System(
            data_dir="BH",
            hamiltonians={
                "dense_rhf_basis_closed": Hamiltonian("afqmc_H_rhf_closed.h5", S.CLOSED, HC.GENERIC_DENSE),
                "dense_rhf_basis_collinear": Hamiltonian("afqmc_H_rhf_collinear.h5", S.COLLINEAR, HC.GENERIC_DENSE),
                "dense_rhf_basis_noncollinear": Hamiltonian("afqmc_H_rhf_noncollinear.h5", S.NONCOLLINEAR, HC.GENERIC_DENSE),
            },
            wavefunctions={
                "rhf_nomsd": Wavefunction("afqmc_rhf_nomsd.h5", S.CLOSED, WC.NOMSD),
                "uhf_nomsd": Wavefunction("afqmc_uhf_nomsd.h5", S.COLLINEAR, WC.NOMSD),
                "ghf_nomsd": Wavefunction("afqmc_ghf_nomsd.h5", S.NONCOLLINEAR, WC.NOMSD),
                "rcasci_rhf_phmsd": Wavefunction("afqmc_casci_rhf_phmsd.h5", S.CLOSED, WC.PHMSD),
                "rcasci_uhf_phmsd": Wavefunction("afqmc_casci_uhf_phmsd.h5", S.COLLINEAR, WC.PHMSD),
                "rcasci_ghf_phmsd": Wavefunction("afqmc_casci_ghf_phmsd.h5", S.NONCOLLINEAR, WC.PHMSD),
                "rcasci_rhf_nomsd": Wavefunction("afqmc_casci_rhf_nomsd.h5", S.CLOSED, WC.NOMSD),
                "rcasci_uhf_nomsd": Wavefunction("afqmc_casci_uhf_nomsd.h5", S.COLLINEAR, WC.NOMSD),
                "rcasci_ghf_nomsd": Wavefunction("afqmc_casci_ghf_nomsd.h5", S.NONCOLLINEAR, WC.NOMSD),
                "rcasci_rhf_1phmsd": Wavefunction("afqmc_casci_rhf_1phmsd.h5", S.CLOSED, WC.PHMSD),
                "rcasci_uhf_1phmsd": Wavefunction("afqmc_casci_uhf_1phmsd.h5", S.COLLINEAR, WC.PHMSD),
                "rcasci_ghf_1phmsd": Wavefunction("afqmc_casci_ghf_1phmsd.h5", S.NONCOLLINEAR, WC.PHMSD),
            },
            walkers=[S.CLOSED, S.COLLINEAR, S.NONCOLLINEAR],
            bp=True,
        ),
        "Li": System(
            data_dir="Li",
            hamiltonians={"hamil_closed": Hamiltonian("hamil_closed.h5", S.CLOSED, HC.GENERIC_DENSE)},
            wavefunctions={"rohf_wfn_polarized": Wavefunction("rohf_nomsd_polarized.h5", S.COLLINEAR, WC.NOMSD)},
            walkers=[S.COLLINEAR],
            bp=True,
        ),
        "Pb": System(
            data_dir="Pb",
            hamiltonians={
                "dense_rhf_basis_noncollinear_sf": Hamiltonian("afqmc_H_rhf_basis_noncollinear_sf.h5", S.NONCOLLINEAR, HC.GENERIC_DENSE),
                "dense_rhf_basis_noncollinear_soc": Hamiltonian("afqmc_H_rhf_basis_noncollinear_soc.h5", S.NONCOLLINEAR, HC.GENERIC_DENSE),
            },
            wavefunctions={
                "uhf_nomsd": Wavefunction("afqmc_uhf_nomsd.h5", S.COLLINEAR, WC.NOMSD),
                "ghf_sf_nomsd": Wavefunction("afqmc_ghf_sf_nomsd.h5", S.NONCOLLINEAR, WC.NOMSD),
                "ghf_soc_nomsd": Wavefunction("afqmc_ghf_soc_nomsd.h5", S.NONCOLLINEAR, WC.NOMSD),
            },
            walkers=[S.CLOSED, S.COLLINEAR, S.NONCOLLINEAR],
        ),
        "hubbard": System(
            data_dir="square_4x4_hubbard_nup5_ndn5",
            hamiltonians={
                "ham_closed": Hamiltonian("ham_closed.h5", S.CLOSED, HC.MODEL),
                "ham_collinear": Hamiltonian("ham_collinear.h5", S.COLLINEAR, HC.MODEL),
                "ham_collinear_cont_spin": Hamiltonian("ham_collinear_cont_spin.h5", S.COLLINEAR, HC.MODEL),
                "ham_noncollinear": Hamiltonian("ham_noncollinear.h5", S.NONCOLLINEAR, HC.MODEL),
            },
            wavefunctions={
                "fe_collinear": Wavefunction("wfn_fe_collinear.h5", S.COLLINEAR, WC.NOMSD),
                "fe_noncollinear": Wavefunction("wfn_fe_noncollinear.h5", S.NONCOLLINEAR, WC.NOMSD),
            },
            walkers=[S.CLOSED, S.COLLINEAR, S.NONCOLLINEAR],
            bp=True,
        ),
        "hubbard_charge": System(
            data_dir="square_4x4_hubbard_nup5_ndn5",
            hamiltonians={
                "ham_collinear_disc_charge": Hamiltonian("ham_collinear_Um4_disc_charge.h5", S.COLLINEAR, HC.MODEL),
                "ham_collinear_cont_charge": Hamiltonian("ham_collinear_Um4_cont_charge.h5", S.COLLINEAR, HC.MODEL),
            },
            wavefunctions={
                "hf_U0.1_collinear": Wavefunction("uhf_U0.1_wfn_nup5_ndn5.h5", S.COLLINEAR, WC.NOMSD),
            },
            walkers=[S.COLLINEAR],
            bp=True,
        ),
        "hubbard_kanamori": System(
            data_dir="square_6x1_hubbard_kanamori_nup6_ndn6",
            hamiltonians={
                "ham_collinear": Hamiltonian("ham_collinear.h5", S.COLLINEAR, HC.MODEL),
                "ham_noncollinear": Hamiltonian("ham_noncollinear.h5", S.NONCOLLINEAR, HC.MODEL),
            },
            wavefunctions={
                "fe_collinear": Wavefunction("wfn_fe_collinear.h5", S.COLLINEAR, WC.NOMSD),
                "fe_noncollinear": Wavefunction("wfn_fe_noncollinear.h5", S.NONCOLLINEAR, WC.NOMSD),
            },
            walkers=[S.COLLINEAR, S.NONCOLLINEAR],
            bp=True,
        ),
        "rashba_soc": System(
            data_dir="rashba_soc",
            hamiltonians={
                "lambda0.1sqrt3_noncollinear": Hamiltonian("afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5", S.NONCOLLINEAR, HC.MODEL),
            },
            wavefunctions={
                "ghf_nomsd_free_elec_trial": Wavefunction("afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5", S.NONCOLLINEAR, WC.NOMSD),
            },
            walkers=[S.NONCOLLINEAR],
        ),
        "diamond": System(
            data_dir="C_diamond_coqui",
            hamiltonians={
                "ham_chol_closed": Hamiltonian("ham_chol_1e-5.h5", S.CLOSED, HC.KPFAC_CHOL),
                "ham_thc_closed": Hamiltonian("ham_thc_1e-6.h5", S.CLOSED, HC.THC),
            },
            wavefunctions={
                "pbe_closed_nomsd": Wavefunction("wfn_mf_pbe_closed.h5", S.CLOSED, WC.NOMSD),
                "pbe_collinear_nomsd": Wavefunction("wfn_mf_pbe.h5", S.COLLINEAR, WC.NOMSD),
                "pbe_collinear_nomsd_noncollinear": Wavefunction("wfn_mf_pbe_noncollinear.h5", S.NONCOLLINEAR, WC.NOMSD),
            },
            walkers=[S.CLOSED, S.COLLINEAR, S.NONCOLLINEAR],
            bp=True,
        ),
        "diamond_2x2x2": System(
            data_dir="C_diamond_coqui",
            hamiltonians={
                "ham_2x2x2_chol_closed": Hamiltonian("ham_2x2x2_chol_1e-5.h5", S.CLOSED, HC.KPFAC_CHOL),
            },
            wavefunctions={"pbe_wfn_2x2x2_collinear": Wavefunction("wfn_mf_2x2x2_pbe.h5", S.COLLINEAR, WC.NOMSD)},
            walkers=[S.COLLINEAR],
            bp=True,
        ),
    }
