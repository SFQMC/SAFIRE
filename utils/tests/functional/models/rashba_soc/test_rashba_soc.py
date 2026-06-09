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
Defines functional tests for the Rashba SOC model.
"""
from pathlib import Path
from warnings import warn

import pytest

from afqmctools.hamiltonian.model.ham_class import SpinSymm

# Import centralized testing infrastructure
from dev_tools.test_infrastructure import (
    HamiltonianClass,
    WavefunctionClass,
    AFQMCHamiltonian,
    AFQMCWavefunction,
    AFQMCWalker,
    AFQMCInputSet,
)

THIS_TEST_DIR = Path(__file__).resolve().parent
INPUTS_DIR = THIS_TEST_DIR / "afqmc_inputs"
REF_DATA_DIR = THIS_TEST_DIR / "afqmc_ref_runs"

@pytest.mark.dev
@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case",[
    AFQMCInputSet(
        hamiltonian=AFQMCHamiltonian(
            path=INPUTS_DIR/"afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5",
            spin_symm=SpinSymm.NONCOLLINEAR,
            type=HamiltonianClass.MODEL
        ),
        wavefunction=AFQMCWavefunction(
            path=INPUTS_DIR/"afqmc_U1.0_lambda0.1sqrt3_free_elec_trial.h5",
            spin_symm=SpinSymm.NONCOLLINEAR,
            type=WavefunctionClass.NOMSD
        ),
        walker=AFQMCWalker(
            name="NONCOLLINEAR",
            spin_symm=SpinSymm.NONCOLLINEAR
        ),
        reference=REF_DATA_DIR/ Path("lambda0.1sqrt3_noncollinear/ghf_nomsd/noncollinear") / "results_free_elec_trial.h5"
    )
])
def test_all_cases(afqmc_helper,result_checker,afqmc_runmode,tmp_path,case):
    """
    Runs all test cases based on available inputs and reference data.
    """
    warn("Only minimal tests are implemented.")
    afqmc_helper.run_afqmc(
        run_path=tmp_path,
        fname="afqmc.json",
        hamil_file=case.hamiltonian.path,
        wfn_file=case.wavefunction.path,
        walker_type=case.walker.name,
        afqmc_runmode=afqmc_runmode
    )
    assert result_checker.results_are_same(
        fname_test=tmp_path/"results.h5",
        fname_ref=case.reference,
        )
