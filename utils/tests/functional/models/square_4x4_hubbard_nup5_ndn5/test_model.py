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
Define functional tests for Square lattice Hubbard
  for the closed-shell Nup=down=5 case
"""
"""
Define functional tests for 4x4 Hubbard model.
"""
from pathlib import Path
from warnings import warn
from typing import List, Optional


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
    get_all_rules,
    generate_param_list,
    should_run_successfully,
    should_exit_werror,
    should_run_warn,
    should_run_successfully_bp,
)


# Test-specific paths
TEST_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = TEST_ROOT / "afqmc_inputs"
REF_DATA_DIR = TEST_ROOT / "afqmc_ref_runs"


def hamil_type_dictionary():
    """Generates and returns a dict of Hamiltonian
    types to test against. the dict is structured using
    a descriptive name for the Hamiltonian as a key, and 
    the value is the path to the HDF5 file (including it's name)
    in SAFIRE format that contains that wavefunction.
    """
    return {
        "ham_closed" :  AFQMCHamiltonian(
            INPUTS_DIR/"ham_closed.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.MODEL
        ),
        "ham_collinear" :  AFQMCHamiltonian(
            INPUTS_DIR/"ham_collinear.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.MODEL
        ),
        "ham_collinear_cont_spin" :  AFQMCHamiltonian(
            INPUTS_DIR/"ham_collinear_cont_spin.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.MODEL
        ),
        "ham_noncollinear" :  AFQMCHamiltonian(
            INPUTS_DIR/"ham_noncollinear.h5",
            SpinSymm.NONCOLLINEAR,
            HamiltonianClass.MODEL
        )   
    }

def wavefunction_type_dictionary():
    """Generates and returns a dict of wavefunction
    types to test against. the dict is structured using
    a descriptive name for the wavefunction as a key, and 
    the value is the path to the HDF5 file (including it's name)
    in SAFIRE format that contains that wavefunction.
    """
    return {
        "fe_collinear": AFQMCWavefunction(
            INPUTS_DIR/"wfn_fe_collinear.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "fe_noncollinear": AFQMCWavefunction(
            INPUTS_DIR/"wfn_fe_noncollinear.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("CLOSED",SpinSymm.CLOSED),
        AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        AFQMCWalker("NONCOLLINEAR",SpinSymm.NONCOLLINEAR),
    ]

# specific test cases
ExtraInputSetList = [
    AFQMCInputSet( # expected to succeed
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"ham_collinear_Um4_disc_charge.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.MODEL
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"uhf_U0.1_wfn_nup5_ndn5.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        reference=REF_DATA_DIR/"ham_collinear_disc_charge/hf_U0.1_collinear/collinear/results.h5"
    ),
    AFQMCInputSet( # expected to succeed
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"ham_collinear_Um4_cont_charge.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.MODEL
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"uhf_U0.1_wfn_nup5_ndn5.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        reference=REF_DATA_DIR/"ham_collinear_cont_charge/hf_U0.1_collinear/collinear/results.h5"
    )
]

# Use centralized generate_param_list
def generate_test_params(extra_input_set: Optional[List[AFQMCInputSet]] = None) -> List[AFQMCInputSet]:
    """Generate test parameters using the centralized infrastructure."""
    return generate_param_list(
        hamil_type_dictionary=hamil_type_dictionary(),
        wavefunction_type_dictionary=wavefunction_type_dictionary(),
        walker_type_list=walker_type_list(),
        ref_data_dir=REF_DATA_DIR,
        extra_input_set=extra_input_set
    )

def _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case,
        run_bp = False
    ):
    """
    Implements the test for the expected success cases
    """
    print(case)
    afqmc_helper.run_afqmc(
        run_path=tmp_path,
        fname="afqmc.json",
        hamil_file=case.hamiltonian.path,
        wfn_file=case.wavefunction.path,
        walker_type=case.walker.name,
        afqmc_runmode=afqmc_runmode,
        run_bp=run_bp
    )
    return result_checker.results_are_same(
        fname_test=tmp_path/"results.h5",
        fname_ref=case.reference,
        )

def _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case,
        run_bp = False
    ):
    """
    Implements the test for the expected failure cases
    """
    print(case)
    afqmc_helper.run_afqmc(
        run_path=tmp_path,
        fname="afqmc.json",
        hamil_file=case.hamiltonian.path,
        wfn_file=case.wavefunction.path,
        walker_type=case.walker.name,
        afqmc_runmode=afqmc_runmode,
        run_bp=run_bp
    )
    return result_checker.same_error(
        fname_test=tmp_path/"results.h5",
        fname_ref=case.reference,
        )

# per push tests
@pytest.mark.functional
@pytest.mark.push
@pytest.mark.parametrize("case",[
    AFQMCInputSet(
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"ham_closed.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.MODEL
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"wfn_fe_collinear.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        reference=REF_DATA_DIR/"ham_closed/fe_collinear/collinear/results.h5"
    )
])
def test_success_push(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.functional
@pytest.mark.push
@pytest.mark.parametrize("case",[
    AFQMCInputSet(
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"ham_collinear.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.MODEL
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"wfn_fe_collinear.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("NONCOLLINEAR",SpinSymm.NONCOLLINEAR),
        reference=REF_DATA_DIR/"ham_closed/fe_noncollinear/collinear/results.h5"
    )
])
def test_fail_push(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

# Weekly tests
@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case", should_run_successfully(generate_test_params(extra_input_set=ExtraInputSetList)))
def test_success_weekly(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case", should_exit_werror(generate_test_params()))
def test_fail_weekly(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.skip
@pytest.mark.dev
def test_expected_warn(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    should_run_wwarn
    ):
    raise NotImplementedError

## Back-propagation tests - be sparing with these, they are expensive! ##
##   only testing expected success cases with back-propagation, there is
##  nothing unique about BP and expected failure cases.
@pytest.mark.dev
@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case", should_run_successfully_bp(generate_test_params()))
def test_success_weekly_bp(
    afqmc_helper,
    result_checker_bp,
    afqmc_runmode,
    tmp_path,
    case,
    run_bp: bool = True,
):
    assert _expected_success(
        afqmc_helper,
        result_checker_bp,
        afqmc_runmode,
        tmp_path,
        case,
        run_bp=run_bp,
    )
