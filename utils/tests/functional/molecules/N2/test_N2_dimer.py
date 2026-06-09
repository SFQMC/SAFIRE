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
Define functional tests for N2 dimer.
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
    get_all_rules,
    generate_param_list,
    should_run_successfully,
    should_exit_werror,
    should_run_warn,
)

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
        "cas_basis_hamil_closed" :  AFQMCHamiltonian(
            INPUTS_DIR/"cas_basis_hamil.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.GENERIC_DENSE
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
        "cas_wfn_collinear": AFQMCWavefunction(
            INPUTS_DIR/"cas_wfn.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.PHMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
    ]


# Use centralized generate_param_list
def generate_test_params():
    """Generate test parameters using the centralized infrastructure."""
    return generate_param_list(
        hamil_type_dictionary=hamil_type_dictionary(),
        wavefunction_type_dictionary=wavefunction_type_dictionary(),
        walker_type_list=walker_type_list(),
        ref_data_dir=REF_DATA_DIR
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


# Weekly tests
@pytest.mark.dev
@pytest.mark.functional
@pytest.mark.push
@pytest.mark.weekly
@pytest.mark.parametrize("case", should_run_successfully(generate_test_params()))
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

@pytest.mark.dev
@pytest.mark.functional
@pytest.mark.push
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
def test_expected_warn(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path
    ):
    # Use centralized should_run_warn function
    cases = should_run_warn(generate_test_params())
    raise NotImplementedError
