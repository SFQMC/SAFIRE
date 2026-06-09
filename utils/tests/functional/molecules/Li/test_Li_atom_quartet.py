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
Define functional tests for Li atom in quartet state.
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
    should_run_successfully_bp,
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
        "hamil_closed" :  AFQMCHamiltonian(
            INPUTS_DIR/"hamil_closed.h5",
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
        "rohf_wfn_fullypolarized": AFQMCWavefunction(
            INPUTS_DIR/"rohf_nomsd_fullypolarized.h5",
            SpinSymm.FULLYPOLARIZED,
            WavefunctionClass.NOMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("FULLYPOLARIZED",SpinSymm.FULLYPOLARIZED)
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


# Back-propagation case selection is centralized in test_infrastructure.should_run_successfully_bp


def generate_param_list(
        hamil_type_dictionary=hamil_type_dictionary(),
        wavefunction_type_dictionary=wavefunction_type_dictionary(),
        walker_type_list=walker_type_list(),
        ref_data_dir=REF_DATA_DIR
    ):
    """
    Simple function to generate the full list of combinations of Hamiltonian types,
      wavefunction types, and walker types for use with the pytest.mark.parameterize decorator.
    """
    params = []
    for h_name in hamil_type_dictionary:
        hamil_path = ref_data_dir/ h_name
        if not hamil_path.exists:
            warn(f"requested a test against non-existant Hamiltonian directory: {hamil_path}; SKIPPING!")
            continue
        for trial_wfn_name in wavefunction_type_dictionary:
            trial_wfn_path = hamil_path / trial_wfn_name
            if not trial_wfn_path.exists:
                warn(f"requested a test against non-existant wavefunction directory: {trial_wfn_path}; SKIPPING!")
                continue
            for walker in walker_type_list:
                walker_path = trial_wfn_path / walker.name.lower()
                if not walker_path.exists:
                    warn(f"requested a test against non-existant walker type directory: {walker_path}; SKIPPING!")
                    continue
                params.append(
                    AFQMCInputSet(
                        hamiltonian=hamil_type_dictionary[h_name],
                        wavefunction=wavefunction_type_dictionary[trial_wfn_name],
                        walker=walker,
                        reference=walker_path/"results.h5"
                    )
                )
    return params


def should_run_successfully(all_cases=generate_param_list()):
    rules = get_all_rules()
    return [ case for case in all_cases if all( (rule(case) for rule in rules) ) ]

### Removed local should_run_successfully_bp in favor of centralized version


def should_exit_werror(all_cases=generate_param_list()):
    rules = get_all_rules()
    return [ case for case in all_cases if any(( not rule(case) for rule in rules )) ]

def should_run_warn(all_cases=generate_param_list()):
    """Generates list of cases where SAFIRE should run with a warning
    """
    rules = []
    return [ case for case in all_cases if any(( not rule(case) for rule in rules )) ]


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

## Back-propagation tests - be sparing with these, they are expensive! ##
##   only testing expected success cases with back-propagation, there is
##  nothing unique about BP and expected failure cases.
@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case", should_run_successfully_bp(generate_test_params()))
def test_success_weekly_bp(
    afqmc_helper,
    result_checker_bp,
    afqmc_runmode,
    tmp_path,
    case,
    run_bp=True
    ):
    assert _expected_success(
        afqmc_helper,
        result_checker_bp,
        afqmc_runmode,
        tmp_path,
        case,
        run_bp=run_bp
    )
