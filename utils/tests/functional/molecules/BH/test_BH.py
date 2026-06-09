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
Define functional tests for BH molecule.
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

BH_TEST_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = BH_TEST_ROOT / "afqmc_inputs"
#REF_DATA_DIR = BH_TEST_ROOT / "afqmc_ref_runs"
REF_DATA_DIR = Path("/mnt/home/beskridge/ceph/proj/auto_testing/ref_data/BH_new/afqmc_runs")

def hamil_type_dictionary():
    """Generates and returns a dict of Hamiltonian
    types to test against. the dict is structured using
    a descriptive name for the Hamiltonian as a key, and 
    the value is the path to the HDF5 file (including it's name)
    in SAFIRE format that contains that wavefunction.
    """
    return {
        "dense_rhf_basis_closed" : AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_closed.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.GENERIC_DENSE
        ),
        "dense_rhf_basis_collinear" : AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_collinear.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
        "dense_rhf_basis_noncollinear" :  AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_noncollinear.h5",
            SpinSymm.NONCOLLINEAR,
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
        "rhf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_rhf_nomsd.h5",
            SpinSymm.CLOSED,
            WavefunctionClass.NOMSD
        ),
        "uhf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_uhf_nomsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "ghf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_ghf_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "rcasci_rhf_phmsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_rhf_phmsd.h5",
            SpinSymm.CLOSED,
            WavefunctionClass.PHMSD
        ),
        "rcasci_uhf_phmsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_uhf_phmsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.PHMSD
        ), 
        "rcasci_ghf_phmsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_ghf_phmsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.PHMSD
        ),
        "rcasci_rhf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_rhf_nomsd.h5",
            SpinSymm.CLOSED,
            WavefunctionClass.NOMSD
        ), 
        "rcasci_uhf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_uhf_nomsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "rcasci_ghf_nomsd" : AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_ghf_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        ), 
        # These are ph-mSD wavefunctions with a single SD
        "rcasci_rhf_1phmsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_rhf_1phmsd.h5",
            SpinSymm.CLOSED,
            WavefunctionClass.PHMSD
        ),
        "rcasci_uhf_1phmsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_uhf_1phmsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.PHMSD
        ),
        "rcasci_ghf_1phmsd" : AFQMCWavefunction(
            INPUTS_DIR/"afqmc_casci_ghf_1phmsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.PHMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("CLOSED",SpinSymm.CLOSED),
        AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        AFQMCWalker("NONCOLLINEAR",SpinSymm.NONCOLLINEAR),
    ]


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

### Back-propagation subset selection is centralized; using should_run_successfully_bp


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
        run_bp=False):
    """
    Implents the test for the expected success cases
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
        run_bp=False
    ):
    """
    Implents the test for the expected failure cases
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
            INPUTS_DIR/"afqmc_H_rhf_closed.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.GENERIC_DENSE
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"afqmc_uhf_nomsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker=AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        reference=REF_DATA_DIR/"dense_rhf_basis_closed/uhf_nomsd/collinear/results.h5"
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
            INPUTS_DIR/"afqmc_H_rhf_collinear.h5",
            SpinSymm.COLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"afqmc_rhf_nomsd.h5",
            SpinSymm.CLOSED,
            WavefunctionClass.NOMSD
        ),
        walker=AFQMCWalker("CLOSED",SpinSymm.CLOSED),
        reference=REF_DATA_DIR/"dense_rhf_basis_collinear/rhf_nomsd/closed/results.h5"
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
@pytest.mark.parametrize("case",should_run_successfully())
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
@pytest.mark.parametrize("case",should_exit_werror())
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
@pytest.mark.parametrize("case",should_run_successfully_bp(generate_param_list()))
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

