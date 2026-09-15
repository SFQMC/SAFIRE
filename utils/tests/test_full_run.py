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
Full Tests of the AFQMC code + setup tooling

author: Kyle Eskridge
GitHub: bkesk

Use PyTest to automate running full AFQMC calculations for a set of representative examples.

Key design notes:
1. tests are detected by the presence of a subdirectory within a specific test_files directory. 
    Each subdirectory is interpreted as a system to run tests on.
2. The AFQMC executable will be invoked to get real data. A consistent random seed should be used.
3. Both the path to the AFQMC executable and the test files directory can be set at run time via:
`$ AFQMC_EXEC=/path/to/afqmc/build/qmcapp TEST_FILE_DIR=/path/to/test_files` pytest ... 
This enables us to currate a set of public tests that live in the repo, and private internal tests 
that live elsewhere.

#TODO: re-organize this! A lot of this code would be generally useful outside of the tests. Maybe add an automation submodule?
"""
import os
from time import perf_counter_ns
from pathlib import Path
import subprocess as sp
import enum
import copy as cp

import pytest
import numpy as np
import pdb

from afqmctools.utils.io import read_input_params,write_model_hamiltonian
from stats.scalar_dat import analyze_scalar_data
from afqmctools.analysis.rdm import average_afqmc_rdm
from afqmctools.inputs.from_hdf import write_json
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

pytest.importorskip("optax")
pytest.importorskip("autohf")

from autohf import lattice_hf, AutoHFHamiltonian
from afqmctools.wavefunction.free_electron import free_electron


pytestmark = pytest.mark.skip("all tests still WIP")

# HERE
def test_dirs_list(test_files=os.environ.get('AFQMC_TEST_FILES')):
    if test_files is not None:
        test_files_path = Path(test_files)
    else:
        return []

    if not test_files_path.is_dir():
        raise ValueError(f"[Developers] test_dirs_list was given a non-directory argument")
    else:
        return [item for item in test_files_path.iterdir() if item.is_dir()]

# HERE
class _ExpectedResult:
    """
    Base class for the expected results from AFQMC tests.

    Concrete result classes are responsible for both storing the expected
    result, and for extracting results from the AFQMC test run.
    """

    def __init__(self) -> None:
        raise ValueError("[For Developers] Constructed from base _ExpectedResult class")
    
    def check_result(self,run_path:Path):
        return self._check_result(run_path)

# HERE
class DummyResult(_ExpectedResult):
    def __init__(self,name) -> None:
        self.name = name
    
    def _check_result(self,run_path):
        print(self)
        return True
    
    def __str__(self) -> str:
        return f"A Dummy Result: {self.name}"
        
# HERE
class InitialEnergyResult(_ExpectedResult):
    """
    Expected result class for the AFQMC energy.

    Uses stats.analyze_scalar_data(...) based on the settings
      in `stats_settings` to compute the energy.
    """
    def __init__(self,energy,stochastic_std_error,stats_settings:dict=None) -> None:
        self._energy=energy

    def _check_result(self,run_path):
        # comptue energy
        # TODO: regex to get initial energy
        pass
    
    def __str__(self) -> str:
        return f"Expected Energy: {self._energy} +/- {self._stochastic_std_error}"


class AFQMCEnergyResult(_ExpectedResult):
    """
    Expected result class for the AFQMC energy.

    Uses stats.analyze_scalar_data(...) based on the settings
      in `stats_settings` to compute the energy.
    """
    def __init__(self,energy,stochastic_std_error,stats_settings:dict=None) -> None:
        self._energy=energy
        self._stochastic_std_error=stochastic_std_error
        self._stats_settings=stats_settings

    def _check_result(self,run_path=None):
        if run_path is None:
            run_path= Path("./")

        # comptue energy
        y,y_err = analyze_scalar_data(dict(
            fname = self._stats_settings.get("fname",str(run_path/"qmc.s000.scalar.dat")),
            xaxis = self._stats_settings.get("xaxis","time"),
            nequil = self._stats_settings.get("nequil",0)
        ))

        print(self)
        print(f"actual AFQMC Energy result is {y} {y_err}")

        # tight criteria okay as long as we set seeds AND set number of nodes, etc.
        assert np.isclose(y,self._energy) 
        assert np.isclose(y_err,self._stochastic_std_error)
    
        return True

    def __str__(self) -> str:
        return f"Expected Energy: {self._energy} +/- {self._stochastic_std_error}"


class AFQMCErrorResult(_ExpectedResult):
    """
    Expecte result class for AFQMC errors.

    Used when a specific error message is expected from the AFQMC code.
    """
    def __init__(self,error_msg) -> None:
        self._error_msg = error_msg

    def _check_result(self,run_path):
        # use regex to search for error message
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Expected AFQMC Error Message: {self._error_msg}"


class AFQMCOneRdmResult(_ExpectedResult):
    """
    Expected result class for the AFQMC one-body reduced density matrix (1rdm)

    Uses afqmctools.analysis.rdm.average_afqmc_rdm(...) based on the settings
      in `stats_settings` to compute the 1rdm.
    """

    def __init__(self,rdm_file,stats_settings=None) -> None:
        self._rdm_file=rdm_file
        self._stats_settings=stats_settings

    def _check_result(self,run_path):
        # comptue 1rdm
        rho_avg, delta_rho = average_afqmc_rdm(**self._stats_settings)

        # TODO: compare to rho,drho in self._rdm_fle
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"Expected one-body reduced density matrix."

# HERE
def result_type_map(type:str):
    type = type.lower()

    if type in ("afqmc_energy","afqmc energy"):
        return AFQMCEnergyResult
    elif type in ("initial_energy","initial energy"):
        return InitialEnergyResult
    elif type in ("afqmc_error","afqmc error","error"):
        return AFQMCErrorResult
    elif type in ("1rdm","one_rdm","one rdm"):
        return AFQMCOneRdmResult
    elif type in ("dummy"):
        return DummyResult
    else:
        raise ValueError(f"Unknown result type: {type}")

# HERE
def parse_expeted_results(settings:dict):
    """
    parse expected results from settings dcitionary.

    Note: should pass full settings dictionary
    (i.e. not settings["expected_results"])
    """
    results = []
    for result_type,result_params in settings["expected_results"].items():
        results.append(result_type_map(result_type)(**result_params))
    return results


class AFQMCRunToken:
    """
    Simple class to determine if an AFQMC run has been performed.

    This class is intentionally un-copyable
    """

    def __init__(self,run_path=None) -> None:
        self._has_run = False
        self._run_path = run_path

    @property
    def has_run(self):
        return self._has_run

    @has_run.setter
    def has_run(self,value):
        self._has_run = value

    @has_run.deleter
    def has_run(self):
        del self._has_run

    @property
    def run_path(self):
        return self._run_path

    @run_path.setter
    def run_path(self,value):
        self._run_path = value

    @run_path.deleter
    def run_path(self):
        del self._run_path

    def __copy__(self):
        raise ValueError("Copying AFQMCRunToken is not allowed") 
    
    def __deepcopy__(self):
        raise ValueError("Copying AFQMCRunToken is not allowed") 


# HERE
class AFQMCTestCase:
    """
    A Container class to hold:
    - parameters for setup tooling (i.e. SCF checkpoint files, integral files, etc.)
    - AFQMC run parameters
    - expected results
    """

    def __init__(self,name,settings:dict=None,base_settings:dict=None) -> None:
        self.name = name
        
        if base_settings is None:
            raise ValueError("AFQMCTestCase requires base_settings to generate a test")
        
        # need a deep copy! otherwise, references to the sub-dictionaries are the same object!
        _settings = cp.deepcopy(base_settings)
        for key,value in settings.items():
            if key in _settings and isinstance(value,dict):
                _settings[key].update(value)
            else:
                _settings[key] = value
        if not _settings.get("expected_results",False):
            raise ValueError(
                "AFQMCTestCase requires expected results. "
                f"please add a `expected_results` block to afqmc_test.toml"
            )
        self.expected_results = parse_expeted_results(_settings)
        self.settings = _settings

        # NOTE: self._has_run needs to a type which is NOT copied in a shallow copy
        #         this allows AFQMCTestCases to be split into multple AFQMCTestCases
        #         while maintaining global awareness of whether AFQMC has been run yet
        #         for the given input parameters
        self.afqmc_run_token = AFQMCRunToken()

    def __str__(self) -> str:
        return self.name

    # I need a way to know if they AFQMC has been run already or not -> add an attribute to the test case
    #    that says "run" or "not run". Make sure that it isn't a simple type so that a shallow
    #     copy will reference the same object for all children of the original AFQMCTestCase
    def split(self):
        """
        'splits' AFQMCTestCase instance into a list of AFQMCTestCase instances
            where the list constains one AFQMCTestCase per expected_result in
            the original AFQMCTestCase.

        Always returns a list even if there is only one expected_result.

        New AFQMCTestCase instances are **shallow** copies of the orignal AFQMCTestCase.
        """
        new_test_cases = []
        for result in self.expected_results:
            new_test_case = cp.copy(self)
            new_test_case.expected_results = [result]
            new_test_cases.append( new_test_case )
        return new_test_cases

    def __deepcopy__(self):
        raise ValueError(
            "AFQMCTestCase may not be 'deep copied'."
            " This is allows an AFQMCTestCase containing multiple "
            "expected results to be split into multiple AFQMCTestCases "
            " which share the same AFQMC run data."
            )

# TODO: still need a way to 'split' tests based on multiple expected
#        results without actually re-running the AFQMC code.
#           - an idea: add a split() method to AFQMCTestCase that returns a list
#                of new test case instances which are **shallow copies** of the original,
#                but with one expected result each. should ALWAYS return a list even if there
#                is only one expected result to begin with

# HERE
def test_cases():
    """
    Generate AFQMCTestCase instances based on the settings in "afqmc_test.toml"
     
    The "afqmc_test.toml" file may contain multiple test cases with the following conventions.

    1. A "base" block will contain the default (or "base") settings for the set of test cases
      defined within "afqmc_test.toml"
    2. one or more "test[x]" blocks, where [x] = 1,2,3,... etc. are defined and correspond to a 
      test case. Each may optionally contain settings which will overried the base settings in 
      the "base" block for this test only (i.e. the base settings are not changed). 
    3. each test block must contain at least expected result - i.e. energy, an expected error 
      message, etc. but may define multiple expected results.
    """
    test_cases=[]
    for dir in test_dirs_list():
        test_file = Path(dir) / "afqmc_test.toml"

        if test_file.exists():
            params = read_input_params( str(test_file) )
        else:
            raise ValueError(
                f"Test file does not exist within: {Path(dir)}"
                " please add an `afqmc_test.toml` file"
            )

        # parse into defaults
        base_settings = params.get("base",None)

        # parse test cases
        for key in params.keys():
            if isinstance(key,str) and key.startswith("test"):
                test_cases.extend(AFQMCTestCase(name=f"{test_file}::{key}",settings=params[key],base_settings=base_settings).split())

    return test_cases

# to automation
class SystemType(enum.Enum):

    LatticeModel = enum.auto()
    Molecule = enum.auto()
    SolidGamma = enum.auto()
    SolidKPoint = enum.auto()

# to automation
def _get_system_type_enum(type_description=None) -> SystemType:
    if type_description is None:
        raise ValueError("[Developers] _get_system_type_enum got type_description None")
    
    if isinstance(type_description, SystemType):
        return type_description

    if type_description.lower() in ('model','lattice_model','lattice model','latticemodel'):
        return SystemType.LatticeModel
    elif type_description.lower() in ('mol','molecule','quantum chemistry','quantum_chemistry'):
        return SystemType.Molecule
    elif type_description.lower() in ('solid_gamma','gamma','solid gamma','gamma_point','gamma point'):
        return SystemType.SolidGamma
    elif type_description.lower() in ('solid','solid_k_point','k-point','k_point','solid k_point','solid k-point'):
        return SystemType.SolidKPoint
    else:
        raise ValueError(f"Invalid SystemType {type_description}. Possible choices are: `model`, `molecule`, `solid_gamma`, `solid_k_point`")


# TODO: This is generally useful -> move it (and related) to an actual submodule
# to automation
def system_setup_dispatcher(test_case:AFQMCTestCase,system_type:SystemType):
    system_type = _get_system_type_enum(system_type)

    if system_type == SystemType.LatticeModel:
        return setup_lattice_model(test_case)
    elif system_type == SystemType.Molecule:
        raise NotImplementedError
    elif system_type == SystemType.SolidGamma:
        raise NotImplementedError
    elif system_type == SystemType.SolidKPoint:
        raise NotImplementedError
    
#TODO: to automation
def setup_lattice_model(test_case:AFQMCTestCase):
    """
    Setup a lattice model AFQMC calculation based on the settings defined
       in 'test_case'
    """

    input_params = test_case.settings

    hamiltonian = HamiltonianBuilder.from_input(
        source=input_params
    ).hamiltonian

    misc_params = input_params["misc_params"]
    nelec = misc_params["nelec"]
    spin_symm = misc_params["spin_symm"]
    hdf5_hamil_outfile = hdf5_wfn_outfile = misc_params["hdf5_outfile"]
    write_model_hamiltonian(
        hamiltonian=hamiltonian,
        fname=hdf5_wfn_outfile,
        nelec=nelec,
        spin_symm=spin_symm
    )
    trial_wfn_type = get_trial_wfn_type(input_params["trial_wavefunction"]["type"])
    if trial_wfn_type == WfnType.FreeElectron:
        free_elec_settings = input_params["trial_wavefunction"].get("free_elec",dict())
        free_electron(
            source=free_elec_settings.get("hamiltonian",input_params),
            nelec=free_elec_settings.get("nelec",nelec),
            twist=free_elec_settings.get("twist",None),
            spin_symm=free_elec_settings.get("spin_symm",spin_symm),
            use_dense=free_elec_settings.get("use_dense",False),
            output=free_elec_settings.get("output","afqmc.h5")
        )
    elif trial_wfn_type == WfnType.HatreeFock:
        hf_settings = input_params["trial_wavefunction"]["hartree_fock"]
        effective_hamiltonian = input_params["trial_wavefunction"].get("hamiltonian",None)
        if effective_hamiltonian is not None:
            new_in = {"lattice":input_params["lattice"],
                     "hamiltonian":effective_hamiltonian}
            hf_hamiltonian = HamiltonianBuilder.from_input(
               source=new_in
            ).hamiltonian
        else:
            hf_hamiltonian = hamiltonian
        
        if "nelec" not in hf_settings:
            hf_settings["nelec"] = nelec
        lattice_hf(
            hamiltonian=AutoHFHamiltonian(source=hf_hamiltonian),
            settings=hf_settings,
            suppress_logo=True,
        )
        if "output" in hf_settings:
            hdf5_wfn_outfile = hf_settings["output"]

    return hdf5_wfn_outfile,hdf5_hamil_outfile

# to automation - lattice
class WfnType(enum.Enum):
    FreeElectron = enum.auto()
    HatreeFock = enum.auto()
    Unknown = enum.auto()

# to automation - lattice
def get_trial_wfn_type(type_name):

    if isinstance(type_name, WfnType):
        return type_name

    if type_name.lower() in {"free-electron", "free_electron", "fe"}:
        return WfnType.FreeElectron
    elif type_name.lower() in {"hartree-fock", "hartree_fock", "hf"}:
        return WfnType.HatreeFock
    else:
        return WfnType.Unknown

# to automation - general
def make_afqmc_inputs(test_case:AFQMCTestCase,path:Path):
    """
    Make AFQMC inputs based on the settings in the given AFQMCTestCase instance
    """
    wfn_file,hamil_file = system_setup_dispatcher(test_case,test_case.settings.get("type",None))
    write_json(
        path/"afqmc.json",
        fwfn0=wfn_file,
        fham0=hamil_file,
        exec_opts=test_case.settings.get("afqmc",None)
    )

# to automation
# TODO: add run type options -> see run_lattice_benchmark.py
def run_afqmc_exec(run_path,afqmc_exec):
    ts = perf_counter_ns()
    print("Launching AFQMC code...",flush=True)
    _infile = run_path / "afqmc.json"
    with open(run_path/"afqmc.out",'w') as f:
        assert sp.run(
            [afqmc_exec,"--filenames",_infile],
            cwd=run_path,
            stdout=f,
            stderr=f
            )
    tf = perf_counter_ns()
    print(f" ... AFQMC finished in {1.0e-9*(tf-ts)} seconds")


# HERE
def check_afqmc_results(test_case:AFQMCTestCase,run_path:Path):
    """
    For each result in test_case, check if it passed
    """
    for result in test_case.expected_results:
        assert result.check_result(run_path=run_path)

# HERE
@pytest.mark.weekly
@pytest.mark.parametrize("test_case",test_cases())
def test_given_cases(test_case,afqmc_exec,tmp_path):
    '''
    runs the test case in the 'test_case' arguments, and checks the results
    '''
    print(f"Running test case: {test_case}  in temporary path: {tmp_path}") # TODO: the tmp_path is unused if afqmc has been previously run. Update this to account for this.
    make_afqmc_inputs(test_case,tmp_path)
    run_token = test_case.afqmc_run_token
    if not run_token.has_run:
        run_afqmc_exec(tmp_path,afqmc_exec)
        run_token.has_run = True
        run_token.run_path = tmp_path
    else:
        print("AFQMC has already been run... skipping run")
    check_afqmc_results(test_case,run_path=run_token.run_path)


