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
Centralized infrastructure for AFQMC functional testing.

This module provides:
- Enums for categorizing Hamiltonians and wavefunctions
- Dataclasses for organizing test inputs
- Rules for filtering valid/invalid test combinations
- Functions for generating parameterized test cases
"""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from warnings import warn

from afqmctools.hamiltonian.model.ham_class import SpinSymm

# ============================================================================
# Enumerations for test categorization
# ============================================================================

class HamiltonianClass(enum.Enum):
    """
    Classification of Hamiltonian types supported in AFQMC.
    """
    GENERIC_DENSE = enum.auto()
    GENERIC_SPARSE = enum.auto()
    MODEL = enum.auto()
    KPFAC_CHOL = enum.auto()
    THC = enum.auto()


class WavefunctionClass(enum.Enum):
    """
    Classification of wavefunction types supported in AFQMC.
    """
    NOMSD = enum.auto()
    PHMSD = enum.auto()


# ============================================================================
# Dataclasses for test organization
# ============================================================================

@dataclass
class AFQMCHamiltonian:
    """
    Metadata for an AFQMC Hamiltonian input.
    
    Attributes
    ----------
    path : Path
        Path to the HDF5 file containing the Hamiltonian
    spin_symm : SpinSymm
        Spin symmetry of the Hamiltonian
    type : HamiltonianClass
        Classification of the Hamiltonian type
    runparams : dict, optional
        Additional runtime parameters specific to this Hamiltonian
    """
    path: Path
    spin_symm: SpinSymm
    type: HamiltonianClass
    runparams: dict = None


@dataclass
class AFQMCWavefunction:
    """
    Metadata for an AFQMC trial wavefunction input.
    
    Attributes
    ----------
    path : Path
        Path to the HDF5 file containing the wavefunction
    spin_symm : SpinSymm
        Spin symmetry of the wavefunction
    type : WavefunctionClass
        Classification of the wavefunction type
    runparams : dict, optional
        Additional runtime parameters specific to this wavefunction
    """
    path: Path
    spin_symm: SpinSymm
    type: WavefunctionClass
    runparams: dict = None


@dataclass
class AFQMCWalker:
    """
    Metadata for an AFQMC walker configuration.
    
    Attributes
    ----------
    name : str
        Name of the walker type (e.g., "CLOSED", "COLLINEAR")
    spin_symm : SpinSymm
        Spin symmetry of the walker
    runparams : dict, optional
        Additional runtime parameters specific to this walker type
    """
    name: str
    spin_symm: SpinSymm
    runparams: dict = None


@dataclass
class AFQMCInputSet:
    """
    Complete set of inputs for an AFQMC test case.
    
    Holds metadata for Hamiltonian, wavefunction, and walker
    configurations to enable categorization into 'correct behavior'
    groups for testing.
    
    Attributes
    ----------
    hamiltonian : AFQMCHamiltonian
        Hamiltonian configuration
    wavefunction : AFQMCWavefunction
        Trial wavefunction configuration
    walker : AFQMCWalker
        Walker configuration
    reference : Path
        Path to reference results file for comparison
    runparams : dict, optional
        Additional runtime parameters for this specific test case
    """
    hamiltonian: AFQMCHamiltonian
    wavefunction: AFQMCWavefunction
    walker: AFQMCWalker
    reference: Path
    runparams: dict = None


# ============================================================================
# Filtering rules for test cases
# ============================================================================

class TestRule:
    """
    A rule for filtering AFQMC test cases.
    
    Attributes
    ----------
    name : str
        Human-readable name of the rule
    description : str
        Detailed description of what the rule checks
    check_func : Callable[[AFQMCInputSet], bool]
        Function that returns True if the case passes the rule
    """
    def __init__(self, name: str, description: str, check_func: Callable[[AFQMCInputSet], bool]):
        self.name = name
        self.description = description
        self.check_func = check_func
    
    def __call__(self, case: AFQMCInputSet) -> bool:
        return self.check_func(case)
    
    def __repr__(self) -> str:
        return f"TestRule('{self.name}')"


def _wfn_is_implemented(case: AFQMCInputSet) -> bool:
    """
    Check if the wavefunction type is implemented.
    
    Currently, only collinear PHMSD wavefunctions are implemented.
    NOMSD wavefunctions are implemented for all spin symmetries.
    """
    if case.wavefunction.type == WavefunctionClass.PHMSD:
        # currently, only collinear PHMSD wavefunctions are implemented
        return case.wavefunction.spin_symm == SpinSymm.COLLINEAR
    if case.wavefunction.type == WavefunctionClass.PHMSD and case.walker.spin_symm == SpinSymm.CLOSED:
        # this was intentionally not implemented since it would require special handling - i.e. perfect pairing
        return False
    elif case.wavefunction.type == WavefunctionClass.NOMSD:
        return True
    else:
        raise ValueError(f"Unrecognized wavefunction type: {case.wavefunction.type}")


def _not_closed_thc_with_noncollinear_wfn(case: AFQMCInputSet) -> bool:
    """
    Filter out closed THC Hamiltonians with non-collinear wavefunctions.
    
    This combination is known to be not implemented.
    """
    if case.hamiltonian.type == HamiltonianClass.THC and \
       case.hamiltonian.spin_symm == SpinSymm.CLOSED:
        if case.wavefunction.spin_symm == SpinSymm.NONCOLLINEAR:
            return False
    return True


def _compatible_spin_H_wfn(case: AFQMCInputSet) -> bool:
    """
    Check spin symmetry compatibility between Hamiltonian and wavefunction.
    
    The wavefunction must have equal or less spin symmetry than the Hamiltonian.
    (Higher enum value means less spin symmetry)
    """
    return case.wavefunction.spin_symm >= case.hamiltonian.spin_symm


def _compatible_spin_H_walker(case: AFQMCInputSet) -> bool:
    """
    Check spin symmetry compatibility between Hamiltonian and walker.
    
    The walker must have equal or less spin symmetry than the Hamiltonian.
    (Higher enum value means less spin symmetry)
    """
    return case.walker.spin_symm >= case.hamiltonian.spin_symm


def _noncollinear_walker_wfn_must_match(case: AFQMCInputSet) -> bool:
    """
    Ensure that non-collinear walkers are only used with non-collinear wavefunctions,
    and vice versa.

    Note: This is not desirable behavior, but it is how SAFIRE is currently implemented.
    Future versions may relax this restriction.
    """
    noncollinear_walker = case.walker.spin_symm == SpinSymm.NONCOLLINEAR
    noncollinear_wfn = case.wavefunction.spin_symm == SpinSymm.NONCOLLINEAR
    return noncollinear_walker == noncollinear_wfn

# Define all rules with human-readable descriptions
WFN_IS_IMPLEMENTED = TestRule(
    name="Wavefunction Implementation",
    description="Checks if the wavefunction type is implemented in SAFIRE. "
                "Currently only collinear PHMSD and all NOMSD types are supported.",
    check_func=_wfn_is_implemented
)

NOT_CLOSED_THC_WITH_NONCOLLINEAR_WFN = TestRule(
    name="THC-Noncollinear Compatibility",
    description="Filters out closed THC Hamiltonians with non-collinear wavefunctions, "
                "which is a known unimplemented combination.",
    check_func=_not_closed_thc_with_noncollinear_wfn
)

NOT_CLOSED_FOR_LATTICE_HAMILTONIANS = TestRule(
    name="Lattice Hamiltonian Closed Spin Symmetry",
    description="Ensures that lattice Hamiltonians do not use CLOSED spin symmetry, "
                "which is not supported for lattice models.",
    check_func=lambda case: not (
        case.hamiltonian.type == HamiltonianClass.MODEL and
        case.hamiltonian.spin_symm == SpinSymm.CLOSED
    )
)

COMPATIBLE_SPIN_H_WFN = TestRule(
    name="Hamiltonian-Wavefunction Spin Compatibility",
    description="Ensures the wavefunction spin symmetry is compatible with the Hamiltonian. "
                "The wavefunction must have equal or less spin symmetry.",
    check_func=_compatible_spin_H_wfn
)

COMPATIBLE_SPIN_H_WALKER = TestRule(
    name="Hamiltonian-Walker Spin Compatibility",
    description="Ensures the walker spin symmetry is compatible with the Hamiltonian. "
                "The walker must have equal or less spin symmetry.",
    check_func=_compatible_spin_H_walker
)

COMPATIBLE_FULLYPOLARIZED_WFN = TestRule(
    name="Fully Polarized Wavefunction Compatibility",
    description="Ensures that fully polarized wavefunctions and walkers are only used together. "
                "A fully polarized wavefunction requires a fully polarized walker, and vice versa.",
    check_func=lambda case: (
        (case.wavefunction.spin_symm == SpinSymm.FULLYPOLARIZED and
         case.walker.spin_symm == SpinSymm.FULLYPOLARIZED) or
        (case.wavefunction.spin_symm != SpinSymm.FULLYPOLARIZED and
         case.walker.spin_symm != SpinSymm.FULLYPOLARIZED)
    )
)

NONCOLLINEAR_WALKER_WFN_MUST_MATCH = TestRule(
    name="Non-collinear Walker-Wavefunction Symmetry Match",
    description="Ensures that non-collinear walkers are only used with non-collinear wavefunctions, and vice versa.",
    check_func=_noncollinear_walker_wfn_must_match
)

def get_all_rules() -> List[TestRule]:
    """
    Get the complete list of test filtering rules.
    
    Returns
    -------
    List[TestRule]
        All active filtering rules for determining valid test cases
    """
    return [
        WFN_IS_IMPLEMENTED,
        COMPATIBLE_SPIN_H_WFN,
        COMPATIBLE_SPIN_H_WALKER,
        NOT_CLOSED_THC_WITH_NONCOLLINEAR_WFN,
        NOT_CLOSED_FOR_LATTICE_HAMILTONIANS,
        COMPATIBLE_FULLYPOLARIZED_WFN,
        NONCOLLINEAR_WALKER_WFN_MUST_MATCH,
    ]


def check_rules(case: AFQMCInputSet, rules: List[TestRule], 
                verbose: bool = False) -> bool:
    """
    Check if a test case passes all provided rules.
    
    Parameters
    ----------
    case : AFQMCInputSet
        Test case to check
    rules : List[TestRule]
        Rules to apply
    verbose : bool, optional
        If True, print detailed information about failed rules
        
    Returns
    -------
    bool
        True if the case passes all rules, False otherwise
    """
    if verbose:
        print(f"\n=== Checking rules for test case ===")
        print(f"Hamiltonian: {case.hamiltonian.type.name}, spin={case.hamiltonian.spin_symm.name}")
        print(f"Wavefunction: {case.wavefunction.type.name}, spin={case.wavefunction.spin_symm.name}")
        print(f"Walker: {case.walker.name}, spin={case.walker.spin_symm.name}")
        print()
    
    passed = True
    for rule in rules:
        result = rule(case)
        if verbose:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {rule.name}")
            if not result:
                print(f"         {rule.description}")
        if not result:
            passed = False
    
    if verbose:
        print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
        print("=" * 40)
    
    return passed


# ============================================================================
# Test case generation and filtering
# ============================================================================

def generate_param_list(
        hamil_type_dictionary: dict,
        wavefunction_type_dictionary: dict,
        walker_type_list: List[AFQMCWalker],
        ref_data_dir: Path,
        extra_input_set: Optional[List[AFQMCInputSet]] = None
) -> List[AFQMCInputSet]:
    """
    Generate the full list of test case combinations.
    
    Creates all combinations of Hamiltonian types, wavefunction types,
    and walker types for use with pytest.mark.parametrize.
    
    Parameters
    ----------
    hamil_type_dictionary : dict
        Dictionary mapping names to AFQMCHamiltonian objects
    wavefunction_type_dictionary : dict
        Dictionary mapping names to AFQMCWavefunction objects
    walker_type_list : List[AFQMCWalker]
        List of walker configurations to test
    ref_data_dir : Path
        Root directory containing reference data
    extra_input_set : List[AFQMCInputSet], optional
        Additional specific test cases to include beyond the combinatorial generation
        
    Returns
    -------
    List[AFQMCInputSet]
        All possible test case combinations
    """
    params = []
    for h_name in hamil_type_dictionary:
        hamil_path = ref_data_dir / h_name
        if not hamil_path.exists():
            warn(f"Requested test against non-existent Hamiltonian directory: {hamil_path}; SKIPPING!")
            continue
        
        for trial_wfn_name in wavefunction_type_dictionary:
            trial_wfn_path = hamil_path / trial_wfn_name
            if not trial_wfn_path.exists():
                warn(f"Requested test against non-existent wavefunction directory: {trial_wfn_path}; SKIPPING!")
                continue
            
            for walker in walker_type_list:
                walker_path = trial_wfn_path / walker.name.lower()
                if not walker_path.exists():
                    warn(f"Requested test against non-existent walker type directory: {walker_path}; SKIPPING!")
                    continue
                
                # Get custom run parameters if they exist
                # Order of precedence: Walker > Wavefunction > Hamiltonian
                runparams = hamil_type_dictionary[h_name].runparams or {}
                runparams.update(wavefunction_type_dictionary[trial_wfn_name].runparams or {})
                runparams.update(walker.runparams or {})

                params.append(
                    AFQMCInputSet(
                        hamiltonian=hamil_type_dictionary[h_name],
                        wavefunction=wavefunction_type_dictionary[trial_wfn_name],
                        walker=walker,
                        reference=walker_path / "results.h5",
                        runparams=runparams
                    )
                )
    if extra_input_set is not None:
        params.extend(extra_input_set)
    return params


def should_run_successfully(all_cases: List[AFQMCInputSet], 
                           verbose: bool = False) -> List[AFQMCInputSet]:
    """
    Filter test cases that should run successfully.
    
    Parameters
    ----------
    all_cases : List[AFQMCInputSet]
        All test cases to filter
    verbose : bool, optional
        If True, print detailed rule checking information
        
    Returns
    -------
    List[AFQMCInputSet]
        Cases that pass all rules and should run successfully
    """
    rules = get_all_rules()
    return [case for case in all_cases if check_rules(case, rules, verbose)]

def should_exit_werror(all_cases: List[AFQMCInputSet],
                      verbose: bool = False) -> List[AFQMCInputSet]:
    """
    Filter test cases that should exit with an error.
    
    Parameters
    ----------
    all_cases : List[AFQMCInputSet]
        All test cases to filter
    verbose : bool, optional
        If True, print detailed rule checking information
        
    Returns
    -------
    List[AFQMCInputSet]
        Cases that fail at least one rule and should error
    """
    rules = get_all_rules()
    return [case for case in all_cases if not check_rules(case, rules, verbose)]


def should_run_warn(all_cases: List[AFQMCInputSet],
                   verbose: bool = False) -> List[AFQMCInputSet]:
    """
    Filter test cases that should run with a warning.
    
    Currently no warning-only rules are defined.
    
    Parameters
    ----------
    all_cases : List[AFQMCInputSet]
        All test cases to filter
    verbose : bool, optional
        If True, print detailed rule checking information
        
    Returns
    -------
    List[AFQMCInputSet]
        Cases that should run with warnings
    """
    # No warning-only rules currently defined
    rules = []
    return [case for case in all_cases if any(not rule(case) for rule in rules)]


def should_run_successfully_bp(
    all_cases: Optional[List[AFQMCInputSet]] = None,
    verbose: bool = False,
    cherry_picker: Optional[Callable[[AFQMCInputSet], bool]] = None,
) -> List[AFQMCInputSet]:
    """Filter test cases that should run successfully with back-propagation.

    Back-propagation (BP) runs are more expensive, so we *cherry-pick* a
    representative subset of the full successful space to keep weekly test
    runtime reasonable while exercising distinct spin-symmetry transitions.

    The default cherry-pick heuristic selects:
      - CLOSED Hamiltonian + COLLINEAR wavefunction with non-CLOSED walker
      - CLOSED Hamiltonian + NONCOLLINEAR wavefunction with NONCOLLINEAR walker
      - COLLINEAR Hamiltonian + COLLINEAR wavefunction with COLLINEAR or NONCOLLINEAR walker
      - NONCOLLINEAR Hamiltonian + NONCOLLINEAR wavefunction + NONCOLLINEAR walker

    You can override the selection logic by supplying `cherry_picker`, a
    callable returning True for cases to include.

    Parameters
    ----------
    all_cases : List[AFQMCInputSet], optional
        Complete list of candidate cases; if None, the caller is expected
        to generate them beforehand.
    verbose : bool, optional
        Print rule audit details.
    cherry_picker : Callable[[AFQMCInputSet], bool], optional
        Custom selection function; if provided replaces the default logic.

    Returns
    -------
    List[AFQMCInputSet]
        Subset of cases expected to succeed and chosen for BP execution.
    """
    if all_cases is None:
        raise ValueError("all_cases must be provided to should_run_successfully_bp; generate it with generate_param_list().")

    def _default_cherry_pick(case: AFQMCInputSet) -> bool:
        if case.hamiltonian.spin_symm == SpinSymm.CLOSED:
            if case.wavefunction.spin_symm == SpinSymm.COLLINEAR:
                if case.walker.spin_symm != SpinSymm.CLOSED:
                    return True
            elif case.wavefunction.spin_symm == SpinSymm.NONCOLLINEAR:
                if case.walker.spin_symm == SpinSymm.NONCOLLINEAR:
                    return True
        elif case.hamiltonian.spin_symm == SpinSymm.COLLINEAR:
            if case.wavefunction.spin_symm == SpinSymm.COLLINEAR:
                if case.walker.spin_symm in (SpinSymm.COLLINEAR, SpinSymm.NONCOLLINEAR):
                    return True
        elif (
            case.hamiltonian.spin_symm == SpinSymm.NONCOLLINEAR
            and case.wavefunction.spin_symm == SpinSymm.NONCOLLINEAR
            and case.walker.spin_symm == SpinSymm.NONCOLLINEAR
        ):
            return True
        return False

    picker = cherry_picker or _default_cherry_pick
    bp_rule = TestRule(
        name="Cherry-pick for back-propagation",
        description="Limits the number of back-propagation test cases to run",
        check_func=picker,
    )
    rules = [bp_rule] + get_all_rules()
    return [case for case in all_cases if check_rules(case, rules, verbose)]

