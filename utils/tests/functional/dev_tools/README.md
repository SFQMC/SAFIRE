# Functional Testing Infrastructure

This directory contains shared infrastructure for AFQMC functional tests.

## Files

### `run_and_record.py`
Contains the `AFQMCHelper` class for executing AFQMC runs and recording results.

### `test_infrastructure.py`
Centralized infrastructure for organizing and filtering functional test cases.

**Key components:**

1. **Enumerations**
   - `HamiltonianClass`: Categories of Hamiltonians (KPFAC_CHOL, THC, MODEL, etc.)
   - `WavefunctionClass`: Categories of wavefunctions (NOMSD, PHMSD)

2. **Dataclasses**
   - `AFQMCHamiltonian`: Metadata for Hamiltonian inputs
   - `AFQMCWavefunction`: Metadata for wavefunction inputs
   - `AFQMCWalker`: Metadata for walker configurations
   - `AFQMCInputSet`: Complete test case combining all inputs

3. **Test Rules**
   - `TestRule`: Class for defining filtering rules with human-readable descriptions
   - Pre-defined rules:
     - `WFN_IS_IMPLEMENTED`: Check if wavefunction type is implemented
     - `COMPATIBLE_SPIN_H_WFN`: Check Hamiltonian-wavefunction spin compatibility
     - `COMPATIBLE_SPIN_H_WALKER`: Check Hamiltonian-walker spin compatibility
     - `COMPATIBLE_SPIN_WFN_WALKER`: Check wavefunction-walker spin compatibility
     - `NOT_CLOSED_THC_WITH_NONCOLLINEAR_WFN`: Filter known unsupported combinations

4. **Functions**
   - `get_all_rules()`: Returns all active filtering rules
   - `check_rules(case, rules, verbose=False)`: Check if a case passes all rules
   - `generate_param_list(...)`: Generate all test case combinations
   - `should_run_successfully(all_cases, verbose=False)`: Filter cases that should pass
   - `should_exit_werror(all_cases, verbose=False)`: Filter cases that should fail
   - `should_run_warn(all_cases, verbose=False)`: Filter cases that should warn

## Usage

### Basic usage in a test file:

```python
from pathlib import Path
import pytest
from afqmctools.hamiltonian.model.ham_class import SpinSymm

# Import centralized infrastructure
from dev_tools.test_infrastructure import (
    HamiltonianClass,
    WavefunctionClass,
    AFQMCHamiltonian,
    AFQMCWavefunction,
    AFQMCWalker,
    AFQMCInputSet,
    generate_param_list,
    should_run_successfully,
    should_exit_werror,
)

# Define your test-specific Hamiltonians, wavefunctions, and walkers
TEST_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = TEST_ROOT / "inputs"
REF_DATA_DIR = TEST_ROOT / "reference"

def hamil_type_dictionary():
    return {
        "my_hamiltonian": AFQMCHamiltonian(
            INPUTS_DIR / "hamiltonian.h5",
            SpinSymm.CLOSED,
            HamiltonianClass.KPFAC_CHOL
        )
    }

def wavefunction_type_dictionary():
    return {
        "my_wavefunction": AFQMCWavefunction(
            INPUTS_DIR / "wavefunction.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("COLLINEAR", SpinSymm.COLLINEAR),
    ]

# Generate test parameters
def generate_test_params():
    return generate_param_list(
        hamil_type_dictionary=hamil_type_dictionary(),
        wavefunction_type_dictionary=wavefunction_type_dictionary(),
        walker_type_list=walker_type_list(),
        ref_data_dir=REF_DATA_DIR
    )

# Use in pytest
@pytest.mark.parametrize("case", should_run_successfully(generate_test_params()))
def test_success(afqmc_helper, result_checker, afqmc_runmode, tmp_path, case):
    # Your test implementation
    pass
```

### Using verbose mode for debugging:

```python
from dev_tools.test_infrastructure import should_run_successfully

# Print detailed rule checking information
cases = should_run_successfully(all_cases, verbose=True)
```

This will print output like:
```
=== Checking rules for test case ===
Hamiltonian: KPFAC_CHOL, spin=CLOSED
Wavefunction: NOMSD, spin=COLLINEAR
Walker: COLLINEAR, spin=COLLINEAR

✓ PASS: Wavefunction Implementation
✓ PASS: Hamiltonian-Wavefunction Spin Compatibility
✓ PASS: Hamiltonian-Walker Spin Compatibility
✓ PASS: Wavefunction-Walker Spin Compatibility
✓ PASS: THC-Noncollinear Compatibility

Overall: PASS
========================================
```

### Creating custom rules:

```python
from dev_tools.test_infrastructure import TestRule, get_all_rules

def my_custom_check(case):
    # Your custom logic
    return case.hamiltonian.type != HamiltonianClass.MODEL

my_rule = TestRule(
    name="No Model Hamiltonians",
    description="Excludes all model Hamiltonians from this test suite",
    check_func=my_custom_check
)

# Use with existing rules
custom_rules = get_all_rules() + [my_rule]
valid_cases = [case for case in all_cases if all(rule(case) for rule in custom_rules)]
```

### Back-propagation subset with a custom `cherry_picker`

Back-propagation (BP) runs are expensive. By default, `should_run_successfully_bp(...)` applies a conservative subset selection. You can override the selection logic by passing your own `cherry_picker` callable that receives an `AFQMCInputSet` and returns `True` to include the case.

```python
from dev_tools.test_infrastructure import should_run_successfully_bp

# Define your subset selection
def my_bp_picker(case):
    # Example: include only COLLINEAR wavefunction cases with non-CLOSED walkers
    from afqmctools.hamiltonian.model.ham_class import SpinSymm
    return (
        case.wavefunction.spin_symm == SpinSymm.COLLINEAR
        and case.walker.spin_symm != SpinSymm.CLOSED
    )

# Generate all combinations (example helper shown earlier)
all_cases = generate_test_params()

# Get the BP subset using the custom picker
bp_cases = should_run_successfully_bp(
    all_cases,
    cherry_picker=my_bp_picker,   # override default selection
    verbose=False                 # set True to audit rule checks
)

@pytest.mark.parametrize("case", bp_cases)
def test_bp(afqmc_helper, result_checker_bp, afqmc_runmode, tmp_path, case):
    # run with back-propagation enabled in your test helper
    pass
```

## Benefits

1. **DRY (Don't Repeat Yourself)**: Define test infrastructure once, use everywhere
2. **Human-readable rules**: Each rule has a name and description for better error messages
3. **Verbose debugging**: Turn on verbose mode to see which rules are failing
4. **Maintainability**: Update rules in one place, all tests benefit
5. **Consistency**: All functional tests use the same filtering logic
6. **Extensibility**: Easy to add new rules or custom filtering for specific test suites
