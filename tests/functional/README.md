# Functional Tests

The goal of the functional tests is to check that the ecosystem does what is intended -
i.e. that we can actually run an AFQMC calculation and get the correct numbers. We're
testing the full system, and proving that the code does what we claim it does.

The unit tests live elsewhere: the C++ unit tests in `tests/` (built by CMake and run via
`ctest`), the Python ones in `utils/tests/` (run via `pytest` from `utils`).

## Layout

- `run_functional.py` : the test runner (see `--help`).
- `functional_cases.py` : declarative definition of the systems, hamiltonians,
  wavefunctions and walker types that get combined into test cases.
- `afqmc_inputs/<system>/` : the hamiltonian and wavefunction HDF5 files each case reads.
- `make_inputs.py` : rebuilds `afqmc_inputs/` into a directory you name (see `--help`).
- `input_recipes/` : one recipe per system directory, saying how its inputs are built.
- `statistical_references/<system>/...` : reference `results.h5` for the default,
  statistical comparison.
- `snapshot_references/<system>/...` : reference `results.h5` for the seeded,
  numerically exact comparison (`--snapshot`).

Note that `afqmc_inputs/` is shared: the C++ unit tests read the same directory
through `unit_test_base()` in `tests/test_common.hpp`. Some files there are used only
by the C++ side and appear in no functional case.

## Requirements

- `AFQMC_EXEC` must point at the SAFIRE executable to test.
- The `afqmctools` and `stats` packages must be importable. Installing `utils` (e.g.
  `pip install -e utils`) is the usual way; otherwise put `utils` on `PYTHONPATH`:
  `export PYTHONPATH=/path/to/SAFIRE/utils:$PYTHONPATH`.

## Local Runs

```bash
rm -rf $OUTPUT_DIR
python tests/functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="mpirun -n 16" \
       --compute cpu \
       --timeout=120
```

`--output-path` is optional; without it the runner writes to a temporary directory that
is discarded on exit. Use `--list` to see the available systems, and pass a single system
key instead of `all` to run just that one. `--dry-run` lists the planned cases without
running AFQMC.

### Snapshot mode

```bash
python tests/functional/run_functional.py all --snapshot
```

This runs each case for a few steps with a fixed seed and requires numerically exact
agreement with `snapshot_references`. It is fast and catches changes the statistical
comparison cannot see, but only reproduces at a fixed rank count.

### Regenerating references

`--regenerate` replaces the comparison step by copying each freshly recorded `results.h5`
into the reference tree (`snapshot_references` with `--snapshot`, `statistical_references`
otherwise). This is how the stored references are produced in the first place.

Before regenerating snapshot files, rerun the full statistical tests and convince yourself
that the new results are still correct.

## Regenerating the inputs

`make_inputs.py` rebuilds the HDF5 files in `afqmc_inputs/` from the calculations that
produced them - pyscf for the molecules, afqmctools for the lattice models, and
Quantum ESPRESSO plus CoQui for diamond. `input_recipes/` holds one recipe per system
directory.

```bash
python tests/functional/make_inputs.py --list          # recipes, and what is runnable here
python tests/functional/make_inputs.py BH  --into DIR  # rebuild BH into DIR
python tests/functional/make_inputs.py all --into DIR  # rebuild everything into DIR
```

`--into` is required and must not be inside `afqmc_inputs/`: the script never writes
into the committed tree. Installing a rebuild is a deliberate `cp -r`, which the script
prints for you, because a rebuild is not byte-identical to what is committed (orbital
gauge, truncated CI expansions and writer changes all move bytes without changing the
physics) and accepting one obliges regenerating the reference results as well. Compare
with `h5diff` before you copy anything in. Intermediate files - pyscf checkpoints, QE
and CoQui runs and their logs - are left under `DIR/_scratch`.

Everything except diamond needs only pyscf, afqmctools and AutoHF, and the whole set
takes about a minute. The two diamond recipes drive external codes and are skipped
unless these are found on `PATH` or named explicitly:

```bash
export QE_BIN_DIR=/path/to/q-e/build/bin      # pw.x and pw2coqui.x
export COQUI_EXEC=/path/to/coqui/bin/coqui
```

Both are run as ordinary serial commands. If they need a particular toolchain, set that
up in your shell before invoking the script.

Two things in `afqmc_inputs/` have no generator anywhere in this repository - the thermal
factorisation `square_2x2_hubbard_Beta3_nt100/wfn_collinear.h5`, and the `TEST_RESULTS`
group inside that directory's `ham_collinear.h5`. Both are checked in under
`input_recipes/assets/finiteT/` and copied into place by the `hubbard_2x2_finite_t`
recipe, so the whole tree can still be rebuilt.

## Sample Functional Test Slurm script

```bash
#!/bin/bash
#SBATCH -J func_tests_safire
#SBATCH --partition=ccq
#SBATCH --constraint=genoa
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH -o functional_tests.o%j

module purge
module load slurm
module load cmake
module load gcc
module load openmpi
module load hdf5
module load boost
module load intel-oneapi-mkl
module load cuda
module load openmpi/cuda

echo "Starting functional tests... "
date

export AFQMC_EXEC="/path/to/safire"

OUTPUT_DIR="/path/to/scratch/dir"
rm -rf $OUTPUT_DIR

export PYTHONPATH="/path/to/SAFIRE/utils:$PYTHONPATH"
python -u /path/to/SAFIRE/tests/functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="srun -n 90 --cpu-bind=cores" \
       --compute cpu \
       --timeout=120

echo "... done!"
date
```

> 💡 Note that the tests write all input and output files to `$OUTPUT_DIR`.
> Using a persistent directory for this is useful because it allows inspecting logs afterwards.
> However, it is recommended to delete the directory before each run to avoid coexisting old and new outputs.

## Adding new cases

See the file `functional_cases.py`.

If the new case needs new input files, add a recipe for them at the same time - write a
`build(ctx)` function in the matching `input_recipes/` module (`molecules.py`,
`models.py` or `solids.py`), register it in that module's `recipes()`, and keep any
intermediate files under `ctx.scratch` so only the finished outputs land in the inputs
tree.
