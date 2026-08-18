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
- `statistical_references/<system>/...` : reference `results.h5` for the default,
  statistical comparison.
- `snapshot_references/<system>/...` : reference `results.h5` for the seeded,
  numerically exact comparison (`--snapshot`).

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
