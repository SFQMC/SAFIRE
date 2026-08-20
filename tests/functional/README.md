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
- `make_inputs.py` : rebuilds `afqmc_inputs/` elsewhere and diffs (see `--help`).
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
directory, each declaring the files it writes.

```bash
python tests/functional/make_inputs.py --list          # recipes, and what is runnable here
python tests/functional/make_inputs.py BH              # rebuild BH in a temp dir and diff
python tests/functional/make_inputs.py all --into DIR  # rebuild everything in DIR and diff
```

**It never writes into `afqmc_inputs/`.** Every run builds somewhere else and reports how
the result differs from what is committed. That is deliberate: a rebuild is not
byte-identical (see below), and accepting one obliges regenerating the reference results
too, so installing it is a deliberate step you take after reading the diff. With `--into`
the script prints the `cp -r` for each system, and leaves the intermediate files - pyscf
checkpoints, QE and CoQui runs and their logs - under `DIR/_scratch`.

Exit status is 0 when everything reproduced, 2 when something differed, 1 when a recipe
failed outright.

Everything except diamond needs only pyscf, afqmctools and AutoHF, and the whole set
takes about a minute. The two diamond recipes drive external codes and are skipped
unless these resolve:

```bash
export QE_BIN_DIR=/path/to/q-e/build/bin
export QE_ENV_SCRIPT=/path/to/q-e/build/env.sh
export COQUI_EXEC=/path/to/coqui.build/bin/coqui
export COQUI_ENV_SCRIPT=/path/to/coqui.build/env_2.4.sh
export COQUI_MPIEXEC="mpirun -n 1"    # at most one rank per irreducible k-point
```

Each code gets its own environment script because they are built against different
toolchains - at Flatiron they sit in module trees that cannot both be loaded in one
shell. Have each script start from `module purge`.

### Do not expect the files back byte for byte

A rebuilt input describes the same physics as the committed one but is not the same
file, and the script will say so. The reasons are worth knowing before you act on a
diff:

- **Orbital gauge.** Every one of these systems has degenerate orbitals - the pi shells
  of BH and N2, the p/d/f shells of the Pb atom, the partly filled shell of the Rashba
  trial. An SCF or a variational solve may return any rotation within a degenerate
  shell, with any sign per orbital. `Hamiltonian/X` and everything written in that basis
  then differ while the physics does not. Confirm by checking that the eigenvalues of
  `Hamiltonian/hcore` still agree (they match to ~1e-14) and that the Cholesky rank is
  unchanged.
- **Truncated CI expansions.** BH's CASCI and N2's CASSCF spread weight across
  degenerate configurations, so an orbital-gauge change alters which determinants
  survive truncation. The trial ends up slightly different in quality, which means the
  stored references have to be regenerated alongside the inputs.
- **Writer changes.** `maximum_connectivity` in the model hamiltonians now has a floor
  of 12 in `write_model_hamiltonian`; the committed files predate it and hold the
  smaller computed value.
- **CoQui version.** The committed diamond files came from the older `aimbes` build.
  Rebuilding with CoQui 2.4 reproduces the wavefunctions exactly but gives a different
  Cholesky basis for the 2x2x2 case (different vector counts per k-point, `System/H0`
  shifted by ~5e-05), and different `Interaction/Vq0` at gamma.

Anything outside those four - a changed nuclear energy, electron count, matrix shape,
or Cholesky rank where none is expected - means the recipe and the committed file
describe different systems, and one of them is wrong.

### What is copied rather than computed

Two things in `afqmc_inputs/` have no generator anywhere in this repository:

- `square_2x2_hubbard_Beta3_nt100/wfn_collinear.h5`, a thermal propagator factorisation
  (UL/UR, VL/VR, DL/DR blocks) in a format afqmctools does not write.
- the `TEST_RESULTS` group inside that directory's `ham_collinear.h5`, holding the
  E1/EJ/EXX/VHS/vbias arrays the finite-temperature C++ unit test asserts against.

Both are checked in under `input_recipes/assets/finiteT/`, and the
`hubbard_2x2_finite_t` recipe copies them onto the hamiltonian it computes. So the whole
tree can be rebuilt, but those two pieces are the same bytes every time - grafting
`TEST_RESULTS` onto a fresh hamiltonian is only sound while that hamiltonian still
matches the one the numbers came from, so read that recipe's diff before trusting a
rebuild of it. `--list` reports any file in the tree that no recipe claims; it should
report none.

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
intermediate files under `ctx.scratch` so only the declared outputs land in the inputs
tree. `make_inputs.py --list` will then flag it if the two ever drift apart.
