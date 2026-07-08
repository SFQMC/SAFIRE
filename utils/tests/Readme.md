# Tests

The tests defined here fall into one of two categories.

1. unit tests : these are relatively fast and can be run very regularly. The goal is to test individual components - i.e. units.
2. functional tests : the goal is to check that the ecosystem does what is intended - i.e. that we can actually run an AFQMC calculation and get the correct numbers. We're testing the full system, and proving that the code does what we claim it does.


## Unit Tests

These are run as usual via pytest: `$ pytest` (from the `utils` directory).
See the `pyproject.toml` file for default configuration.


## Functional Tests

The functional tests are run using the script `functional/run_functional.py`.

### Sample Functional Test Slurm script


```bash
#!/bin/bash
#SBATCH -J func_tests_safire
#SBATCH --partition=ccq
#SBATCH --constraint=genoa
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH -o functional_tests.o%j

senf safire
module load cuda
module load openmpi/cuda

echo "Starting functional tests... "
date

OUTPUT_DIR="/path/to/scratch/dir"
rm -rf $OUTPUT_DIR

set -x PYTHONPATH (pwd) $PYTHONPATH
python functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="srun -n 90 --cpu-bind=cores" \
       --compute cpu \
       --timeout=120 \

echo "... done!"
date
```

> 💡 Note that the tests write all input and output files to `$OUTPUT_DIR`. 
> Using a persistent directory for this is useful because it allows inspecting logs afterwards.
> However, it is recommended to delete the directory before each run to avoid coexisting old and new outputs. 

### Local Runs

```bash
rm -rf $OUTPUT_DIR
python functional/run_functional.py \
       all \
       --output-path=$OUTPUT_DIR \
       --mpiexec="mpirun -n 16" \
       --compute cpu \
       --timeout=120
```

### Adding new cases

See the file `functional/functional_cases.py`.



