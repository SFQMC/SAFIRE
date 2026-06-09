# Tests
-------

The tests defined here fall into one of two categories.

1. unit tests : these are relatively fast and can be run very regularly. The goal is to test individual components - i.e. units.
2. functional tests : the goal is to check that the ecosystem does what is intended - i.e. that we can actually run an AFQMC calculation and get the correct numbers. We're testing the full system, and proving that the code does what we claim it does.


## Unit Tests
-------------

These are run as usual via pytest: `$ pytest` (from the `utils` directory).
See the `pyproject.toml` file for default configuration.


## Functional Tests
-------------------

The functional tests are also invoked via `pytest`; however, the default settings
defined in the `pyrpoject.toml` file will not  run the functional tests by default.
You must explicitly request them using pytest marks.

### Sample Functional Test Slurm script
---------------------------------------

A sample script to run the "weekly" functional tests is as follows.

```bash
#!/bin/bash
#SBATCH -J func_tests_safire
#SBATCH --partition=ccq,gen
#SBATCH --constraint=icelake
#! Number of MPI ranks (= tasks for Slurm)
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH -o functional_tests.o%j

module load safire

echo "====== some git info =======\n\n"
git status
git log -n 1

# This is read by the testing infrastructure!!
export NUM_MPI_TASKS=64

echo "Starting functional tests... "
date
python -c "import sys; print(sys.path)"
python -m pytest --basetemp=/path/to/scratch/dir/ \
       --afqmc-runmode mpi_cpu \
       --a
       -m "functional and weekly" \
       -vvv -rP -rs \
       --html=./.htmlpytest_functional_cpu/pytest.html
echo "... done!"
date
```

> The key detail to notice is the `-m "functional and weekly"` option which
>   selects both functional tests and weekly tests. Using just the `-m "functional"` 
>   option will run all functional tests which will take a long time! This can 
>   be done occassionally (recommended one per month or less frequently)


> 💡 Note that the tests use a temporary directory for scratch work. 
> The directory is cleared upon the next test run.
> These are a very useful place to debug problems in the test.

### Local Runs
--------------

```bash
 NUM_MPI_TASKS=64 python -m pytest --basetemp= \
       --afqmc-runmode mpi_cpu \
       -m "functional and push" \
       -vvv -rP -rs \
       --html=./.htmlpytest_functional_cpu/pytest.html
```

### Adding new cases
--------------------

A collection of developer tools related to the functional tests is provide in `utils/tests/functional/dev_tools`.
It has it's own Readme.md file which explains the functional testing infrastructure.



