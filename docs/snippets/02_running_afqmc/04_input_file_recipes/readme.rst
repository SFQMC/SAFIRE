.. _run_afqmc_ex_4:

Input File Recipes
^^^^^^^^^^^^^^^^^^

This example shows a few recipes for input files to run SAFIRE.



Nested structure with default hamiltonian location
--------------------------------------------------

In this input file, we use a "nested" structure in which we
define the "walker_set" and the "wavefunction" within the "execute"
block. 
This is the simplest input file layout if you don't need to reuse the "walker_set" in a second "execute" block.
Additionally, we allow the "hamiltonian" to default to the "filename" found in the "wavefunction" block by not 
defining an explicit "hamiltonian" block.
If you will only be trying one trial wavefunction, this can be simpler than saving the Hamiltonian and trial
wavefunctions in separate HDF5 files.

.. code-block:: json

    {
      "afqmc": {
        "project": {
          "id": "qmc",
          "series": 0
        },
        "execute": {
          "walker_set": {
            "walker_type": "CLOSED"
          },
          "wavefunction": {
            "filename": "input.h5"
          },
          "timestep": 0.01,
          "steps": 10000,
          "n_walkers_per_mpi_task": 200,
          "seed": 42
        }
      }
    }

Flat structure with multiple "execute" blocks
---------------------------------------------

In the input file below we use a "flat" input structure in the
sense that the "walker_set" and "wavefunction" blocks are defined
outside of the "execute" blocks and are referenced by name within the "execute" blocks.
This input file contains two execute blocks.
The first is used to quickly equilibrate using a fairly large step size.
The second resumes that calculation using an equilibrated population, and a smaller Trotter step size in order to perform measurements.

In this case, it is strictly necessary to define at least the "walker_set" outside of each "execute" block since it is shared between both.
While not functionally necessary, defining the "wavefunction"and "hamiltonian" blocks outside of the "execute" blocks prevents the trial wavefunction and Hamiltonian from being constructed more than once.

.. code-block:: json

    {
      "afqmc": {
        "project": {
          "id": "qmc",
          "series": 0
        },
        "walker_set":{
            "name" : "walkers",
            "walker_type": "CLOSED"
        },
        "wavefunction" : {
            "name" : "my_wavefunction",
            "filename": "files/input.h5"
        },
        "hamiltonian" : {
            "name" : "my_hamiltonian",
            "filename": "files/input.h5"
        },
        "execute": {
          "walker_set": "walkers",
          "wavefunction": "my_wavefunction",
          "hamiltonian" : "my_hamiltonian",
          "timestep": 0.05,
          "steps": 20,
          "n_walkers_per_mpi_task": 200,
          "measure_interval_multiplier": 1,
          "population_control_interval": 1,
          "walker_ortho_interval": 1,
          "seed": 42
        },
        "execute": {
          "walker_set": "walkers",
          "wavefunction": "my_wavefunction",
          "hamiltonian" : "my_hamiltonian",
          "timestep": 0.01,
          "steps": 10000,
          "n_walkers_per_mpi_task": 200,
          "measure_interval_multiplier": 1,
          "population_control_interval": 10,
          "walker_ortho_interval": 10,
          "seed": 43
        }
      }
    }

The walker set is saved after the first execute block and will be used "as is" in
the second execute block.
A common use case for this is to use a large trotter timestep in the first execute block
to equilibrate faster, then switch to a smaller trotter timestep in the second execute block
while collecting samples.
Defining the "wavefunction" input outside of either execute block allows is to be read in
only once, and re-used in both calculations.
This input file will lead to AFQMC energy samples being saved into a file called "qmc.s000.scalar.dat"
where the first 2 samples come from the first execute block (up to total imaginary time $\beta = 1.0$
based on the 20 steps, with $\tau = 0.05$, and the remaining samples are from the second
execution block.

For more detail on the input file, see the [input file description](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/input_description_afqmc.html)


Input file with most parameters exposed
---------------------------------------

Here, we provide a sample input file where we expose most of the possible input blocks.
We use ellipses ( `...`)  in some of the advanced input blocks within some input blocks for visual simplicity.

.. code-block:: json

  {
    "afqmc": {
      "project": {
        "id": "qmc",
        "series": 0
      },
      "walker_set": {
        "name" : "my_walkers",
        "walker_type": "CLOSED",
        "load_balance_type": "async",
        "pop_control_type": "pair",
        "min_weight": "0.05",
        "max_weight": "4"
      },
      "execute": {
        "walker_set" : "my_walkers",
        "wavefunction": {
          "filename": "files/input.h5",
          "ndets_to_read": "-1"
        },
        "hamiltonian" : {
          "filename" : "files/input.h5"
        },
        "timestep": "0.01",
        "steps": "1",
        "population_control_interval": "10",
        "measure_interval_multiplier": "2",
        "walker_ortho_interval": "10",
        "checkpoint_interval": "-1",
        "hdf_write_file": "",
        "hdf_read_file": "",
        "n_walkers_per_mpi_task": "10",
        "seed": "42",
        "propagator" : {
          /* ... */
        },
        "estimator" : {
          "name" : "energy",
          /* ... */
        },
        "estimator" : {
          "name" : "mixed",
          /* ... */
        }
      }
    }
  }


For pedagogical reasons, we have defined a "walker_set" block outside of the "execute" block and have named it "my_walkers".
Notice that within "execute" we are able to reference this "walker_set" by its name using "walker_set" : "my_walkers"
instead of supplying a json input block.


