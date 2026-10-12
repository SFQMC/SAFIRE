.. _run_afqmc_ex_1:

Basic Ground State Energy with CPU Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers running AFQMC for a generic input
Hamiltonian to obtain the ground state energy.
It assumes that you already have an HDF5 file containing 
a Hamiltonian and trial wavefunction.
If not, see the examples in :ref:`setup_exs`.

Here, it is assumed that a Hamiltonian and a trial wavefunction are
both saved in an HDF5 file called ``afqmc.h5``.
SAFIRE uses an input file in .json format to set parameters 
in calculations.
The input file is explained in detail in :ref:`input_afqmc`, but a minimal 
example is shown here.

.. literalinclude:: afqmc.json

Note that all intervals are expressed in units of steps.

A CPU-only build of SAFIRE can be invoked on rusty using the following runscript,

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH -J afqmc
    #SBATCH --partition=ccq
    #SBATCH --constraint=ib
    #! Number of MPI ranks (= tasks for Slurm)
    #SBATCH --ntasks=160
    #SBATCH --time=0:30:00

    export AFQMC_PATH=/path/to/your/afqmc/build

    module purge
    source $AFQMC_PATH/env.bash

    # Launch MPI code...
    srun --cpu-bind=cores $AFQMC_PATH/bin/safire --filenames afqmc.json &> afqmc.out

If the Hamiltonian and trial wavefunction file from :ref:`setup_ex_1` (called ``afqmc.h5``) is used with the input file above, then we will see output similar to this (perhaps with differences in the timing)

.. literalinclude:: afqmc.out


See :ref:`analysis_ex_1` for an example of analyzing this output.

