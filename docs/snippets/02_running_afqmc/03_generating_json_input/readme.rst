.. _run_afqmc_ex_3:

Generating a JSON Input File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SAFIRE uses an input file in .json format to set parameters 
in calculations.
This example covers generating a .json input file for AFQMC.
It assumes that you already have an HDF5 file containing 
a Hamiltonian and trial wavefunction.
If not, see the examples in :ref:`setup_exs`.

For demonstration, it is assumed that a Hamiltonian exists in an HDF5 file called ``afqmc_ham.h5`` and
a trial wavefunction is saved in an HDF5 file called ``afqmc_wfn.h5``.
Of course, the Hamiltonian and wavefunction can exist within the same HDF5 file if desired.

The input file is explained in detail in :ref:`input_afqmc`.
We can use the ``safiretools`` Python module to generate one for us
using the same keywords as in the input file.
In the example below, we show how to set the ``execute`` block parameters
with all possible options explicitly enumerated with the exception
of the ``estimators`` block.
Each estimator in it has its own set of possible arguments.

.. literalinclude:: example.py

Note that all intervals are expressed in units of steps.
Running the example Python script above will generate a file called ``afqmc.json`` with the following content:

.. literalinclude:: afqmc.json

See :ref:`run_afqmc_ex_1` for an example of running AFQMC given an input file.



