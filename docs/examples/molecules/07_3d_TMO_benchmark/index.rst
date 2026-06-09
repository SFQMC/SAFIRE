.. _example_molecule_3d_tmo_benchmark:

3d Transition Metal Oxide Benchmark
===================================

Here we reproduce previous AFQMC benchmark data from a previous Simons Collaboration on the Many-Electron Problem benchmark for 3d transition metal monoxides (TMO) diatomics. 
(Phys. Rev. X 10, 011041).

Running the Example
-------------------

Perform the following steps to run the example. You can see each file below for details.

1. **Generate inputs for all systems**: Execute the benchmark script:

   .. code-block:: bash

      python 06_TMO_benchmark.py

   The script creates a scratch directory for calculation files along with a subdirectory for each TMO system.
   The script creates a separate HDF5 file (``afqmc.h5``) for each transition metal oxide system containing both the Hamiltonian and trial wavefunction.
   Additionally, the script will generate an AFQMC JSON input file (``afqmc.json``) in each system's scratch directory for running SAFIRE.   

2. **Run SAFIRE**: Execute AFQMC calculations for each system using the HDF5 files:

   Run each example separately. From within the scratch directory *corresponding to each molecule*
   created by the benchmark script, execute:

   .. code-block:: bash

      # Example for VO system
      mpirun -n 64 safire afqmc.json

3. **Analyze results**: Use ``get_results.py`` to compare computed energies against reference values from the literature.

You can modify the script to run calculations for specific TMOs by commenting out systems in the loop. 
Reference data from the publication supporting materials are in the ``files/`` directory.
If you used all of the same settings as us, running the analysis script should generate the following table:

.. code-block:: text

   ======== Benchmark Results ========
   | Molecule | E AFQMC | Ref. A AFQMC | abs diff | match? |
   |---|---|---|---|---|
   | VO  | -87.0825 +/- 0.0010 | -87.0818 +/- 0.0010 | 0.0007 +/- 0.0014 | 1 sigma |
   | TiO | -73.9005 +/- 0.0008 | -73.9019 +/- 0.0009 | 0.0014 +/- 0.0012 | 2 sigma |
   | CrO | -102.5534 +/- 0.0010 | -102.5533 +/- 0.0010 | 0.0001 +/- 0.0014 | 1 sigma |
   | MnO | -119.8512 +/- 0.0013 | -119.8515 +/- 0.0014 | 0.0003 +/- 0.0019 | 1 sigma |
   | FeO | -139.4317 +/- 0.0009 | -139.4296 +/- 0.0010 | 0.0021 +/- 0.0013 | 2 sigma |
   | CuO | -213.1260 +/- 0.0015 | -213.1271 +/- 0.0011 | 0.0011 +/- 0.0019 | 1 sigma |
   | ScO | -62.4175 +/- 0.0005 | -62.4192 +/- 0.0013 | 0.0017 +/- 0.0014 | 2 sigma |


Files
-----

**Benchmark Calculation** (``06_TMO_benchmark.py``):

This comprehensive script performs setup of all calculations. SAFIRE must be run separately for each generated set up inputs.

.. literalinclude:: 06_TMO_benchmark.py
   :language: python

Key features:

- Defines reference systems (VO, TiO, CrO, MnO, FeO, CuO, ScO) with equilibrium geometries from Phys. Rev. X 10, 011041
- Specifies spin multiplicities and active space parameters for each system
- Provides reference HF and AFQMC energies for validation. **Importnat:** The HF energies must match the reference values before performing CASSCF to ensure correct trial wavefunctions.
- Performs CASSCF calculations to generate multi-determinant trial wavefunctions
- Writes Hamiltonians and trial wavefunctions for each system

**Analysis** (``get_results.py``):

.. literalinclude:: get_results.py
   :language: python

Utility script for extracting and comparing AFQMC results against benchmark reference values, and generating a summary table.
