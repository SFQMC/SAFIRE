.. _example_molecule_v_fully_polarized:

Vanadium Atom (Fully Polarized)
================================

This example demonstrates a fully polarized AFQMC calculation on an isolated vanadium atom
which has a ground state with :math:`S=3/2` (quartet state).
We will use the frozen core approximation to freeze all electrons except for the :math:`3d^3` valence electrons.
In the cc-pVDZ basis set, this results in 32 active orbitals and 3 active electrons.
We will use CASCI to compute the exact energy within the active space as a reference for comparison with AFQMC.

External Dependencies
---------------------

1. Dice: A selected CI code used to generate the trial wavefunction.
   Installation instructions can be found at `Dice GitHub repository <https://github.com/sanshar/Dice>`_.

Running the Example
-------------------

The workflow follows three steps:

1. **Generate trial wavefunction**: Execute ``scf/scf.py`` to perform ROHF, and CASCI(32o,3e) calculations on the vanadium atom.
   
   The ROHF energy should be around -942.873080272797 Ha, and the CASCI(32o,3e) energy should be 
   around -942.901099297148 Ha were the contribution in the active space is -3.53263111527235 Ha.

2. **Create AFQMC inputs**: Execute ``input/setup.py`` to generate the Hamiltonian and trial wavefunction. This treats the system as having all electrons in one spin channel. 
   The script writes both the Hamiltonian and a fully polarized trial wavefunction to ``afqmc.h5``.

3. **Next, use Dice to generate a selected CI trial wavefunction**: 
   First, convert from the SAFIRE HDF5 Hamiltonian format to FCIDUMP format using the `hdf5_to_dice` tool from afqmctools:
   
   .. code-block:: bash

      $ afqmc_to_fcidump -i inputs/afqmc.h5 -o dice/FCIDUMP

   Then, create a Dice input file `dice/dict_input.dat` with the following content

   .. code-block:: text

      #system - 3d^3 shell of V atom with Sz=3/2
      nocc 3
      0 2 4
      end
      orbitals ./FCIDUMP
      nroots 1

      #variational
      schedule
      1   0.01
      2   0.005
      3   0.001
      5   0.0005
      10  0.0001
      end
      davidsonTol 5e-05
      dE 1e-08
      maxiter 20

      #pt
      nPTiter 500
      epsilon2 1e-08
      epsilon2Large 1000
      targetError 0.00001
      sampleN 300

      #misc
      printBestDeterminants 2000 # this is for the output file in ASCII format


   Run Dice using the provided input file as follows:

   .. code-block:: bash

      $ dice dice/dict_input.dat &> dice/output.dat

   And convert to the SAFIRE format using afqmctools,

   .. code-block:: bash

      $ dice_to_hdf5 -o inputs/shci_wfn.h5 -i dice/output.dat -n 100 -v
      Number of electrons: up:3, down:0
      PHMSD trial wavfunction -> using uhf-like walkers


4. **Run SAFIRE**: Execute SAFIRE using the provided ``afqmc.json`` input file:

   .. code-block:: bash

      mpirun -n 64 safire afqmc.json

   SAFIRE will use fully polarized walkers, which is computationally efficient for ferromagnetic systems.

5. **analyze the results**: Use the `scalar_stats` command-line tool from afqmctools to analyze the energy output:

   .. code-block:: bash

      $ scalar_stats qmc.s000.scalar.dat -s time -e 5.0 -t

      FILL IN!!

   This will compute the average AFQMC energy and stochastic uncertainty, using an equilibration time of 5.0 Ha^{-1}.
   The `-t` flag will generate a plot of the energy as a function of imaginary time if running locally.
   If running remotely, you can save the plot of the energy vs imaginary time to a file using the `--savefig [name].png` option.
   In either case, you should see the following output if you used the same settings as in the provided `afqmc.json` file and
   above.


See ``run.sh`` for execution details. The ``dice/`` directory contains an alternative trial wavefunction generation approach using selected CI.

Files
-----

**SCF/CASSCF Calculation** (``scf/scf.py``):

.. literalinclude:: scf/scf.py
   :language: python

**Hamiltonian and Wavefunction Generation** (``ham/ham.py``):

.. literalinclude:: ham/ham.py
   :language: python




