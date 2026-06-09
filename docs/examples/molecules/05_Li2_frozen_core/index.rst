.. _example_molecule_li2_frozen_core:

Li₂ Molecule with Frozen Core
=============================

This example demonstrates the frozen-core approximation in AFQMC, 
where the energetic core orbitals are excluded from the correlated calculation.
This significantly reduces the computational cost while maintaining high accuracy in relative energies for many systems.
For the Li₂ molecule, we freeze the 1s core orbitals of each lithium atom, treating only the 2s valence electrons explicitly in AFQMC.
The computational savings become more significant for larger atoms with more core electrons.
We reproduce results from Purwanto, Zhang, and Krakauer (J. Chem. Theory Comput. 2013, 9, 4825–4833).
In a cc-pVDZ basis, the FCI energy is -14.9005 Ha and the AFQMC energy is -14.9017(1) Ha.

Running the Example
-------------------

The workflow includes the following steps:

1. **Generate all AFQMC inputs**: Execute the single Python script:

   .. code-block:: bash

      $ python 05_Li2_frozen_core.py

   This creates ``afqmc.h5`` containing the frozen-core Hamiltonian and trial wavefunction.
   You should see that the RHF energy of the full system is -14.8694988585082 Ha,


2. **Run SAFIRE**: Execute the AFQMC calculation using the provided JSON input file:

   .. code-block:: bash

      mpirun -n 16 safire --filenames afqmc.json

   The calculation runs only in the valence space with the core electrons frozen, significantly reducing computational cost.

3. **Analyze the results**: Use the `scalar_stats` command-line tool from afqmctools to analyze the energy output:

   .. code-block:: bash

      $ scalar_stats qmc.s000.scalar.dat -s time -t -e 15.0
      
   
   This computes the average AFQMC energy and stochastic uncertainty, using an equilibration time of 15.0 :math:`Ha^{-1}`.
   The `-t` flag generates a plot of the energy as a function of imaginary time if running locally.
   If running remotely, you can save the plot of the energy vs imaginary time to a file using the `--savefig [name].png` option.
   In either case, you should see output similar to the following if you used the same settings as in the provided `afqmc.json` file and
   above.

   .. code-block:: text

      ====== [analyze_scalar_data Settings] ======

      [+] fname            = qmc.s000.scalar.dat
      [+] mark_header      = #               
      [+] series_column    = time            
      [+] nequil           = 15.0            
      [+] estimate_equil   = False           
      [+] column           = LocalEnergy     
      [+] reblock          = 1               
      [+] ndiscard         = None            
      [+] list             = False           
      [+] trace            = True            
      [+] append           = None            
      [+] dump             = False           
      [+] dump_fname       = trace.dat       
      [+] verbose          = True            
      [+] autocorr         = None            
      [+] savefig          = None            
      [+] dump_avail_columns = False           

      AFQMC Energy   -14.901069 +/-   0.000386 15.93 15.0/70.0

   We note that the AFQMC energy of -14.9011(4) Ha is in excellent agreement with the reference AFQMC energy of -14.9017(1) Ha from the literature.

Files
-----

**Frozen-Core Preparation** (``05_Li2_frozen_core.py``):

.. literalinclude:: 05_Li2_frozen_core.py
   :language: python

This all-in-one script:

- Sets up the Li₂ molecule with PySCF at equilibrium bond length
- Performs RHF calculation to generate molecular orbitals
- Specifies active space using ``cas=(ne, no)`` notation: 2 valence electrons in all remaining orbitals (``no=-1``)
- Applies frozen-core transformation via ``write_hamil_mol`` with the ``cas`` parameter
- Writes Hamiltonian in the active space representation (including the constant core energy) to ``afqmc.h5``
- Generates the trial wavefunction via ``write_wfn_mol`` and the ``cas`` parameter

**AFQMC Input File** (``afqmc.json``):

.. literalinclude:: afqmc.json
   :language: json

This JSON file configures the AFQMC calculation parameters, including time step, number of walkers, and measurement settings.