.. _example_molecule_o_atom:

Oxygen Atom (UHF Trial Wavefunction)
====================================

This example demonstrates an AFQMC calculation of an isolated oxygen atom using an unrestricted Hartree-Fock (UHF) trial wavefunction.
The Oxygen atom has a gorund state with :math:`S=2` (quartet state)

Running the Example
-------------------

The workflow includes the following steps:

1. **Generate the trial wavefunction and orbital basis**: Execute ``scf/scf.py`` to run PySCF calculations generating both ROHF and UHF checkpoint files. 

2. **Create AFQMC input files**: Execute ``ham/ham.py`` to generate the Hamiltonian in Cholesky-decomposed form and format the trial wavefunction. Note that we used the ROHF orbitals as a basis, since they are spin-independent. The UHF wavefunction will be used as the trial wavefunction for AFQMC. The `write_wfn_mol ()` function directly handles the change of basis as demonstrated in this example. This produces ``afqmc.h5`` containing the Hamiltonian and trial wavefunction. 

3. **Run SAFIRE**: Run SAFIRE using the provided json input file (see below):

   .. code-block:: bash

      $ mpirun -n 16 safire afqmc.json

   The AFQMC calculation will perform sampling and output energy estimates and other observables.

4. **analyze the results**: Use the `scalar_stats` command-line tool from afqmctools to analyze the energy output:

   .. code-block:: bash

      $ scalar_stats qmc.s000.scalar.dat -s time -e 5.0 -t

   This will compute the average AFQMC energy and stochastic uncertainty, using an equilibration time of 5.0 :math:`Ha^{-1}`.
   The `-t` flag will generate a plot of the energy as a function of imaginary time if running locally.
   If running remotely, you can save the plot of the energy vs imaginary time to a file using the `--savefig [name].png` option.
   In either case, you should see the following output if you used the same settings as in the provided `afqmc.json` file and
   above.

   .. code-block:: text

      ====== [analyze_scalar_data Settings] ======

      [+] fname            = qmc.s000.scalar.dat
      [+] mark_header      = #               
      [+] series_column    = time            
      [+] nequil           = 5.0             
      [+] estimate_equil   = False           
      [+] column           = LocalEnergy     
      [+] reblock          = 1               
      [+] ndiscard         = None            
      [+] list             = False           
      [+] trace            = False           
      [+] append           = None            
      [+] dump             = False           
      [+] dump_fname       = trace.dat       
      [+] verbose          = True            
      [+] autocorr         = None            
      [+] savefig          = None            
      [+] dump_avail_columns = False           

      AFQMC Energy   -74.909596 +/-   0.000328 4.22  5.0/60.0


Files
-----

**SCF Calculation** (``scf/scf.py``):

.. literalinclude:: scf/scf.py
   :language: python

**Hamiltonian and Wavefunction Generation** (``ham/ham.py``):

.. literalinclude:: ham/ham.py
   :language: python

**AFQMC Input File** (``afqmc.json``):

.. literalinclude:: afqmc/afqmc.json
   :language: json
