.. _example_molecule_o_atom:

Oxygen Atom (UHF Trial Wavefunction)
====================================

This example demonstrates an AFQMC calculation of an isolated oxygen atom using an unrestricted Hartree-Fock (UHF) trial wavefunction.
The Oxygen atom has a ground state with :math:`S=2` (quartet state)

Running the Example
-------------------

The workflow includes the following steps:

1. **Create AFQMC input files**: Execute ``ham/ham.py``. It runs the ROHF and UHF calculations
   with PySCF, then generates the Hamiltonian in Cholesky-decomposed form and formats the
   trial wavefunction.
   Note that we used the ROHF orbitals as a basis, since they are spin-independent.
   The UHF wavefunction will be used as the trial wavefunction for AFQMC.
   ``Wavefunction.from_pyscf()`` directly handles the change of basis as demonstrated in 
   this example. This produces ``afqmc.h5`` containing the Hamiltonian and trial wavefunction. 

2. **Run SAFIRE**: Run SAFIRE using the provided JSON job file (see below):

   .. code-block:: bash

      $ mpirun -n 16 safire afqmc.json

   The AFQMC calculation will perform sampling and output energy estimates and other observables.

3. **Analyze the results**: Use ``safiretools`` to analyze the energy in the results file ``qmc.results.h5``:

   .. code-block:: python

      from safiretools import Results, vis

      results = Results("qmc.results.h5")
      vis.plot_timeseries(results)

   This prints the average AFQMC energy and its stochastic uncertainty and plots the energy as a function of imaginary time.
   The first 500 steps (an imaginary time of 5.0 :math:`\mathrm{Ha}^{-1}`) are discarded as equilibration, as set by
   ``equilibration_steps`` in the provided ``afqmc.json``.
   If you used the same settings, you should get an output compatible with

   .. code-block:: text

      Energy = -74.90960 ± 0.00033


Files
-----

**SCF, Hamiltonian and Wavefunction Generation** (``ham/ham.py``):

.. literalinclude:: ham/ham.py
   :language: python

**AFQMC Input File** (``afqmc.json``):

.. literalinclude:: afqmc/afqmc.json
   :language: json
