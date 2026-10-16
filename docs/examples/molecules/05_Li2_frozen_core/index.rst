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


2. **Run SAFIRE**: Execute the AFQMC calculation using the provided JSON job file:

   .. code-block:: bash

      mpirun -n 16 safire afqmc.json

   The calculation runs only in the valence space with the core electrons frozen, significantly reducing computational cost.

3. **Analyze the results**: Use ``safiretools`` to analyze the energy in the results file ``qmc.results.h5``:

   .. code-block:: python

      from safiretools import Results, vis

      results = Results("qmc.results.h5")
      vis.plot_timeseries(results)

   This prints the average AFQMC energy and its stochastic uncertainty and plots the energy as a function of imaginary time.
   The first 1500 steps (an imaginary time of 15.0 :math:`\mathrm{Ha}^{-1}`) are discarded as equilibration, as set by
   ``equilibration_steps`` in the provided ``afqmc.json``.
   If you used the same settings, you should get an output compatible with

   .. code-block:: text

      Energy = -14.90107 ± 0.00039

   We note that the AFQMC energy of -14.9011(4) Ha is in excellent agreement with the reference AFQMC energy of -14.9017(1) Ha from the literature.

Files
-----

**Frozen-Core Preparation** (``05_Li2_frozen_core.py``):

.. literalinclude:: 05_Li2_frozen_core.py
   :language: python

This all-in-one script:

- Sets up the Li₂ molecule with PySCF at equilibrium bond length
- Performs RHF calculation to generate molecular orbitals
- Specifies active space using ``active_space=(ne, no)`` notation: 2 valence electrons in all remaining orbitals (``no=-1``)
- Applies frozen-core transformation via ``MolecularHamiltonian.from_pyscf`` with the ``active_space`` parameter
- Writes Hamiltonian in the active space representation (including the constant core energy) to ``afqmc.h5``
- Generates the trial wavefunction via ``Wavefunction.from_pyscf`` and the ``active_space`` parameter

**AFQMC Input File** (``afqmc.json``):

.. literalinclude:: afqmc.json
   :language: json

This JSON job file configures the AFQMC calculation parameters, including time step, number of walkers, and measurement settings.