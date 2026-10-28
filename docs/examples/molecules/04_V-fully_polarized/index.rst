.. _example_molecule_v_fully_polarized:

Vanadium Atom (Fully Polarized)
===============================

This example demonstrates a fully polarized AFQMC calculation on an isolated vanadium atom
which has a ground state with :math:`S=3/2` (quartet state).
We will use the frozen core approximation to freeze all electrons except for the :math:`3d^3` valence electrons.
In the cc-pVDZ basis set, this results in 32 active orbitals and 3 active electrons.
We will use CASCI to compute the exact energy within the active space as a reference for comparison with AFQMC.

Running the Example
-------------------

The workflow follows these steps:

1. **Create AFQMC inputs**: Execute ``inputs/setup.py``. It first performs ROHF on the vanadium
   atom, whose orbitals we will use as a basis and from which the trial wavefunction is built.
   It also runs a CASCI(32o,3e) calculation to provide a reference energy.

   The ROHF energy should be around -942.884910 Ha, and the CASCI(32o,3e) energy should be
   around -942.902127 Ha, where the contribution from the active space is -3.535253 Ha.

   .. note::

      An isolated vanadium atom has other ROHF solutions, and the CASCI and
      AFQMC energies quoted in this example all assume the lowest one. ``setup.py`` therefore runs
      an internal stability analysis, re-solving from any instability it finds, and then checks the
      total energy against the value above.

   It then generates the SAFIRE inputs.
   Note that script uses an active space for AFQMC of ``cas_afqmc = (3,32)``.
   This treats the system as having all electrons in the spin-up channel.
   The script writes both the Hamiltonian and a fully polarized trial wavefunction to ``afqmc.h5``.
   A trial with no beta electrons is collinear with ``ndown == 0``, so ``walker_type`` in
   ``afqmc.json`` is ``"collinear"`` and the wavefunction's beta blocks are written with zero width.

2. **Run SAFIRE**: Execute SAFIRE using the provided JSON job file ``afqmc.json``:

   First, go to the ``inputs`` directory, then run SAFIRE with the following command:

   .. code-block:: bash

      mpirun -n 64 safire afqmc.json

3. **Analyze the results**: Use ``safiretools`` to analyze the energy in the results file ``inputs/qmc.results.h5``:

   .. code-block:: python

      from safiretools import Results, vis

      results = Results("qmc.results.h5")
      vis.plot_timeseries(results)

   This prints the average AFQMC energy and its stochastic uncertainty and plots the energy as a function of imaginary time.
   The first 1000 steps (an imaginary time of 5.0 :math:`\mathrm{Ha}^{-1}`) are discarded as equilibration, as set by
   ``equilibration_steps`` in the provided ``afqmc.json``.
   If you used the same settings, you should get an output compatible with

   .. code-block:: text

      Energy = -942.90168 ± 0.00015


4. compare with the reference CASCI energy: The AFQMC energy, -942.9017(2) Ha, agrees with the CASCI energy, -942.902127 Ha,
   which is the exact energy within the active space, to well within chemical accuracy.

``inputs/run_safire_cpu.sh`` is a sample Slurm batch script for step 2.
For an alternative way to build the trial wavefunction, using selected CI instead of a single
determinant, see :doc:`../02_B_atom_SHCI_trial_wfn/06_SHCI_trial_wavefunction`.

Files
-----

**SCF/CASCI Calculation and Input Generation** (``inputs/setup.py``):

.. literalinclude:: inputs/setup.py
   :language: python

