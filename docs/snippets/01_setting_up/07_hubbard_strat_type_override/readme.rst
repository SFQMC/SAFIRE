.. _setup_ex_7:

Hubbard-Stratonovich Transformation Type Override
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WARNING: This functionality is for experts only. Use at your own risk.

This example covers an advanced feature where we force a specific Hubbard-Stratonovich transformation (HST) type.
By default, the Hamiltonian builder will choose an appropriate HST type based on the sign 
of the interaction.
For band-dependent interactions, it will split each interaction into two separate terms if 
there are both positive and negative interaction strengths and assign an HST type separately.
The builder supports forcing a specific HST type; however, it will be applied to the specified 
interaction term regardless of the sign of the interaction strength.

.. literalinclude:: input.toml

safiretools can be invoked within a Python script as

.. code-block:: python

    from safiretools import HamiltonianBuilder, Wavefunction

    infile = "input.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(source=infile).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=infile,
        nelec=hamiltonian.nelec,
    ).to_hdf5("afqmc.h5")

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.
