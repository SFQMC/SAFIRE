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

afqmctools can be invoked within a Python script as

.. code-block:: python

    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    from afqmctools.wavefunction.free_electron import free_electron

    infile = "input.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(source=infile).hamiltonian
    input_params = io.read_input_params(infile)
    nelec = input_params["misc_params"]["nelec"]
    io.write_model_hamiltion(
        hamiltonian=hamiltonian,
        fname="afqmc.h5",
        nelec=nelec
    )

    # compute and save a free-electron trial wfn
    free_electron(
        source=infile,
        nelec=nelec,
        output="afqmc.h5"
    )

If the sample Python script is run above with the sample input file, the following
should be present at the end of the output.

.. code-block:: text

    Running in Slater Detemrinant Mode
    energyCall: Etotal=(-15.444445610046387+0j) with EK=(-18+0j) EU=(2.7777771949768066+0j) EU1=(-0.6666667461395264+0j) EU2=(0.444444477558136+0j) EJ=(5.796812270493889e-33+0j)
    Reference HF Energy = -15.444445610046387

Internally, the AutoHF Hartree-Fock code is used to evaluate the energy of the free-electron
trial wavefunction with respect to the interacting Hamiltonian.
This allows the initial energy to be checked within the AFQMC code.
