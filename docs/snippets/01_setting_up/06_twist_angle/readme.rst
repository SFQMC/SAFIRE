.. _setup_ex_6:

Adding twist angles
^^^^^^^^^^^^^^^^^^^

This example covers building a Hubbard model Hamiltonian on square Lattice with 
twist angles applied and generating a free-electron trial wavefunction.

A lattice model Hamiltonian can be generated using afqmctools and
a toml-based input file.
Below is a sample input file, which we name `input1.toml`, for a Hubbard model 
on a 4x8 square lattice with periodic boundary conditions.

.. literalinclude:: input.toml

We note that both the symbol 'Pi' and fractions 
of the form "numerator/denominator" are parsed and converted
when included as strings.
Explicit decimal inputs are, of course, also allowed

.. literalinclude:: input2.toml

afqmctools can be invoked within a Python script as

.. code-block:: python

    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    from afqmctools.wavefunction.free_electron import free_electron

    infile = "input1.toml"

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
        twist=input_params["lattice"].get("twist",None),
        output="afqmc.h5"
    )

The `free_electron()` function accepts a `twist` keyword argument for the sake of 
reproducibility for open-shell systems. 
Unlike many of the other examples, here we explicitly set the twist to that 
of the lattice.
By default, i.e. if twist is None, a small incommensurate twist angle is used instead.

If the sample Python script is run above with the sample input file, the following
should be present at the end of the output.

.. code-block:: text

    Running in Slater Detemrinant Mode
    energyCall: Etotal=(31.609146118164062-6.798654794692993e-07j) with EK=(-40.39085388183594-2.384185791015625e-07j) EU=(72-4.414469003677368e-07j) EU1=0 EU2=0 EJ=0
    Reference HF Energy = 31.609146118164062


Internally, the AutoHF Hartree-Fock code is used to evaluate the energy of the free-electron
trial wavefunction with respect to the interacting Hamiltonian.
This allows the initial energy to be checked within the AFQMC code.

.. code-block:: python

    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    import afqmctools.utils.io as io
    from afqmctools.wavefunction.free_electron import free_electron

    infile = "input_charge.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(infile).hamiltonian
    nelec = io.read_input_params(infile)["misc_params"]["nelec"]
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
    energyCall: Etotal=(-13.653738975524902+0j) with EK=(-13.951533317565918+0j) EU=(0.2977941334247589+0j) EU1=0 EU2=0 EJ=0
    Reference HF Energy = -13.653738975524902

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
