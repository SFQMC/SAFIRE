.. _setup_ex_3:

Hubbard-Kanamori Model
^^^^^^^^^^^^^^^^^^^^^^
This example covers building a two-band Hubbard-Kanamori model Hamiltonian on a square Lattice, 
and generating a free-electron trial wavefunction.
The lattice is 1-d with periodic boundary conditions only
in the long direction.

A lattice model Hamiltonian can be generated using afqmctools and
a toml-based input file.
Below is a sample input file, which we name `input.toml`, for a Hubbard model 
on a 6x1 square lattice with periodic boundary conditions in the long direction
such that the system is effectively 1-dimensional.

.. literalinclude:: input.toml

afqmctools can be invoked within a Python script as

.. code-block:: python

    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    import afqmctools.utils.io as io
    from afqmctools.wavefunction.free_electron import free_electron

    infile = "input.toml"

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

The `free_electron()` function accepts a `twist` keyword argument for the sake of 
reproducibility for open-shell systems. 
`twist` should be a 2-d iterable containing :math:`\vec{\theta}=(\theta_1,\theta_2)`
in radians even if one component is set to 0.0.
By default, a small incommensurate twist angle is used.

If the sample Python script is run above with the sample input file, the following
should be present at the end of the output.

.. code-block:: text

    Running in Slater Detemrinant Mode
    energyCall: Etotal=(-15.473198890686035+0j) with EK=(-16+0j) EU=(0.5268006920814514+0j) EU1=0 EU2=0 EJ=0
    Reference HF Energy = -15.473198890686035


Internally, the AutoHF Hartree-Fock code is used to evaluate the energy of the free-electron
trial wavefunction with respect to the interacting Hamiltonian.
This allows the initial energy to be checked within the AFQMC code.

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
