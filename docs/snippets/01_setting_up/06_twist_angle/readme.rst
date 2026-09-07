.. _setup_ex_6:

Adding twist angles
^^^^^^^^^^^^^^^^^^^

This example covers building a Hubbard model Hamiltonian on square Lattice with 
twist angles applied and generating a free-electron trial wavefunction.

A lattice model Hamiltonian can be generated using safiretools and
a toml-based input file.
Below is a sample input file, which we name `input.toml`, for a Hubbard model
on a 4x8 square lattice with periodic boundary conditions.

.. literalinclude:: input.toml

We note that both the symbol 'Pi' and fractions 
of the form "numerator/denominator" are parsed and converted
when included as strings.
Explicit decimal inputs are, of course, also allowed

.. literalinclude:: input2.toml

safiretools can be invoked within a Python script as

.. code-block:: python

    import toml

    from safiretools import HamiltonianBuilder, Wavefunction

    input_params = toml.load("input.toml")

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(source=input_params).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=input_params,
        nelec=hamiltonian.nelec,
        twist=input_params["lattice"].get("twist", None),
    ).to_hdf5("afqmc.h5")

`Wavefunction.from_free_electron()` accepts a `twist` keyword argument for the
sake of reproducibility for open-shell systems.
Unlike many of the other examples, here we explicitly set the twist to that 
of the lattice.
By default, i.e. if twist is None, a small incommensurate twist angle is used instead.

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.

.. code-block:: python

    from safiretools import HamiltonianBuilder, Wavefunction

    infile = "input_charge.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(infile).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=infile,
        nelec=hamiltonian.nelec,
    ).to_hdf5("afqmc.h5")

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
