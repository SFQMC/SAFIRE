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

    # Build and save a lattice model Hamiltonian. Its lattice already carries
    #   the twist set in the [lattice] section of input.toml.
    hamiltonian = HamiltonianBuilder.from_input(source=input_params).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

    # compute and save a free-electron trial wfn, into the same file. The
    #   lattice is already twisted, so one Hamiltonian serves both the trial
    #   wavefunction and the AFQMC run.
    Wavefunction.from_free_electron(
        source=hamiltonian,
        nelec=hamiltonian.nelec,
    ).to_hdf5("afqmc.h5")

The twist is a property of the **lattice**:
`Wavefunction.from_free_electron()` uses the Hamiltonian exactly as given and
applies no twist of its own.

In this example, the lattice we want to
simulate at the AFQMC level of theory is already twisted.
When the lattice of interest carries no twist, an open-shell free-electron
which orbitals of a degenerate shell you occupy is
arbitrary, and `from_free_electron()` warns when it has to make that choice.
The fix is to build a **second**, twisted Hamiltonian for the trial
wavefunction alone, and to hand AFQMC the untwisted one. The block below and
the other examples in this section all follow that pattern.

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.

.. code-block:: python

    from safiretools import HamiltonianBuilder, Lattice, Wavefunction
    from safiretools.wavefunction.free_electron import DEFAULT_TWIST

    infile = "input_charge.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(infile).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

  # The trial wavefunction needs its *own* Hamiltonian, built on a lattice
    #   carrying a small irrational twist.
    #   The AFQMC run itself uses the untwisted Hamiltonian above.
    fe_lattice = Lattice.from_dict(
        dict(hamiltonian.lattice_params, twist=list(DEFAULT_TWIST))
    )
    fe_hamiltonian = HamiltonianBuilder.from_input(
        infile,
        lattice=fe_lattice,
    ).get_hamiltonian()

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=fe_hamiltonian,
        nelec=hamiltonian.nelec,
    ).to_hdf5("afqmc.h5")

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
