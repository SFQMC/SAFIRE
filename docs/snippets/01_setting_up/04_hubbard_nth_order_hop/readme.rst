.. _setup_ex_4:

Hubbard with nth-order hopping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers building a Hubbard model Hamiltonian on square Lattice with a t' (i.e next-nearest neighbor hopping), 
and generating a free-electron trial wavefunction.

A lattice model Hamiltonian can be generated using safiretools and
a toml-based input file.
Below is a sample input file, which we name `input.toml`, for a Hubbard model 
on a 3x2 square lattice with periodic boundary conditions.

.. literalinclude:: input.toml

safiretools can be invoked within a Python script as

.. code-block:: python

    from safiretools import HamiltonianBuilder, Lattice, Wavefunction
    from safiretools.wavefunction.free_electron import DEFAULT_TWIST

    infile = "input.toml"

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

`Wavefunction.from_free_electron()` uses the Hamiltonian exactly as given and
applies no twist of its own, so the twist belongs on the lattice you build that
Hamiltonian from. `twist` should be a 2-d iterable containing
:math:`\vec{\theta}=(\theta_1,\theta_2)` in radians even if one component is
set to 0.0.
For an open-shell filling, an untwisted lattice leads to an abitrary choice 
of which degenerate orbitals are occupied.
`from_free_electron()` warns when that happens.

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
