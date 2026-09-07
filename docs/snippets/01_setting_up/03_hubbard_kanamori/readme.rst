.. _setup_ex_3:

Hubbard-Kanamori Model
^^^^^^^^^^^^^^^^^^^^^^
This example covers building a two-band Hubbard-Kanamori model Hamiltonian on a square Lattice, 
and generating a free-electron trial wavefunction.
The lattice is 1-d with periodic boundary conditions only
in the long direction.

A lattice model Hamiltonian can be generated using safiretools and
a toml-based input file.
Below is a sample input file, which we name `input.toml`, for a Hubbard model 
on a 6x1 square lattice with periodic boundary conditions in the long direction
such that the system is effectively 1-dimensional.

.. literalinclude:: input.toml

safiretools can be invoked within a Python script as

.. code-block:: python

    from safiretools import HamiltonianBuilder, Wavefunction

    infile = "input.toml"

    # Build and save a lattice model Hamiltonian
    hamiltonian = HamiltonianBuilder.from_input(infile).get_hamiltonian()
    hamiltonian.to_hdf5("afqmc.h5")

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=infile,
        nelec=hamiltonian.nelec,
    ).to_hdf5("afqmc.h5")

`Wavefunction.from_free_electron()` accepts a `twist` keyword argument for the
sake of reproducibility for open-shell systems.
`twist` should be a 2-d iterable containing :math:`\vec{\theta}=(\theta_1,\theta_2)`
in radians even if one component is set to 0.0.
By default, a small incommensurate twist angle is used.

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
