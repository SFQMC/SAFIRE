.. _setup_ex_5:

Hubbard with Pinning Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers building a Hubbard model Hamiltonian on square Lattice with 
an antiferromagnetic pinning field and a charge pinning field, 
and generating a free-electron trial wavefunction.

Currently, pinning is applied only at the boundaries perpendicular to axis 1, 
and is applied to both boundaries or not at all.
A more flexible pinning field interface is planned.

Currently the following spin pinning are available:

* AFM pinning - alternating :math:`\pm1` on each site. 
  
  * :code:`h_afm_pin` - magnitude of pinning field

  * :code:`afm_pin_type` options:

    * "staggered" or "afm" - each edge has the opposite signs (*default*)

    * "same" or "fm" - each edge has the same pinning

* FM pinning - a constant Sz field on each edge. 
  
  * :code:`h_fm_pin` - magnitude of pinning field

  * :code:`fm_pin_type` options:

    * "staggered" - +1 on one edge and -1 on the other (*default*)

    * "same" or "fm" - +1 on both edges

TODO add a picture illustrating the pinning field location!

A lattice model Hamiltonian can be generated using safiretools and
a toml-based input file.
Below is a sample input file, which we name `input_afm.toml`, for a Hubbard model 
on a 8x4 square lattice with periodic boundary conditions.

.. literalinclude:: input_afm.toml

safiretools can be invoked within a Python script as

.. code-block:: python

    from safiretools import HamiltonianBuilder, Wavefunction

    infile = "input_afm.toml"

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


**Charge Pinning**

Charge pinning is applying a constant charge field :math:`h \hat{n}_i` applied along the edge. More flexible pinning is planned for the future.
The charge pinning case can be set up similarly using

.. literalinclude:: input_charge.toml

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
