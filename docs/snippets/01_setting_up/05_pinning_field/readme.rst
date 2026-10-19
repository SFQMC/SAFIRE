.. _setup_ex_5:

Hubbard with Pinning Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers building a Hubbard model Hamiltonian on square Lattice with 
an antiferromagnetic pinning field and a charge pinning field, 
and generating a free-electron trial wavefunction.

Pinning fields are on-site potentials on the edges of the lattice.
They are passed to the Hamiltonian as ``pinning=(field_0, field_1)``, with one
entry per lattice axis: ``pinning[a]`` acts on the two edges at the ends of
axis :math:`a`, i.e. the sites at lattice coordinate 0 (edge 0) and
:math:`L_a-1` (edge 1) along that axis.
An entry of ``None`` leaves that axis unpinned.
A site on the edges of both axes gets the sum of the two fields, and every
band of a pinned site gets the same field.

Each field is a ``PinningField``. The following are available:

* ``PinningField.afm(h, staggered=...)`` - antiferromagnetic field,
  :math:`\pm h/2` on spin up (and the opposite sign on spin down),
  alternating from site to site along the edge.
  If ``staggered``, the two edges have opposite signs;
  otherwise both edges have the same pattern.

* ``PinningField.fm(h, staggered=...)`` - ferromagnetic field,
  :math:`\pm h/2` on spin up (and the opposite sign on spin down),
  constant along the edge.
  If ``staggered``, the two edges have opposite signs;
  otherwise both edges have the same sign.

* ``PinningField.charge(h, staggered=...)`` - charge field,
  :math:`\pm h/2` on both spins, constant along the edge.
  If ``staggered``, the two edges have opposite signs;
  otherwise both edges have the same sign.

* ``PinningField(func)`` - a custom field. ``func(n, edge)`` returns the pair
  ``(h_up, h_down)`` of on-site potentials of the edge site at coordinate ``n``
  along the edge, on edge ``0`` or ``1``.
  For example, ``PinningField(lambda n, edge: (0.25 * (-1)**n, -0.25 * (-1)**n))``
  is the same as ``PinningField.afm(0.5, staggered=False)``.

TODO add a picture illustrating the pinning field location!

A lattice model Hamiltonian can be generated using safiretools and
a toml-based input file.
Below is a sample input file, which we name ``input_afm.toml``, for a Hubbard model
on a 8x4 square lattice with periodic boundary conditions.

.. literalinclude:: input_afm.toml

A ``PinningField`` cannot be written in the input file, so the pinning field is
added when the Hamiltonian is built.
safiretools can be invoked within a Python script as

.. code-block:: python

    import tomllib

    from safiretools import Lattice, LatticeHamiltonian, PinningField, Wavefunction
    from safiretools.wavefunction.free_electron import DEFAULT_TWIST

    infile = "input_afm.toml"
    with open(infile, "rb") as f:
        params = tomllib.load(f)

    # AFM pinning on the two edges at the ends of axis 0
    pinning = (PinningField.afm(0.5, staggered=True), None)

    # Build and save a lattice model Hamiltonian
    lattice = Lattice.from_type(**params["lattice"])
    hamiltonian = LatticeHamiltonian.from_lattice(
        lattice, **params["hamiltonian"], pinning=pinning
    )
    hamiltonian.to_hdf5("afqmc.h5")

    # The trial wavefunction needs its *own* Hamiltonian, built on a lattice
    #   carrying a small irrational twist.
    #   The AFQMC run itself uses the untwisted Hamiltonian above.
    fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=DEFAULT_TWIST))
    fe_hamiltonian = LatticeHamiltonian.from_lattice(
        fe_lattice, **params["hamiltonian"], pinning=pinning
    )

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=fe_hamiltonian,
        nelec=params["nelec"],
    ).to_hdf5("afqmc.h5")

``Wavefunction.from_free_electron()`` uses the Hamiltonian exactly as given and
applies no twist of its own, so the twist belongs on the lattice you build that
Hamiltonian from. ``twists`` should be a 2-d iterable containing
:math:`\vec{\theta}=(\theta_1,\theta_2)` in radians even if one component is
set to 0.0.
For an open-shell filling, an untwisted lattice leads to an abitrary choice 
of which degenerate orbitals are occupied.
``from_free_electron()`` warns when that happens.

Building the trial wavefunction does not measure its energy.
To evaluate the variational energy of a trial wavefunction with respect to the
interacting Hamiltonian, run the AutoHF Hartree-Fock solver explicitly, as shown
in :ref:`setup_ex_9`.


**Charge Pinning**

Charge pinning applies a constant charge field :math:`\frac{h}{2} \hat{n}_i` along the edge.
The charge pinning case can be set up similarly using

.. literalinclude:: input_charge.toml

.. code-block:: python

    import tomllib

    from safiretools import Lattice, LatticeHamiltonian, PinningField, Wavefunction
    from safiretools.wavefunction.free_electron import DEFAULT_TWIST

    infile = "input_charge.toml"
    with open(infile, "rb") as f:
        params = tomllib.load(f)

    # charge pinning on the two edges at the ends of axis 0
    pinning = (PinningField.charge(0.5, staggered=False), None)

    # Build and save a lattice model Hamiltonian
    lattice = Lattice.from_type(**params["lattice"])
    hamiltonian = LatticeHamiltonian.from_lattice(
        lattice, **params["hamiltonian"], pinning=pinning
    )
    hamiltonian.to_hdf5("afqmc.h5")

    # The trial wavefunction needs its *own* Hamiltonian, built on a lattice
    #   carrying a small irrational twist. The twist lifts the k-space
    #   degeneracies that leave an open-shell free-electron determinant
    #   ill-defined; the AFQMC run itself uses the untwisted Hamiltonian above.
    fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=DEFAULT_TWIST))
    fe_hamiltonian = LatticeHamiltonian.from_lattice(
        fe_lattice, **params["hamiltonian"], pinning=pinning
    )

    # compute and save a free-electron trial wfn, into the same file
    Wavefunction.from_free_electron(
        source=fe_hamiltonian,
        nelec=params["nelec"],
    ).to_hdf5("afqmc.h5")

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
