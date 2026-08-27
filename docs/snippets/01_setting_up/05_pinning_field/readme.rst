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

A lattice model Hamiltonian can be generated using afqmctools and
a toml-based input file.
Below is a sample input file, which we name `input_afm.toml`, for a Hubbard model 
on a 8x4 square lattice with periodic boundary conditions.

.. literalinclude:: input_afm.toml

afqmctools can be invoked within a Python script as

.. code-block:: python

    from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
    import afqmctools.utils.io as io
    from afqmctools.wavefunction.free_electron import free_electron

    infile = "input_afm.toml"

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
    energyCall: Etotal=(-14.568331718444824+0j) with EK=(-14.848626136779785+0j) EU=(0.2802944779396057+0j) EU1=0 EU2=0 EJ=0
    Reference HF Energy = -14.568331718444824


Internally, the AutoHF Hartree-Fock code is used to evaluate the energy of the free-electron
trial wavefunction with respect to the interacting Hamiltonian.
This allows the initial energy to be checked within the AFQMC code.


**Charge Pinning**

Charge pinning is applying a constant charge field :math:`h \hat{n}_i` applied along the edge. More flexible pinning is planned for the future.
The charge pinning case can be set up similarly using

.. literalinclude:: input_charge.toml

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
