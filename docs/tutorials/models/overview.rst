.. _lattice_model_overview:

SAFIRE for lattice models
==========================

.. image:: /images/shared/00_top_level/lattice_model.png
   :width: 500

|

Some preliminaries
==================

SAFIRE can handle model Hamiltonians of the form

.. math::

   \hat{H} = \hat{H}_t + \hat{H}_U + \hat{H}_J,

where

.. math::

   \hat{H}_t = \sum_{\mu\nu,\sigma \sigma'}t^{\sigma \sigma'}_{\mu\nu}\hat{c}^\dagger_{\mu\sigma}\hat{c}_{\nu\sigma'},

.. math::

   \hat{H}_U =  \sum_{\mu} U_i \hat{n}_{\mu\uparrow} \hat{n}_{\mu\downarrow}
   + \sum_{\mu<\nu} U_{\mu\nu}^1 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\downarrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\uparrow} )
   + \sum_{\mu<\nu} U_{\mu\nu}^2 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\uparrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\downarrow} ),

and,

.. math::

   \hat{H}_J = \sum_{\mu<\nu} J_{\mu\nu} (
        \hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\nu\uparrow}
        +\hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\nu\uparrow}
        +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\mu\uparrow}
        +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\mu\uparrow}
        ).

:math:`\mu`, :math:`\nu` are compound indices, :math:`\mu = (i,p,m)`, :math:`\nu = (j,q,n)`, that include the lattice site index, (:math:`i`, :math:`j`), sublattice, (:math:`p`, :math:`q`) and band (:math:`m`, :math:`n`) indices.
This is known as the Hubbard-Kanamori Hamiltonian.
Many standard model Hamiltonians can be written in this language including the standard Hubbard model,
extended Hubbard with nearest-neighbor :math:`V`, etc.
SAFIRE is agnostic to the details of the compound indices.
It simply accepts Hamiltonian "components" which are interpreted as above.
This provides great flexibility in specifying Hamiltonians; however, if Hamiltonians are generated externally, care must be taken
to ensure that consistent index conventions are used.

``afqmctools`` provides a framework for building lattice model Hamiltonians that can generate broad classes of lattice model Hamiltonians
on a variety of lattices, using consistent conventions for indexing.
The framework consists of a Lattice class which is responsible for geometry (see the `lattice tutorial <03_setting_up_a_lattice/03_setting_up_a_lattice_executed.html>`_),
and a Hamiltonian *Builder* which is responsible for generating specific Hamiltonian terms given a specific Lattice instance.
The ``HamiltonianBuilder.from_input()`` classmethod accepts a set of input Hamiltonian parameters (as a Python
dict or a toml file) and automatically runs the corresponding build steps, which covers most use cases.
Users who want direct control over the Hamiltonian build steps can instead construct a ``HamiltonianBuilder``
and invoke build steps directly. See the `Hamiltonian Builder <05_hamiltonian_builder/05_hamiltonian_builder.html>`_ for
more detail.

Imaginary-Time Propagation
--------------------------

Explain the Trotterization, and Hubbard-Stratonovich transformation for the lattice Hamiltonian. 
Be sure to explicitly show each type of Hubbard-Stratonovich transformation (discrete spin, continuous charge, discrete charge, continuous spin).

Trial wavefunctions
-------------------

A trial wavefunction is used in AFQMC which is, in general, some linear
combination of Slater determinants,

.. math::

   | \Psi_\mathrm{T} \rangle = \sum_n C_n |\Phi_m\rangle,

where :math:`|\Phi_m\rangle` are Slater determinants,
and :math:`C_n` is a coefficient.
There is no need for the set of Slater determinants to be mutually orthogonal.

Using a trial wavefunction which accurately captures the ordering and symmetries 
of the target system is critical for achieving good AFQMC results.
Care must always be taken to ensure that the trial wavefunction has the desired properties.

Common methods for computing a trial wavefunction include:

- Hartree-Fock
- free electron / noninteracting solution

among others.

Typical Workflow
================

|

Software prerequisites
======================

The afqmctools Python package provides tools for generating a broad range of lattice model Hamiltonians.
Additionally, afqmctools and the autoHF Hartree-Fock solver allow single-Slater determinant trial wavefunctions to be computed.
In these tutorials, we will use these tools exclusively;
however, from the tutorials, it should also be clear how to import
and use wavefunctions and/or Hamiltonians which are generated externally by the user.

See the User manual for details on `the Hamiltonian HDF5 format <https://users.flatironinstitute.org/~beskridge/auxiliary_fields/afqmc.html#lattice-model-hamiltonian>`_ and the possible `Trial wavefunction formats <https://users.flatironinstitute.org/~beskridge/auxiliary_fields/afqmc.html#wavefunction-file-formats>`_.

|

The Tutorials
=============

The following tutorials will guide you through AFQMC calculations
using SAFIRE in order to teach you the typical workflow,
and some of the main features of SAFIRE.
Each "basic" tutorial builds on the previous one.
We recommend going through them in order.

.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1
   :titlesonly:

   */*

.. seealso::
   Worked examples for :ref:`examples_models`
