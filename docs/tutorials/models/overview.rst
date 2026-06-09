.. _lattice_model_overview:

SAFIRE for lattice models
==========================

.. image:: https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/00_top_level/lattice_model.png
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

:math:`\mu` / :math:`\nu` are compound indices, :math:`\mu = (i,p,m)` / :math:`\mu = (j,q,n)`, that include the lattice site index, (:math:`i` / :math:`j`), sublattice, (:math:`p` / :math:`q`) and band (:math:`m` / :math:`n`) indices.
This is known as the Hubbard-Kanamori Hamiltonian.
Many standard model Hamiltonians can be written in this language including the standard Hubbard model,
extended Hubbard with nearest-neighbor :math:`V`, etc.
SAFIRE is agnostic to the details of the compound indices.
It simply accepts Hamiltonian "components" which are interpreted as above.
This provides great flexibility in specifying Hamiltonians; however, if Hamiltonians are generated externally, care must be taken
to ensure that consistent index conventions are used.

``afqmctools`` provides a framework for building lattice model Hamiltonians that can generate broad classes of lattice model Hamiltonians
on a variety of lattices, using consistent conventions for indexing.
The framework consists of a Lattice class which is responsible for geometry (see the `lattice tutorial <https://colab.research.google.com/drive/1fefjMiXED0shSl4RJKv3Zu93l9FOw5sj>`_),
a Hamiltonian builder which is responsible for generating specific Hamiltonian terms on demand given a specific Lattice instance (See the `Hamiltonian Builder <https://colab.research.google.com/drive/1iUBLuxhuvfbNOYgT-K63_HrL-pzVq25g>`_),
and a Hamiltonian "Director" which is responsible for choosing which build steps to perfrom based on
a set of input Hamiltonian parameters.
Each component of the framework can be used directly; however
the Director class represents the highest-level interface of the lattice Hamiltonian framework and
can manage the underlying Lattice instance and Builder instance.
Users who want direct control over the Hamiltonian build steps can directly
use the Hamiltonian "Builder". See the `Hamiltonian Builder <https://colab.research.google.com/drive/1iUBLuxhuvfbNOYgT-K63_HrL-pzVq25g>`_ for
more detail.
It is recommended to use the Director whenever possible.

.. image:: https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/models/00_overview/HamilDirectorSystem.png
   :width: 1000


Imaginary-Time Propagation
--------------------------

Explain the trotterization, and Hubbard-Stratonovich transformation for the lattice Hamiltonian. 
Be sure to explicitly show each type of Hubbard-Stratonovich transformation (discrete spin, continuous charge, discrete charge, continuous spin).

Trial wavefunctions
-------------------

A trial wavefunction is used in AFQMC which is, in general, some linear
combination of Slater determinants,

.. math::

   | \Psi_T \rangle = \sum_n C_n |\Phi_m\rangle,

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
   :hidden:

   01_hello_safire/01_hello_safire_executed.ipynb
   02_understanding_the_input_file/02_understanding_the_input_file_executed.ipynb
   03_setting_up_a_lattice/03_setting_up_a_lattice_executed.ipynb
   04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian_executed.ipynb
   06_writing_a_trial_wavefunction/06_writing_a_trial_wavefunction_executed.ipynb
   07_computing_observables/07_computing_observables_executed.ipynb

1. `Hello SAFIRE <01_hello_safire/01_hello_safire_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1zfdEh-c1mClzhXGOVdzANNzyhG1gBMBG>`__ 🧑‍💻
2. `Understanding the input file <02_understanding_the_input_file/02_understanding_the_input_file_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1rWeqD-DVQNMN8ILqelEZ56OURmCYpozp>`__ 🧑‍💻
3. `Setting up a Lattice <03_setting_up_a_lattice/03_setting_up_a_lattice_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1fefjMiXED0shSl4RJKv3Zu93l9FOw5sj>`__ 🧑‍💻
4. `Building and writing a Hamiltonian <04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1LMxAtaSkE1-zIKAlEBu66MFarzgHZ-8i>`__ 🧑‍💻
5. *(optional) Finer control of the Hamiltonian* | `Colab <https://colab.research.google.com/drive/1iUBLuxhuvfbNOYgT-K63_HrL-pzVq25g>`__ 🛠️
6. `Writing a Trial wavefunction <06_writing_a_trial_wavefunction/06_writing_a_trial_wavefunction_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1OxejnAc2_3a5OsCFligaHTwOkZJduMea>`__ 🧑‍💻
7. *autoHF crash course* | `Colab <https://colab.research.google.com/drive/1M9j2Hud7omJPCIkd9mdeL53K62KiMntu#scrollTo=d3YMrpqs5xcm>`__ 🛠️
8. `Computing Observables <07_computing_observables/07_computing_observables_executed.html>`__ | `Open in Colab <https://colab.research.google.com/drive/1XlSH9RKjSpcd_CmNWgFT6MQmAVqMXCM8>`__ 🧑‍💻

Worked Examples
===============

We provide the following worked examples in which we
go through the entire workflow. We use autoHF to perform
HF calculations, and afqmctools to generate Hamiltonians and write trial wavefunctions.

1. `4x4 Hubbard Model with open shell <https://colab.research.google.com/drive/1s9bdH7XgEj4qP982ZZTOGSsQkeMgAJdX>`_
2. `Stripe Ordering <https://colab.research.google.com/drive/1HrfXBp0SkiGzYWvKKJvKBQwdL-psWV-n>`_ `🧑‍💻 ready for edits 🧑‍💻`
3. `Hubbard at half-filling <https://colab.research.google.com/drive/1MjA5PnZC5V1qRjINxXdqbs1majV_FTWf>`_ `🧑‍💻 ready for edits 🧑‍💻`
4. `pair correlation functions <https://colab.research.google.com/drive/1rI5zbAB6tMAFJWRGOp1zxIpDAZIh9LTW>`_ `🛠️ Under construction 🛠️`
5. `Multi-Slater determinant trial wavefunction <https://colab.research.google.com/drive/1yTDk1u8Ww-c1t6kUKhBkEAKUIsnmGm3e>`_ `🧑‍💻 ready for edits 🧑‍💻`
6. `Hubbard on a Honeycomb lattice <https://colab.research.google.com/drive/1g9_84kc92vAyDfKHTjDiuWHXeJsp12vZ>`_  `🛠️ Under construction 🛠️`
7. `Hubbard with t-prime <https://colab.research.google.com/drive/1eR1RNS-BiMToOswQ4OyJWeijcpW21a4J>`_ `🛠️ Under construction 🛠️`
8. `Emery Model <https://colab.research.google.com/drive/1-zKd4kHd6Pu8WrDFbQzv3MqeyC4tiiQz>`_ `🛠️ Under construction 🛠️`
