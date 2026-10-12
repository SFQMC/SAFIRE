.. _tutorials:

Tutorials
=========

.. toctree::
   :hidden:

   tutorials/molecules/overview
   tutorials/solids/overview
   tutorials/models/overview

Auxiliary-field quantum Monte Carlo (AFQMC) for general interacting electron systems.

SAFIRE is a fast, efficient implementation of AFQMC written in C++. It uses a general interacting second-quantized Hamiltonian allowing it to be applied to broad classes of interacting-electron systems.

.. image:: /images/shared/00_top_level/software_components.png
   :width: 800px
   :align: center

SAFIRE comes with the safiretools Python package to assist the user in both setting up second quantized Hamiltonians and wavefunctions before running SAFIRE, and in post-processing the results after running SAFIRE.

Typical Workflow
----------------

A typical calculation using SAFIRE consists of three steps as shown in the flowchart below.

.. image:: /images/shared/00_top_level/TopLevelFlowChart_v3.png
   :width: 1000px
   :align: center

Running the SAFIRE executable is generic and the details are largely independent of the intended application area. The most variation occurs in the first step, Setup, since the set of possible starting points is very large. Post processing beyond simple scalar observables similarly has some dependence on the details of the application area. For example, the real-space charge density depends explicitly on the real-space representation of the underlying orbitals.

Common Application Areas
------------------------

While AFQMC can handle any interacting electron system, it is most commonly applied in the following application areas. **Click on one of the boxes below to learn more.**

Quantum Chemistry
~~~~~~~~~~~~~~~~~

:ref:`molecules_overview`.

Molecules using standard quantum chemistry basis sets

.. image:: /images/shared/00_top_level/molecule.png
   :width: 500px
   :align: center

Ab initio solids
~~~~~~~~~~~~~~~~

:ref:`solids_overview`.

Ab initio solids using Kohn-Sham orbital basis sets

.. image:: /images/shared/00_top_level/solid.png
   :width: 500px
   :align: center

Lattice models
~~~~~~~~~~~~~~

:ref:`lattice_model_overview`

Model Hamiltonians defined on a lattice such as the Hubbard model

.. image:: /images/shared/00_top_level/lattice_model.png
   :width: 500px
   :align: center
