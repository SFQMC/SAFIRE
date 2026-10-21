.. _safiretools:

safiretools
===========

safiretools is the companion Python package of SAFIRE. It contains tools to

- prepare the Hamiltonian and Wavefunction HDF5 files SAFIRE reads

- read and postprocess the results of a SAFIRE calculation

- visualize common inputs and outputs such as lattices and timeseries


Installation
------------

safiretools is installed with ``pip`` from the ``utils`` directory of the SAFIRE source
tree:

.. code-block::

    $ pip install "./utils[GROUPNAME]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Additional functionality is enabled by optional dependency groups, installed via

.. code-block::

    $ pip install "./utils[GROUPNAME]"

where ``GROUPNAME`` is one of the groups below.

============ =========================== ===================================================================
 Group Name   installs                    enables
============ =========================== ===================================================================
 pyscf        pyscf>=2.4.0                building Hamiltonians and wavefunctions from PySCF calculations
 vis          matplotlib                  the plotting helpers in ``safiretools.vis``
 testing      pytest                      running the safiretools test suite
 docs         sphinx and its extensions   building this documentation
============ =========================== ===================================================================

The `AutoHF <https://github.com/SFQMC/AutoHF>`__ Hartree-Fock solver for lattice models
is a separate package, installed with

.. code-block::

  pip install sfqmc-autohf

API Reference
-------------

The public interface of safiretools is what the top-level package exports; anything
reachable only through a deeper module path is an implementation detail.

Use this documentation as a reference. For practical instructions, the :ref:`tutorials` and :ref:`user_examples` are a good starting point.

safiretools
~~~~~~~~~~~

.. automodule:: safiretools
   :members:
   :imported-members:
   :show-inheritance:
   :member-order: bysource

safiretools.vis
~~~~~~~~~~~~~~~

.. automodule:: safiretools.vis
   :members:
