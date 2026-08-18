.. _Walker-classes:

Walker Classes
--------------

Several types of Slater determinant random walkers are implemented 
based on how spin-symmetry is handled.
Each type is described below.
The following list includes all walker types.

#. :ref:`RHF/Closed Walkers <closed_walker>`
#. :ref:`UHF/Collinear Walkers <collinear_walker>`
#. :ref:`GHF/Noncollinear Walkers <noncollinear_walker>`

The walker type can be set in the AFQMC input file using the ``walker_type`` keyword 
within a "walker_set" input block.
The walker type can be set to one of the following values:

.. _closed_walker:

RHF/Closed Walkers
~~~~~~~~~~~~~~~~~~

RHF/Closed walkers are treated as having identical spin-up and spin-down channels.
Formally,

.. math::
  
   |\Phi_k\rangle = |\Phi_{k \uparrow}\rangle \otimes |\Phi_{k \tilde{\uparrow}}\rangle,

where :math:`|\Phi_{k \uparrow}\rangle` is the spin-up channel walker with walker index :math:`k`, and 
:math:`|\Phi_{k \tilde{\uparrow}}\rangle` is the spin-down channel walker with walker index :math:`k`.
The notation :math:`\tilde{\uparrow}` is used to denote that the spin-down channel Slater matrix is identical to
the spin-up channel Slater matrix.

.. note::

   Closed walkers can only be used with Hamiltonians that have a "Closed" representation.

.. warning::

   Closed walkers are not allowed in calculations using Lattice Model Hamiltonians.

.. _collinear_walker:

UHF/Collinear Walkers
~~~~~~~~~~~~~~~~~~~~~

UHF/Collinear walkers are treated as having independent spin-up and spin-down channels.
Formally,

.. math::
  
   |\Phi_k\rangle = |\Phi_{k \uparrow}\rangle \otimes |\Phi_{k \downarrow}\rangle,

where :math:`|\Phi_{k \uparrow}\rangle` is the spin-up channel walker with walker index :math:`k`, and 
:math:`|\Phi_{k \downarrow}\rangle` is the spin-down channel walker with walker index :math:`k`.
Both spin channels have independent Slater matrices which are separately updated during
the random walk.

.. note::

   Collinear walkers can only be used with Hamiltonians that have a "Collinear" or "Closed" representation.

.. _noncollinear_walker:

GHF/Noncollinear Walkers
~~~~~~~~~~~~~~~~~~~~~~~~

GHF/Noncollinear walkers include both spin-up, spin-down channels as well as spin-flip channels within
a single Slater matrix.
Formally,

.. math:: |\Phi_k\rangle = \begin{bmatrix}
   \Phi^{\uparrow \uparrow}_k & \Phi^{\uparrow \downarrow}_k \\
   \Phi^{\downarrow \uparrow}_k & \Phi^{\downarrow \downarrow}_k 
   \end{bmatrix}  
   
where $\Phi^{\sigma \sigma'}_k$ represent each respective spin channel.
In practice, the entire Slater matrix is stored and propagated during the random walk
with no partitioning between spin channels.

.. note::

   Noncollinear walkers can be used with Hamiltonians that have any spin symmetry representation ("Closed", "Collinear", "Noncollinear").
