.. _Walker-classes:

Walkers
-------

SAFIRE supports several representations of walkers in the :ref:`afqmc` algorithm. Unless ``walker_type`` is set in the :ref:`walker_set_block` block of the job file, the code infers the walker type from the input files described in :ref:`Hamiltonian-classes` and :ref:`Wavefunction-classes`, choosing the highest spin symmetry that all of them allow.

The currently supported walkers are all Slater determinants of different spin symmetries.


.. _closed_walker:

Closed Walkers
~~~~~~~~~~~~~~

Closed walkers have identical spin-up and spin-down orbitals,

.. math::
  
   |\Phi_\mathrm{closed}\rangle = |\Phi\rangle \otimes |\Phi\rangle
   = \prod_{\sigma} \prod_{k=1}^{N_\uparrow} \Big( \sum_{i} \Phi_{ik} \, c^\dagger_{i\sigma} \Big) |0\rangle,

where both spins share the same
:math:`M \times N_\uparrow` Slater matrix :math:`\Phi`, and :math:`N_\uparrow = N_\downarrow`.

.. note::

   Closed walkers are also called restricted Hartree-Fock (RHF) walkers due to their resemblance to the RHF wavefunction.


.. _collinear_walker:

Collinear Walkers
~~~~~~~~~~~~~~~~~

Collinear walkers have :math:`N_\uparrow` spin-up and :math:`N_\downarrow` spin-down orbitals,

.. math::
  
   |\Phi_\mathrm{collinear}\rangle = |\Phi^{\uparrow}\rangle \otimes |\Phi^{\downarrow}\rangle
   = \prod_{\sigma} \prod_{k=1}^{N_\sigma} \Big( \sum_{i} \Phi^{\sigma}_{ik} \, c^\dagger_{i\sigma} \Big) |0\rangle,

where :math:`|\Phi^{\uparrow}\rangle` is the spin-up and :math:`|\Phi^{\downarrow}\rangle` the spin-down Slater
determinant, with the :math:`M \times N_\sigma` Slater matrices :math:`\Phi^\sigma`. This structure causes all electron spins to be collinear, i.e. aligned with the quantization axis.

.. note::

   Collinear walkers are also called unrestricted Hartree-Fock (UHF) walkers due to their resemblance to the UHF wavefunction.

.. _noncollinear_walker:

Noncollinear Walkers
~~~~~~~~~~~~~~~~~~~~~~~~

Noncollinear walkers include spinor orbitals that mix spin flavor. There is no clear notion of spin-up or down electrons.

.. math:: |\Phi_\mathrm{noncollinear}\rangle = \Bigg|\begin{matrix}
   \Phi^{\uparrow} \\
   \Phi^{\downarrow}
   \end{matrix} \Bigg\rangle
   = \prod_{k=1}^{N} \Big( \sum_{i\sigma} \Phi^{\sigma}_{ik} \, c^\dagger_{i\sigma} \Big) |0\rangle,

where the :math:`M \times N` blocks :math:`\Phi^{\sigma}` of the :math:`2M \times N` Slater matrix represent the
spin-up and spin-down parts of each of the :math:`N` spinor orbitals. Unlike for the
collinear walker, every orbital :math:`k` is a superposition of both spins.

.. note::

   Noncollinear walkers are also called general Hartree-Fock (GHF) walkers due to their resemblance to the GHF wavefunction.
