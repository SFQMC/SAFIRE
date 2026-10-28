.. _Wavefunction-classes:

Wavefunction Files
------------------

The wavefunction input to SAFIRE determines how the :ref:`afqmc` random walk is importance-sampled and constrained. Wavefunctions also define the initial condition of the :ref:`Walker-classes`.

SAFIRE supports wavefunctions which are linear combinations

.. math::
  |\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle

of Slater determinants :math:`|\Phi_n\rangle` with complex coefficients :math:`C_n`.

There are two kinds of these multi determinant wavefunctions:

1. **Nonorthogonal multi-Slater determinant (NOMSD) wavefunctions**
   consist of *strictly* nonorthogonal Slater determinants. Each individual determinant is specified as a complete matrix of orbitals.
2. **Particle-hole multi-Slater determinant (PHMSD) wavefunctions**
   are configuration-interaction-like wavefunctions where all individual Slater determinants are mutually orthogonal and specified in terms of occupancy vectors of a set of reference orbitals.

General Storage Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For general HDF5 format conventions, see :ref:`storage_conventions` in :ref:`Hamiltonian-classes`.

.. _nomsd_wavefunction:

NOMSD
~~~~~

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``Wavefunction/``

  * ``NOMSD/``

    * ``🏷️ format_version`` 1
    * ``🏷️ spin_type``
    * ``ci_coeffs`` :math:`(N_\mathrm{det},)`
    * ``PsiT_<k>/`` :math:`(N_\sigma, n_\mathrm{pol} M)`, a :ref:`CSR matrix <csr_storage>`

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``spin_type``
    - The spin symmetry of the wavefunction, ``"closed"``, ``"collinear"`` or ``"noncollinear"``.
  * - ``ci_coeffs``
    - The complex coefficients :math:`C_n` of the determinants.
  * - ``PsiT_<k>``
    - The **conjugate transpose** of the orbital matrix of spin :math:`\sigma` of determinant
      :math:`n`, with :math:`k = n_\mathrm{spin} n + \sigma`: ``PsiT_<2n>`` (:math:`\uparrow`) and
      ``PsiT_<2n+1>`` (:math:`\downarrow`) for collinear wavefunctions, ``PsiT_<n>`` otherwise. It has one
      row per electron of that spin, :math:`N_\sigma = N_\uparrow` or :math:`N_\downarrow`, and one row per
      electron, :math:`N_\sigma = N`, for noncollinear wavefunctions.

.. _phmsd_wavefunction:

PHMSD
~~~~~

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``Wavefunction/``

  * ``PHMSD/``

    * ``🏷️ format_version`` 1
    * ``🏷️ spin_type`` ``"collinear"``
    * ``🏷️ number_of_orbitals`` :math:`M`
    * ``ci_coeffs`` :math:`(N_\mathrm{det},)`
    * ``occa`` :math:`(N_\mathrm{det}, N_\uparrow)`
    * ``occb`` :math:`(N_\mathrm{det}, N_\downarrow)`
    * ``PsiT_<k>/`` :math:`(M_\mathrm{ref}, n_\mathrm{pol} M)`, optional, a
      :ref:`CSR matrix <csr_storage>`

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``spin_type``
    - The spin symmetry of the wavefunction, ``"collinear"``.
  * - ``number_of_orbitals``
    - The number of spatial orbitals :math:`M`.
  * - ``ci_coeffs``
    - The complex coefficients :math:`C_n` of the determinants.
  * - ``occa``, ``occb``
    - The occupied orbitals of each determinant, integers. For example if :math:`N_\uparrow=N_\downarrow=2`,
      :math:`N_\mathrm{det}=2`, :math:`M=4`, and
      :math:`|\Psi_\mathrm{T}\rangle = |0,1\rangle|0,1\rangle + |0,1\rangle|0,2\rangle` then
      ``occa`` = :math:`[[0, 1], [0, 1]]` and ``occb`` = :math:`[[0, 1], [0, 2]]`.
  * - ``PsiT_<k>``
    - By default, the occupancies refer to the basis the Hamiltonian is written in. If these entries are present, they specify orbitals that should be used instead.
      Each is the **conjugate transpose** of the matrix of the :math:`M_\mathrm{ref}`
      reference orbitals in the basis of the Hamiltonian. There is either one, ``PsiT_0``, shared by both
      spins, or ``PsiT_0`` (:math:`\uparrow`) and ``PsiT_1`` (:math:`\downarrow`) for a spin-resolved
      reference.

