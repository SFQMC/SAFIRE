.. _Hamiltonian-classes:


Hamiltonian Files
-----------------

The Hamiltonian is one of the key inputs to SAFIRE that describes the system to be simulated. In general, it takes the form of an interacting many-electron problem

.. math::
  :label: eq-h012

  \hat{H} =& E_0 + \hat{H}_1 + \hat{H}_2 \\
    =& E_0 + \sum\limits_{pq\sigma\sigma'} h^{\sigma \sigma'}_{pq} \hat{c}^\dagger_{p\sigma}\hat{c}_{q\sigma'}
    + \frac{1}{2} \sum\limits_{pqrs\sigma\sigma'} v_{pqrs}^{\sigma\sigma'} \hat{c}^\dagger_{p\sigma}
      \hat{c}^\dagger_{q\sigma'}\hat{c}_{s\sigma'}\hat{c}_{r\sigma},

with an energy offset :math:`E_0`, a one-body Hamiltonian :math:`h^{\sigma \sigma'}_{pq}` and an interaction tensor :math:`v_{pqrs}^{\sigma\sigma'}`.

To perform the :ref:`afqmc` algorithm, :math:`v_{pqrs}^{\sigma\sigma'}` has to be factorized into a sum over squared one-body operators. There are different such factorizations that have different accuracy and performance characteristics. In SAFIRE, each of these factorizations corresponds to a different input file format, which has to be prepared before the simulation is started.

In the following, we will describe the different available Hamiltonian factorizations and their on-disk format. Knowing these details is unnecessary for most users who will instead use :ref:`safiretools` or `CoQuí <https://github.com/AbInitioQHub/coqui>`__ to generate factorized Hamiltonians from a variety of sources automatically (see :ref:`tutorials` and :ref:`user_examples`).

.. _storage_conventions:

General Storage Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Data is stored in an HDF5 file.
- :math:`n_\mathrm{spin}` and :math:`n_\mathrm{pol}` express the spin structure: :math:`n_\mathrm{spin} = n_\mathrm{pol} = 1` for closed-shell (spin-independent), :math:`n_\mathrm{spin} = 2`, :math:`n_\mathrm{pol} = 1` for collinear (spin up first), and :math:`n_\mathrm{spin} = 1`, :math:`n_\mathrm{pol} = 2` for noncollinear systems.
- Complex arrays can either follow the ``h5py`` compound type format or the `TRIQS/h5 <https://github.com/TRIQS/h5>`__ interleaved complex format with an ``__complex__`` attribute.
- Scalar parameters are usually stored as an attribute, denoted by 🏷️ in the schemas below.

.. _csr_storage:

Sparse Matrices
_______________

Sparse matrices are stored in the compressed sparse row (CSR) format, as a group holding

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``<matrix>/``

  * ``shape`` :math:`(2,)`
  * ``values`` :math:`(N_\mathrm{nz},)`
  * ``column_indices`` :math:`(N_\mathrm{nz},)`
  * ``row_pointers`` :math:`(M_1 + 1,)`

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``shape``
    - The dimensions :math:`(M_1, M_2)` of the matrix.
  * - ``values``
    - The :math:`N_\mathrm{nz}` non-zero matrix elements.
  * - ``column_indices``
    - The column index of each element of ``values``.
  * - ``row_pointers``
    - Row :math:`i` holds the entries ``row_pointers[i]`` up to (not including)
      ``row_pointers[i+1]`` of ``values`` and ``column_indices``, so ``row_pointers[0] = 0`` and
      ``row_pointers[M_1]`` :math:`= N_\mathrm{nz}`.

.. _dense-cholesky:

Dense Cholesky
~~~~~~~~~~~~~~

This Hamiltonian type is based on the modified-Cholesky
factorization :cite:`BeebeCholesky1977,KochCholesky2003,AquilanteMOLCAS2009,PurwantoCa2011,PurwantoDownfolding2013` of the interaction
tensor:

.. math::
  :label: eq-dense-chol

  v_{pqrs} = V_{(pr),(sq)} \approx \sum_n^{N_{\mathrm{chol}}} L_{pr,n} L^{*}_{sq,n},

where the sum is truncated at :math:`N_{\mathrm{chol}} = x_\mathrm{c} M`,
:math:`x_\mathrm{c}` is typically between :math:`5` and :math:`10`, and :math:`M` is
the number of single-particle basis functions. The storage requirement
thus scales like :math:`\mathcal{O}(M^3)`. 

The SAFIRE implementation currently assumes the common case that :math:`v_{pqrs}^{\sigma\sigma'}` is real-valued and has two-spin-index structure. There is no such restriction on the single particle Hamiltonian.


.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``Hamiltonian/``

  * ``🏷️ format_version`` 1
  * ``🏷️ type`` ``"dense"``
  * ``🏷️ nuclear_energy`` optional
  * ``🏷️ frozen_core_energy`` optional
  * ``hcore`` :math:`(n_\mathrm{spin}, n_\mathrm{pol}, M, n_\mathrm{pol}, M)`
  * ``DenseFactorized/``

    * ``L`` :math:`(n_\mathrm{spin}, 1, M, 1, M, N_\mathrm{chol})`

  * ``BasisRotation`` :math:`(n_\mathrm{pol} N_\mathrm{AO}, n_\mathrm{pol} M)`, optional

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    -  **Description**
  * - ``nuclear_energy``, ``frozen_core_energy``
    - Energy offset  :math:`E_0 = E_\mathrm{nuclear} + E_\mathrm{frozen-core}`

  * - ``hcore``
    - The one-body Hamiltonian.
  * - ``DenseFactorized/L``
    - The factorized interaction Cholesky operators. The leading :math:`n_\mathrm{spin}` is either 1,
      for vectors shared by every spin sector, or that of ``hcore``, for one set per spin sector. Only
      :math:`n_\mathrm{pol} = 1` is supported, including for noncollinear systems, whose Cholesky
      vectors are spin independent.
  * - ``BasisRotation``
    - Connects the computational basis to the natural basis of the Hamiltonian, of size
      :math:`N_\mathrm{AO}`, for example non-orthogonal atomic orbitals.

Despite the name, this Hamiltonian format can be used for any factorization that has three indices on the factors. 

.. seealso::

   The primary use case for this kind of Hamiltonian is in quantum chemistry. :ref:`molecules_overview` describes many ways to construct it.


.. _k_point_factorized:

Periodic Cholesky
~~~~~~~~~~~~~~~~~

In periodic systems, there is an alternative way to factorize the interaction that makes use of translation symmetry :cite:`MottaKPoint2019,MaloneGPU2020`. Given a periodic basis :math:`|\mathbf{k} p\rangle` we can write the momentum conserving interaction tensor as

.. math::
  :label: eq-kp-chol

  v^{\sigma\sigma'}_{(\mathbf{k}+\mathbf{Q})p,\,(\mathbf{k}'-\mathbf{Q})q,\,\mathbf{k}r,\,\mathbf{k}'s}
    = \sum_n L^{\mathbf{Q},\mathbf{k}+\mathbf{Q}}_{pr,n} \left(L^{\mathbf{Q},\mathbf{k}'}_{sq,n}\right)^{*},

where :math:`\mathbf{k}`, :math:`\mathbf{k}'` and :math:`\mathbf{Q}` are
vectors in the first Brillouin zone, and :math:`\mathbf{k}+\mathbf{Q}`, :math:`\mathbf{k}'-\mathbf{Q}` are folded back into it.
The one-body Hamiltonian is block diagonal in :math:`\mathbf{k}`. Because each factor only holds a single :math:`\mathbf{k}` index per two single-particle basis indices, it can bring significant memory savings compared to the dense factorization.

For :math:`\mathbf{Q}` vectors that
satisfy :math:`\mathbf{Q} \ne -\mathbf{Q}`, we have
:math:`{L^{\mathbf{Q},\mathbf{k}}_{sq,n}}^{*} = {L^{-\mathbf{Q},\mathbf{k}-\mathbf{Q}}_{qs,n}}`,
which requires us to store Cholesky vectors for either one of the
:math:`(\mathbf{Q},-\mathbf{Q})` pair, but not both. For highly-symmetric :math:`\mathbf{Q}` vectors, :math:`\mathbf{Q} = -\mathbf{Q}` and we store just that :math:`\mathbf{Q}` vector. Similar reductions can be performed for other symmetries of the lattice.

In what follows let :math:`m` denote the number of basis functions per
:math:`k`-point, which has to be the same at every :math:`k`-point,
:math:`N_k` the number of :math:`k`-points and
:math:`n_{\mathrm{chol}}^{\mathbf{Q}_j}` the number of Cholesky vectors
for momentum transfer :math:`\mathbf{Q}_j`. The one-body Hamiltonian and
the Cholesky vectors take the layouts of the
:ref:`dense-cholesky` format with a leading
:math:`k`-point axis; the format holds only spin-independent
Hamiltonians, so :math:`n_\mathrm{spin} = n_\mathrm{pol} = 1` throughout. The file format for this
factorization is as follows:

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``Hamiltonian/``

  * ``🏷️ format_version`` 1
  * ``🏷️ type`` ``"kpoint"``
  * ``🏷️ nuclear_energy`` optional
  * ``🏷️ frozen_core_energy`` optional
  * ``🏷️ madelung_constant`` optional
  * ``hcore`` :math:`(N_k, 1, 1, m, 1, m)`
  * ``KPFactorized/``

    * ``L<j>`` :math:`(N_k, 1, 1, m, 1, m, n^{\mathbf{Q}_j}_\mathrm{chol})`, one per
      :math:`\mathbf{Q}_j`

  * ``KPoints`` :math:`(N_k, 3)`
  * ``MinusK`` :math:`(N_k,)`
  * ``QKTok2`` :math:`(N_k, N_k)`
  * ``BasisRotation`` :math:`(N_k, N_\mathrm{AO}, m)`, optional

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``nuclear_energy``, ``frozen_core_energy``, ``madelung_constant``
    - Energy offset :math:`E_0 = E_\mathrm{nuclear} + E_\mathrm{frozen-core} - N_\mathrm{e} E_\mathrm{Madelung}`, where the electron self-interaction is scaled by the number of electrons :math:`N_\mathrm{e}`.
  * - ``hcore``
    - The one-body Hamiltonian :math:`h_{(\mathbf{k}p)(\mathbf{k}q)}`, one block per :math:`k`-point.
  * - ``KPFactorized/L<j>``
    - The Cholesky operators :math:`L^{\mathbf{Q}_j,\mathbf{k}}_{pr,n}` for momentum transfer
      :math:`\mathbf{Q}_j`, indexed by the :math:`k`-point :math:`\mathbf{k}` of :math:`p`. Only one
      of each :math:`(\mathbf{Q}, -\mathbf{Q})` pair, the one with :math:`j \le` ``MinusK[j]``, has
      to be stored. The normalization :math:`1/\sqrt{N_k}` is included.
  * - ``KPoints``
    - The :math:`k`-points :math:`\mathbf{k}_i` sampling the Brillouin zone.
  * - ``MinusK``
    - Maps a :math:`k`-point to its inverse: :math:`\mathbf{k}_i + \mathbf{k}_{\mathrm{MinusK}[i]}
      = \mathbf{0} \mod \mathbf{G}.`
  * - ``QKTok2``
    - ``QKTok2[j, i]`` is the index of :math:`\mathbf{k}_i - \mathbf{Q}_j \mod \mathbf{G}`, the
      :math:`k`-point of :math:`r` when :math:`p` is at :math:`\mathbf{k}_i`.
  * - ``BasisRotation``
    - Connects the computational basis to the natural basis of the Hamiltonian, of size
      :math:`N_\mathrm{AO}`, at each :math:`k`-point.

The number of electrons :math:`N_\mathrm{e}` in the electron self-interaction offset is taken from the trial wavefunction.

.. seealso::

   The primary use case for this Hamiltonian is simulating realistic materials. :ref:`solids_overview` gives examples of generating it with both `CoQuí <https://github.com/AbInitioQHub/coqui>`__ and :ref:`safiretools`.



.. _lattice_model_hamiltonian:

Lattice Model Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~

The Lattice Model Hamiltonian format implements general multi-band, Hubbard-Kanamori (HK) models. The short-range interactions in these
models have a special structure that allows for optimizations and bespoke factorizations that are not applicable in the general case. SAFIRE will also adjust some of its :ref:`configuration defaults <propagator_block>` to better suit HK-type calculations when a Lattice Model Hamiltonian is simulated.

The HK Hamiltonian has the general form,

.. math::
  :label: eq-hk

   \hat{H} &= \sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'}
   + \sum_{i} U_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} \\
   &+ \sum_{i<j} U_{ij}^{(1)} (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} )
   + \sum_{i<j} U_{ij}^{(2)} (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} ) \\
   &+ \sum_{i<j} J_{ij} ( 
     \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
     +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} 
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
     ),

where :math:`i` and :math:`j` are combined lattice and band indices, :math:`\hat{n}_{i\sigma}` is the number operator,
:math:`t^{\sigma \sigma'}_{ij}` includes all one-body terms
(:math:`n^\mathrm{th}`-order neighbor hopping, spin-orbit coupling, etc.),
:math:`U_i` is the traditional on-site Hubbard interaction, :math:`U^{(1)}` is
a density-density interaction, :math:`U^{(2)}` is
a spin-spin interaction, and :math:`J` is a Hund's coupling term.

Exactly one one-body term is required. Each interaction term is optional;
however, including at least one interaction term is required.
Model Hamiltonians are saved in HDF5 using the following format.

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``Hamiltonian/``

  * ``🏷️ format_version`` 1
  * ``🏷️ type`` ``"model"``
  * ``🏷️ spin_type``
  * ``🏷️ nuclear_energy`` optional
  * ``🏷️ frozen_core_energy`` optional
  * ``ModelHamiltonian/``

    * ``🏷️ number_of_sites`` :math:`n_\mathrm{sites}`
    * ``🏷️ number_of_bands`` :math:`n_\mathrm{bands}`
    * ``Lattice/`` optional

      * ``L`` :math:`(d,)`
      * ``boundaries`` :math:`(d,)`
      * ``twist`` :math:`(d,)`
      * ``lattice_vectors`` :math:`(d, d)`
      * ``basis`` :math:`(n_\mathrm{basis}, d)`

    * ``ModelComponent_<n>/`` one per term

      * ``🏷️ model_type``
      * ``🏷️ spin_type``
      * ``🏷️ hst_type`` interactions only
      * ``tij/`` or ``Uij/``, ``Jij/``; a :ref:`CSR matrix <csr_storage>`

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``nuclear_energy``, ``frozen_core_energy``
    - Energy offset :math:`E_0 = E_\mathrm{nuclear} + E_\mathrm{frozen-core}`, generally set to 0 for lattice models.
  * - ``spin_type``
    - The spin symmetry of the Hamiltonian, ``"collinear"`` or ``"noncollinear"``.
  * - ``number_of_sites``, ``number_of_bands``
    - The number of lattice sites and of bands per site. The number of basis functions is
      :math:`M = n_\mathrm{sites} n_\mathrm{bands}`.
  * - ``Lattice/``
    - The lattice geometry: the extent ``L``, the ``boundaries``, ``twist``,
      ``lattice_vectors`` and ``basis``. SAFIRE itself does not read it. See :py:class:`safiretools.Lattice`.
  * - ``ModelComponent_<n>/``
    - One term of the Hamiltonian. The components are numbered from 0 without gaps; the number of
      components is where the numbering stops.
  * - ``model_type``
    - The kind of term, ``"one_body"``, ``"hubbard_u"`` or ``"hubbard_j"``.
  * - ``spin_type`` of a component
    - The spin symmetry of the term, ``"closed"``, ``"collinear"`` or ``"noncollinear"``.
  * - ``hst_type``
    - The Hubbard-Stratonovich transformation of an interaction term, ``"continuous_charge"``,
      ``"continuous_spin"``, ``"discrete_charge"`` or ``"discrete_spin"`` (the discrete ones for
      ``"hubbard_u"`` only).
  * - ``tij/``, ``Uij/``, ``Jij/``
    - The matrix of the term, named after ``model_type``: ``tij`` for ``"one_body"``, ``Uij`` for
      ``"hubbard_u"`` and ``Jij`` for ``"hubbard_j"``. Their shapes are described below.

Model Component Conventions
___________________________

-  The model components must either be all real or all complex valued.  
-  For all interaction terms except for on-site, on-band Hubbard U, we take the convention
   that :math:`i < j` as shown in the general Hubbard-Kanamori Hamiltonian above. If entries
   with :math:`i \geq j` are supplied, they will be ignored with a warning.
-  For the one-body term ``tij``, if :math:`M` is the total basis set size, including bands and sites,
   the CSR matrix may have dimension :math:`(M, M)`, :math:`(2M, M)`, or :math:`(2M, 2M)` 
   corresponding to ``"closed"``, ``"collinear"``, and ``"noncollinear"`` spin symmetry,
   respectively. For a ``"closed"`` hopping matrix, the same hopping will be applied to 
   both spin channels. For a ``"collinear"`` hopping matrix, the upper half of ``tij``
   will be applied to the spin up channel, and the lower half to the spin-down channel. 
   For a ``"noncollinear"`` hopping matrix, ``tij`` is explicitly represented in a 
   spin-orbital basis and is applied as is.
-  For the ``Uij`` terms, the CSR matrix may have dimensions :math:`(M, M)`, or :math:`(2M, M)`. If the dimension
   is :math:`(2M,M)`, then the upper half of the array is interpreted as :math:`U_i` on the
   diagonal and :math:`U^{(1)}_{ij}` off it, with :math:`i,j` running over all basis set
   functions, and the lower half is interpreted as :math:`U^{(2)}_{ij}` with :math:`i,j` running
   over all basis set functions.
   If :math:`U^{(2)}_{ij}` is desired without :math:`U_i` or :math:`U^{(1)}_{ij}`, the matrix must
   still have dimension :math:`(2M,M)`, but the upper half should have no non-zero elements.
   If dimension is :math:`(M,M)`, then the CSR matrix is interpreted as :math:`U_i` and
   :math:`U^{(1)}_{ij}`.
-  For ``Jij``, the CSR matrix must have dimension :math:`(M, M)` and is interpreted as
   the :math:`J_{ij}` from the Hubbard-Kanamori term.


.. seealso::

   :ref:`lattice_model_overview` gives an overview of generating Hamiltonians in this format.

.. _thc_hamiltonian:

Tensor Hypercontraction
~~~~~~~~~~~~~~~~~~~~~~~

The tensor hypercontraction (THC) factorization is an efficient method for representing two-electron integrals that exploits the low-rank structure of the electron repulsion integral (ERI) tensor :cite:`HohensteinTHCI2012,ParrishTHCII2012,HohensteinTHCIII2012`. THC factorizes the four-center electron repulsion integrals into a multi-tensor product :cite:`MaloneISDF2019`

.. math::
  :label: eq-thc

  v^{\sigma\sigma'}_{pqrs} \approx \sum_{u,u'}^{N_u} X^{\sigma*}_{pu} X^{\sigma}_{ru} M_{uu'} X^{\sigma'*}_{qu'} X^{\sigma'}_{su'},
  \qquad M_{uu'} = \sum_n^{N_v} L_{un} L^*_{u'n},

where :math:`X^{\sigma}_{pu}` are collocation matrices that map orbitals to interpolation points, :math:`M_{uu'}` is the Coulomb matrix at the interpolation points, and :math:`L_{un}` is its factorization.
:math:`N_u` is the number of interpolation points and :math:`N_v` the number of inner Cholesky operators. The approximation quality depends on the choice of interpolation points and the rank of the factorization.

THC Hamiltonians are stored in HDF5 using the following format:

.. rst-class:: filetree-caption

``afqmc.h5``

.. rst-class:: filetree

* ``System/``

  * ``🏷️ number_of_bands`` :math:`M`
  * ``🏷️ number_of_spins`` :math:`n_\mathrm{spin}`
  * ``🏷️ nuclear_energy`` optional
  * ``🏷️ frozen_core_energy`` optional
  * ``🏷️ madelung_constant`` optional
  * ``BZ/``

    * ``🏷️ number_of_kpoints`` 1

  * ``H0`` :math:`(n_\mathrm{spin}, 1, M, M)`

* ``Interaction/``

  * ``🏷️ number_of_bands`` :math:`M`
  * ``🏷️ number_of_spins`` :math:`n_\mathrm{spin}`
  * ``🏷️ number_of_kpoints`` 1
  * ``🏷️ number_of_qpoints`` 1
  * ``collocation_matrix`` :math:`(n_\mathrm{spin}, 1, M, N_u)`
  * ``factorized_coulomb_matrix`` :math:`(1, N_u, N_v)`
  * ``coulomb_matrix`` :math:`(1, N_u, N_u)`

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``number_of_bands``, ``number_of_spins``
    - The number of basis functions :math:`M` and of spin sectors :math:`n_\mathrm{spin}`, the same
      in ``System`` and ``Interaction``.
  * - ``nuclear_energy``, ``frozen_core_energy``, ``madelung_constant``
    - Energy offset :math:`E_0 = E_\mathrm{nuclear} + E_\mathrm{frozen-core} - N_\mathrm{e} E_\mathrm{Madelung}`,
      as in :ref:`k_point_factorized`.
  * - ``number_of_kpoints``, ``number_of_qpoints``
    - Must be 1: this format holds a single :math:`k`-point.
  * - ``H0``
    - The one-body Hamiltonian.
  * - ``collocation_matrix``
    - The collocation matrix :math:`X_{pu}` mapping orbitals to interpolation points.
  * - ``factorized_coulomb_matrix``
    - The factorized Coulomb matrix :math:`L_{un}` at the interpolation points.
  * - ``coulomb_matrix``
    - The full Coulomb matrix :math:`M_{uu'}`, which allows reusing
      the same collocation matrix for both electron coordinates.

The number of electrons :math:`N_\mathrm{e}` is taken from the wavefunction.

The THC factorization provides substantial memory and computational savings compared to storing the full four-index ERI tensor, especially for large systems where the number of interpolation points :math:`N_u \ll M^2`.

.. seealso::

   :ref:`solids_overview` shows how to use `CoQuí <https://github.com/AbInitioQHub/coqui>`__ to generate this Hamiltonian.
