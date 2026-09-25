.. _Hamiltonian-classes:


Hamiltonian File formats
------------------------

The Hamiltonian is, of course, one of the key inputs to SAFIRE.
SAFIRE is aware only of the generic class of Hamiltonian, so that it can take advantage of 
class-specific algorithms while remaining general.
Python-based tools to generate Hamiltonians, among other inputs, are provided in the `afqmctools` module.
See the afqmctools documentation :ref:`afqmctools` and the tutorials :ref:`tutorials` for more details.

Several generic classes of Hamiltonians are implemented and described in detail below.

#. :ref:`Dense Cholesky <dense-cholesky>`
#. :ref:`K-Point Factorized (KP) <k_point_factorized>`
#. :ref:`Lattice Model Hamiltonian <lattice_model_hamiltonian>`
#. :ref:`THC Hamiltonian <thc_hamiltonian>`

.. _dense-cholesky:

Dense Cholesky
~~~~~~~~~~~~~~

A common Hamiltonian type that is implemented in SAFIRE is based on the modified-Cholesky
factorization :cite:`BeebeCholesky1977,KochCholesky2003,AquilanteMOLCAS2009,PurwantoCa2011,PurwantoDownfolding2013` of the ERI
tensor:

.. math::
  :label: eq-dense-chol

  v_{pqrs} = V_{(pr),(sq)} \approx \sum_n^{N_{\mathrm{chol}}} L_{pr,n} L^{*}_{sq,n},

where the sum is truncated at :math:`N_{\mathrm{chol}} = x_c M`,
:math:`x_c` is typically between :math:`5` and :math:`10`, :math:`M` is
the number of basis functions and we have assumed that the
single-particle orbitals are in general complex. The storage requirement
is thus naively :math:`\mathcal{O}(M^3)`. Note we follow the usual
definition of :math:`v_{pqrs} = \langle pq | rs \rangle = (pr|qs)`. With
this form of factorization, SAFIRE allows for the integrals to be stored
in either dense or sparse format.

The dense case is the simplest and is only implemented for Hamiltonians
with *real* integrals (and basis functions, i.e. not the homogeneous
electron gas which has complex orbitals but real integrals). The file
format is given as follows:

.. code-block:: text
  :caption: Sample Dense Cholesky AFQMC Hamiltonian.
  :name: Listing_Dense_Cholesky

  $ h5dump -n afqmc.h5
  HDF5 "afqmc.h5" {
      FILE_CONTENTS {
          group      /
          group      /Hamiltonian
          dataset    /Hamiltonian/dims
          dataset    /Hamiltonian/Energies
          dataset    /Hamiltonian/hcore
          group      /Hamiltonian/DenseFactorized
          dataset    /Hamiltonian/DenseFactorized/L
      }
  }

**Required datasets:**

-  ``/Hamiltonian/dims`` Descriptor array of length 8 containing :math:`[0,0,0,M,N_\alpha,N_\beta,0,N_\mathrm{chol}]`. :math:`M` is the number of basis functions, :math:`N_\alpha` and :math:`N_\beta` are the numbers of spin-up and spin-down electrons, and :math:`N_\mathrm{chol}` is the number of Cholesky vectors.

-  ``/Hamiltonian/Energies`` Array containing :math:`[E_{II}, E_{\mathrm{core}}]`. :math:`E_{II}` should contain ion-ion repulsion energy and any additional constant terms which have to be added to the total energy. :math:`E_{\mathrm{core}}` is deprecated and not used.

-  ``/Hamiltonian/hcore`` One-body Hamiltonian matrix elements :math:`h^{s}_{\sigma p,\tau r}`, with dimensions :math:`[N_s, N_p, M, N_p, M]`. :math:`N_s = 2` for collinear systems (spin up first) and 1 otherwise, and :math:`N_p = 2` for noncollinear systems and 1 otherwise. For **complex integrals**, an additional dimension of size 2 is added to store real and imaginary parts separately.

-  ``/Hamiltonian/DenseFactorized/L`` The Cholesky decomposition tensor :math:`L^{s,n}_{\sigma p,\tau r}`, with dimensions :math:`[N_s, N_p, M, N_p, M, N_\mathrm{chol}]`. :math:`N_s` is either 1, for vectors shared by every spin sector, or that of ``hcore``, for one set per spin sector. Only :math:`N_p = 1` is supported, including for noncollinear systems, whose Cholesky vectors are spin independent.

**Storage conventions:**

- The Dense Cholesky format only supports real-valued Cholesky vectors.
- Complex one-body Hamiltonians should be stored with an additional dimension of size 2, where ``array[..., 0]`` contains real parts and ``array[..., 1]`` contains imaginary parts.
- All arrays are stored in row-major order, the last index running fastest.

.. seealso::

   See the `Writing a Hamiltonian tutorial <../tutorials/molecules/03_writing_a_hamiltonian/03_writing_a_hamiltonian.html>`_  from the :ref:`molecules_overview` tutorials.


.. _k_point_factorized:

K-Point Factorized (KP)
~~~~~~~~~~~~~~~~~~~~~~~

We have implemented an explicitly :math:`k`-point dependent factorization for periodic systems :cite:`MottaKPoint2019,MaloneGPU2020`

.. math::
  :label: eq62

  (\textbf{k}_p p \textbf{k}_r r| \textbf{k}_q q \textbf{k}_s s) = \sum_n L^{\textbf{Q},\textbf{k}}_{pr,n} {L^{\textbf{Q},\textbf{k}'}_{sq,n}}^{*}

where :math:`\textbf{k}`, :math:`\textbf{k}'` and :math:`\textbf{Q}` are
vectors in the first Brillouin zone. The one-body Hamiltonian is block
diagonal in :math:`\textbf{k}` and in :eq:`eq62` we have used
momentum conservation
:math:`(\textbf{k}_p - \textbf{k}_r + \textbf{k}_q - \textbf{k}_s) = \textbf{G}`
with :math:`\textbf{G}` being some vector in the reciprocal lattice of
the simulation cell. The convention for the Cholesky matrix
:math:`L^{\textbf{Q},\textbf{k}}_{pr,\gamma}` is as follows:
:math:`\textbf{k}_r = \textbf{k}_p - \textbf{Q}`, so the vector
:math:`\textbf{k}` labels the *k*-point of the first band index,
:math:`\textit{p}`, while the *k*-point vector of the second band index,
:math:`\textit{r}`, is given by :math:`\textbf{k} - \textbf{Q}`.
Electron repulsion integrals at different :math:`\textbf{Q}` vectors are
zero by symmetry, resulting in a reduction in the number of required
:math:`\mathbf{Q}` vectors. For certain :math:`\textbf{Q}` vectors that
satisfy :math:`\textbf{Q} \ne -\textbf{Q}` (this is not satisfied at the
origin and at high symmetry points on the edge of the 1BZ), we have
:math:`{L^{\textbf{Q},\textbf{k}}_{sq,\gamma}}^{*} = {L^{-\textbf{Q},\textbf{k}-\textbf{Q}}_{qs,\gamma}}`,
which requires us to store Cholesky vectors for either one of the
:math:`(\textbf{Q},-\textbf{Q})` pair, but not both.

In what follows let :math:`m_{\mathbf{k}}` denote the number of basis
functions for basis functions of a given :math:`k`-point (these can in
principle differ for different :math:`k`-points due to linear
dependencies), :math:`n^{\alpha}_{\mathbf{k}}` the number of
:math:`\alpha` electrons in a given :math:`k`-point and
:math:`n_{\mathrm{chol}}^{\mathbf{Q}_n}` the number of Cholesky vectors
for momentum transfer :math:`\mathbf{Q}_n`. The file format for this
factorization is as follows (for a :math:`2\times2\times2`
:math:`k`-point mesh, for denser meshes generally there will be far
fewer symmetry inequivalent momentum transfer vectors than there are
:math:`k`-points):

.. code-block::
  :caption: Sample Dense :math:`k`-point dependent Cholesky SAFIRE Hamiltonian.
  :name: Listing 55

  $ h5dump -n afqmc.h5
  HDF5 "afqmc.h5" {
      FILE_CONTENTS {
          group      /
          group      /Hamiltonian
          group      /Hamiltonian/KPFactorized
          dataset    /Hamiltonian/KPFactorized/L0
          dataset    /Hamiltonian/KPFactorized/L1
          dataset    /Hamiltonian/KPFactorized/L2
          dataset    /Hamiltonian/KPFactorized/L3
          dataset    /Hamiltonian/KPFactorized/L4
          dataset    /Hamiltonian/KPFactorized/L5
          dataset    /Hamiltonian/KPFactorized/L6
          dataset    /Hamiltonian/KPFactorized/L7
          dataset    /Hamiltonian/NCholPerKP
          dataset    /Hamiltonian/MinusK
          dataset    /Hamiltonian/NMOPerKP
          dataset    /Hamiltonian/QKTok2
          dataset    /Hamiltonian/H1_kp0
          dataset    /Hamiltonian/H1_kp1
          dataset    /Hamiltonian/H1_kp2
          dataset    /Hamiltonian/H1_kp3
          dataset    /Hamiltonian/H1_kp4
          dataset    /Hamiltonian/H1_kp5
          dataset    /Hamiltonian/H1_kp6
          dataset    /Hamiltonian/H1_kp7
          dataset    /Hamiltonian/ComplexIntegrals
          dataset    /Hamiltonian/KPoints
          dataset    /Hamiltonian/dims
          dataset    /Hamiltonian/Energies
      }
  }

-  ``/Hamiltonian/KPFactorized/L[n]`` This series of datasets store elements of the Cholesky tensors
   :math:`L[\mathbf{Q}_n,\mathbf{k},pr,n]`. Each data set is of
   dimension
   :math:`[N_k,m_{\mathbf{k}}\times m_{\mathbf{k}'},n^{\mathbf{Q}_n}_\mathrm{chol}]`,
   where, again, :math:`k` is the :math:`k`-point associated with basis
   function :math:`p`, the :math:`k`-point of basis function :math:`r`
   is defined via the mapping ``QKtok2``.

-  ``/Hamiltonian/NCholPerKP`` :math:`N_k` length array giving number of Cholesky vectors per
   :math:`k`-point.

-  ``/Hamiltonian/MinusK``: :math:`N_k` length array mapping a
   :math:`k`-point to its inverse: :math:`\mathbf{k}_i+`\ ``MinusK[i]``
   :math:`= \mathbf{0} \mod \mathbf{G}`.

-  ``/Hamiltonian/NMOPerKP``: :math:`N_k` length array listing number of
   basis functions per :math:`k`-point.

-  ``/Hamiltonian/QKTok2``: :math:`[N_k,N_k]` dimensional array.
   ``QKtok2[i,j]`` yields the :math:`k` point index satisfying
   :math:`\mathbf{k}=\mathbf{Q}_i-\mathbf{k}_j+\mathbf{G}`.

-  ``/Hamiltonian/dims``: Descriptor array of length 8 containing
   :math:`[0,0,0,M,N_\alpha,N_\beta,0,0]`.

-  ``/Hamiltonian/H1_kp[n]`` Contains the :math:`[m_{\mathbf{k}_n},m_{\mathbf{k}_n}]`
   dimensional one-body Hamiltonian matrix elements
   :math:`h_{(\mathbf{k}_{n}p)(\mathbf{k}_{n}q)}`.

-  ``/Hamiltonian/ComplexIntegrals`` Length 1 array that specifies if integrals are complex valued. 1
   for complex integrals, 0 for real integrals.

-  ``/Hamiltonian/KPoints`` :math:`[N_k,3]` Dimensional array containing :math:`k`-points used to
   sample Brillouin zone.

-  ``/Hamiltonian/dims`` Descriptor array of length 8 containing
   :math:`[0,0,N_k,M,N_\alpha,N_\beta,0,N_\mathrm{nchol}]`. Note that
   :math:`M` is the total number of basis functions, i.e.
   :math:`M=\sum_\mathbf{k} m_\mathbf{k}`, and likewise for the number
   of electrons.

-  ``/Hamiltonian/Energies`` Array containing :math:`[E_{II}, E_{\mathrm{core}}]`.
   :math:`E_{II}` should contain ion-ion repulsion energy and any
   additional constant terms which have to be added to the total energy
   (such as the electron-electron interaction Madelung contribution of
   :math:`\frac{1}{2} N \xi )`. :math:`E_{\mathrm{core}}` is deprecated
   and not used.

Complex integrals should be written as an array with an additional dimension, e.g., a 1D array should be written as a 2D array with ``array_hdf5[:,0]=real(1d_array)`` and ``array_hdf5[:,1]=imag(1d_array)``. The functions ``afqmctools.utils.misc.from_complex`` and ``afqmctools.utils.misc.to_complex`` can be used to transform from the internal complex format to complex valued numpy arrays of the appropriate shape and vice versa.

.. seealso::

   See the following from the :ref:`solids_overview` tutorials.

   * (under construction) `Writing a Hamiltonian with CoQuí <#>`_



.. _lattice_model_hamiltonian:

Lattice Model Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~

General multi-band, Hubbard-Kanamori (HK) lattice model Hamiltonians are implemented.
The HK Hamiltonian has the general form,

.. math::

   \hat{H} = \sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'} 
   + \sum_{i} U_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} \\
   + \sum_{i<j} U_{ij}^1 (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} ) 
   + \sum_{i<j} U_{ij}^2 (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} ) \\
   + \sum_{i<j} J_{ij} ( 
     \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
     +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} 
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
     ),

where :math:`i`,\ :math:`j` are combined lattice and band indices,
:math:`\hat{c}^\dagger_{i\sigma}`, :math:`\hat{c}_{j\sigma'}`
create/annihilate and electron on the site (and band) corresponding to
:math:`i`/:math:`j` with spin :math:`\sigma`/:math:`\sigma'`,
:math:`\hat{n}_{i\sigma}` is the number operator,
:math:`t^{\sigma \sigma'}_{ij}` includes all one-body terms
(:math:`n^{th}`-order neighbor hopping, spin-orbit coupling, etc.),
:math:`U` is the traditional on-site hubbard interaction, :math:`U^1` is
a density-density interaction, :math:`U^2` is
a spin-spin interaction, and :math:`J` is a Hund's coupling term.
:math:`U^1`, :math:`U^2`, and :math:`J` are typically only non-zero between
bands on the same lattice site, but are by no means limited in this way.

Each individual term is optional (including the hopping term); 
however, including at least one interaction term is required.
Model Hamiltonians are saved in HDF5 using the following format.

.. code-block::
  :caption: Sample Hubbard-Kanamori SAFIRE Hamiltonian.
  :name: Listing 57

  $ h5dump -n afqmc.h5 
    HDF5 "afqmc.h5" {
      FILE_CONTENTS {
          group      /
          group      /Hamiltonian
          dataset    /Hamiltonian/Energies
          group      /Hamiltonian/ModelHamiltonian
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_0
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/model_type
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/spin_type
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/data_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/dims
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/jdata_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/pointers_begin_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/pointers_end_
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_1
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij/data_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij/dims
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij/jdata_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij/pointers_begin_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/Jij/pointers_end_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/hst_type
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/model_type
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_1/spin_type
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_2
          group      /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij/data_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij/dims
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij/jdata_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij/pointers_begin_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/Uij/pointers_end_
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/hst_type
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/model_type
          dataset    /Hamiltonian/ModelHamiltonian/ModelComponent_2/spin_type
          dataset    /Hamiltonian/ModelHamiltonian/maximum_connectivity
          dataset    /Hamiltonian/ModelHamiltonian/number_of_components
          dataset    /Hamiltonian/dims
          dataset    /Hamiltonian/spin_type
    }
  }

-  ``/Hamiltonian/Energies``: Array containing :math:`[E_{II}, E_{\mathrm{core}}]`.
   :math:`E_{II}` should contain ion-ion repulsion energy and any
   additional constant terms which have to be added to the total energy
   (such as the electron-electron interaction Madelung contribution of
   :math:`\frac{1}{2} N \xi )`. :math:`E_{\mathrm{core}}` is deprecated
   and not used. In the context of lattice models, both :math:`E_{II}` and 
   :math:`E_{\mathrm{core}}` while generally be set to 0.0.

-  ``/Hamiltonian/dims``: Descriptor array of length 8 containing
   :math:`[0,0,0,M,M,0,0,0]`. Note that
   :math:`M` is the total number of basis functions, i.e. number of sites * number of bands

-  ``/Hamiltonian/spin_type``: Is a string containing the spin symmetry type of the Hamiltonian.
   possible values are "collinear", or "noncollinear".

-  ``/Hamiltonian/ModelHamiltonian/maximum_connectivity``: A scalar integer containing the maximum 
   number of possible connections in the Hamiltonian. If in doubt, the number of non-zero elements
   can be safely used here.

-  ``/Hamiltonian/ModelHamiltonian/number_of_components``: A scalar integer containing the number 
   of "components" (i.e. terms) in the Hamiltonian. This should be consistent with the number of 
   :math:` /Hamiltonian/ModelHamiltonian/ModelComponent_n/` groups, where :math:`n` is replaced with an
   integer.

-  ``/Hamiltonian/ModelHamiltonian/ModelComponent_n``: (where :math:`n` are replaced by a specific integer index) 
   are HDF5 groups which contain a Hamiltonian component (i.e. a term). Each component consists of some metadata
   and a compressed sparse row (CSR) matrix representation of that Hamiltonian component. Each group has the 
   following datasets and groups.

    -  ``/spin_type``: a string containing the spin symmetry type of the component.
       Possible values are "collinear", or "noncollinear".

    -  ``/model_type``: a string describing the type of model term. 
       Possible values are: "one_body", "hubbard_u", and "hubbard_j".

    -  a matrix group. The name of the group depends on the ``/model_type``. 
       Possible values are: ``/tij/`` (``"one_body"``), ``/Uij/`` (``"hubbard_u"``), 
       and ``/Jij/`` (``"hubbard_j"``). Conventions for the shape of each matrix are
       described below. The matrix group will always have the following datasets:
    
      -  ``dims``:  Descriptor array of length 3 containing :math:`[M_1,M_2,N_{nz}]`. 
         where :math:`M_1` and :math:`M_2` are the dimensions of the CSR matrix, and
         :math:`N_{nz}` is the number of non-zero entries.

      -  ``data_``: an array containing the non-zero matrix elements.

      -  ``jdata_``: an array containing the column indices of each non-zero matrix element.

      -  ``pointers_begin_``: an array containing "pointers" to the beginning of each row
         within ``data_`` / ``jdata_``.

      -  ``pointers_end_``: an array containing "pointers" to the end of each row
         within ``data_`` / ``jdata_``.

Model Component Conventions
___________________________

The following Conventions apply to the CSR matrix representations of each model component.

-  both real- and complex-valued matrices are supported; however, all components must have 
   the same type. i.e. if the hopping matrix is complex-valued, due to a twist, all interaction 
   terms must also be complex-valued even of the imaginary part is identically zero.
  
-  for all interaction terms except for on-site, on-band Hubbard U, we take the convention 
   that :math:`i < j` as shown in the general Hubbard-Kanamori Hamiltonian above. If entries
   with :math:`i >= j` are supplied, they will be ignored with a warning.

-  For ``/tij/``, if :math:`M` is the total basis set size, including bands and sites,
   the CSR matrix may have dimension :math:`M x M`, :math:`2M x M`, or :math:`2M x 2M` 
   corresponding to ``"closed"``, ``"collinear"``, and ``"noncollinear"`` spin symmetry,
   respectively. For a ``"closed"`` hopping matrix, the same hopping will be applied to 
   both spin channels. For a ``"collinear"`` hopping matrix, the upper half of ``/tij/``
   will applied to the spin up channel, and the lower half to the spin-down channel. 
   For a ``"noncollinear"`` hopping matrix, ``/tij/`` is explicitly represented in a 
   spin-orbital basis and is applied as is. We note that ``/tij/`` may contain arbitrary
   one-body terms if desired; for example, a pinning field.

   the CSR matrix may have dimension :math:`M x M`, or :math:`2M x M`. If the dimension
   is :math:`2M x M`, then the upper half of the array is interpreted as :math:`U_{ii}+U1_{ij}`
   with :math:`i,j` running over all basis set functions, and the lower half 
   is interpreted as :math:`U2_{ij}`with :math:`i,j` running over all basis set functions.
   If :math:`U2_{ij}` is desired without :math:`U_{ii}` or :math:`U1_{ij}`, the matrix must
   still have dimension :math:`2M x M`, but the upper half should have no non-zero elements.
   If dimension is :math:`M x M`, then the CSR matrix is interpreted as :math:`U_{ii}+U1_{ij}`.

-  for ``/Jij``, the CSR matrix must have dimension :math:`M x M` and is interpreted as the 
   the :math:`J_{ij}` from the Hubbard-Kanamori term.

The other groups and datasets represent the Hamiltonian "components" (i.e. the terms which
are included in the specific Hamiltonian given to AFQMC).


.. seealso::

   See the following from the :ref:`lattice_model_overview` tutorials.

   * `Setting up a Lattice <../tutorials/models/03_setting_up_a_lattice/03_setting_up_a_lattice.html>`_
   * `Building and writing a Hamiltonian <../tutorials/models/04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian.html>`_
   * `Finer control of the Hamiltonian <../tutorials/models/05_hamiltonian_builder/05_hamiltonian_builder.html>`_

.. _thc_hamiltonian:

Tensor Hyper Contraction (THC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tensor Hyper-Contraction (THC) approach is an efficient method for representing two-electron integrals that exploits the low-rank structure of the electron repulsion integral (ERI) tensor. THC factorizes the four-center electron repulsion integrals into products involving lower-dimensional tensors, providing significant computational savings for AFQMC calculations.

THC represents the ERI tensor as:

.. math::
  :label: eq-thc

  v_{pqrs} \approx \sum_{u}^{N_u} \sum_{v}^{N_v} X_{pu} L_{uv} X_{rv}^*

where :math:`X_{pu}` are collocation matrices that map orbitals to interpolation points, :math:`L_{uv}` is the factorized Coulomb matrix at interpolation points, :math:`N_u` and :math:`N_v` are the numbers of interpolation points, and the approximation quality depends on the choice of interpolation points and the rank of the factorization.

The THC format supports two main variants:

**Option 1: Direct THC factorization**
  Uses the factorized Coulomb matrix :math:`L_{uv}` directly and reuses the same collocation matrix for both electron coordinates.

**Option 2: Half-rotated THC factorization**  
  Uses a half-rotated Coulomb matrix and separate collocation matrices that have been pre-rotated to a specific orbital space.

THC Hamiltonians are stored in HDF5 using the following format:

.. code-block:: text
  :caption: Sample THC AFQMC Hamiltonian.
  :name: Listing_THC

  $ h5dump -n afqmc.h5
  HDF5 "afqmc.h5" {
      FILE_CONTENTS {
          group      /
          group      /System
          dataset    /System/H0
          dataset    /System/atomic_positions
          dataset    /System/lattice_vectors
          [additional system datasets...]
          group      /Interaction
          dataset    /Interaction/factorized_coulomb_matrix
          dataset    /Interaction/collocation_matrix
          dataset    /Interaction/coulomb_matrix [Option 1 only]
          dataset    /Interaction/half_rotated_coulomb_matrix [Option 2 only]
          dataset    /Interaction/collocation_matrix_half_rotated [Option 2 only]
      }
  }

**Required datasets:**

-  ``/System/H0`` One-body Hamiltonian matrix elements. Dimensions are ``[nspin, nkpts, norb, norb, complex_dim]`` where ``complex_dim=2`` for complex integrals and is omitted for real integrals. For molecular calculations, ``nkpts=1``.

-  ``/Interaction/factorized_coulomb_matrix`` The factorized Coulomb tensor :math:`L_{uv}`. Dimensions are ``[nq, nu, nv, complex_dim]`` where ``nq`` should be 1 for current implementation. For complex integrals, the last dimension has size 2 storing real and imaginary parts.

-  ``/Interaction/collocation_matrix`` The collocation matrix :math:`X_{pu}` mapping orbitals to interpolation points. Dimensions are ``[nspin, nkpts, norb, nu, complex_dim]``.

**Option 1 additional dataset:**

-  ``/Interaction/coulomb_matrix`` The full Coulomb matrix :math:`M_{uv} = L_{uv} \cdot L_{uv}^{\dagger}`. This allows reuse of the same collocation matrix. Dimensions are ``[nq, nu, nu, complex_dim]``.

**Option 2 additional datasets:**

-  ``/Interaction/half_rotated_coulomb_matrix`` A half-rotated version of the Coulomb matrix for specialized THC factorizations. Dimensions are ``[nq, nu_rot, nv_rot, complex_dim]``.

-  ``/Interaction/collocation_matrix_half_rotated`` The corresponding half-rotated collocation matrix. Dimensions match the half-rotated coulomb matrix structure.

**Storage conventions:**

- Complex integrals should be stored with an additional dimension of size 2, where ``array[..., 0]`` contains real parts and ``array[..., 1]`` contains imaginary parts.
- For spin-collinear calculations with ``nspin=2``, if only one spin component is provided in the file (``nspin_in_file=1``), it will be automatically duplicated for the second spin channel.
- The parameter ``cutoff_cholesky`` can be specified to control the accuracy of Cholesky-related operations (default: ``1e-6``).

The THC representation provides substantial memory and computational savings compared to storing the full four-index ERI tensor, especially for large systems where the number of interpolation points :math:`N_u \ll M^2` (where :math:`M` is the number of orbitals).

.. seealso::

   See the following from the :ref:`solids_overview` tutorials.

   * (under construction) `Writing a Hamiltonian with CoQuí <#>`_

Final Notes on Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code CoQuí is able to generate and output both the K-Point Factorized, and THC Hamiltonian format directly.

Lattice model Hamiltonians can be generated using the afqmctools Python package which is included with SAFIRE. 
See the lattice model tutorials
for more information on how to generate these Hamiltonians.

Finally, if using external tools to generate this file format, 
we provide a sanity checker script in ``utils/bin/test_afqmc_input.py`` 
which will raise errors if the format does not conform to what is being used internally.
