.. _estimators:

Estimators
==========

The results of a SAFIRE calculation are obtained by performing stochastic measurements on the :ref:`afqmc` random walk. For this purpose SAFIRE implements different estimators.

Estimators are enabled, disabled, and configured in the :ref:`estimator_block` block of the job file. Their measurement results are written to the :ref:`output_afqmc`. Refer to these sections for details on the input and output and the :ref:`tutorials` and :ref:`user_examples` for practical use cases. On this page, we will instead describe the behavior of each estimator and the definitions of the observables that can be measured.

Each estimator uses the trial wavefunction :math:`\Psi_\mathrm{T}` and Hamiltonian :math:`H` of its execute block, unless it names different ones with its own ``wavefunction`` and ``hamiltonian`` entries.

.. _energy_estimator:

Energy Estimator
----------------

The energy estimator is the only estimator that is enabled by default. It measures the local energy

.. math::
  :name: energyEstimator

  E = \frac{\langle \Psi_\mathrm{T} | H | \Psi(\tau) \rangle}{\langle \Psi_\mathrm{T} | \Psi(\tau) \rangle} \approx
  \frac{1}{\sum_n W_n(\tau)} \sum_n W_n(\tau) \frac{\langle \Psi_\mathrm{T} | H | \Phi_n(\tau) \rangle}{\langle \Psi_\mathrm{T} | \Phi_n(\tau) \rangle},

of the walkers and its components for a given Hamiltonian :math:`H` and trial wavefunction :math:`\Psi_\mathrm{T}`,
where :math:`|\Phi_n(\tau)\rangle` are the walkers at imaginary time :math:`\tau` and :math:`W_n(\tau)` their weights.

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - **Observable**
     - **Description**
   * - ``Energy``
     - The total local energy :math:`E`, the sum of the three components below.
   * - ``OnebodyEnergy``
     - The one-body part of the local energy. It also includes the constant energy offset of the
       Hamiltonian.
   * - ``CoulombEnergy``
     - The direct (Hartree) part of the two-body local energy.
   * - ``ExchangeEnergy``
     - The exchange part of the two-body local energy.
   * - ``ProjectionTime``
     - The imaginary time :math:`\tau` of the measurement, counted from the start of the
       execution stage: :math:`\tau = s\Delta\tau` after :math:`s` propagation steps. Useful for plotting time series.

.. note::
   Since the one and two-body terms of the Hamiltonian generally do not commute, the ``OnebodyEnergy``, ``CoulombEnergy`` and ``ExchangeEnergy`` component estimators carry a mixed-estimator bias that the total energy does not. See :ref:`mixed_estimators`.
  
Additionally, the following diagnostic observables are available.

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - **Observable**
     - **Description**
   * - ``Overlap``
     - The weighted average
       :math:`\sum_n W_n(\tau) \langle \Psi_\mathrm{T} | \Phi_n(\tau) \rangle / \sum_n W_n(\tau)` of the
       walker overlaps with the trial wavefunction. Reset by reorthogonalization.
   * - ``TotalWeight``
     - The total walker weight :math:`\sum_n W_n(\tau)`.
   * - ``Phase``
     - The average phase :math:`\sum_n W_n(\tau) / \sum_n |W_n(\tau)|` of the walker weights. A
       magnitude well below one signals a phase problem. As long as a constraint is active, this is one.
   * - ``EffectiveNumWalkers``
     - The effective number of walkers
       :math:`\left(\sum_n |W_n(\tau)|\right)^2 / \sum_n |W_n(\tau)|^2`, which equals the number of
       walkers for uniform weights and drops as the weights spread out.

.. _mixed_estimators:

Mixed Estimator
----------------

SAFIRE implements more general mixed estimators,

.. math::
  \langle O \rangle_\mathrm{Mixed} = \frac{1}{\sum_n W_n(\tau)} \sum_n W_n(\tau) \frac{\langle \Psi_\mathrm{T} | O | \Phi_n(\tau) \rangle }{\langle \Psi_\mathrm{T} | \Phi_n(\tau) \rangle},

where :math:`O` is one of several :ref:`observables` described below.

In the output file, all mixed observables are saved under a ``MixedEstimator/`` prefix.

Mixed Estimator Bias
~~~~~~~~~~~~~~~~~~~~

Because the mixed estimator is based on a one-sided projection, even if the walker ensemble exactly represents the ground-state wavefunction, the mixed average will generally not match the true ground-state average. This bias vanishes only when :math:`[O,H] = 0`, such as for the total energy, :math:`O=H`, or when :math:`O` represents a symmetry of the Hamiltonian. The latter property means that mixed averages — despite being numerically biased — can still be used to detect the symmetry sector of the ground state.

The mixed estimator bias can be mitigated by using the back-propagation estimator described in the next section.

.. _backprop_estimator:

Back-Propagation Estimator
--------------------------

The Back-Propagation (BP) estimator :cite:`ZhangConstrained1997,MottaBackPropagation2017` replaces the one-sided projection of the mixed estimator by a two-sided projection of the form

.. math::

  \langle O \rangle_\mathrm{BP} = \frac{1}{\sum_n W_n(\tau+\tau_\mathrm{BP})} \sum_n W_n(\tau+\tau_\mathrm{BP}) \frac{\langle \tilde{\Phi}_n(\tau_\mathrm{BP}) | O | \Phi_n(\tau) \rangle }{\langle \tilde{\Phi}_n(\tau_\mathrm{BP}) |\Phi_n(\tau)\rangle}


where :math:`| \Phi_n(\tau) \rangle` are the usual forward-projected Slater determinant random walkers,
and :math:`| \tilde{\Phi}_n(\tau_\mathrm{BP}) \rangle` are the back-propagated walkers given by,

.. math::

  | \tilde{\Phi}_n(\tau_\mathrm{BP}) \rangle = B^\dagger\big( x_n(\tau)\big) \cdots B^\dagger\big(x_n(\tau+\tau_\mathrm{BP}-\Delta\tau) \big) | \Psi_\mathrm{T} \rangle.

Here :math:`\tau` is the imaginary time of the forward walkers,
and :math:`\tau_\mathrm{BP} = m_\mathrm{BP}\Delta\tau` is the back-propagation time over :math:`m_\mathrm{BP}` steps.
We note that each random walker has a corresponding back-propagated partner
which shares the same path in auxiliary-field space.

Path Restoration
~~~~~~~~~~~~~~~~

In every propagation step, the phaseless approximation (see :ref:`afqmc`) multiplies the walker weights by the
cosine projection factor :math:`\max\{0, \cos\Delta\theta\}` and keeps only the magnitude of the local energy
factor :math:`e^{-\Delta\tau (E_\mathrm{L} - E_\mathrm{shift})}`. Along the back-propagation path, both
modifications bias the back-propagated observables. With ``path_restoration``, which is enabled by default, the
weights :math:`W_n(\tau+\tau_\mathrm{BP})` in the back-propagated average are multiplied by

.. math::

  \prod_{\tau'} \frac{e^{-i \Delta\tau \, \mathrm{Im} E_{\mathrm{L},n}(\tau')}}{\max\{0, \cos\Delta\theta_n(\tau')\}},

with the product running over the :math:`m_\mathrm{BP}` steps
:math:`\tau' = \tau, \dots, \tau + \tau_\mathrm{BP} - \Delta\tau` of the back-propagation path. This restores the
phase of the local energy factor and undoes the cosine projection :cite:`MottaBackPropagation2017`. A walker whose weight the projection has set
to zero keeps a zero weight. ``extra_path_restoration`` extends the product to the :math:`2 m_\mathrm{BP}` steps
:math:`\tau' = \tau - \tau_\mathrm{BP}, \dots, \tau + \tau_\mathrm{BP} - \Delta\tau`, so that it also covers the
:math:`m_\mathrm{BP}` steps before the forward walkers :math:`|\Phi_n(\tau)\rangle`.


Back-Propagation Lengths
~~~~~~~~~~~~~~~~~~~~~~~~

The result of the back-propagation estimator depends on the correct choice of the back-propagation length. Short back-propagation lengths will approach the mixed estimator while setting the length too long is costly. Therefore, it is advisable to calculate multiple back-propagation lengths at the same time and check their convergence. SAFIRE supports calculating multiple lengths in a single calculation (see :ref:`backprop_block`). The longest back-propagation length of a simulation simultaneously acts as the measurement interval of the estimator.

In the output file, all back-propagated observables are saved under a ``BackPropEstimator/Steps=<m_BP>/`` prefix.

.. _observables:

Observables
-----------

Mixed and back-propagated estimators share a set of general observables they can measure. Here we give the definition of each of them. The most important observables are the one- and two-body reduced density matrices ``OneRDM``, ``DiagonalTwoRDM``, and ``TwoRDM`` from which all one-body and two-body properties can be calculated. Additional, more specialized observables are defined for subclasses of systems that can be simulated with SAFIRE.

Below, :math:`\langle \cdot \rangle` is the mixed or back-propagated average of the estimator measuring the
observable. The orbital indices :math:`i, j, k, l` run over the :math:`M` orbitals of the Hamiltonian
and :math:`\sigma, \sigma'` over the spins. For noncollinear walkers, spin and orbital are combined into a
single spin-orbital index :math:`\sigma M + i` with :math:`\uparrow = 0` and :math:`\downarrow = 1`.

Every observable can be measured with any Hamiltonian. The wavefunctions and walker spin symmetries it
supports are listed with each one.

.. _onerdm_obs:

``OneRDM``: One-Body Reduced Density Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the spin blocks of the one-particle reduced density matrix,

.. math::

    \mathtt{OneRDM}[s, i, j] = \langle c^\dagger_{i\sigma} c_{j\sigma} \rangle,

where :math:`s = 0` is :math:`\sigma = \uparrow` and :math:`s = 1` is :math:`\sigma = \downarrow`.
Closed walkers store only the :math:`\uparrow` block, which equals the :math:`\downarrow` block.
Noncollinear walkers store a single block over spin-orbitals that includes the spin-flip elements,
:math:`\mathtt{OneRDM}[0, \sigma M + i, \sigma' M + j] = \langle c^\dagger_{i\sigma} c_{j\sigma'} \rangle`.

If a ``RotationMatrix`` :math:`R` is given, the density matrix is measured in the rotated basis
:math:`d_{s,a} = \sum_i R_{s,a,i} c_{i\sigma}`,

.. math::

    \mathtt{OneRDM}[s, a, b] = \langle d^\dagger_{s,a} d_{s,b} \rangle = \sum_{ij} R^*_{s,a,i} R_{s,b,j} \langle c^\dagger_{i\sigma} c_{j\sigma} \rangle.

For noncollinear walkers, the sums run over all :math:`2M` spin-orbitals, so the rotation can mix the spins.

Compatibility
_______________

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Wavefunctions**
    - **Spin Symmetry**
  * - NOMSD
    - closed, collinear, noncollinear
  * - PHMSD
    - collinear




.. _twordm_obs:

``TwoRDM``: Two-Body Reduced Density Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the full two-particle reduced density matrix,

.. math::

    \mathtt{TwoRDM}[b, i, k, j, l] = \langle c^\dagger_{i\sigma} c^\dagger_{j\sigma'} c_{l\sigma'} c_{k\sigma} \rangle,

where the spin block :math:`b = 0, 1, 2` is :math:`\sigma\sigma' = {\uparrow\uparrow}, {\uparrow\downarrow}, {\downarrow\downarrow}`.
Note the index order: the pair :math:`(i, k)` belongs to spin :math:`\sigma` and :math:`(j, l)` to :math:`\sigma'`.
The :math:`{\downarrow\uparrow}` block follows from
:math:`\langle c^\dagger_{i\downarrow} c^\dagger_{j\uparrow} c_{l\uparrow} c_{k\downarrow} \rangle = \mathtt{TwoRDM}[1, j, l, i, k]`.
Closed walkers omit the :math:`{\downarrow\downarrow}` block, which equals the :math:`{\uparrow\uparrow}` block, so
only :math:`b = 0, 1` are stored.

.. caution::

	Computing and storing the full 2-RDM can be very memory intensive and greatly increases the run time.
	Use only when absolutely necessary, and restrict to small systems.


Compatibility
_____________

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Wavefunctions**
    - **Spin Symmetry**
  * - NOMSD
    - closed, collinear
  * - PHMSD
    - collinear


.. _diag_twordm_obs:

``DiagonalTwoRDM``: Diagonal Two-Body Reduced Density Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the diagonal elements of the two-particle reduced density matrix (2-RDM),

.. math::

    D^{\sigma\sigma'}_{ij} = \langle c^\dagger_{i\sigma} c^\dagger_{j\sigma'} c_{j\sigma'} c_{i\sigma} \rangle
    = \langle n_{i\sigma} n_{j\sigma'} \rangle - \delta_{ij} \delta_{\sigma\sigma'} \langle n_{i\sigma} \rangle.

The same-spin blocks are symmetric, :math:`D^{\sigma\sigma}_{ij} = D^{\sigma\sigma}_{ji}`, and vanish for
:math:`i = j`, so only their symmetrized upper triangle :math:`i < j` is stored. The :math:`{\uparrow\downarrow}`
block is stored whole. ``DiagonalTwoRDM`` is a flat array that holds, for collinear and closed walkers,

1. for each :math:`i = 0, \dots, M-1` in turn: :math:`D^{\uparrow\uparrow}_{ij}` for :math:`j = i+1, \dots, M-1`,
   followed by :math:`D^{\uparrow\downarrow}_{ij}` for :math:`j = 0, \dots, M-1`,
2. then, for collinear walkers only, :math:`D^{\downarrow\downarrow}_{ij}` for :math:`i < j`, row by row.

Closed walkers omit the :math:`{\downarrow\downarrow}` block, which equals the :math:`{\uparrow\uparrow}` block.
Noncollinear walkers have a single block over spin-orbitals, :math:`D_{pq} = \langle c^\dagger_p c^\dagger_q c_q c_p \rangle`
with :math:`p, q = 0, \dots, 2M-1`, which is stored as its upper triangle :math:`p < q`, row by row.

Compatibility
_____________

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Wavefunctions**
    - **Spin Symmetry**
  * - NOMSD
    - closed, collinear, noncollinear
  * - PHMSD
    - collinear


.. _spincorr_obs:

``SpinCorr``: Spin-Spin Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the spin-spin correlation function :math:`\langle \vec{S}_i \cdot \vec{S}_j \rangle`, where

.. math::

    \vec{S}_i = \frac{1}{2} \sum_{\sigma,\sigma'} c_{i\sigma}^\dagger \vec{\sigma}_{\sigma\sigma'} c_{i\sigma'},

split into the XY and the Z channel,

.. math::

    \mathtt{SpinCorr}[0, n] &= \langle S^x_i S^x_j + S^y_i S^y_j \rangle = \frac{1}{2} \langle S^+_i S^-_j + S^-_i S^+_j \rangle, \\
    \mathtt{SpinCorr}[1, n] &= \langle S^z_i S^z_j \rangle,

so that :math:`\langle \vec{S}_i \cdot \vec{S}_j \rangle = \mathtt{SpinCorr}[0, n] + \mathtt{SpinCorr}[1, n]`.
Since the correlation is symmetric in :math:`i` and :math:`j`, only the upper triangle :math:`i \le j` is
stored, row by row, at :math:`n = iM - i(i-1)/2 + j - i`.



Compatibility
_____________

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Wavefunctions**
    - **Spin Symmetry**
  * - NOMSD
    - closed, collinear, noncollinear
  * - PHMSD
    - collinear


.. _paircorr_obs:

``PairCorr``: Pair Correlation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

	Pair correlation functions are an experimental feature.

Calculates singlet pair correlation functions

.. math::

	\mathtt{PairCorr/}\alpha\mathtt{\_}\beta[i, j] = \langle \Delta^\dagger_{i,\alpha} \Delta_{j,\beta} \rangle

between the pair creation and annihilation operators

.. math::

	\Delta^\dagger_{i,\alpha} = \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{\bar{i}_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{\bar{i}_\alpha \uparrow} \right), \qquad
	\Delta_{j,\beta} = \frac{1}{\sqrt{2}} \left( c_{\bar{j}_\beta\downarrow} c_{j \uparrow}  - c_{\bar{j}_\beta \uparrow} c_{j \downarrow} \right).

Here :math:`\alpha` and :math:`\beta` are *pair maps* that assign every orbital :math:`i` a partner orbital
:math:`\bar{i}_\alpha`. On a lattice, a pair map typically pairs each site with its neighbor at a fixed
offset such as :math:`+x`, and the pair map :math:`\bar{i}_\alpha = i` gives on-site (s-wave) pairs.

The pair maps are read from the HDF5 file and group given by the ``pairs`` entry of the :ref:`paircorr input block <observables_block>`.

.. rst-class:: filetree-caption

``pairs.h5``

.. rst-class:: filetree

* ``<group>/``

  * ``<α>`` :math:`(M, 1)`
  * ``<β>`` :math:`(M, 1)`
  * …

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``<α>``
    - One pair map, named freely: the partner orbital :math:`\bar{i}_\alpha` of each orbital :math:`i`,
      integers. A negative entry means :math:`i` has no partner, which leaves the corresponding row and
      column of the correlation function zero.

Every ordered pair of maps :math:`(\alpha, \beta)` is measured and stored as its own
:math:`(M, M)` array ``PairCorr/<α>_<β>``.

For collinear walkers, only the elements :math:`i < j` are measured, and the diagonal and lower triangle are
zero. Noncollinear walkers measure all :math:`i, j`.

Compatibility
_____________

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Wavefunctions**
    - **Spin Symmetry**
  * - NOMSD
    - collinear, noncollinear
  * - PHMSD
    - collinear

Pair correlation functions are most useful for lattice models, but they can be measured with any
Hamiltonian for which pair maps are provided.

