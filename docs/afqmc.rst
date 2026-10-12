.. _afqmc:

Auxiliary-Field Quantum Monte Carlo
===================================

The auxiliary-field quantum Monte Carlo (AFQMC) method is a wavefunction-based many-body method which is formulated in terms of a generic, interacting second-quantized Hamiltonian.
For a detailed description of the algorithm, we refer the reader to :cite:`ZhangConstrained1997,PhysRevLett.90.136401,AFQMC_review,motta_initio_2018`. Here we will restrict ourselves to a coarse overview of the method.

AFQMC calculates the many-body ground-state wavefunction, :math:`| \Psi_0 \rangle`, by performing an imaginary-time projection,

.. math::
  :label: eq-imag-time-prop

  |\Psi_0⟩ \propto \lim_{\beta \to \infty} e^{-\beta H} | \Psi_\mathrm{I} \rangle, 

on an initial wavefunction :math:`| \Psi_\mathrm{I} \rangle` with a many-body Hamiltonian, which in second quantization can be written,

.. math::
  :label: eq-h012-af

  H &= H_1 + H_2 \\
    &= \sum\limits_{ij} h_{ij} c^\dagger_{i}c_{j}
    + \frac 1 2 \sum\limits_{ijkl} v_{ijkl} c^\dagger_{i}
      c^\dagger_{j}c_{l}c_{k}

where the indices are combined orbital and spin indices.

Performing this projection in the many-body Hilbert space as written is as prohibitively expensive as calculating the ground state directly. There is, however, a way to reformulate this single many-body imaginary-time projection in terms of an integral over many efficiently solvable one-body time evolutions. Sampling this integral stochastically is the main idea behind the AFQMC method.

Decomposing the Imaginary-Time Evolution
----------------------------------------

There are two kinds of decompositions that will allow us to rewrite the many-body time evolution. The first one is the Trotter decomposition :cite:`Trotter1959,Suzuki1976` that we use to split up the time evolution operator into a one-body and a two-body step:

.. math::

  e^{-β H} = \prod_p e^{-Δτ H} = \prod_p e^{-Δ\tau H_1/2} e^{-Δ\tau H_2} e^{-Δ\tau H_1 / 2} + O(Δ\tau^2)

In this form, if our initial wavefunction is a Slater determinant, we can already evaluate the one-body propagators :math:`e^{-Δ\tau H_1 /2}` very efficiently due to Thouless' theorem :cite:`Thouless`: the exponential of a one-body operator maps a Slater determinant to another Slater determinant whose orbitals are transformed by the exponential of the one-body matrix.

The two-body propagator :math:`e^{-Δ\tau H_2}`, on the other hand, has no such simple action and we need an additional way to simplify it. The Hubbard-Stratonovich decomposition :cite:`Hubbard1959,Stratonovich1957` does this for us by expressing exponentials of squares of one-body operators :math:`v_n` by a Gaussian integral over the one-body exponentials we want,

.. math::
  :label: eqHS

  e^{\sum_n v_n^2/2} = \int_{ℝ^N} \frac{d\mathbf{x}}{(2π)^{N/2}}\, e^{-\frac 12 \sum_n x_n^2} e^{\sum_n x_n v_n},

where the :math:`x_n` are the auxiliary fields that give AFQMC its name.

To use :eq:`eqHS` we need to rewrite :math:`H_2` as a sum of squares, but this is always possible by subtracting a one-body operator and using one of several ways of decomposing the two-body interaction tensor (see :ref:`Hamiltonian-classes` for the supported options in SAFIRE).

We can finally write the whole projection as an integral over one-body propagators :math:`B(\mathbf{x}) = e^{-Δτ H_1/2} e^{\sum_n x_n v_n} e^{-Δτ H_1/2}`,

.. math::
  :label: eqProp

  e^{-β H} \approx \prod_p\int \frac{d\mathbf{x}_p}{(2π)^{N/2}} e^{-\frac{\mathbf{x}_p^2}{2}} B(\mathbf{x}_p).

Since each :math:`B(\mathbf{x})` consists of one-body exponentials, we can successively use Thouless' theorem to propagate a Slater determinant through the integrand. We use Monte Carlo sampling to evaluate the whole integral.

Random Walk in Slater-Determinant Space
---------------------------------------

To sample the integral from :eq:`eqProp`, we employ a branching random walk: starting from an ensemble of Slater determinant walkers :math:`|Φ_w⟩`, we successively sample the auxiliary fields :math:`\mathbf{x}_p` at each step and apply the propagator :math:`B(\mathbf{x}_p)`. The wavefunction at imaginary time :math:`τ = Δτ p` is then given as

.. math::

  |Ψ(τ)⟩ \approx \sum_w |Φ_w(τ)⟩

where the :math:`|Φ_w(τ)⟩` are samples from the random walk. This sampling alone is not statistically efficient, so we apply importance sampling: Given a trial wavefunction :math:`|Ψ_\mathrm{T}⟩` that is a good approximation of the ground state, we can assume that Slater determinants that have a low overlap with the trial wavefunction will also contribute little to the full wavefunction. We can therefore absorb this overlap into a weight :math:`W_w`

.. math::

  |Ψ(τ)⟩ = \sum_w W_w(τ) \frac{|Φ_w(τ)⟩}{⟨Ψ_\mathrm{T}|Φ_w(τ)⟩}

and then adjust our random sampling so that the variance of the :math:`W_w(τ)` is low. We do this in two ways:

1. **Force bias**: we sample the auxiliary fields from a shifted Gaussian that minimizes fluctuations to leading order in :math:`Δτ`
2. **Population control**: we periodically resample the walkers, deleting low-weight walkers and duplicating high-weight ones, so that only walkers with :math:`O(1)` weights remain.

For algorithmic details, refer to the references cited above, in particular :cite:`AFQMC_review`.

Observables are evaluated from the walkers periodically as weighted averages, see :ref:`estimators`.

The Sign or Phase Problem
--------------------------

The way we introduced AFQMC above is exact up to the Trotter time step error and a small population control bias :cite:`UmrigarDMC1993` that vanishes inversely with the number of walkers. However, we have left out one important complication. For almost all systems of interest, the weights :math:`W_w` will be either signed or even complex numbers. We can use statistical importance sampling to keep the absolute weights :math:`|W_w(τ)| \sim O(1)` but if their sign or phase fluctuation is high, the total weight will average out to zero and lead to an exponential decrease in the signal-to-noise ratio in weighted averages. This is known as the sign or phase problem :cite:`LohSign1990`.

There are two approximations that allow us to get around the sign or phase problem. The first is the constrained-path approximation :cite:`ZhangConstrained1997` for cases when the weights are real but signed. In this case, we realize that if our trial wavefunction were the ground state, we could exactly eliminate the sign problem by removing paths that cross the nodal surface because once there is a crossing at any point in time, :math:`⟨Ψ_0|Φ_w(τ)⟩=0`, all descendant paths cancel too, :math:`⟨Ψ_0|e^{-(τ'-τ)H}|Φ_w(τ)⟩=0` for all :math:`τ' > τ`. The constrained-path approximation applies the same cutoff logic but with the approximate nodal surface :math:`⟨Ψ_\mathrm{T}|Φ_w(τ)⟩ = 0` of the trial wavefunction.

For the case where the weights are complex, a second approximation is necessary. Even if we cut off paths at the nodal surface, they can still freely rotate their phase and will do so uncontrollably as the random walk progresses. The phaseless approximation :cite:`PhysRevLett.90.136401` does not completely neglect these phase fluctuations but suppresses the weight of walkers that stray too far from the positive real line using a weight factor :math:`\max\{0, \cos(\Delta \theta)\}`, where
:math:`\Delta \theta = \arg\left[\frac{⟨Ψ_\mathrm{T}|Φ_w(τ+Δτ)⟩}{⟨Ψ_\mathrm{T}|Φ_w(τ)⟩}\right]`.

Both of these approximations introduce a bias that can be reduced by using more accurate trial wavefunctions. However, in practice, even very simple Hartree–Fock-based trial wavefunctions frequently give good results.

Bibliography
------------

.. bibliography:: /bibs/afqmc.bib

