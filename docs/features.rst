.. _features:

Current Features
================

SAFIRE is formulated in the language of second quantization in terms of the generic
interacting fermionic Hamiltonian. 
The code is written as generically as possible to reflect the inherent flexibility 
of the AFQMC method.
SAFIRE's features can be categorized into either generic AFQMC implementation features, 
or concrete implementations of Hamiltonians, trial wavefunctions, walkers, and observables.
We begin with a list of the generic AFQMC implementation features.
Next, we list the concrete implementations of each type of object.
We conclude by listing which :ref:`combinations of these components and features are currently compatible <supported-combinations>` with each other.

.. seealso::

   * See the :ref:`User Manual <user_manual>` for a detailed and comprehensive reference of SAFIRE's features.
   * For detailed input file specifications, see :ref:`Input File Description <input_afqmc>`.
   * For tutorials on using these features, see :ref:`SAFIRE Tutorials <tutorials>`.

General Features
----------------

**Numerical Features**
  - GPU acceleration - see :doc:`installation` for instructions on enabling GPU acceleration.
  - Fast Woodbury updates for multi-Slater determinant trial wavefunctions - see :ref:`list of Supported Combinations <supported-combinations>` for which Hamiltonians and computational architectures are supported.

**AFQMC Methodological Features**
  - Importance sampling with either hybrid or local energy importance function
  - Modified Gram-Schmidt orthogonalization of Walkers with configurable intervals
  - Restart capabilities from previous calculations
  - Multiple population control algorithms (pair branching, serial comb method)
  - Configurable weight thresholds and branching criteria
  - Load balancing across MPI processes (asynchronous and blocking modes)
  - Configurable measurement intervals for performance optimization
  - Mixed estimators and back-propagated estimators (can be used in the same calculation)

For more information on configuring these features, see the :ref:`Input File Description <input_afqmc>`.

Concrete Implementations of Hamiltonians, Wavefunctions, Walkers, and Observables
---------------------------------------------------------------------------------

SAFIRE supports multiple Hamiltonian types, wavefunction formats, walker types, and observables.
The specific **combinations** of these components that are supported depend on the computational 
architecture (CPU vs GPU).
Below, we provide a comprehensive list of supported combinations and features.

Supported Hamiltonian Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SAFIRE supports the following Hamiltonian formats.

**Dense Cholesky Hamiltonian**
   Molecular and solid systems with Cholesky decomposed two-body integrals.
**K-Point Factorized Hamiltonian** 
   For periodic solid systems exploiting k-point symmetry.
   Can directly read Hamiltonian files from CoQuí in this format.
**Lattice Model Hamiltonian**
   For lattice models (Hubbard, t-J, extended Hubbard, etc.).
**Tensor Hypercontraction (THC) Hamiltonian**
   For systems using THC factorization of the electron repulsion integrals.
   Provides memory-efficient representation for large systems.
   Can directly read Hamiltonian files from CoQuí in this format.

See :ref:`Hamiltonian File Formats <Hamiltonian-classes>` for format details.

Supported Wavefunction Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NOMSD (Non-Orthogonal Multi-Slater Determinant)**
   Single and multi-determinant trial wavefunctions represented 
   in terms of explicit Slater matrices.
   Compatible with all Hamiltonian types and acceleration modes.
**PHMSD (Particle-Hole Multi-Slater Determinant)**
   Single and (typically) multi-determinant trial wavefunctions built from 
   occupation strings ( i.e. explicit lists of occupied orbital indices ).
   Fast Woodbury updates implemented for some Hamiltonians / computational architectures (See below).

See :ref:`Wavefunction File Formats <Wavefunction-classes>` for format specifications.

Supported Walker Types
^^^^^^^^^^^^^^^^^^^^^^

**CLOSED (RHF/Closed Shell)**
   For closed-shell systems where all electrons are paired.
   Uses the same Slater matrix for both the up and down spin sectors.
   System must have an even number of electrons with :math:`N^\uparrow = N^\downarrow`.
**COLLINEAR (UHF/Open Shell)**
   For open-shell systems with collinear (up/down) spins.
   Uses separate :math:`\alpha` and :math:`\beta` Slater determinants per walker.
**NONCOLLINEAR (GHF/Noncollinear)**
   For systems with noncollinear spin arrangements.
   Uses a single Slater matrix represented in an explicit spin-orbital basis.
   Essential for systems with spin-orbit coupling or frustrated magnetism.

See :ref:`Random Walker Classes <Walker-classes>` for details.

Supported Estimators
^^^^^^^^^^^^^^^^^^^^

**Mixed Estimators**
   Ground state expectation values using mixed estimation.
   Provides approximate but efficient observable calculations.
   Only suitable for observables that commute with the interacting Hamiltonian.

**Back-Propagation Estimators**
   Pure ground state estimators via back-propagation technique.
   Multiple back-propagation lengths supported for convergence testing.
   Higher computational cost but improved accuracy for ground state properties.

For detailed information on configuring estimators, see :ref:`Estimators reference <estimators>` 
and :ref:`Observables reference <observables>`.

Supported Observables
~~~~~~~~~~~~~~~~~~~~~

SAFIRE supports a comprehensive set of physical observables for quantum many-body systems:

**Full 1-RDM**
   Complete one-body reduced density matrix (all walker types)
**Two-body RDM**
   Full two-body reduced density matrix (CLOSED/COLLINEAR only)
**Diagonal Two-body RDM**
   Diagonal elements of 2-RDM (all walker types)
**Spin-Spin Correlation**
   Magnetic correlation functions (all walker types)
**Pair Correlation Functions**
   Superconducting pair correlators - recommended for lattice models only (COLLINEAR/NONCOLLINEAR only)

For detailed mathematical definitions, input specifications, and compatibility matrices, 
see :ref:`Observables reference <observables>`.

.. _supported-combinations:

Supported Combinations
----------------------

Here, we list the supported **combinations** of Hamiltonians, trial wavefunctions, 
and walker types. If a combination is not listed, it should be assumed that the combination is not supported.
We indicate in parentheses whether the combination supports CPU, GPU, or is in testing phase.

**Dense Cholesky Hamiltonian**
    **NOMSD Wavefunction**
        * CLOSED walkers (CPU, GPU)
        * COLLINEAR walkers (CPU, GPU)  
        * NONCOLLINEAR walkers (CPU, GPU)
    **PHMSD Wavefunction**
        * COLLINEAR walkers - with fast Woodbury algorithm (CPU, GPU)

**K-Point Factorized Hamiltonian**
    **NOMSD Wavefunction**
        * CLOSED walkers (CPU, GPU)
        * COLLINEAR walkers (CPU, GPU)
        * NONCOLLINEAR walkers (CPU, GPU)

**Lattice Model Hamiltonian**
    **NOMSD Wavefunction**
        * COLLINEAR walkers (CPU, GPU for Continuous Hubbard-Stratonovich Transformations)
        * NONCOLLINEAR walkers (CPU, GPU for Continuous Hubbard-Stratonovich Transformations)

**THC Hamiltonian**
    **NOMSD Wavefunction**
        * CLOSED walkers (CPU, GPU)
        * COLLINEAR walkers (CPU, GPU)

Not Currently Supported
^^^^^^^^^^^^^^^^^^^^^^^

* Discrete Hubbard-Stratonovich Transformation on GPU for Lattice Model Hamiltonian **please use CPU instead**.
* PHMSD trial wavefunctions with THC Hamiltonians or Lattice Model Hamiltonians
* PHMSD trial wavefunctions with K-Point Factorized Hamiltonians *on CPU* - **please use GPU instead.**
