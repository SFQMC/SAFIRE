.. _solids_overview:

SAFIRE for Ab Initio Solids
============================

.. image:: /images/shared/00_top_level/solid.png
   :width: 500


Some Preliminaries
------------------

The Hamiltonian
~~~~~~~~~~~~~~~

SAFIRE uses an orthonormal orbital basis to represent solid-state Hamiltonians in second quantization.
These orbitals are typically obtained from plane-wave density functional theory (DFT) calculations
(such as Kohn-Sham orbitals), but SAFIRE itself is agnostic to how the orbitals were originally computed.

For crystalline solids, Bloch's theorem and translational symmetry allow the problem to be expressed
in terms of k-points in the Brillouin zone. The Hamiltonian takes a k-point factorized form:

.. math::

   \hat{H} = E_0 + \sum_{\mathbf{k}} \sum_{pq\sigma} h^{\mathbf{k}}_{pq} \hat{c}^\dagger_{\mathbf{k}p\sigma}\hat{c}_{\mathbf{k}q\sigma}
   + \frac{1}{2}\sum_{\mathbf{k}\mathbf{k}'\mathbf{Q}} \sum_{pqrs\sigma\sigma'} v_{(\mathbf{k}+\mathbf{Q})p,\,(\mathbf{k}'-\mathbf{Q})q,\,\mathbf{k}r,\,\mathbf{k}'s}
   \hat{c}^\dagger_{(\mathbf{k}+\mathbf{Q})p\sigma} \hat{c}^\dagger_{(\mathbf{k}'-\mathbf{Q})q\sigma'} \hat{c}_{\mathbf{k}'s\sigma'} \hat{c}_{\mathbf{k}r\sigma},

where :math:`\mathbf{k}`, :math:`\mathbf{k}'` and :math:`\mathbf{Q}` are vectors of the k-point mesh sampling the first Brillouin zone,
with :math:`\mathbf{k}+\mathbf{Q}` and :math:`\mathbf{k}'-\mathbf{Q}` folded back into it,
:math:`p,q,r,s` are orbital indices at each k-point,
:math:`\sigma, \sigma'` are spin indices,
:math:`E_0` includes nuclear repulsion and other constant terms,
:math:`h^{\mathbf{k}}_{pq}` are one-body matrix elements (kinetic energy and electron-ion interactions),
and :math:`v` are two-body electron-electron interaction matrix elements.
The interaction transfers the momentum :math:`\mathbf{Q}` from one electron to the other, so crystal momentum is conserved.
See :ref:`k_point_factorized` for how SAFIRE factorizes this interaction.

The underlying plane-wave DFT calculations typically use pseudopotentials to replace core electrons,
reducing the computational cost while maintaining accuracy for valence properties.
The orbital basis and matrix elements provided to SAFIRE already incorporate these pseudopotential effects.

**CoQuí's Role**: The conversion from plane-wave DFT output to this k-point factorized orbital representation
is handled by `CoQuí <https://github.com/AbInitioQHub/coqui>`_, which extracts the orbitals computed in the
plane-wave basis and computes the necessary Hamiltonian matrix elements in the format required by SAFIRE.

|

Trial Wavefunctions
~~~~~~~~~~~~~~~~~~~

A trial wavefunction is used in AFQMC which is, in general, some linear
combination of Slater determinants,

.. math::

   | \Psi_\mathrm{T} \rangle = \sum_n C_n |\Phi_n\rangle,

where :math:`|\Phi_n\rangle` are Slater determinants,
and :math:`C_n` is a coefficient.

For solid-state systems, the trial wavefunction typically consists of Kohn-Sham orbitals
obtained from DFT calculations with various exchange-correlation functionals:

- **Local Density Approximation (LDA)**: Simple and computationally efficient
- **Generalized Gradient Approximation (GGA)**: Improved accuracy for many systems
- **Hybrid functionals**: Incorporate exact exchange, often better for band gaps

The quality of the trial wavefunction is crucial for AFQMC accuracy.
For solids, the trial wavefunction must properly capture:

- **K-point sampling**: The trial includes orbitals at all k-points in the mesh.
  CoQuí handles the extraction and formatting of these k-point dependent orbitals.
- **Symmetries**: Crystal symmetry, spin symmetry, and other relevant quantum numbers
- **Boundary conditions**: Finite-size effects can be mitigated using twisted boundary conditions

While single-determinant trial wavefunctions from DFT are most common for solids,
multi-determinant trial wavefunctions may be beneficial for strongly correlated systems
or for capturing specific symmetry-breaking states.

Note that while the orbitals may be computed using a plane-wave basis in the underlying DFT calculation,
they are represented as orbitals (not plane waves) in the SAFIRE HDF5 files prepared by CoQuí.

Typical Workflow
----------------

A typical AFQMC calculation for solids using SAFIRE follows these steps:

1. **Setup crystal structure**: Define the crystal lattice parameters, atomic positions, and choose appropriate pseudopotentials for your system.

2. **Run plane-wave DFT calculation**: Use a plane-wave DFT code (e.g., Quantum Espresso, VASP) with:

   - Appropriate k-point mesh for Brillouin zone sampling
   - Converged plane-wave energy cutoff
   - Suitable exchange-correlation functional

3. **Convert with CoQuí**: Use CoQuí to:

   - Extract the Kohn-Sham orbitals from the DFT calculation
   - Compute Hamiltonian matrix elements in the orbital basis
   - Write k-point factorized HDF5 files for SAFIRE

4. **Prepare the SAFIRE job file**: Create a JSON job file specifying:

   - Hamiltonian and trial wavefunction file paths
   - AFQMC parameters (timestep, number of walkers, etc.)
   - Desired observables

5. **Run AFQMC calculation**: Execute SAFIRE (typically in parallel using MPI)

6. **Analyze results**: Extract energies, observables, and apply finite-size corrections as needed

|

Software Prerequisites
----------------------

Many mature electronic structure codes exist and are widely used for solid-state calculations.
For this reason, SAFIRE does not implement effective one-body methods
such as density functional theory (DFT).
Instead, SAFIRE is designed to use externally generated Hamiltonians and trial wavefunctions
for easy integration into existing workflows.

SAFIRE can directly run AFQMC calculations using Hamiltonians and trial wavefunctions generated using CoQuí.

**Required Software:**

1. **CoQuí** (`https://github.com/AbInitioQHub/coqui <https://github.com/AbInitioQHub/coqui>`_):

   - Essential tool for converting plane-wave DFT output to SAFIRE format
   - Extracts Kohn-Sham orbitals from DFT calculations
   - Computes Hamiltonian matrix elements in k-point factorized orbital representation
   - Writes HDF5 files with orbitals and matrix elements compatible with SAFIRE
   - Handles k-point sampling and orbital indexing conventions

2. **Plane-wave DFT code** (via CoQuí support):

   - **Quantum Espresso**: Widely used open-source plane-wave DFT code
   - **PySCF**: Python-based electronic structure package with pGTO capabilities
   - Other codes as supported by CoQuí

**Important Considerations:**

- **K-point convergence**: Ensure the underlying DFT calculation uses a sufficiently dense k-point mesh
- **Plane-wave cutoff**: Ensure the DFT plane-wave cutoff is converged, since it determines the quality of the orbitals and matrix elements
- **Pseudopotentials**: Choose well-tested pseudopotentials appropriate for your system
- **File formats**: CoQuí outputs HDF5 files containing the k-point factorized Hamiltonian and trial wavefunction

Orbital Basis Considerations
----------------------------

**Number of Orbitals per K-point**:

The number of orbitals (bands) at each k-point is chosen when converting the DFT output, e.g. with ``nbnd`` in CoQuí,
and can be at most the number of bands computed in the DFT calculation.
More orbitals give a more complete basis at a higher cost. Convergence studies are essential.

**K-point Mesh Density**:

- The k-point mesh must be dense enough to capture the physics of the system
- Metals typically require denser meshes than insulators
- Convergence should be checked by comparing results with different mesh densities

**Orbital Truncation**:

- High-energy orbitals can sometimes be truncated to reduce computational cost
- Care must be taken to ensure important physics is not lost
- Convergence with respect to the number of retained orbitals should be verified

|

The Tutorials
-------------

The following tutorials will guide you through AFQMC calculations
using SAFIRE in order to teach you the typical workflow,
and some of the main features of SAFIRE.
We assume that you are familiar with typical electronic structure calculations using pseudopotentials and a plane-wave basis.
We also assume that you
have access to CoQuí
and that you are familiar with its basic use.
Each tutorial builds on the previous one.
We recommend going through them in order.

.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   */*

.. seealso::
   Worked examples for :ref:`examples_solids`
