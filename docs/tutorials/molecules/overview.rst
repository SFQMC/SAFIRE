.. _molecules_overview:

SAFIRE for Quantum Chemistry
============================

.. image:: /images/shared/00_top_level/molecule.png
   :width: 500

This document is the entry point for learning how to use SAFIRE for quantum chemistry calculations. We will first go over the basics of setting up a calculation before branching out to more specific applications at the end.

Preliminaries
-------------

Typical quantum chemistry calculations are performed in a basis
of contracted Gaussian-type orbitals (cGTOs), :math:`g_\mu(\vec{r})`;
however, AFQMC is formulated in the language of second quantization
and requires an orthonormal basis.
Some common choices of orthonormal basis are the set of canonical Hartree-Fock orbitals,
or orthogonalized atomic orbitals, but in principle any set of orthonormal orbitals can be used.
An orbital from the orthonormal orbital basis, :math:`\phi_i(\vec{r})`,
is represented in the basis of cGTOs as,

.. math::

   \phi_i(\vec{r}) = \sum_{\mu} C_{\mu i } g_{\mu} (\vec{r}).

The Hamiltonian is in turn expressed in the orthonormal orbital basis as

.. math::

   \hat{H} = H^0 + \sum_{ij\sigma} H^1_{ij} \hat{c}^{\dagger}_{i\sigma}\hat{c}_{j\sigma} + \frac{1}{2}\sum_{ijkl\sigma\sigma'} H^2_{ijkl} \hat{c}^{\dagger}_{i\sigma} \hat{c}^{\dagger}_{j\sigma'} \hat{c}_{k\sigma'} \hat{c}_{l\sigma},

where, :math:`\hat{c}^{\dagger}_{i\sigma}` (:math:`\hat{c}_{i\sigma}`) create (annihilate) electrons with spin :math:`\sigma` in orbital :math:`\phi_i(\vec{r})`,
:math:`H^0` is a constant energy (typically from nuclear repulsion),
:math:`H^1_{ij}` are one-body matrix elements,
and :math:`H^2_{ijkl}` are electron-electron interaction matrix elements.
All standard forms of quantum chemistry Hamiltonians can be written in this form.

The Hamiltonian
~~~~~~~~~~~~~~~

In these tutorials, we will be using the standard Born-Oppenheimer Hamiltonian, unless otherwise stated, which is given in first quantization
by

.. math::

   H_\mathrm{Born-Oppenheimer} = \sum_p^{N_\mathrm{e}}\left(  -\frac{1}{2} \nabla^2_p - \sum_A^{N_\mathrm{atom}} \frac{Z_A}{ | \vec{R}_A - \vec{r}_p | }  \right) + \frac{1}{2} \sum^{N_\mathrm{e}}_{q \neq p} \left(\frac{1}{r_{pq}}\right) + \frac{1}{2}\sum_{A' \neq A}^{N_\mathrm{atom}} \frac{Z_{A} Z_{A'}}{| R_A - R_{A'} |},

where,
:math:`N_\mathrm{e}` is the number of electrons,
:math:`\vec{r}_p` is the position of electron :math:`p`,
:math:`N_\mathrm{atom}` is the number of atomic nuclei,
and :math:`Z_A` and :math:`\vec{R}_A` are the atomic number and position, respectively, of atomic nuclei, :math:`A`.
This Hamiltonian can be expressed in the language of second quantization by making the identifications,

.. math::

   H^0 = \frac{1}{2}\sum_{A' \neq A}^{N_\mathrm{atom}} \frac{Z_{A} Z_{A'}}{| R_A - R_{A'} |},

is the constant nuclear repulsion energy,

.. math::

   H^1_{ij} = \int d\vec{r}_p  \phi^*_i (\vec{r}_p) \left(  -\frac{1}{2} \nabla^2_p - \sum_A^{N_\mathrm{atom}} \frac{Z_A}{ | \vec{R}_A - \vec{r}_p | }  \right) \phi_j (\vec{r}_p),

contains the one-body kinetic term and electron-nuclei interactions,
and

.. math::

   H^2_{ijkl} = (il|jk) = \int \int d\vec{r}_p d\vec{r}_q  \left( \phi^*_i (\vec{r}_p)  \phi_l (\vec{r}_p)  \frac{1}{|\vec{r}_p - \vec{r}_q|}  \phi^*_j (\vec{r}_q) \phi_k (\vec{r}_q)  \right)

is the electron-electron interaction.

Trial Wavefunctions
~~~~~~~~~~~~~~~~~~~

A trial wavefunction is used in AFQMC which is, in general, some linear
combination of Slater determinants,

.. math::

   | \Psi_\mathrm{T} \rangle = \sum_n C_n |\Phi_n\rangle,

where :math:`|\Phi_n\rangle` are Slater determinants,
and :math:`C_n` is a coefficient.
This can either be a configuration interaction-type expansion
or a linear combination of nonorthogonal Slater determinants.

In the former case, it is convenient to specify Slater determinants in terms of occupation vectors as,

.. math::

   | \Psi_\mathrm{T} \rangle = \sum_n C_n | O_\alpha \rangle \otimes | O_\beta \rangle,

where :math:`O_\sigma = [o_0, o_1, ..., o_{N_\sigma-1}]` is the set of orbitals which are occupied for :math:`\sigma=\alpha, \beta`.

In the latter case, Slater determinants are often expressed explicitly by their respective Slater matrices,
:math:`[\Phi^\sigma_n]_{ip}`, where :math:`p` is the electron index.

Common methods for computing a trial wavefunction include:

- Hartree-Fock (HF)
- Kohn-Sham density functional theory (DFT)
- Complete-active space self-consistent field (CASSCF)
- Semistochastic heat-bath configuration interaction (SHCI)

among others.

Typical Workflow
----------------

.. image:: /images/shared/00a_molecules_intro/QChemWorkflow_v3.png
   :width: 900

SAFIRE reads :math:`\hat{H}` in generic second-quantized form from an HDF5 file.
This allows SAFIRE to use any Hamiltonian that can be expressed in this form.
The ``safiretools`` Python package can write a Hamiltonian to the SAFIRE format
given arrays containing the matrix elements,
or from the standard FCIDUMP format.
If the optional dependency PySCF is installed, ``safiretools`` is able to use PySCF's
interface to ``libcint`` to generate the required matrix elements.

The trial wavefunction is also read from an HDF5 file.
Similar to the Hamiltonian, ``safiretools`` includes tools for writing wavefunctions
in this format as will be seen in the tutorials.

Software Prerequisites
----------------------

Many mature quantum chemistry codes exist and are widely used in the quantum chemistry community.
For this reason, SAFIRE does not implement common quantum chemistry methods,
such as Hartree-Fock (HF), density functional theory (DFT), complete active space (CAS) methods, etc.
Instead, SAFIRE is designed to use externally generated Hamiltonians and trial wavefunctions
for easy integration into existing workflows.

To use SAFIRE, you will need a quantum chemistry code that can:

1. generate / output Hamiltonian matrix elements, :math:`H^0`, :math:`H^1_{ij}`, and :math:`H^2_{ijkl}`. SAFIRE comes with a converter from the common FCIDUMP format to its internal format via safiretools.
2. compute / output wavefunctions to use as trial wavefunctions. This can either be:

   - a list of CI coefficients, :math:`C_n`, and occupation strings :math:`O_\sigma = [o_0, o_1, ..., o_{N_\sigma-1}]`
   - a set of CI coefficients, :math:`C_n`, with corresponding non-orthogonal Slater determinant Slater matrices, :math:`[\Phi^\sigma_n]_{ip}`.

For all tutorials except for Hello SAFIRE we assume that you have access to a quantum
chemistry code that can do all of this.
If you do not have access to a quantum chemistry code that can do all of this,
PySCF is a possible choice which is free and open source.
The tutorials will teach you how to input this information using the ``safiretools`` Python package.

The Tutorials
-------------

The following tutorials will guide you through AFQMC calculations
using SAFIRE in order to teach you the typical workflow,
and some of the main features of SAFIRE.
We assume that you are familiar with typical quantum chemistry calculations
in a standard cGTO basis including Hartree-Fock,
and CAS-methods.
We also assume that you
have access to a quantum chemistry code that can output a FCIDUMP file
and that you are familiar with using it.
Each tutorial builds on the previous one.
We recommend going through them in order.

.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   */*

.. seealso::
   Worked examples for :ref:`examples_molecules`
