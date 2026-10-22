.. _getting_started:

Getting Started with SAFIRE
================================

.. toctree::
  :maxdepth: 3
  :hidden:

  installation
  tutorials
  examples
  user_manual
  safiretools/safiretools
  changelog
  afqmc

.. _anatomy_of_safire:

**\ S**\ tochastic **A**\ uxiliary-\ **F**\ ields for \ **I**\ nte\ **R**\ acting **E**\ lectrons (SAFIRE)
is a flexible, high-performance implementation of the Auxiliary Field Quantum Monte Carlo (AFQMC) method. 
SAFIRE is organized into two basic components.
The first is the SAFIRE executable which is a high-performance code
implemented in terms of generic classes of interacting many-electron Hamiltonians 
and wavefunctions. It is designed as generically as possible to take full advantage of
AFQMC's flexibility in treating general interacting electron systems. 
The second component is the :ref:`safiretools` Python package which allows users to both generate the inputs used 
by SAFIRE and analyze the outputs.
A formal overview of AFQMC is provided in :ref:`afqmc`.

How to Run an AFQMC Calculation
-------------------------------

After installing SAFIRE (see :ref:`install`), a basic AFQMC calculation
can be performed by following the steps below.
The :ref:`tutorials` walk through each step, and the :ref:`user_examples` show them in complete workflows.

1. Set up the AFQMC calculation by specifying a Hamiltonian and a trial wavefunction using safiretools.
2. Run the SAFIRE executable.
3. Analyze the stochastic data.

Tutorials
---------

One of AFQMC's greatest strengths is its flexibility in treating a very wide range of interacting electron problems.
There are, of course, some domain specific considerations in performing AFQMC calculations. 
We have provided a series of tutorials spanning applications of AFQMC to lattice models, solids, and molecules
in order to illustrate the practical knowledge necessary to perform high-quality AFQMC calculations in each respective application domain.

You can find the tutorials here: :ref:`tutorials`.

Examples
--------

The :ref:`user_examples` illustrate complete workflows in various application domains.

Exploring the examples is a great way to familiarize yourself with the capabilities of SAFIRE and see how to apply them to your own problems.


