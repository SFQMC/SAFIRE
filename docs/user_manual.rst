.. _user_manual:

User Manual
===========

.. toctree::
   :maxdepth: 3
   :hidden:
   
   user_manual/input_description_afqmc      
   user_manual/walkers
   user_manual/estimators
   user_manual/observables
   user_manual/hamiltonians
   user_manual/wavefunctions

(S)tochastic (A)uxiliary-(F)ields for (I)nte(R)acting (E)lectrons (SAFIRE)  
is a flexible, high-performance implementation of the Auxiliary Field Quantum Monte Carlo (AFQMC) method. 
It is designed to allow users to study a wide range of physical systems.
This manual provides detailed information on the various components of SAFIRE, including its input parameters, 
Hamiltonian, wavefunction, walker classes, and observables.
This user manual assumes a basic familiarity with AFQMC.
For a brief introduction to AFQMC, please see :ref:`afqmc`.


Input File
----------

SAFIRE uses a json-based input file to specify the parameters of the AFQMC calculation. 
This section contains parameters that control the calculation including choosing methodological parameters,
Hamiltonian, trial wavefunction, walker type, and observables.
A sample input file is shown below.
See the ":ref:`Input file description <input_afqmc>`" for more.

.. code-block::
  :caption: Sample input file for SAFIRE.
  :name: Listing 1

  {
    "afqmc": {
      "project": {
        "id": "my_qmc_run",
        "series": 0
      },
      "execute": {
        "walker_set": {
          "walker_type": "COLLINEAR"
        },
        "wavefunction": {
          "filename": "wfn.h5"
        },
        "hamiltonian": {
          "filename": "hamiltonian.h5"
        },
        "timestep": 0.01,
        "steps": 10000,
        "measure_interval": 10,
        "population_control_interval" : 10,
        "walker_ortho_interval" : 10 ,
        "num_walkers": 200,
        "estimator": {
          "name": "mixed",
          "onerdm" : {
            "name" : "my_onerdm"
          }
        }
      }
    }
  }



Hamiltonian Classes
-------------------

AFQMC is formulated in terms of a general second-quantized Hamiltonian,

.. math::
  :label: eq-h012

  \hat{H} =& H_0 + \hat{H}_1 + \hat{H}_2 \\
    =& E_0 + \sum\limits_{il\sigma\sigma'} h^{\sigma \sigma'}_{il} \hat{c}^\dagger_{i\sigma}\hat{c}_{l\sigma'}
    + \sum\limits_{ijkl\sigma\sigma'\sigma'\sigma} v_{ijkl}^{\sigma\sigma'\sigma'\sigma} \hat{c}^\dagger_{i\sigma}
      \hat{c}^\dagger_{j\sigma'}\hat{c}_{k\sigma'}\hat{c}_{l\sigma},

However, there are class-specific optimizations
that can be employed for different types of Hamiltonians.
This is especially true in the case of lattice model Hamiltonians where the form of the interactions is significantly simpler 
than the long-range Coulomb interaction found in ab initio Hamiltonians, for example.

See the :ref:`Hamiltonian classes <Hamiltonian-classes>` reference for more details.

Wavefunction Classes
--------------------

The trial wavefunction in AFQMC is typically a linear combination of Slater determinants,

.. math::
  :label: eq-trial-wf

  |\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle

where :math:`C_n` is a complex-valued coefficient, and :math:`|\Phi_n\rangle` are Slater determinants which are not necessarily orthogonal to each other. Of course, each Slater determinant consists of some set of single-particle orbitals, :math:`\{ \psi_p \}`, such that,

.. math::
  :label: eq-orbitals

  \psi_{p} = \sum_i \bar{C}_{ip} \phi_i

where :math:`\{\phi_i\}` are the chosen orthonormal basis set orbitals.
SAFIRE implements a few different forms for the trial wavefunction.

See the :ref:`Wavefunction classes <Wavefunction-classes>` reference for more details.

Walkers
-------

In AFQMC, walkers are typically single Slater determinants.

.. math::
  :label: eq-slater-matrix

  \Phi = \begin{bmatrix}
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0N} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1N} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M0} & \bar{C}_{M1} & \bar{C}_{M2} & \dots  & \bar{C}_{MN}
  \end{bmatrix}

SAFIRE allows one of several type of Slater determinant walkers to be used in calculations based on their spin symmetry.

See the :ref:`Walker classes <Walker-classes>` reference for more details.

Observables
-----------

In AFQMC, physical observables are computed on the fly using a Monte Carlo representation of the many-body wavefunction.
Both mixed estimators of the form, 

.. math::
  :name: mixedEstimators

  \langle \hat{O} \rangle_\mathrm{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_\mathrm{T} | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_\mathrm{T} | \Phi_{n,k} \rangle}

where :math:`n` is the projection step index,
:math:`| \Phi_{n,k} \rangle` are Slater determinant walkers with weight :math:`W_{n,k}`,
and :math:`| \Psi_\mathrm{T} \rangle` is the trial wavefunction,
and pure estimators using the Back-Propagation (BP) algorithm of the form,

.. math::
  :label: bpEstimators

  \langle \hat{O} \rangle_\mathrm{BP} = \frac{1}{\sum_k W_{n+m,k}} \sum_k W_{n+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{n,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{n,k}\rangle},


where :math:`| \tilde{\Phi}_{m,k} \rangle` are the back-propagated walkers, given by

.. math::

   | \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{n,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{n+m-1,k} ) | \Psi_\mathrm{T} \rangle,

are implemented.

Several observables, :math:`\hat{O}`, are implemented in SAFIRE which can each be used with either type of estimator.

See the :ref:`Observables <observables>` reference for more details.
