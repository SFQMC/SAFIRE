.. _estimators:

Estimators
==========

This page is organized as follows.
We begin by explaining the different types of estimators available in SAFIRE.
Estimators are used to compute observables.
An estimator requires at least one observable to compute.
Below, we explain each :ref:`type of observable <observables>` that can be computed using these estimators.
All observables can be used with any of the estimators.

.. important::

    Unlike all other types of input blocks, the "name" parameter is used to determine the type of estimator - i.e. 
    NOT to define an identifier that can be used elsewhere in the input file.

Mixed Estimators
----------------

SAFIRE implements mixed estimators, which are used to compute observables :math:`\hat{O}` such that :math:`[\hat{O},\hat{H}] = 0`.
In this case, a mixed estimator is an unbiased estimator.
Formally, mixed estimators evaluate an observable :math:`\hat{O}` as

.. math::
  \langle \hat{O} \rangle_\mathrm{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_\mathrm{T} | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_\mathrm{T} | \Phi_{n,k} \rangle}


Settings
~~~~~~~~

.. code-block:: json
  :caption: Sample input block for Mixed Estimator.  
  
  "estimator": {
    "name": "mixed",
    "measure_interval" : 10,
    "onerdm": {
      "name": "one_rdm",
    }
  }


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **measure_interval**
     - Inherited from execute block (default: 10)
     - Number of projection steps between measurements. Measurement is the most expensive operation in AFQMC, so a larger "measure_interval" will reduce the CPU time necessary to perform AFQMC calculations.

Energy Estimator
----------------

SAFIRE implements a specialized mixed estimator for the energy, which is always added 
to an execute block by default.
The only reason to explicitly define it is to customize its settings.
Formally, it evaluates

.. math::
  :name: energyEstimator

  E = \frac{\langle \Psi_\mathrm{T} | \hat{H} | \Psi^s \rangle}{\langle \Psi_\mathrm{T} | \Psi^s \rangle} \approx 
  \frac{1}{\sum_n W^s_n} \sum_n W^s_n \frac{\langle \Psi_\mathrm{T} | \hat{H} | \Phi^s_n \rangle}{\langle \Psi_\mathrm{T} | \Phi^s_n \rangle},

Settings
~~~~~~~~

.. code-block:: json
  :caption: Sample input block for Energy Estimator.  
  
  "estimator": {
    "name": "energy",
    "measure_interval": 10,
    "print_components": true
  }


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **measure_interval**
     - Inherited from execute block (default: 10)
     - Number of projection steps between measurements. Measurement is the most expensive operation in AFQMC, so a larger "measure_interval" will reduce the CPU time necessary to perform AFQMC calculations.
   * - **print_components**
     - false
     - if true, print the one-body and two-body direct, and two-body exchange components of the energy separately (in addition to the total energy). Note: the one-body energy also includes any constant energy contributions.
 
.. only:: developer

    Advanced / developer setting
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
      :header-rows: 1
      :widths: 25 20 55

      * - **Parameter**
        - **Default**
        - **Description**
      * - print_sign
        - false
        - if true, print detailed information about the phase and constraint in the scalar.dat output file.
      * - truncate
        - false
        -

Back-Propagation (BP) Estimators
--------------------------------

Mixed estimators are biased for observables :math:`\hat{O}` such that :math:`[\hat{O},\hat{H}] \neq 0`, 
which requires the use of pure estimators. The Back-Propagation (BP) algorithm is used to compute pure estimators in SAFIRE.
The back-propagated estimator has the form,

.. math::

  \langle \hat{O} \rangle_\mathrm{BP} = \frac{1}{\sum_k W_{s+m,k}} \sum_k W_{s+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{s,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{s,k}\rangle}


where :math:`| \Phi_{s,k} \rangle` are the usual forward-projected Slater determinant random walkers,
and :math:`| \tilde{\Phi}_{m,k} \rangle` are the back-propagated walkers given by,

.. math::

  | \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{s,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{s+m-1,k} ) | \Psi_\mathrm{T} \rangle.

The index :math:`s` corresponds to the current forward projection step,
and :math:`m` is the back-propagated step index.
We note that each random walker has a corresponding back-propagated partner
which share the same path in auxiliary-field space.

Sample Input File
~~~~~~~~~~~~~~~~~

.. code-block:: json
    :caption: Sample input block for Back-Propagation (BP) Estimator.

    "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "propagation_steps": [800],
        "onerdm": {
            "name": "one_rdm"
        }
    }


Configuration
~~~~~~~~~~~~~

BP is invoked in SAFIRE by including a "back_propagation" estimator in the input file.
Observables are added to the BP block in the same way as for mixed estimators.

Back-Propagation Lengths
~~~~~~~~~~~~~~~~~~~~~~~~

Since BP involves an additional projection, a BP estimator has no ``measure_interval``: what it
is given instead is ``propagation_steps``, the number of steps to project back over. That
length doubles as the measurement interval, because a new BP window starts as soon as the
previous one has been covered.

``propagation_steps`` has no default -- a back-propagation length is a property of the estimate
and is never inherited from the execute block -- so a BP estimator has to state it.

Multiple BP Lengths
~~~~~~~~~~~~~~~~~~~

SAFIRE implements the capability of running BP with multiple BP measurement lengths within the same calculation.
We call each measurement with a different BP length an "average".
This allows the forward projection to be reused while checking for convergence in the BP length.
To define multiple BP lengths/averages, simply list all of the BP lengths that you would like
to use, in steps. The longest one is how often a new window starts:

.. code-block:: json
    :caption: Sample input block for Back-Propagation (BP) Estimator with multiple BP lengths.

    "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "propagation_steps": [600, 700, 800],
        "onerdm": {
            "name": "one_rdm"
        }
    }


Equilibration
~~~~~~~~~~~~~

No observable is measured during the equilibration phase at the beginning of an AFQMC
calculation, back-propagated ones included. This is recommended since computing observables is
typically expensive and since AFQMC needs to equilibrate before samples can meaningfully
contribute to the average. The length of that phase is the execute block's
``equilibration_steps``, in steps; back propagation anchors its first window at the end of it.


Settings
~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **path_restoration**
     - false
     - if true, use path restoration in the back-propagation algorithm.
   * - **bp_walker_ortho_interval**
     - 10
     - Interval for walker orthogonalization during back-propagation.
   * - **propagation_steps**
     - required
     - An array of back-propagation lengths, in steps. Each one is measured on its own, and the longest one is how often a new back-propagation window starts.

.. only:: developer

    Advanced / developer settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
      :header-rows: 1
      :widths: 25 20 55

      * - **Parameter**
        - **Default**
        - **Description**
      * - **extra_path_restoration**
        - false
        - if true, perform an extra path restoration in the back-propagation algorithm.
