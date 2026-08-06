.. _input_afqmc:

Input file
==========

The AFQMC executable runs AFQMC calculations based on the settings provided in an
input file written in json.
The input file is described in detail below.
This reference document assumes familiarity with the AFQMC method.
An explanation of the AFQMC method is provided elsewhere in the :ref:`AFQMC overview <afqmc>`.
The structure of the input file is described below in  :ref:`input_file_strcuture`.

.. seealso::

   The :ref:`SAFIRE tutorials <tutorials>` covers the input file along with other relevant topics.

.. _input_file_strcuture:

Input File Structure
--------------------

The input file is json format and consists of several json "blocks".
Some input blocks may have sub-blocks as described below.
The highest level block determines what kind of calculation to run.
Currently, "afqmc" is the only supported option.

.. image:: /images/shared/02_input_file/input_file_01_all.png
   :width: 800px
   :alt: Input file structure diagram
   :align: center

*Input file structure overview*

.. important::

  The wavefunction, hamiltonian, and walker_set blocks may be defined either within an execute block or outside of an execute block.
  If defined outside of an execute block, they must be given a name via the "name" parameter.
  They can then be referenced by name within an execute block.

Below is a sample input file for an AFQMC calculation in SAFIRE. 
We will explore the details of this input file in the following sections.

.. code-block::json
  :caption: Sample input file for AFQMC.
  :name: Listing 1

  {
    "afqmc": {
      "project": {
        "id": "qmc",
        "series": 0
      },
      "execute": {
        "walker_set": {
          "walker_type": "COLLINEAR"
        },
        "wavefunction": {
          "filename": "afqmc.h5"
        },
        "hamiltonian": {
          "filename": "afqmc.h5"
        },
        "timestep": 0.01,
        "steps": 10000,
        "measure_interval_multiplier": 1,
        "population_control_interval" : 10,
        "walker_ortho_interval" : 10 ,
        "n_walkers_per_mpi_task": 10
      }
    }
  }


.. seealso::

  The sample input file above is just one possible input file layout.
  See the :ref:`input file recipes <run_afqmc_ex_4>` example for other possible layouts.

.. only:: developer

  An experimental driver, called "csafqmc", also exists to perform a (c)orrelated (s)ampling afqmc calculation.


.. code-block::json
  :caption: Sample input file for AFQMC.
  :name: Listing 1

  {
    "afqmc": {
      "project": {
         # project settings here
      },
      "execute": {
         # execution settings here
      }
    }
  }

.. _project_block:

Project block
-------------

The "project" block is optional and can be used to specify miscellaneous high-level settings.
It contains the following options.

.. code-block::json
  :caption: Sample "project" block
  :name: Listing 200

  {
    "afqmc": {
      "project": {
        "id": "afqmc",
        "series": 0
      },

    ...

  } 


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **id**
     - afqmc
     - A string used to identify the project. AFQMC output files are prefixed by "id" and "series" (below) ``[id].[series].*``
   * - **series**
     - 0
     - An up to 3-digit integer which may be used to identify the specific calculation that is being performed. AFQMC output files are prefixed by "id" and "series" (below) ``[id].[series].*``

.. _execute_block:

Execute block
-------------

The "execute" block consists of methodological parameters for AFQMC
and several sub-blocks which are each explained below.
The sub-blocks within the "execute" block represent different mathematical objects 
which are used in AFQMC such as the Hamiltonian and the trial wavefunction.

.. image:: /images/shared/02_input_file/input_file_02_execute.png
   :width: 800px
   :alt: Execute block structure diagram
   :align: center

*Execute block structure overview*


.. code-block::json
  :caption: Sample input file for AFQMC with Back-Propagation
  :name: Listing 201

  {
    "afqmc": {
      ...
      "execute": {
        "walker_set": {
          ...
        },
        "wavefunction": {
          ...
        },
        "hamiltonian": {
          ...
        },
        "timestep": 0.01,
        "steps": 10000,
        "population_control_interval" : 10,
        "measure_interval_multiplier": 1,
        "walker_ortho_interval" : 10 ,
        "n_walkers_per_mpi_task": 10 ,
        "seed" : 42,
        "estimator": {
          ...
        },
        "estimator": {
          ...
        }
      }
    }
  }


Externally defined blocks
~~~~~~~~~~~~~~~~~~~~~~~~~

The wavefunction, hamiltonian, and walker_set blocks may be defined either within an execute block or outside of an execute block.
If defined outside of an execute block, they must be given a name via the "name" parameter.
They can then be referenced by name within an execute block instead of defining a json block.
For example, in the input file below, the walker_set is defined outside of the execute block and given the name "my_walkers".

.. code-block:: json
  :caption: Sample input file for AFQMC with Externally Defined Blocks

  {
    "afqmc": {
      "walker_set": {
          "name" : "my_walker_set"
      },
      "execute": {
        "walker_set" : "my_walker_set",
        "wavefunction": {
          /* ... */
        },
        "hamiltonian": {
          /* ... */
        },
        "timestep": 0.01,
        "steps": 10000,
        "population_control_interval" : 10,
        "measure_interval_multiplier": 1,
        "walker_ortho_interval": 10,
        "n_walkers_per_mpi_task": 10,
        "seed" : 42,
        "estimator": {
          /* ... */
        },
        "estimator": {
          /* ... */
        }
      }
    }
  }

.. note::

  Externally defined blocks can be reused in multiple execute blocks.
  The underlying objects have a persistent state between execute blocks.
  This allows a walker population to be reused across different simulation runs,
  picking up where the last projection left off.


Settings
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **timestep**
     - 
     - The Trotter step size, :math:`\tau`, in inverse energy units. The specific unit depends on what units are used to represent the Hamiltonian such that :math:`\tau \hat{H}` is dimensionless.
   * - **steps**
     - 
     - The total number of projection steps, :math:`N_{steps}` to perform. Together, the "steps" setting and the "timestep" setting determine the total projection time :math:`\beta = N_{steps} \tau`
   * - **population_control_interval**
     - 10
     - The number of projection steps between population control operations. Population control is relatively inexpensive, and it is typically okay to allow this interval to remain small.
   * - **measure_interval_multiplier**
     - 1
     - Used to determine the number of projection steps between measurements using the formula below. Measurement is the most expensive operation in AFQMC. A larger "measure_interval_multiplier" will reduce the CPU time necessary to perform AFQMC calculations.
   * - **walker_ortho_interval**
     - 10
     - The number of projection steps between application of the modified Gram-Schmidt (mGS) orthogonalization procedure. The mGS procedure is relatively inexpensive computationally and frequent orthogonalization is recommended.
   * - **n_walkers_per_mpi_task**
     - 10
     - The number of random walkers to use per MPI task. This value should be chosen such that the total population is reasonably large. The choice also depends on whether SAFIRE has been compiled for CPUs or GPUs. For GPUs, the goal is to saturate the device memory and could be on the order of 10000 for small systems. For CPUs and many MPI tasks, this value will typically be on the order of 10.
   * - **seed**
     - 
     - The seed for the random number generator. This value only needs to be set when strict reproducibility is necessary.

.. include:: ../_include/measure_interval_admonition.rst

.. _wavefunction_block:

Wavefunction block
------------------

The wavefunction block contains settings related to the trial wavefunction used in the AFQMC code.
**It is one of the only mandatory sub-blocks within the "execute" block.**
The wavefunction block must at least specify an HDF5 file containing the trial wavefunction.
Some settings are relevant only to specific classes of trial wavefunctions as indicated in the settings
listings below.

.. image:: /images/shared/02_input_file/input_file_03_wavefunction.png
   :width: 800px
   :alt: Wavefunction block structure diagram
   :align: center

*Wavefunction block structure overview*

Settings
~~~~~~~~

.. code-block::
  :caption: Sample execute.wavefunction input block with settings exposed.

  "wavefunction": {
    "filename" : "afqmc.h5",
    "ndets_to_read" : -1,
  }


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **filename** (**mandatory**) 
     - 
     - name of the HDF5 file containing the trial wavefunction.
   * - **ndets_to_read**
     - -1
     - The number of Slater determinants to read from the HDF5 file. If set to -1, all available determinants will be read.


.. only:: developer

  Advanced / developer settings
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Advanced settings for developers.

  .. code-block::
    :caption: Sample execute.wavefunction input block with advanced / developer settings exposed.
    :name: Listing 211

    "wavefunction": {
      "filename" : "afqmc.h5",
      "rediag" : false ,
      "ndets_to_read" : -1,
      "algorithm" : 1,
      "dense_trial" : true
    }

  .. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **restart_file**
     - 
     - **Currently disabled.** Name of restart file to generate.
   * - **rediag**
     - 
     - For PHMSD, rediagonalize the Hamiltonian in the basis of Slater determinants included in the trial wavefunction.
   * - **algorithm**
     - auto
     - Energy evaluation algorithm. Choices are 0 (slow) or 1. If no value is specified, "algorithm" is determined internally from Hamiltonian.
   * - **dense_trial**
     - auto
     - For non-orthogonal multi-Slater determinant trial wavefunctions, Boolean value to treat trial wavefunction as dense (true) or sparse (false). Determined internally based on the trial Hamiltonian type.

.. _walker_set_block:

Walker set block
----------------

The Walker set block contains settings for the Slater determinant random walkers in 
the AFQMC code.

.. image:: /images/shared/02_input_file/input_file_04_walker.png
   :width: 800px
   :alt: Walker set block structure diagram
   :align: center

*Walker set block structure overview*


See :ref:`Walker-classes` for more detail on the types of Slater determinant random walkers available.
Below is a sample "walker_set" block with all settings explicitly set.

.. code-block::json
  :caption: Sample execute.walker_set input block
  :name: Listing 202
  
  "walker_set": {
    "walker_type": "COLLINEAR",
    "load_balance_type" : "async",
    "pop_control_type" : "pair",
    "min_weight" : 0.05,
    "max_weight" : 4.0
  }


Settings
~~~~~~~~

These are the most common Walker set options that a typical user will interact with.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - **Parameter**
     - **Default**
     - **Description**
   * - **walker_type**
     - Collinear
     - The type of walker to use in AFQMC. Options are Closed, Collinear, Noncollinear. See :ref:`Walker-classes` for more detail.
   * - **name**
     - n/a
     - The name to assign to the current walker_set block. This allows it to be referenced by name in execute blocks. A name is generated internally if not set here.
   * - **load_balance_type**
     - async
     - Choose which load balancing algorithm to use. Choices are "async" for the asynchronous non-block swap load balancing algorithm and "simple" for a blocking (1-1) swap load balancing algorithm.
   * - **pop_control_type**
     - pair
     - Choose population control algorithm to use. Choices are "pair" and "serial_comb". The "pair" algorithm uses paired walker branching. The "serial_comb" algorithm uses the comb method from Booth, Gubernatis, PRE 2009.
   * - **min_weight**
     - 0.05
     - Minimum walker weight for population control
   * - **max_weight**
     - 4.0
     - Maximum walker weight for population control



.. _hamiltonian_block:

Hamiltonian block
-----------------

The Hamiltonian block contains settings related to the Hamiltonian used in the AFQMC code.
The main purpose of the block is to tell the AFQMC code which file to find the Hamiltonian in.

.. image:: /images/shared/02_input_file/input_file_05_hamiltonian.png
   :width: 800px
   :alt: Hamiltonian block structure diagram
   :align: center

*Hamiltonian block structure overview*

Settings
~~~~~~~~

These are the most common hamiltonian settings that a typical user will interact with.

.. code-block::json
  :caption: Sample execute.hamiltonian input block with settings exposed
  :name: Listing 221

  "hamiltonian": {
    "name" : "my_hamiltonian",
    "filename" : "afqmc.h5",
  }


.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - **Parameter**
     - **Default**
     - **Description**
   * - **name**
     - 
     - The name to assign to the current walker_set block. This allows it to be referenced by name in execute blocks. A name is generated internally if not set here.
   * - **filename** 
     - 
     - name of the HDF5 file containing the hamiltonian. If not specified, then the hamiltonian must exist within the same hdf5 file as the trial wavefun. See :ref:`wavefunction_block`. 


.. _estimator_block:

Estimator block
---------------

The purpose of the estimator block(s) is to choose and configure estimators to use during AFQMC calculations.
Specific observables to be measured are chosen within each estimator block.

.. image:: /images/shared/02_input_file/input_file_07_estimator.png
   :width: 800px
   :alt: Estimator block structure diagram
   :align: center

*Estimator block structure overview*


Multiple estimator blocks may be specified and configured separately.
Regardless of which estimator blocks are specified, a "basic" estimator block and an "energy"
estimator block will be generated.
Settings vary considerably based on the type of estimator used.
See the :ref:`Estimators reference <estimators>` for details on the various estimators available,
and the :ref:`Observables reference <observables>` for details on the various observables available.

.. _projector_block:

Projector block
---------------

.. caution::

  Changing projector settings is considered advanced. Be sure that you know what you are doing.


The projector block is used to set properties of the AFQMC projector.
A typical user will not need to interact with this input block

.. image:: /images/shared/02_input_file/input_file_06_propagator.png
   :width: 800px
   :alt: Projector block structure diagram
   :align: center

*Projector block structure overview*


Settings
~~~~~~~~

.. code-block:: json

  "projector": {
    "name": "my_projector",
    "nbatch": 0,
    "nbatch_qr": 0,
    "vbias_bound": 50.0,
    "upper_cutoff_scale": 10.0,
    "lower_cutoff_scale": 1.0,
    "apply_constrain": true,
    "importance_sampling": true,
    "substractMF": true,
    "hybrid": true,
    "printP1eigval": false,
    "denseP1": false,
  }

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **nbatch**
     - 0 for CPU builds or -1 for GPU builds 
     - Batch size for matrix operations
   * - **nbatch_qr**
     - 0 for CPU builds or -1 for GPU builds 
     - Batch size for QR decomposition operations
   * - **vbias_bound**
     - 50.0
     - Bound for the force bias.
   * - **upper_cutoff_scale**
     - 10.0
     - Upper cutoff scaling factor for pseudo local energy updates
   * - **lower_cutoff_scale**
     - 1.0
     - Lower cutoff scaling factor for pseudo local energy updates
   * - **apply_constrain**
     - true
     - Whether to apply constraints during propagation
   * - **importance_sampling**
     - true
     - Whether to use importance sampling
   * - **substractMF**
     - true
     - Whether to subtract mean field contribution
   * - **hybrid**
     - true
     - Whether to use hybrid propagation method
   * - **printP1eigval**
     - false
     - Whether to print P1 eigenvalues for debugging
   * - **denseP1**
     - false
     - Whether to use dense P1 matrix representation

.. only:: developer

  Advanced / developer settings
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  .. code-block:: json

    "projector": {
      "name": "my_projector",
      "nbatch": 0,
      "nbatch_qr": 0,
      "vbias_bound": 50.0,
      "external_field_scale": 1.0,
      "upper_cutoff_scale": 10.0,
      "lower_cutoff_scale": 1.0,
      "apply_constrain": true,
      "importance_sampling": true,
      "substractMF": true,
      "hybrid": true,
      "printP1eigval": false,
      "free_projection": false,
      "denseP1": false,
      "debug_verbosity": false
    }

  .. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - **Parameter**
      - **Default**
      - **Description**
    * - **nbatch**
      - 0 for CPU builds or -1 for GPU builds 
      - Batch size for matrix operations
    * - **nbatch_qr**
      - 0 for CPU builds or -1 for GPU builds 
      - Batch size for QR decomposition operations
    * - **i**
      - -1
      - excited orbital index for excited state calculation. Must correspond to a virtual / unoccupied orbital.
    * - **a**
      - -1
      - occupied orbital index for excited state calculation. Must correspond to a filled orbital.
    * - **vbias_bound**
      - 50.0
      - Bound for the force bias.
    * - **external_field_scale**
      - 1.0
      - Scaling factor for external field
    * - **upper_cutoff_scale**
      - 10.0
      - Upper cutoff scaling factor for pseudo local energy updates
    * - **lower_cutoff_scale**
      - 1.0
      - Lower cutoff scaling factor for pseudo local energy updates
    * - **apply_constrain**
      - true
      - Whether to apply constraints during propagation
    * - **importance_sampling**
      - true
      - Whether to use importance sampling
    * - **substractMF**
      - true
      - Whether to subtract mean field contribution
    * - **hybrid**
      - true
      - Whether to use hybrid propagation method
    * - **printP1eigval**
      - false
      - Whether to print P1 eigenvalues for debugging
    * - **free_projection**
      - false
      - Whether to use free projection or not.
    * - **denseP1**
      - false
      - Whether to use dense P1 matrix representation
    * - **debug_verbosity**
      - false
      - Whether to enable debug verbosity
    * - **external_field**
      - ""
      - External field specification (string)
    * - **excited_file**
      - ""
      - File containing excited state information


