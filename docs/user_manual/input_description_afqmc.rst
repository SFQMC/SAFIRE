.. _input_afqmc:

The Job File
============

The SAFIRE executable runs :ref:`afqmc` calculations based on the settings provided in a
JSON job file.

.. code-block::

  $ safire afqmc.json

This page serves as a reference for its format.

.. seealso::

   The :ref:`tutorials` and :ref:`user_examples` contain many examples of practical job files. Often you can use one of them as a starting point for your own calculation.
   ``safiretools`` provides a helper function ``write_jobfile()`` that simplifies writing job files from Python.

A Minimal Job File
------------------

Most parameters in the job file have sensible defaults so that for a standard calculation of the energy of a system, only a small subset of options has to be spelled out.

.. code-block:: json
  :caption: ``afqmc.json``: A minimal job file containing all the necessary information for a basic calculation.

  {
    "execute": {
      "wavefunction": {
        "filename": "afqmc.h5"
      },
      "hamiltonian": {
        "filename": "afqmc.h5"
      },
      "timestep": 0.01,
      "steps": 10000,
      "equilibration_steps": 1000,
      "measure_interval": 10,
      "population_control_interval": 10,
      "walker_ortho_interval": 10,
      "num_walkers": 200
    }
  }

At the start of a calculation, SAFIRE will fill in all the default values and print the full job file in the log output.

.. _input_file_structure:

The Full Job File Structure
---------------------------

The following tree shows every parameter of the job file. Below, we will list the meaning and default of every parameter.

.. rst-class:: filetree-caption

``afqmc.json``

.. rst-class:: filetree filetree-wide

* ``driver``
* ``output_name``
* ``seed``
* ``wavefunctions`` list

  * ``name``
  * ``filename`` required
  * ``ndets_to_read``

* ``hamiltonians`` list

  * ``name``
  * ``filename`` required

* ``walker_sets`` list

  * ``name``
  * ``walker_type``
  * ``from``

    * ``wavefunction`` required

* ``propagators`` list

  * ``name``
  * ``taylor_n``
  * ``vbias_bound``
  * ``upper_cutoff_scale``
  * ``lower_cutoff_scale``
  * ``weight_bound_floor``
  * ``weight_bound_fraction``
  * ``subtractMF``
  * ``hybrid``
  * ``free_projection``
  * ``natural_shift``
  * ``use_cp_constraint``
  * ``project_force_bias``

* ``execute`` required, a block or a list of blocks

  * ``walker_set``
  * ``wavefunction`` required
  * ``hamiltonian`` required
  * ``propagator``
  * ``timestep`` required
  * ``steps`` required
  * ``equilibration_steps``
  * ``population_control_interval``
  * ``measure_interval``
  * ``binsize``
  * ``walker_ortho_interval``
  * ``num_walkers`` required
  * ``initial_Eshift``
  * ``Eshift_relaxation_factor``
  * ``estimators``

    * ``energy``

      * ``wavefunction``
      * ``hamiltonian``
      * ``measure_interval``

    * ``mixed``

      * ``wavefunction``
      * ``hamiltonian``
      * ``measure_interval``
      * ``onerdm``

        * ``rotation``

          * ``filename`` required
          * ``group``

      * ``diag_twordm``
      * ``twordm``
      * ``paircorr``

        * ``pairs`` required

          * ``filename`` required
          * ``group``

      * ``spincorr``

    * ``backprop``

      * ``propagation_steps`` required
      * ``wavefunction``
      * ``hamiltonian``
      * ``walker_ortho_interval``
      * ``path_restoration``
      * ``extra_path_restoration``
      * ``onerdm`` as in ``mixed``
      * ``diag_twordm``
      * ``twordm``
      * ``paircorr`` as in ``mixed``
      * ``spincorr``

.. note::

  There may be some parameters that appear in the SAFIRE log but not here. These parameters are considered experimental and changing them is not supported.


In the following we will traverse the job file structure section by section and list all the keys with their defaults at each level.

At the top level, we have the following general parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **driver**
     - ``"afqmc"``
     - The driver that runs the execute blocks: ``"afqmc"`` for ground state AFQMC.
   * - **output_name**
     - name of the job file without its extension
     - The results are written to ``<output_name>.results.h5``. Unlike every other path in the job file, it is relative to the directory SAFIRE is launched from. See :ref:`output_afqmc`.
   * - **seed**
     - random
     - The seed for the random number generator. Can be used to reproduce simulations with the same random number stream. To reproduce an existing calculation, you can reuse the seed in its log and run it with the same number of MPI tasks.

The remaining top-level entries are the subblocks ``execute``, ``wavefunctions``, ``hamiltonians``, ``walker_sets``, and ``propagators``.

.. admonition:: Inline vs Out-of-Line Definitions
  :class: note

  Hamiltonians, wavefunctions, walker sets, and propagators can either be defined directly inline where they are needed, deep in the job file, or declared out-of-line with a name at the top level, in the corresponding subblocks. This is useful when one input component is shared in multiple parts of the calculation. For this overview, we will assume all components are declared out-of-line.

.. _execute_block:

``execute``
~~~~~~~~~~~

An execute block represents one execution stage of an AFQMC calculation. It contains all essential parameters describing a simulation.


.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **hamiltonian**
     - *required*
     - Hamiltonian to use for the AFQMC propagation. Either the name or an inline block. See :ref:`hamiltonian_block`.
   * - **wavefunction**
     - *required*
     - The AFQMC trial wavefunction. Either the name or an inline block. See :ref:`wavefunction_block`.
   * - **walker_set**
     - a walker set initialized from **wavefunction**, or the walker set of the previous execute block
     - The walker set to propagate. Either the name or an inline block. See :ref:`walker_set_block`.
   * - **propagator**
     - a propagator with default settings
     - The propagator settings. Either the name or an inline block. See :ref:`propagator_block`.
   * - **estimators**
     - energy estimator enabled
     - Specifies which estimators will measure what observables. See :ref:`estimator_block`.
   * - **timestep**
     - *required*
     - The Trotter step size, :math:`\Delta\tau`, in inverse energy units. The specific unit depends on what units are used to represent the Hamiltonian such that :math:`\Delta\tau \hat{H}` is dimensionless.
   * - **steps**
     - *required*
     - The total number of propagation steps, :math:`N_\mathrm{steps}`, to perform. Together, ``steps`` and ``timestep`` determine the total projection time :math:`\beta = N_\mathrm{steps} \Delta\tau`.
   * - **equilibration_steps**
     - 100
     - The number of propagation steps at the start of the execute block during which nothing is measured.
   * - **population_control_interval**
     - 10
     - The number of propagation steps between population control operations. Population control is relatively inexpensive, and it is typically okay to allow this interval to remain small.
   * - **measure_interval**
     - 10
     - The number of propagation steps between measurements, for every estimator that does not set one of its own. Measurement is the most expensive operation in AFQMC, so a larger ``measure_interval`` will speed up an AFQMC calculation, but increasing it above the Monte Carlo autocorrelation time brings no benefit.
   * - **binsize**
     - 1
     - The number of consecutive measurements averaged into one bin of the results file. Binning reduces the size of the results file and the autocorrelation between neighbouring bins. A trailing partial bin is not written out.
   * - **walker_ortho_interval**
     - 10
     - The number of propagation steps between orthogonalizing the walker Slater determinants. Orthogonalization is relatively inexpensive so performing it frequently is recommended.
   * - **num_walkers**
     - *required*
     - The total number of random walkers. The walkers are split evenly over the tasks, and there have to be at least as many walkers as tasks. This value should be chosen such that the total population is reasonably large. The choice also depends on whether SAFIRE has been compiled for CPUs or GPUs. For GPUs, the goal is to saturate the device bandwidth, which could take on the order of 10000 walkers per GPU for small systems. For CPUs and many MPI tasks, a share on the order of 10 walkers per task is typical.
   * - **initial_Eshift**
     - initial local energy of the walkers
     - The initial value of the energy shift.
   * - **Eshift_relaxation_factor**
     - set so that :math:`E_\mathrm{shift}` relaxes on a scale of **equilibration_steps**/10 steps
     - The fraction :math:`d` of the gap to the average energy that the energy shift closes once per step of the equilibration phase under the update :math:`E_\mathrm{shift} \mathrel{+}= d (E - E_\mathrm{shift})`.

.. _estimator_block:

``estimators``
^^^^^^^^^^^^^^

The estimator block configures which measurements are performed during the AFQMC calculation.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **energy**
     - ``{}``
     - The energy estimator. It is enabled by default, ``"energy": null`` disables it.
   * - **mixed**
     - disabled
     - The mixed estimator.
   * - **backprop**
     - disabled
     - The back-propagated estimator.

See :ref:`estimators` for the theory behind each estimator.

``energy``
""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **wavefunction**
     - the execute block's **wavefunction**
     - Name of the trial wavefunction used to evaluate the energy.
   * - **hamiltonian**
     - the execute block's **hamiltonian**
     - Name of the Hamiltonian whose energy is measured.
   * - **measure_interval**
     - the execute block's **measure_interval**
     - The number of propagation steps between measurements.

``mixed``
"""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **wavefunction**
     - the execute block's **wavefunction**
     - Name of the trial wavefunction of the mixed estimate.
   * - **hamiltonian**
     - the execute block's **hamiltonian**
     - Name of the Hamiltonian the estimator uses.
   * - **measure_interval**
     - the execute block's **measure_interval**
     - The number of propagation steps between measurements.
   * - **onerdm**, **diag_twordm**, **twordm**, **paircorr**, **spincorr**
     - disabled
     - The observables to measure. See :ref:`observables_block`.

.. _backprop_block:

``backprop``
""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **propagation_steps**
     - *required*
     - List of back-propagation lengths in propagation steps. Every length is measured separately and stored in its own ``BackPropEstimator/Steps=<m>`` group of the results file. The longest one is also the measurement interval: a new back-propagation window starts once it has been covered.
   * - **wavefunction**
     - the execute block's **wavefunction**
     - Name of the trial wavefunction that is back-propagated.
   * - **hamiltonian**
     - the execute block's **hamiltonian**
     - Name of the Hamiltonian the estimator uses.
   * - **walker_ortho_interval**
     - the execute block's **walker_ortho_interval**
     - The number of back-propagation steps between orthogonalizations.
   * - **path_restoration**
     - true
     - Restore the walker paths during back propagation. Advanced option, keep the default.
   * - **extra_path_restoration**
     - false
     - Apply an additional path restoration. Advanced option, keep the default.
   * - **onerdm**, **diag_twordm**, **twordm**, **paircorr**, **spincorr**
     - disabled
     - The observables to measure. See :ref:`observables_block`.

.. _observables_block:

Observables
"""""""""""

The **mixed** and **backprop** estimators measure an observable if and only if its block is present. Observables without parameters are enabled with an empty block, e.g. ``"twordm": {}``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **onerdm**
     - disabled
     - The one-body reduced density matrix. Takes an optional **rotation** block with **filename** and **group** (default ``"/"``) of an HDF5 group containing a ``RotationMatrix`` dataset of shape :math:`(n_\mathrm{spin}, n, n_\mathrm{pol} M)` that maps the :math:`M` orbitals of the Hamiltonian to :math:`n` rotated orbitals. If given, the density matrix is measured in the rotated basis.
   * - **diag_twordm**
     - disabled
     - The diagonal of the two-body reduced density matrix.
   * - **twordm**
     - disabled
     - The full two-body reduced density matrix.
   * - **paircorr**
     - disabled
     - Pair correlation functions. Requires a **pairs** block with **filename** and **group** of an HDF5 group in which every dataset defines one orbital pair map. Each dataset becomes a correlator named after it.
   * - **spincorr**
     - disabled
     - Spin-spin correlation functions.

See :ref:`observables` for details on each observable.

.. _wavefunction_block:

``wavefunctions``
~~~~~~~~~~~~~~~~~

A wavefunction block describes a wavefunction used in SAFIRE. It can either be defined inline where it is needed or out-of-line in the top-level ``wavefunctions`` list.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **name**
     - ``wavefunction_*``
     - The name of the wavefunction, required to refer to it elsewhere.
   * - **filename**
     - *required*
     - Path of the HDF5 file containing the wavefunction.
   * - **ndets_to_read**
     - -1
     - The number of Slater determinants to read from the HDF5 file. If set to -1, all available determinants will be read.


.. _hamiltonian_block:

``hamiltonians``
~~~~~~~~~~~~~~~~

A Hamiltonian block describes a Hamiltonian used in SAFIRE. It can either be defined inline where it is needed or out-of-line in the top-level ``hamiltonians`` list.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - **Parameter**
     - **Default**
     - **Description**
   * - **name**
     - ``hamiltonian_*``
     - The name of the Hamiltonian, required to refer to it elsewhere.
   * - **filename**
     - *required*
     - Path of the HDF5 file containing the Hamiltonian.
    
.. _walker_set_block:

``walker_sets``
~~~~~~~~~~~~~~~

A walker set block describes an ensemble of AFQMC walkers that can be propagated. It can either be defined inline where it is needed or out-of-line in the top-level ``walker_sets`` list. Walker sets are mutable. If they are shared between different execution stages, later stages will carry on the state from previous stages.

For many basic calculations, this block can be omitted.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - **Parameter**
     - **Default**
     - **Description**
   * - **name**
     - ``walker_set_*``
     - The name of the walker set, required to refer to it elsewhere.
   * - **from**
     - the execute block's **wavefunction**
     - What the walkers are initialized from, as an object with a single key. ``{"wavefunction": ...}`` starts every walker from the determinant of the wavefunction with the largest coefficient magnitude; the wavefunction is given by name or as an inline block. Initializing the walkers from a different wavefunction can be very beneficial for performance. See :ref:`initial_walkers`.
   * - **walker_type**
     - the highest spin symmetry viable for simulating all wavefunctions and Hamiltonians in the job file
     - Spin symmetry of the walkers to use in AFQMC. Options are ``"closed"``, ``"collinear"``, or ``"noncollinear"``. See :ref:`Walker-classes`.


Only the first execute block falls back to a default walker set. A later execute block that
names no walker set continues with the one of the execute block before it, so consecutive
stages share their walkers unless a stage asks for a walker set of its own.

.. _initial_walkers:

Initial Walkers
^^^^^^^^^^^^^^^

Initializing walkers from a different wavefunction than the trial wavefunction using the ``from`` key can be crucial for performance for symmetry-broken trials.
For example, while an unrestricted Hartree-Fock trial wavefunction may give a better description of the ground state than a restricted one, it may be contaminated by soft Goldstone modes that are not excited in the ground state but take a long time to project out in the AFQMC propagation. In such cases, initializing the walkers from a restricted Hartree-Fock calculation can greatly reduce the equilibration time.

.. _propagator_block:

``propagators``
~~~~~~~~~~~~~~~

.. caution::

  Changing propagator settings is considered advanced. Be sure that you know what you are doing.

A propagator block configures the detailed behavior of the AFQMC propagator. It can either be defined inline where it is needed or out-of-line in the top-level ``propagators`` list.


.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **name**
     - ``propagator_*``
     - The name of the propagator, required to refer to it elsewhere.
   * - **taylor_n**
     - 6
     - Order of the Taylor expansion used to apply the exponential of the propagator to the walkers.
   * - **vbias_bound**
     - 50.0, or 100.0 for model Hamiltonians
     - Bound for the force bias.
   * - **upper_cutoff_scale**
     - 10.0, or 50.0 for model Hamiltonians
     - Upper cutoff scaling factor for pseudo local energy updates.
   * - **lower_cutoff_scale**
     - 1.0, or 50.0 for model Hamiltonians
     - Lower cutoff scaling factor for pseudo local energy updates.
   * - **weight_bound_floor**
     - 100.0
     - Lower limit of the walker weight bound. After every propagation step, walker
       weights are capped at ``max(weight_bound_floor, weight_bound_fraction*N)``, where
       ``N`` is the global target population, which is the total weight the population is
       rescaled to at every population control interval. A walker over the bound is
       rescaled onto it and keeps its phase.
   * - **weight_bound_fraction**
     - 0.1
     - Largest fraction of a normalized population's total weight that a single walker is
       allowed to carry. See **weight_bound_floor**.
   * - **subtractMF**
     - true
     - Whether to subtract the mean-field contribution.
   * - **hybrid**
     - true
     - Whether to use hybrid propagation method.
   * - **free_projection**
     - false
     - Whether to use free projection or not. Requires **hybrid**. Disables the force bias,
       the phaseless constraint and the local-energy bound, and leaves the phase of each
       step in the walker weight, which therefore turns complex.
   * - **natural_shift**
     - true
     - For the discrete Hubbard-Stratonovich fields of model Hamiltonians, shift by the natural
       mean field (a density of 1 for the charge and 0 for the spin decomposition) instead of the
       mean field of the trial wavefunction.
   * - **use_cp_constraint**
     - false
     - Use the constrained-path approximation: a walker whose overlap phase changes by
       :math:`|\Delta\theta| \geq \pi/2` in one step is given zero weight, instead of scaling the
       weight by :math:`\max(0, \cos\Delta\theta)` as in the phaseless approximation. Implies
       **project_force_bias**.
   * - **project_force_bias**
     - false
     - Keep only the real part of the force bias for continuous charge decompositions, and only
       the imaginary part for continuous spin decompositions.

.. _multi_stage_runs:

Multiple Execution Stages
-------------------------

The ``execute`` block can be a list of multiple entries in the format above. These will then define multiple execution stages
that are run sequentially. Stages which share the same ``walker_set`` will continue propagating it so that this feature can be used
to define propagation protocols for quicker equilibration or sweeping different Hamiltonian parameters.

.. code-block:: json
  :caption: Equilibrating with a larger time step before measuring with a smaller one.

  {
    "wavefunctions": [
      {
        "name": "trial",
        "filename": "afqmc.h5"
      }
    ],
    "hamiltonians": [
      {
        "name": "hamiltonian",
        "filename": "afqmc.h5"
      }
    ],
    "execute": [
      {
        "wavefunction": "trial",
        "hamiltonian": "hamiltonian",
        "timestep": 0.02,
        "steps": 500,
        "equilibration_steps": 500,
        "num_walkers": 200
      },
      {
        "wavefunction": "trial",
        "hamiltonian": "hamiltonian",
        "timestep": 0.005,
        "steps": 10000,
        "equilibration_steps": 100,
        "num_walkers": 200
      }
    ]
  }

The first stage only equilibrates, since all of its steps are equilibration steps. The second stage continues with its walkers and
equilibrates for a short time to adapt to the smaller time step before it starts measuring.

Measurement results are recorded and saved to the results file separately per stage. See :ref:`output_afqmc`.
