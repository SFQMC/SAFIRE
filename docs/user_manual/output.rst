.. _output_afqmc:

Results File
============

A SAFIRE run writes every measurement it takes into a single HDF5 file at the path ``<output_name>.results.h5`` relative to where the executable was run. The ``output_name`` defaults to the name of the job file without extension, and can also be set explicitly in the :ref:`job file <input_afqmc>`.

.. seealso::

  The :class:`safiretools.Results` class provides tools to quickly extract measurements from the results file and perform statistical postprocessing on them.

.. _results_file_layout:

File Layout
-----------

.. rst-class:: filetree-caption

``<output_name>.results.h5``

.. rst-class:: filetree

* ``/``

  * ``🏷️ format_version`` 1
  * ``Measurements/``

    * ``Stage<s>/``

      * ``<observable>/``

        * ``bins`` :math:`(N_\mathrm{bins}, \dots)`

      * ``MixedEstimator/``

        * ``<observable>/``
        
          * ``bins`` :math:`(N_\mathrm{bins}, \dots)`

        * ``...``


      * ``BackPropEstimator/``

        * ``Steps=<m>/``

          * ``<observable>/``
          
            * ``bins`` :math:`(N_\mathrm{bins}, \dots)`
          
          * ``...``

.. list-table::
  :header-rows: 1
  :widths: 30 55

  * - **Entry**
    - **Description**
  * - ``format_version``
    - The version of the results file layout.
  * - ``Stage<s>``
    - The measurements of one execution stage, numbered :math:`s = 0, 1, \dots` in the order of the execute
      blocks. See :ref:`multi_stage_runs`.
  * - ``Stage<s>/<observable>``
    - An observable of the energy estimator. See :ref:`energy_estimator`.
  * - ``MixedEstimator/<observable>``
    - An observable of the mixed estimator. See :ref:`observables`.
  * - ``BackPropEstimator/Steps=<m>/<observable>``
    - An observable of the back-propagation estimator with a back-propagation length of :math:`m` steps,
      one group per entry of ``propagation_steps``. See :ref:`backprop_estimator`.
  * - ``bins``
    - The binned measurements of the observable. The leading axis is the bin, followed by the axes of a
      single measurement, indexed as in the definition of the observable. For complex observables, the array is stored in the `TRIQS/h5 <https://github.com/TRIQS/h5>`__ interleaved format with a trailing dimension of 2 and a ``__complex__`` attribute.

The group of an observable can itself be nested, as for the pair correlation functions
``PairCorr/<α>_<β>/bins``.

The results file of a previous run with the same ``output_name`` is deleted when the run starts. The
measurements of an execution stage are written once, when the stage has finished, so a run that is aborted
keeps only the stages it completed.

Each bin holds the average of ``binsize`` consecutive measurements, set in the :ref:`execute_block` block of the job file. A bin is
written out only once it is full, so the measurements of a trailing partial bin do not appear in the file.

Every measurement is stored as defined in :ref:`estimators`, so the weighted averages are already divided by
the total walker weight at the time of the measurement, and that denominator is not stored. Averaging an
observable over its bins is therefore an ordinary unweighted mean.
