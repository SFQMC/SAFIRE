.. _output_afqmc:

Output File
===========

A SAFIRE run writes every measurement it takes into a single HDF5 file.
This page describes how that file is named and how it is organized.

.. _results_file_name:

File Name
---------

The results file is named after the ``id`` parameter of the :ref:`project block <project_block>`

.. code-block:: text

  [id].results.h5

so the default ``id`` of ``qmc`` produces ``qmc.results.h5``.
The ``series`` parameter is *not* part of the file name; it selects the index of the first
stage group inside the file, as described in :ref:`results_file_layout`.

There is one results file per run, not one per ``execute`` block.
A results file left over from a previous run is deleted before the first ``execute`` block
writes to it, so running twice in the same directory with the same ``id`` discards the
results of the first run.
Give each run its own ``id`` or its own directory to keep both.

The file is written once at the end of each ``execute`` block, by the root MPI rank only.

.. _results_file_layout:

File Layout
-----------

Every measurement lives below the ``Measurements`` group, in a subgroup per execution
stage.
One ``execute`` block is one stage, numbered starting from the ``series`` parameter of the
:ref:`project block <project_block>`, so an ordinary single-``execute`` input produces only
``Stage0``.

Below the stage group, each estimator writes under a prefix of its own, and each observable
it measures gets a group holding a single ``bins`` dataset.
An input requesting a mixed estimator and a back-propagation estimator with two
back-propagation lengths gives a file like

.. code-block:: text

  Measurements/
    Stage0/
      Energy/bins
      OnebodyEnergy/bins
      ExchangeEnergy/bins
      CoulombEnergy/bins
      Overlap/bins
      MixedEstimator/
        OneRDM/bins
        DiagonalTwoRDM/bins
      BackPropEstimator/
        Steps=20/
          OneRDM/bins
        Steps=40/
          OneRDM/bins

Estimator Prefixes
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Estimator**
     - **Prefix below** ``Stage<N>``
   * - ``energy``
     - none, its datasets sit directly in the stage group
   * - ``mixed``
     - ``MixedEstimator``
   * - ``backprop``
     - ``BackPropEstimator/Steps=<m>``
   * - ``time_evolved_bp``
     - ``TimeEvolvedBP/Steps=<m>``

For the two back-propagation estimators, ``<m>`` is one entry of ``propagation_steps``, so
there is one ``Steps=<m>`` group per back-propagation length requested, and ``<m>`` is that
length in projection steps; see :ref:`estimators`.

Observable Datasets
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - **Requested by**
     - **Group**
     - **Shape of one bin**
   * - the ``energy`` estimator
     - ``Energy``, ``OnebodyEnergy``, ``ExchangeEnergy``, ``CoulombEnergy``, ``Overlap``
     - scalar. ``OnebodyEnergy`` includes any constant energy contribution, and ``Energy``
       is the sum of the three components
   * - ``onerdm``
     - ``OneRDM``
     - :math:`(n_\mathrm{spin}, n_\mathrm{pol} N_\mathrm{MO}, n_\mathrm{pol} N_\mathrm{MO})`,
       or :math:`(n_\mathrm{spin}, n_\mathrm{out}, n_\mathrm{out})` if a ``rotation`` was given
   * - ``twordm``
     - ``TwoRDM``
     - :math:`(3, N_\mathrm{MO}, N_\mathrm{MO}, N_\mathrm{MO}, N_\mathrm{MO})`, indexed
       ``[spin block][i][k][j][l]`` with the spin blocks ordered
       :math:`(\alpha\alpha\alpha\alpha)`, :math:`(\alpha\alpha\beta\beta)`,
       :math:`(\beta\beta\beta\beta)`
   * - ``diag_twordm``
     - ``DiagonalTwoRDM``
     - a flat array of length :math:`N_\mathrm{MO}(2 N_\mathrm{MO} - 1)`, minus
       :math:`N_\mathrm{MO}(N_\mathrm{MO}-1)/2` for closed walkers. See
       :ref:`diag_twordm_obs`
   * - ``spincorr``
     - ``SpinCorr``
     - :math:`(2, N_\mathrm{MO}(N_\mathrm{MO}+1)/2)`. Row 0 is the XY channel and row 1 the
       Z channel; the second index runs over the upper triangle including the diagonal,
       row by row
   * - ``paircorr``
     - ``PairCorr/<a>_<b>``
     - :math:`(N_\mathrm{MO}, N_\mathrm{MO})`, one group per ordered pair of orbital pair
       maps :math:`a`, :math:`b`

The ``bins`` Dataset
~~~~~~~~~~~~~~~~~~~~

Every observable is stored as one growable dataset called ``bins``.
Its leading axis is the bin index, and the remaining axes are the shape of a single
measurement as listed above.
Each bin holds the average of ``binsize`` consecutive measurements, which is one
measurement by default. A bin is written out only once it is full, so the measurements of
a trailing partial bin do not appear in the file.

All measurements are complex, and are stored with the usual trailing axis of size two plus
a ``__complex__`` attribute, so ``h5py`` and ``nda`` read them back as complex arrays.
A scalar observable therefore has an on-disk shape of ``(nbins, 2)``.

Each bin has already been divided by the summed walker weight at the time it was measured,
and that denominator is not stored.
Averaging an observable over its bins is therefore an ordinary unweighted mean.

Nothing is measured during the equilibration phase set by ``equilibration_steps``, so every
bin in the file is already equilibrated.

Analyzing Results
-----------------

TODO(postprocessing): document the analysis interface for ``results.h5`` once it is settled.
