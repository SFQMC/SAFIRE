.. _analysis_ex_1:

Analyzing Scalar Data
^^^^^^^^^^^^^^^^^^^^^
This example covers analyzing the scalar stochastic data output by SAFIRE.

This example assumes that you have already run an AFQMC calculation using
SAFIRE.
If not, see :ref:`run_afqmc_exs` for examples on running it.
SAFIRE saves all of its results in the ``[output_name].results.h5`` file, where ``output_name`` is set in the JSON job file.
If ``output_name`` is not set, it defaults to the name of the job file without the ``.json`` extension.

For example, using

.. code-block:: json

    "output_name": "qmc"

the data will be written to a file called ``qmc.results.h5``.
We assume this name for the remainder of the example.

The results file holds a series of bins for each observable, one bin per measurement.
``Results.observable_names()`` lists all of them.
A typical user will usually be most interested in the following scalar observables.

* ``Energy`` is the energy estimate.
* ``ProjectionTime`` gives the total projection time, in units of inverse energy, at which the bin was measured.
* ``TotalWeight`` gives the total walker weight. This can be useful for troubleshooting.

The ``safiretools`` Python package includes functions to assist in analyzing the results.
The following sample Python script computes the average AFQMC energy and its stochastic
uncertainty and plots the energy versus the projection time.

.. code-block:: python

    from safiretools import Results, vis

    results = Results("qmc.results.h5")
    print(results.observable_names())

    vis.plot_timeseries(results)
    E, dE = results.average("Energy")

Running the script above on the results of the example :ref:`run_afqmc_ex_1`, using inputs from setup example :ref:`setup_ex_1`, prints an output compatible with

.. code-block:: text

    Energy = -8.85051 ± 0.00043

The plot shows the energy bins together with their average and its error band.
It should be visually inspected to check that the AFQMC energy has equilibrated.
Bins from the ``equilibration_steps`` set in the job file are not written to the results file.
If the energy still drifts at the start of the series, more leading bins can be discarded from the average with ``skip``:

.. code-block:: python

    vis.plot_timeseries(results, skip=50)
    E, dE = results.average("Energy", skip=50)

The average accounts for the autocorrelation of successive bins by rebinning them.
A ``RebinningWarning`` is raised if the series is too short for its autocorrelation time or still drifts after ``skip``.

Any other scalar observable can be analyzed similarly by passing its name:

.. code-block:: python

    vis.plot_timeseries(results, obs="TotalWeight")
    weight, delta_weight = results.average("TotalWeight")
