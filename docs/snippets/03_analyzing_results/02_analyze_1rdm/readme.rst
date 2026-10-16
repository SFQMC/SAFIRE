.. _analysis_ex_2:

Analyzing One-Body Reduced Density Matrix Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers analyzing the one-rdm stochastic data output by SAFIRE
when back-propagation or mixed estimators are used.

This example assumes that you have already run an AFQMC calculation using
using SAFIRE.
If not, see :ref:`run_afqmc_ex_2` for an example on running AFQMC with back-propagation.
SAFIRE saves all of its results in the ``[output_name].results.h5`` file, where ``output_name`` is set in the
JSON job file and defaults to the name of the job file without the ``.json`` extension.

The one-rdm data can be analyzed via the ``afqmctools`` Python package as 
shown below.

.. code-block:: python

    from afqmctools.analysis.rdm import average_afqmc_rdm

    rho_avg, delta_rho = average_afqmc_rdm(
        rdm_file="qmc.s000.stat.h5"
    )

    # Note: the AFQMC input file may already include
    #         an equilibration time for back-propagation.
    #         If not, or if you want to increase the size 
    #         of the equilibration phase ...

    # ... you can specify the Number of equilibration blocks
    rho_avg, delta_rho = average_afqmc_rdm(
        rdm_file="qmc.s000.stat.h5",
        Neq=5
    )

    # ... OR you can specify the equilibration time
    rho_avg, delta_rho = average_afqmc_rdm(
        rdm_file="qmc.s000.stat.h5",
        Teq=10.0
    )

    # ... if both are given, Neq is used.
    rho_avg, delta_rho = average_afqmc_rdm(
        rdm_file="qmc.s000.stat.h5",
        Neq=5,
        Teq=10.0
    )
