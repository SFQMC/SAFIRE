.. _analysis_ex_2:

Analyzing One-Body Reduced Density Matrix Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers analyzing the one-rdm stochastic data output by SAFIRE
when back-propagation or mixed estimators are used.

This example assumes that you have already run an AFQMC calculation using
using SAFIRE.
If not, see :ref:`run_afqmc_ex_2` for an example on running AFQMC with back-propagation.
While the AFQMC executable runs, it prints scalar data samples to a text-based file.
The name of the file depends on the parameters set in the ``project`` block of
the AFQMC input file and follows the pattern, ``[id].s[###].scalar.dat`` where
``[id]`` is replaced by the ``id`` string and ``[###]`` is replaced by the value of
``series`` but at a fixed width of 3.
Similarly, if back-propagation is used, the one-rdm data is written to a file called,
``[id].s[###].stat.h5``.

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
