.. _changelog:

Changelog
=========

1.0.0 - 2026-10-28
------------------

Added
~~~~~

- ``safiretools`` Python package, replacing ``afqmctools``
- Walker weight bound.
- Versioning of all HDF5 files written by SAFIRE and safiretools. Starting at format version 1.
- ``binsize`` input parameter.
- BCC Na momentum distribution example.

Changed
~~~~~~~

- The executable takes a single positional argument, the job file (`--filenames` is gone).
- Various changes to the job file format

  - ``timestep`` and ``steps`` are now mandatory.
  - ``n_walkers_per_mpi_task`` was replaced by ``num_walkers``, the total number of walkers.
  - ``walker_type`` is now optional and inferred from the wavefunction and Hamiltonian.
  - ``seed`` moved to the top level. The driver moved from a top-level key into a ``driver`` field, and ``project`` was renamed to ``output_name``.
  - Out-of-line ``ham``, ``wfn``, ``wset``, and ``prop`` blocks were renamed to their plurals. Input paths are resolved
    relative to the job file. A wavefunction-only file can fall back to the Hamiltonian filename.
  - New estimator parameter format. ``measurement_interval`` is decoupled from the population control interval.
  - Renamed ``use_real_vbias`` to ``project_force_bias``, ``substractMF`` to ``subtractMF``, and ``dshift`` to
    ``Eshift_relaxation_rate``. ``equilibration_time`` has a reasonable default.
- HDF5 formats: scalars are stored as attributes, Cholesky storage is unified, the sparse format is simplified, the
  dense ``hcore`` is rank 5, and the k-point factorized format resembles the dense one.
- Walkers are always initialized from a wavefunction. The PHMSD type is inferred from ``Psi_T``.
- All results of a run are written to a single output file in a new measurement format.
- Population control was simplified to the comb method with asynchronous exchange. ``serial_comb`` is the automatic
  choice.
- The :math:`\mathrm{Re}(v_\mathrm{MF}^2/2)` factori and :math:`E_0` are now always included in the hybrid energy so that it matches the local
  energy. ``Eshift`` is always initialized from the starting energy.
- The vbias bound no longer includes :math:`\sqrt{\Delta\tau}`.
- Lattice basis vectors are given in fractional coordinates. ``from_lattice`` exposes all terms, and ``spin_symm`` is
  inferred in ``HamiltonianBuilder.from_input``.
- PySCF SCF objects are accepted directly instead of checkpoint files.
- The Python sources moved from ``utils`` to ``python``.
- The default colormap for lattice visualizations is ``RdBu_r``.
- Functional tests use ``safiretools``, with a continuous crossover between machine and statistical tolerance.
- Documentation was rewritten for ``safiretools`` and the new input format. The HamiltonianBuilder tutorial was merged
  into the models tutorial on building and writing a Hamiltonian.

Removed
~~~~~~~

- ``afqmctools``.
- ``--filenames`` command line option.
- ``importance_sampling``, ``apply_constraint``, ``external_field``, excitations, and CSAFQMC leftovers.
- ``Psi0``, ``dims`` arrays, and other unnecessary HDF5 datasets.
- AutoHF, ``nelec`` on Hamiltonians, MPI in the periodic Cholesky factorization, the numba dependency, and
  ``force_hermitian`` from ``safiretools``.

Fixed
~~~~~

- ``G_MF`` with multiple determinants.
- Estimator GPU path.
- ``free_propagation``.
- ``Eshift`` initialization.
- Out-of-bounds PHMSD excitations.
- SAFIRE now aborts when the Hamiltonian has no interaction.
