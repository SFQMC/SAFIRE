3d TM Oxide Diatomics (AFQMC Benchmark)
=======================================

This example reproduces AFQMC benchmark calculations for seven 3d transition-metal oxide diatomic molecules:
VO, TiO, CrO, MnO, FeO, CuO, and ScO. The workflow generates inputs with PySCF + AutoHF (for CASSCF trials),
runs SAFIRE, and summarizes the resulting energies.

Prerequisites
-------------

- SAFIRE built and available (e.g., ``build/CPU/bin/safire`` or a GPU build).
- The utilities package from this repo installed so ``stats`` and ``tutorial_utils`` are importable:

  .. code-block:: bash

	  # From the repository root
	  python -m venv .venv && source .venv/bin/activate
	  pip install -e utils

- PySCF for integrals and CASSCF (optional extra ``PYSCF`` via utils):

  .. code-block:: bash

	  pip install 'pyscf>=2.4.0'

- AutoHF for the HF reference used by CASSCF (optional extra ``AUTOHF`` via utils):

  .. code-block:: bash

	  # Install AutoHF and its JAX dependencies
	  pip install 'jax' 'jaxlib' 'optax' 'jaxopt'
	  # If you prefer installing AutoHF from source, install the project
	  # and ensure it is on your Python path (e.g., editable install)
	  pip install -e utils/AutoHF

- afqmctools (provided in this repo via ``utils``): installed with the step above.

Notes
-----

- If using GPUs, install the CUDA-compatible ``jaxlib`` wheel per JAX documentation for your CUDA/cuDNN version.
- MPI runtime (e.g., OpenMPI or MPICH) must be available for multi-rank runs.
- The example scripts write and read from a scratch directory under ``~/.scratch``. This directory will be created if it does not exist.

Step 1 — Generate AFQMC inputs
------------------------------

Run the setup script to generate Hamiltonians, CAS trial wavefunctions, and input JSON files for the 7 molecules.
This creates one subfolder per molecule in a scratch directory and writes ``afqmc.h5`` and ``afqmc.json``.

.. code-block:: bash

	# From this example directory
	cd examples/molecule/05_3d_TM_diatomics
	python 20_TMO_benchmark.py

Outputs are written to:

- ``~/.scratch/20_3d_tm_oxides/VO``
- ``~/.scratch/20_3d_tm_oxides/TiO``
- ``~/.scratch/20_3d_tm_oxides/CrO``
- ``~/.scratch/20_3d_tm_oxides/MnO``
- ``~/.scratch/20_3d_tm_oxides/FeO``
- ``~/.scratch/20_3d_tm_oxides/CuO``
- ``~/.scratch/20_3d_tm_oxides/ScO``

Each contains at minimum:

- ``afqmc.h5``: CAS trial wavefunction and molecular Hamiltonian.
- ``afqmc.json``: SAFIRE input file with execution options.

Step 2 — Run SAFIRE (preferably in parallel on a cluster)
--------------------------------------------------------

For each molecule subfolder, run SAFIRE against the generated input. Example single-node run:

.. code-block:: bash

	# Adjust np and path to the SAFIRE binary as needed
	mpirun -np 64 /path/to/SAFIRE/build/CPU/bin/safire ~/.scratch/20_3d_tm_oxides/VO/afqmc.json

Repeat for all seven subfolders. To run them all in parallel, submit a Slurm job array (example):

.. code-block:: bash

	#!/bin/bash
	#SBATCH --job-name=tmo_afqmc
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=64
	#SBATCH --time=02:00:00
	#SBATCH --partition=<your_partition>
	#SBATCH --array=0-6

	set -euo pipefail

	MOLS=(VO TiO CrO MnO FeO CuO ScO)
	MOL=${MOLS[$SLURM_ARRAY_TASK_ID]}
	SCRATCH="$HOME/.scratch/20_3d_tm_oxides/$MOL"

	cd "$SCRATCH"
	mpirun -np "$SLURM_NTASKS" /path/to/SAFIRE/build/CPU/bin/safire afqmc.json

Alternatively, submit a separate job per molecule without an array:

.. code-block:: bash

	for MOL in VO TiO CrO MnO FeO CuO ScO; do
	  sbatch -J "tmo_${MOL}" <<'EOF'
	  #!/bin/bash
	  #SBATCH --nodes=1
	  #SBATCH --ntasks-per-node=64
	  #SBATCH --time=02:00:00
	  #SBATCH --partition=<your_partition>
	  set -euo pipefail
	  SCRATCH="$HOME/.scratch/20_3d_tm_oxides/${MOL}"
	  cd "$SCRATCH"
	  mpirun -np "$SLURM_NTASKS" /path/to/SAFIRE/build/CPU/bin/safire afqmc.json
	  EOF
	done

On completion, each directory should contain a file like ``qmc.s000.scalar.dat`` plus other run artifacts.

Step 3 — Summarize results
--------------------------

Run the summary script to analyze the scalar output files and print a comparison table to reference AFQMC values.

.. code-block:: bash

	# From this example directory
	cd examples/molecule/05_3d_TM_diatomics
	python get_results.py

This scans each molecule folder under the configured scratch directory, parses ``qmc.s000.scalar.dat``,
and prints energies with stochastic uncertainties alongside published references.

Scratch Directory Tag Consistency
---------------------------------

The setup and summary scripts use a tag to name the scratch directory. Ensure they match before running.

- In ``20_TMO_benchmark.py`` the tag is ``"20_3d_tm_oxides"``.
- In ``get_results.py`` the default tag is ``"3d_tm_oxides"``.

You can either change the summary script to match the setup tag (recommended):

.. code-block:: python

	# get_results.py (near the top)
	home = Path.home()
	scratch_dir = get_scratch_dir("20_3d_tm_oxides", home / ".scratch")

Or, change the setup tag to match the summary script. The key is that both point to the same base directory
under ``~/.scratch`` so the results can be found.

Troubleshooting
---------------

- If AFQMC results are incorrect, check that the HF converged in AutoHF and that CASSCF completed without errors. Rerun individual cases in step 1 as needed.
- Missing ``qmc.s000.scalar.dat``: The SAFIRE run likely did not finish or wrote to a different folder.
  Verify you ran from the molecule’s scratch subfolder and used the correct ``afqmc.json``.
- Import errors for ``stats`` or ``tutorial_utils``: Ensure ``pip install -e utils`` was done in the active environment.
- PySCF or AutoHF issues: Confirm versions and JAX installation; GPU JAX requires the correct CUDA/cuDNN wheel.
