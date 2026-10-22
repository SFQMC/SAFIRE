---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: ipython3
  name: python3
---

# Hello SAFIRE

<!--
This is the main version of the Hello SAFIRE tutorial. The solids and lattice-model versions
include the regions between the "BEGIN hello-NAME" and "END hello-NAME" markers below,
so keep those regions free of system-specific content and code cells. Code cells and
system-specific text go between the regions, and every version repeats them.
-->

<!-- BEGIN hello-intro -->

<b>Goal:</b>
At the end of this tutorial, you will know how to run an AFQMC calculation using SAFIRE and extract the AFQMC energy from the output
given all necessary inputs.

## What You Will Learn

1. what are the inputs and outputs of SAFIRE.
2. how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s)
3. how to extract the AFQMC energy from the outputs of SAFIRE using safiretools

+++

## Introduction
The purpose of this tutorial is to run a first AFQMC simulation without going into low-level details.

In order to run a simulation, SAFIRE needs at least three inputs: A Hamiltonian, a trial wavefunction, and a job file defining all run parameters. The output of a simulation is a single HDF5 file containing the Monte Carlo samples of observables such as the energy.

We will learn more about the details of all of these files in future tutorials. For this tutorial, pregenerated input files are provided together with the `*.ipynb` file.

+++

### Hamiltonian File: `hamiltonian.h5`

This HDF5 file contains the second-quantized Hamiltonian $\hat{H}$.

<!-- END hello-intro -->

For this tutorial, we have supplied an HDF5 file containing the Born-Oppenheimer
Hamiltonian for the STO-3G basis hydrogen dimer at a bondlength of $\delta_\mathrm{H-H} = 1.4$ $a_0$
in `hamiltonian.h5`.

![](files/HyrdogenMolecule_v2.png)

<!-- BEGIN hello-wavefunction -->

For technical details on the Hamiltonian HDF5 format, see {doc}`/user_manual`.

### Wavefunction File: `wavefunction.h5`

This HDF5 file contains the trial wavefunction, $| \Psi_\mathrm{T} \rangle$.

<!-- END hello-wavefunction -->

For this tutorial, we have supplied an HDF5 file containing the RHF
ground state as the trial wavefunction.

$$
| \Psi_\mathrm{T} \rangle = | \Phi_\mathrm{RHF} \rangle
$$

<!-- BEGIN hello-jobfile-and-running -->

For technical details on the wavefunction HDF5 format, see {doc}`/user_manual`.

### Job File: `afqmc.json`

The JSON job file sets AFQMC run parameters and references the locations of the Hamiltonian and trial wavefunction files. It is organized into a hierarchy of input blocks which each control specific details of the calculation.

**For now, we focus on the `wavefunction` and `hamiltonian` blocks.**

The `wavefunction` input block is used to tell SAFIRE where to find the HDF5 file containing the trial wavefunction.
Similarly, the `hamiltonian` input block is used to tell SAFIRE where to find the HDF5 file containing the Hamiltonian.

The sample JSON job file that we have provided is reproduced here.

```{literalinclude} files/afqmc.json
:language: json
```

Notice that the `wavefunction` and `hamiltonian` input blocks point to the HDF5 files where we have saved the trial
wavefunction and second-quantized Hamiltonian, respectively.

+++

## Running SAFIRE

Now that we have learned what the inputs of SAFIRE are,
we are ready to learn how to run the SAFIRE executable given the inputs.

Because the path to the SAFIRE executable depends on your installation, in the tutorials, we will assume that you have defined the environment variable
`$AFQMC_EXEC` that contains `/path/to/SAFIRE/build/bin/safire`. So the following should print the path to SAFIRE and then display the help.

<div class="alert alert-block alert-info">
<b>Note:</b>

In a Jupyter notebook, the `!` symbol will run commands in
a shell instead of in Python.

</div>

<!-- END hello-jobfile-and-running -->

```{code-cell} ipython3
!echo Path to SAFIRE: $AFQMC_EXEC
!$AFQMC_EXEC --help
```

<!-- BEGIN hello-running -->

### Running SAFIRE in Serial

SAFIRE takes the JSON job file as its only positional argument,

```bash
    $ $AFQMC_EXEC afqmc.json
```

When you run the above, an AFQMC calculation will be performed based on the contents of `afqmc.json`,
using the Hamiltonian and trial wavefunction specified there.

### Running SAFIRE in Parallel

SAFIRE can be run with MPI in the usual ways.
For example, using mpirun, we could run SAFIRE with 64 MPI tasks as

```bash
    $ mpirun -n 64 $AFQMC_EXEC afqmc.json
```

It is important to note that `num_walkers` in the job file is the total number of walkers
over all MPI tasks.
SAFIRE splits them evenly over the tasks, so that the walker counts of two tasks differ by at
most one.
So, running the sample job file above with 64 MPI tasks gives each task 3 or 4 of its 200
walkers.
For calculations that run on a CPU only build of SAFIRE, `num_walkers` is typically chosen
such that every task carries on the order of 10-50 walkers.

### Running SAFIRE on GPUs

To run on GPUs, you must compile SAFIRE for the GPU (See {ref}`install` for instructions).

GPU builds of SAFIRE run on the GPU by default, but using the `--compute cpu` command line flag, it is possible to run on the CPU without rebuilding. When running on the GPU, it is important to saturate the compute bandwidth to gain an advantage. This requires calculations on large systems or with a high number of walkers (`num_walkers` in the job file). For small systems, this could mean using on the order of 1000 walkers or more per GPU. For very large systems, 10 walkers per GPU may be enough.

Multi GPU calculations are supported and should use one MPI task per GPU.

+++

### Your Turn

Run SAFIRE with the provided input files on 4 MPI tasks.

<!-- END hello-running -->

```{code-cell} ipython3
:tags: [scroll-output]
:scrolled: true

!mpirun -n 4 $AFQMC_EXEC files/afqmc.json
```

<!-- BEGIN hello-run-output -->

You should see the following at the top of the
output after running SAFIRE.

```
███████╗ █████╗ ███████╗██╗██████╗ ███████╗
██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝
███████╗███████║█████╗  ██║██████╔╝█████╗
╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝
███████║██║  ██║██║     ██║██║  ██║███████╗
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝

Version:     <version>
Git branch:  <branch>
Git commit:  <commit> (<date>)
Build type:  <build type>
Features:    <features>

Compute device:    cpu
MPI ranks:         4
MPI nodes:         1
MPI ranks/node:    4
OMP_NUM_THREADS:   1 (default)
TBLIS_NUM_THREADS: (unset)
```

Congratulations, you have now run an AFQMC calculation using SAFIRE!
In the directory that you ran this notebook from, you should see a file called
`afqmc.results.h5`.
We will return to how to use `safiretools` to extract the AFQMC energy later in this tutorial.

<!-- END hello-run-output -->

<!-- BEGIN hello-postprocessing -->

+++

## Basic Post-Processing

Now that we've learned how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s),
we are ready to learn how to extract the AFQMC energy from the outputs using safiretools.

The run above generated a results file, `afqmc.results.h5`, which contains the measurement data that was accumulated during the AFQMC simulation, including the total energy of the system.

safiretools provides the {py:class}`safiretools.Results` class to extract this information.

<!-- END hello-postprocessing -->

```{code-cell} ipython3
from safiretools import Results

results = Results("afqmc.results.h5")

energy, energy_err = results.average("Energy")

print(f"E = {energy.real} ± {energy_err.real}")
```

<!-- BEGIN hello-timeseries -->

It is a good idea to not take this value blindly but to instead inspect the Monte Carlo timeseries.

<!-- END hello-timeseries -->

```{code-cell} ipython3
# manual
import matplotlib.pyplot as plt
plt.plot(results.timeseries("ProjectionTime"), results.timeseries("Energy").real)
plt.show()

# visualization helper
from safiretools import vis
vis.plot_timeseries(results, skip=0)
plt.show()
```

You should get an output compatible with `Energy = -1.13683 ± 0.00037`.

<!-- BEGIN hello-equilibration -->

The time series should be converged to that energy. If it is not, we can pass the `skip` parameter to the {py:class}`safiretools.Results` functions or increase `equilibration_steps` in the job file. Note also that the bin axis in the timeseries plot does not coincide with the number of steps. This is because SAFIRE measures only every `measure_interval` steps (10 in this job file). You are encouraged to play with these values and rerun the simulation.

<!-- END hello-equilibration -->

You will learn how to measure observables other than the energy in {doc}`../05_computing_observables/05_computing_observables`.

+++

<!-- BEGIN hello-summary -->

## Summary

In this tutorial we have learned,

1. what are the inputs and outputs of SAFIRE.
2. how to invoke the SAFIRE executable from the command line in serial, using MPI, and using GPU(s)
3. how to extract the AFQMC energy from the outputs of SAFIRE using safiretools

---

Next, we will take a more detailed look at the JSON job file in {doc}`../02_understanding_the_input_file/02_understanding_the_input_file`.

<!-- END hello-summary -->
