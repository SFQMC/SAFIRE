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

+++ {"id": "16a2dc3c-fe26-455b-bade-b59769494e25"}

# Understanding the JSON Job File

<b>Goal:</b>
Understand how to control SAFIRE via the JSON job file.

## What You Will Learn

1. The structure of the JSON job file for SAFIRE
2. The minimal amount of information needed to run an AFQMC calculation using SAFIRE
3. The most common parameters that you will interact with as a user
4. How to use the {py:func}`safiretools.write_jobfile` helper

+++ {"id": "5DZWyTSwFaBI"}

## Introduction

As we saw in {doc}`../01_hello_safire/01_hello_safire`, SAFIRE simulations are configured via a JSON job file. This job file specifies which Hamiltonian should be simulated with which trial wavefunction for how long. In this tutorial, we will explore this job file further and give some practical tips on how to choose common parameters. For a complete reference of all settings, see {ref}`input_afqmc`. Later tutorials will cover how to generate the Hamiltonian and wavefunction inputs.

## A Minimal Job File

While there are many configuration parameters to customize SAFIRE simulations, most of them have sensible defaults. Therefore, basic simulations of the ground state energy of a system can be run with a minimal job file containing only the most important parameters.

```json
{
  "execute": {
    "wavefunction": {
      "filename": "wavefunction.h5"
    },
    "hamiltonian": {
      "filename": "hamiltonian.h5"
    },
    "timestep": 0.01,
    "steps": 10000,
    "equilibration_steps": 1000,
    "num_walkers": 200
  }
}
```

Here,

- `wavefunction` and `hamiltonian` point to the HDF5 files containing the trial wavefunction and the Hamiltonian. The paths are relative to the directory of the job file, not to the directory SAFIRE is launched from. Both blocks may also name the same file if it contains both.
- `timestep` is the Trotter step size $\Delta\tau$ in units of inverse energy, where the energy unit is the one the Hamiltonian is written in. Since the Trotter decomposition introduces a timestep error, it is good practice to check that the result does not change within error bars when repeating the calculation with a smaller timestep.
- `steps` is the total number of propagation steps, including the equilibration steps. Together with `timestep`, it sets the total projection time $\beta = N_\mathrm{steps}\Delta\tau$.
- `equilibration_steps` is the number of steps at the start of the calculation during which nothing is measured. During this phase, the walkers are projected towards the ground state and the energy shift of the propagator relaxes to the average energy. The equilibration should be long enough that the energy timeseries has converged, which you can check as in {doc}`../01_hello_safire/01_hello_safire`. If you choose this too small, you can always skip more samples using the `skip` parameter in the postprocessing tools.
- `num_walkers` is the total number of walkers over all MPI tasks. See {doc}`../01_hello_safire/01_hello_safire` for typical values on CPUs and GPUs. AFQMC simulations have a small finite-population bias that scales as $1/N_\text{walkers}$, so for simulations with few walkers, convergence with number of walkers should be confirmed.


## More Advanced Examples

In the following, we give some common examples of going beyond the minimal configuration.

### Initializing Walkers from a Different Wavefunction

By default, all walkers start from the trial wavefunction (or its determinant with the largest coefficient, for multi-determinant trials). The `walker_set` block allows starting them from a different wavefunction instead.

```json
{
  "execute": {
    "wavefunction": {
      "filename": "uhf.h5"
    },
    "hamiltonian": {
      "filename": "hamiltonian.h5"
    },
    "walker_set": {
      "from": {
        "wavefunction": {
          "filename": "rhf.h5"
        }
      }
    },
    "timestep": 0.01,
    "steps": 10000,
    "equilibration_steps": 1000,
    "num_walkers": 200
  }
}
```

This is useful when the trial wavefunction breaks a symmetry of the Hamiltonian, as an unrestricted Hartree-Fock (UHF) trial often breaks spin symmetry. Such a trial may give a better description of the ground state than a symmetric one, but walkers started from it are contaminated by soft Goldstone modes. These modes are not present in the ground state, but they are low in energy and therefore take a long projection time to decay. Starting the walkers from a symmetric wavefunction, such as the restricted Hartree-Fock (RHF) determinant, avoids this contamination and can greatly reduce the necessary number of equilibration steps.

The walker type is chosen automatically as the most symmetric one that can represent all wavefunctions in the job file, here collinear walkers for the UHF trial. See {ref}`initial_walkers` for more details.

### Stabilization and Measurement Intervals

Several operations of the AFQMC algorithm are performed only every few steps. Their intervals are set in the execute block and are given in units of propagation steps.

```json
"execute": {
  "measure_interval": 20,
  "population_control_interval": 10,
  "walker_ortho_interval": 5,
  "binsize": 10
}
```

- `measure_interval` (default 10) sets how often the observables are measured. Measurement is usually the most expensive operation in AFQMC, so measuring less often speeds up the calculation. Because consecutive measurements are correlated, there is no benefit to measuring more often than the Monte Carlo autocorrelation time. An estimator can also override this interval with its own `measure_interval`.
- `population_control_interval` (default 10) sets how often the walker population is resampled according to the walker weights. Population control is relatively cheap, so this interval can stay small.
- `walker_ortho_interval` (default 10) sets how often the walker Slater determinants are orthogonalized. Without it, the propagation becomes numerically unstable, which happens sooner for large systems and large timesteps. Orthogonalization is relatively cheap, so performing it frequently is recommended.
- `binsize` (default 1) sets how many consecutive measurements are averaged into a single bin of the results file. Larger bins reduce the size of the results file, which can become large when measuring large observables such as density matrices.

### Output Name

By default, the results are written to `<job file name>.results.h5`, e.g. `afqmc.results.h5` for `afqmc.json`. The `output_name` parameter at the top level of the job file sets a different name.

```json
{
  "output_name": "dt0.005",
  "execute": { /* ... */ }
}
```

This writes the results to `dt0.005.results.h5`. Unlike all other paths in the job file, `output_name` is relative to the directory SAFIRE is launched from. Be aware that SAFIRE overwrites an existing results file of the same name. See {ref}`output_afqmc` for the content of the results file.

### Reproducibility

AFQMC is a stochastic method, so two runs of the same job file give different, statistically compatible results. To reproduce a calculation exactly, the random seed can be fixed at the top level.

```json
{
  "seed": 42,
  "execute": { /* ... */ }
}
```

If no seed is given, SAFIRE chooses one at random and prints it in the log. Reusing that seed reproduces the earlier run, provided it is run with the same number of MPI tasks.

### Advanced Estimators

Measuring observables beyond the energy or switching off energy estimation requires additional settings in the job file that will be covered in a later tutorial. For all estimator settings, see {ref}`estimator_block`.

### Multiple Execute Blocks

It is possible to make `"execute"` a list of blocks to carry on simulations in multiple stages. The results of each stage are stored separately in the results file. This is discussed in {ref}`multi_stage_runs`.

### Out-of-Line Definitions

So far, we have defined the wavefunction and the Hamiltonian inline in the execute block. Alternatively, they can be defined out-of-line with a `name` in the top-level `wavefunctions` and `hamiltonians` lists and referred to by that name elsewhere. This becomes useful in advanced estimator and multiple execute blocks configurations where the same Hamiltonian, wavefunction or walker set may appear more than once.

```json
{
  "wavefunctions": [
    {
      "name": "trial",
      "filename": "wavefunction.h5"
    }
  ],
  "hamiltonians": [
    {
      "name": "hamiltonian",
      "filename": "hamiltonian.h5"
    }
  ],
  "execute": {
    "wavefunction": "trial",
    "hamiltonian": "hamiltonian",
    "timestep": 0.01,
    "steps": 10000,
    "equilibration_steps": 1000,
    "num_walkers": 200
  }
}
```

## Writing Job Files with `safiretools`

Rather than writing JSON by hand, it is often useful to generate them using the same python script that generates the Hamiltonian and wavefunction inputs. One way to do this is to just use the [json](https://docs.python.org/3/library/json.html) module. Another way is the {py:func}`safiretools.write_jobfile` helper, which brings some additional convenience features.

```python
from safiretools import write_jobfile
hamiltonian = ...  # covered in following tutorials
wavefunction = ...  # covered in following tutorials

write_jobfile("afqmc.json", {
    "execute": {
        "wavefunction": wavefunction,
        "hamiltonian": hamiltonian,
        "timestep": 0.01,
        "steps": 10000,
        "equilibration_steps": 1000,
        "num_walkers": 200
    }
})
```

The parameters follow the same structure as the job file, but `safiretools` Hamiltonian and wavefunction objects can be passed directly in place of their blocks or their `filename`. `write_jobfile` writes each of these objects to its own HDF5 file next to the job file and puts the name of that file into the job file. Here, these are `afqmc_wavefunction0.h5` and `afqmc_hamiltonian0.h5`. If a block has a `name`, the file is called `afqmc_<name>.h5` instead. An object that appears in several places is written only once.

Alternatively, use {py:meth}`safiretools.Hamiltonian.to_hdf5` and {py:meth}`safiretools.Wavefunction.to_hdf5` to write them to HDF5 files and specify the input path manually.

+++ {"id": "17b6510b"}

## Summary

In this tutorial, you learned

1. The structure of the JSON job file for SAFIRE
2. The minimal amount of information needed to run an AFQMC calculation using SAFIRE
3. The most common parameters that you will interact with as a user
4. How to use the {py:func}`safiretools.write_jobfile` helper

See {ref}`input_afqmc` for more information on the JSON job file.
