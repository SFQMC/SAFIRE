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

# Understanding the input file

<b>Goal:</b>
Understand how to control SAFIRE via the input file.

## What you will learn

1. The structure of the json input file for SAFIRE
2. The minimal amount of information needed to run an AFQMC calculation using SAFIRE
3. The most common parameters that you will interact with as a user

+++ {"id": "5DZWyTSwFaBI"}

## Introduction

SAFIRE uses a json-based input file with modular input blocks to control AFQMC calculations.
The input file is organized into several hierarchical input blocks.
Certain types of block may be defined at different levels of the
hierarchy.
Additionally, some types of block may be defined multiple times.
We will begin with a brief explanation of how the blocks relate to each other, followed by an explanation
of each individual input block type.
We will explore the most common settings in each input block as well.

+++ {"id": "30febdd4"}

## Input block hierarchy

<!--
<div>
<img src="./figs/input_file_01_all.png" width="800"/>
</div>
-->

![](files/input_file_01_all.png)

The name of the highest level block tells SAFIRE which "driver" to run.
In this case, we are running with the "afqmc" driver.

Below the "afqmc" block are the "project" block and
one or more "execute" block(s).
We saw the "project" block in the previous tutorial,
where we used it to set the names of the output files via the "id" and "series" parameters.
As a reminder, the output files are named:

- `[id].s[series].scalar.dat` for scalar output data (energy, overlap, etc.)
- `[id].s[series].stat.h5` for non-scalar output data (one-body reduced density matrix, etc.)

"execute" block(s) define an actual calculation.
They consist of both AFQMC runtime parameters and references to the following five low-level blocks:

1. wavefunction - **always required!**
2. hamiltonian
3. walker_set
4. projector
5. estimator

The wavefunction, hamiltonian, and walker_set blocks can be defined either directly within
an execute block, by providing a json input block,
or defined in the "afqmc" block and referenced by "name" within one
or more "execute" blocks.
In the later case, these blocks must be named by setting the
"name" parameter in the corresponding input block.

We will explain each type of block in more detail below
including the most common settings that you will use.
For completeness, we will describe additional settings.
First, let's look at a sample input file for a typical calculation.

+++ {"id": "15c30cf5"}

## Sample input file - typical calculation

In a typical calculation. you will only need to set a few parameters.
In the input file below, we include the default value of each common parameter where possible.
We use  ellipses ( `...`) for the "estimator" input block for visual simplicity.
This input block will be explored in more detail in later tutorials.

```json
{
    "afqmc": {
      "project": {
        "id": "qmc",
        "series": 0
      },
      "execute": {
        "walker_set" : {
          "walker_type" : "CLOSED"
        },
        "wavefunction": {
          "filename": "files/input.h5"
        },
        "timestep": "0.01",
        "steps": "1",
        "population_control_interval": "10",
        "measure_interval_multiplier": "1",
        "walker_ortho_interval": "10",
        "n_walkers_per_mpi_task": "10",
        "seed": "42",
        "estimator" : {
          "name" : "mixed",
          /* ... */
        }
      }
    }
  }
```

Note that we did not define a "hamiltonian" block at all.
In this case, SAFIRE will look for the Hamiltonian in the same
file as the trial wavefunction.
We defined the "walker_set" block within the "execute" block.
We could have defined it within the "afqmc" block and referenced it by "name"
within the execute block instead.
This is only necessary if
you will use multiple execute blocks and want to use the same walker set in both.
For example, you may want to continue a
second AFQMC calculation starting from the final walkers of the
first AFQMC calculation.

<div class="alert alert-block alert-info">
<b>Note:</b>
  Essentially all parameters are able to default to some value except
  for the "filename" parameter in the "wavefunction" block. This must always be
  set to the name of the HDF5 file containing the trial wavefunction.
</div>

### Minimal information needed to run.

The bare minimum input blocks to run is

1. at least one execute block
2. at least wavefunction block with filename set to the HDF5 file containing a trial wavefunction (assuming the Hamiltonian is in the same HDF5 file)

Usually, one will also often provide:

1. a "walker_set" block
2. an "estimator" block (for any observables beside the energy)

+++ {"id": "d79cb60c"}

## The "project" block

The project block is used to set some project-level information.
For example, "id" and "series" are used to make the prefix of output
files as `[id].s[series].scalar.dat` and `[id].s[series].stat.h5`.

```json
"project" : {
  "id" : "your_project_name",
  "series" : 0,
  "ncores" : 1
}
```

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
| <b>Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>id</b> |  "afqmc" | The base name for the `[id].s[series].scalar.dat` and `[id].s[series].stat.h5` files  |
| <b>series</b> |0 |  The series number for the `[id].s[series].scalar.dat` and `[id].s[series].stat.h5` files. The filename will print as width three, padded with 0s if needed  |

### Less Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>ncores</b> | `1` | Set the number of cores to use per MPI task |

+++ {"id": "6981f73f"}

## The "execute" block

<!--
<div>
<img src="./figs/input_file_02_execute.png" width="800"/>
</div>
-->

![](files/input_file_02_execute.png)

The execute block is used to define an AFQMC calculation.
As described earlier, it needs a wavefunction, walker_set, hamiltonian,
projector, and estimator.
Of these, **the wavefunction must always be defined** while the others have default values.
The blocks can either be defined within the "execute" block, or defined
outside and referenced by name.

In addition to these blocks, the "execute" block is used to define several AFQMC methodological parameters.

Here is a sample execute block with options exposed and default values where appropriate.

```json
"execute" : {
  "wavefunction": { /* ... */ },
  "hamiltonian" : { /* ... */ },
  "walker_set" : { /* ... */ },
  "estimator" : { /* ... */ },
  "projector" : { /* ... */ },
  "timestep": "0.01",
  "steps": "1",
  "n_walkers_per_mpi_task": "10",
  "measure_interval_multiplier": "1",
  "population_control_interval": "10",
  "walker_ortho_interval": "10",
  "seed": "42",
  "checkpoint_interval": "-1",
  "hdf_write_file": "",
  "hdf_read_file": ""
}
```

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  

| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>timestep</b> |  0.01  |  The trotter step size in units of inverse energy (depending on the Hamiltonian's units)  |
| <b>steps</b> |  1  |   The number of imaginary time steps to take  |
| <b>n_walkers_per_mpi_task</b> |  10  |   number of Slater determinant random walkers to use per MPI task  |
| <b>measure_interval_multiplier</b> |  20  |  Used to determine the number of projection steps between measurements using the formula below. Measurement is the most expensive operation in AFQMC. A larger "measure_interval_multiplier" will reduce the CPU time necessary to perform AFQMC calculations.  |
| <b>population_control_interval</b> |  10  |  Interval to perform population control at in units of steps.  |
| <b>walker_ortho_interval</b> | 10  |  Interval to stabilize walkers at via a modified Gram-Schmidt procedure in units of steps |

$$
\text{measure\_interval} = \text{measure\_interval\_multiplier} \times \text{population\_control\_interval}
$$

### Less Common Settings

<style>
td, th {
   border: none!important;
}
</style>

| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
|  <b>seed</b> |  n/a  | Random seed for auxiliary fields. A seed is generated if not provided.  |
| <b>checkpoint_interval </b> |  -1  | Interval at which to write to checkpoint file in units of steps. An interval less than 0 will result in no checkpointing  |
| <b>hdf_write_file</b> |  ""  | Name of checkpoint HDF5 file to write. No checkpoint is written if the name is ""  |
| <b>hdf_read_file</b> |  ""  | Name of checkpoint HDF5 file to restart from. No checkpoint is read if the name is "" |

+++ {"id": "c9d95fca"}

## The "wavefunction" block

<!--
<div>
<img src="./figs/input_file_03_wavefunction.png" width="800"/>
</div>
-->

![](files/input_file_03_wavefunction.png)

The wavefunction block is the only of the 5 low-level blocks
that must always be specified.
The wavefunction block is used to point to the input HDF5 file containing
the desired trial wavefunction via the "filename" keyword.
If no hamiltonian block is provided, SAFIRE assumes that this HDF5 file
also contains the Hamiltonian.

Here is a sample wavefunction block with options exposed and default values where appropriate.

```json
"wavefunction" : {
    "name": "my_wavefunction",
    "filename": "afqmc.h5",
    "ndets_to_read": "-1"
}
```

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
|<b>filename</b> |  n/a  | **Required** The name of the file (possibly including a path) to the HDF5 file containing the trial wavefunction, and possibly containing the Hamiltonian as well |
| <b>name</b> |   n/a  | The name to assign to the current wavefunction block. This allows it to be referenced by name in execute blocks. A name is generated internally if not set here. |
| <b>ndets_to_read</b> |   -1  | The number of Slater determinants to read and from the HDF5 file and use in the trial wavefunction. If less than 0, all Slater determinants in the HDF5 file will be read and used. Must be less than or equal to the number of Slater determinants in the input HDF5 file |

+++ {"id": "67898650"}

## The "walker_set" block

<!--
<div>
<img src="./figs/input_file_04_walker.png" width="800"/>
</div>
-->

![](files/input_file_04_walker.png)

The walker_set block is used to set general properties of the random walkers.
The most common setting that you will change is the "walker_type" keyword
which allows you to specify the type of walker to use.
The possible walker types are as follows:

<b>Closed (i.e. RHF-like) </b>

$$
| \Phi_k \rangle = | \Phi^\uparrow \rangle \otimes | \Phi^{\downarrow = \uparrow} \rangle,
$$

where $\downarrow = \uparrow$ indicates that the $\beta$-sector is identical to the
$\alpha$-sector.

<b>Collinear (i.e. UHF-like) </b>

$$
| \Phi_k \rangle = | \Phi^\uparrow \rangle \otimes | \Phi^\downarrow \rangle
$$

<b>Noncollinear (i.e. GHF-like) </b>

$$
| \Phi_k \rangle = \begin{bmatrix}
\Phi^{\uparrow \uparrow} & \Phi^{\uparrow \downarrow} \\
\Phi^{\downarrow \uparrow} & \Phi^{\downarrow \downarrow} \\
\end{bmatrix}
$$

Here is a sample walker_set block with options exposed and default values where appropriate.

```json
"walker_set" : {
    "name": "your_walker_name",
    "walker_type": "COLLINEAR",
    "load_balance_type": "async",
    "pop_control_type": "pair",
    "min_weight": "0.05",
    "max_weight": "4"
}
```

<div class="alert alert-block alert-info">
<b>Note:</b>
    Setting the "name" parameter allows the current walker_set block to be reference in
    multiple execute blocks / calculations.
    This allows one to perform two successive AFQMC calculations which share the same
    walkers.
</div>

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
|<b>walker_type</b> |  Collinear  | The type of walker to use in AFQMC. Options are Closed, Collinear, Noncollinear |
|<b>name</b> | n/a  | The name to assign to the current walker_set block. This allows it to be referenced by name in execute blocks. A name is generated internally if not set here. |

### Less Common Settings

<style>
td, th {
   border: none!important;
}
</style>

| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>load_balance_type</b> |  "async"  |  Choose which load balancing algorithm to use, Choices are "async" for the asynchronous non-block swap load balancing algorithm and "simple" for a blocking (1-1) swap load balancing algorithm.  |
| <b>pop_control_type</b> |"pair"  |   choose population control algorithm to use. Choices are "pair", AND "serial_comb". The "pair" algorithm uses paired walker branching. The "serial_comb" algorithm uses the comb method from Booth, Gubernatis, PRE 2009. |
| <b>min_weight</b> | 0.05 |   Minimum walker weight for population control  |
| <b>max_weight</b> |4 |  Maximum walker weight for population control  |

+++ {"id": "b003de2e"}

## The "hamiltonian" block

<!--
<div>
<img src="./figs/input_file_05_hamiltonian.png" width="800"/>
</div>
-->

![](files/input_file_05_hamiltonian.png)

The hamiltonian block is used to provide settings related to the Hamiltonian.
This mostly entails pointing to the input HDF5 file containing
the desired hamiltonian via the "filename" keyword.

Here is a sample hamiltonian input block with
options exposed and default values where appropriate.

```json
"hamiltonian" : {
    "name": "my_hamiltonian",
    "filename": "afqmc.h5"
}
```

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>filename</b> |  n/a  |  The name of the file (possibly including a path) to the HDF5 file containing the Hamiltonian  |
| <b>name</b> |  n/a  |  The name to assign to the current hamiltonian block. This allows it to be referenced by name in execute blocks. A name is generated internally if not set here.  |

+++ {"id": "72f853ff"}

## The "projector" block

<!--
<div>
<img src="./figs/input_file_06_propagator.png" width="800"/>
</div>
-->

![](files/input_file_06_propagator.png)

The projector block is used to set properties of the AFQMC projector.
A typical user will not need to interact with this input block;
however, we include it here for completeness.

See the [input file description](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/input_description_afqmc.html) of the User manual for more information.

+++ {"id": "a59c54b9"}

## The "estimator" block

<!--
<div>
<img src="./figs/input_file_07_estimator.png" width="800"/>
</div>
-->

![](files/input_file_07_estimator.png)

The estimator block is used to specify which
estimator type(s) to use during a calculation
as well as which observables to compute.
This is a complicated topic and we devote
a later tutorial to Estimators in general,
and yet another tutorial to the back propagation estimator.

For now, we will discuss a relatively simple estimator, the Energy estimator,
in order to give you a sense of how they work.

Here is a sample input block for the energy estimator with
options exposed and default values where appropriate.

```json
"estimator" : {
    "name" : "energy",
    "print_components": "false",
    "print_sign": "false",
    "equil": "0",
    "skip": "0"
}
```

**It it important to note that the "name" parameter works very differently in the "estimator" blocks
than in "wavefunction", "walker_set", and "hamiltonian".**
In the later cases, "name" is used as an identifier to refer to blocks within a "execute" block.
For estimators, "name" is used to define the type of estimator being used.
Valid names include "energy", "mixed", and "back_propagated".

### Estimator types

The estimator type is selected using the "name" parameter.
The available "name" / types are:

- **"energy"** : this is a specialized mixed-estimator which specifically evaluates the energy. It also provides some granular control over what details of the energy are printed. For example, you can print the 1-body, exchange, and direct Coulomb interaction energies, in addition to the total energy, by setting `"print_components" : true` within an energy estimator input block.

- **"mixed"** : This adds a generic mixed estimator where the user may select which observables to compute. For observable, $\hat{O}$. the mixed estimator is given by
$$
\langle \hat{O} ⟩_\mathrm{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_\mathrm{T} | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_\mathrm{T} | \Phi_{n,k} \rangle}
$$
where $n$ is the projection step index,
$| \Phi_{n,k} \rangle$ are Slater determinant random walkers with weight $W_{n,k}$ (from importance sampling),
and $| \Psi_\mathrm{T} \rangle$ is the trial wavefunction.

- **"back-propagated"** : Similar to the "mixed" estimator, but the back-propagation algorithm is used to evaluate user-specified observables. The back-propagated estimator has the form,
$$
\langle O_\mathrm{BP} \rangle = \frac{1}{\sum_k W_{n+m,k}} \sum_k W_{n+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{n,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{n,k}\rangle}
$$
where $| \Phi_{n,k} \rangle$ are the usual forward-projected Slater determinant random walkers,
and $| \tilde{\Phi}_{m,k} \rangle$ are the back-propagated walkers given by,
$$
| \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{n,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{n+m-1,k} ) | \Psi_\mathrm{T} \rangle.
$$
The index $n$ corresponds to the current forward projection step,
and $m$ is the back-propagated step index.

### Observables

For the "mixed" and "back-propagated" estimators, specific observables are selected by defining the corresponding input block within the desired estimator block.
For example, one can define a mixed estimator which computes the one-body reduces density matrix as

```json
"estimator" : {
    "name" : "mixed",
    "onerdm" : {
        "name" : "my_onerdm"
    }
}
```

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>

| <b>        Parameter</b>   |  Default | Description |
|--:|---|:--|
| <b>name</b> |  n/a  |  The type of estimator to use. Choices are  "energy", "mixed", and "back_propagated" (see brief descriptions above)  |
| <b> print_components </b> |  false  | (energy estimator) If true, print the one-body, direct Coulomb, and exchange energy components to the *.scalar.dat file in addition to the total energy.   |
| <b> equil </b> |  0  | The number of initial steps to skip (i.e. the size of the Equilibration phase)   |

+++ {"id": "2e46faa0"}

## Some input file recipes

While there are many possible ways to set up an input file,
some structures will come up more often than others.
The [User manual](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/examples/02_running_afqmc/04_input_file_recipes/readme.html)
includes
several input file "recipes" / templates that you can use as a starting point for your calculations.
We recommend taking a look at them to complete the information provided here.

+++ {"id": "17b6510b"}

## Summary

In this tutorial, you learned

1. The structure of the json input file for SAFIRE
2. The minimal amount of information needed to run an AFQMC calculation using SAFIRE
3. The most common parameters that you will interact with as a user

See the [input file description](https://users.flatironinstitute.org/~beskridge/auxiliary_fields/input_description_afqmc.html) of the User manual for more information on the input file.

```{code-cell} ipython3
:id: 7rqqHZirH3zt


```
