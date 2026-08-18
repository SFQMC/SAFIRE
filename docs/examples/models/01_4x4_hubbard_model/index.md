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

+++ {"id": "-CSom7nd5xcl"}

# 4x4 Square Lattice Hubbard Model

Author: Kyle Eskridge

In this tutorial, we will perform AFQMC calculations for the Hubbard model on
a 4x4 lattice.
The Hamiltonian is given by,
$$
\hat{H} = -t \sum_{\langle i,j\rangle} \hat{c}^\dagger_i \hat{c}_j + U \sum_{i} \hat{n}^{\uparrow}_i \hat{n}^{\downarrow}_i,
$$
where $i,j$ are lattice site indices, angle brackets indicate nearest-neighbors,
$\hat{c}^\dagger_i$/$\hat{c}_i$ are electronic creation/annihilation operators,
and $\hat{n}^{\sigma}_i$ are spin-resolved number operators corresponding to site $i$.
The 4x4 lattice is small enough to make exact diagonalization possible.
We will use exact results for this system as reference data throughout
this tutorial.

The tutorial is structured as follows.
We will begin with an explanation of how to build lattice model Hamiltonians
using the `afqmctools` Python package.
Next, we will walk through computing the ground state energy of this Hamiltonian for a particular filling and value of U,
comparing a couple of options for trial wavefunctions.
Next, we will continue the walk through by computing the ground state charge,
and spin density for the same filling and value of U.
Next, we will demonstrated how to obtain exact results using AFQMC
for the special case of half filling for which, if care is taken,
there is no sign problem.

```{code-cell} ipython3
:id: d3YMrpqs5xcm

from pathlib import Path
# simple setup
from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "dfgyQ5OC5xcn"}

## Building a lattice model Hamiltonian

`afqmctools` provides a convenience lattice model Hamiltonian builder and writer which
will allow us to construct the "standard" Hubbard model with nearest-neighbor hopping and on-site U, among
other Hamiltonians.
The basic steps for using the lattice model Hamiltonian builder are:
1. define lattice parameters
2. define Hamiltonian parameters
3. build the lattice model Hamiltonian

Step 3. can optionally be broken into two sub-steps:  

- construct the lattice instance using the lattice parameters
- construct the Hamiltonian using the lattice instance, and the Hamiltonian parameters

Breaking step 3. into substeps gives us easy access to the lattice instance
which will facilitate visualizing the charge and spin densities latter on,
so we will use this approach.

```{code-cell} ipython3
---
id: tNji__B25xcn
colab:
  base_uri: https://localhost:8080/
  height: 855
outputId: c0edb976-6ddc-4ada-eb2f-66c8587a93b7
---
from afqmctools.systems.lattice import get_lattice
import afqmctools.utils.visualize as vis

# Step 1. define the lattice parameters
lattice_params = dict(
    L1 = 4,              # lattice dimension along a1
    L2 = 4,              # lattice dimension along a2
    boundary1 = 'PBC',   # boundary condition for the boundary *normal to* a1
    boundary2 = 'PBC',    # similarly for a2
)

# Step 3.a. build the lattice
lattice = get_lattice(
    params=lattice_params
)

vis.plot_lattice(lattice)
```

```{code-cell} ipython3
---
id: IMqV9fCc5xcn
colab:
  base_uri: https://localhost:8080/
outputId: 601a13a7-15f4-49d9-ba6a-01fb4ba3b173
---
from afqmctools.hamiltonian.model.director import HamiltonianDirector
import afqmctools.utils.io as io

print(scratch_dir)

fe_scratch_dir = scratch_dir / "fe"
fe_scratch_dir.mkdir(exist_ok=True)

# Step 2. define Hamiltonian parameters
hamiltonian_params = {
    'hamiltonian' : {
        "t" : 1.0,  # note: we could omit this, nearest-neighbor hoping with t=1 is included by default
        "U" : 4.0
    }
}

# Step 3.b. build the lattice model Ha
hamiltonian = HamiltonianDirector(
    lattice=lattice,
    source=hamiltonian_params
).build()

# set the number of electrons - we'll use this again
nelec = (8,8)

# AND save it!
io.write_model_hamiltonian(
    hamiltonian=hamiltonian,
    fname=fe_scratch_dir/"afqmc.h5",
    nelec=nelec
)
```

+++ {"id": "MqK7sb3Z5xcn"}

We now have a "standard" Hubbard model with nearest-neighbor hopping and onsite U saved in an HDF5 file, "afqmc.h5"
which can be directly read by SAFIRE.
Next, we'll walk through an AFQMC calculation where we'll compute the ground state energy.

+++ {"id": "9KnqEEIS5xcn"}

## AFQMC Walkthrough : ground state energy

Here, we will compute the ground state energy of the "standard" Hubbard model
based on the Hamiltonian that we made previously - i.e.
$$
\hat{H} = -t \sum_{\langle i,j\rangle} \hat{c}^\dagger_i \hat{c}_j + U \sum_{i} \hat{n}^{\uparrow}_i \hat{n}^{\downarrow}_i,
$$
with $t = 1.0$, and $U = 4.0$.
We will perform the calculation at half-filling.

AFQMC uses a trial wavefunction to control the sign problem at the cost of introducing a bias;
however, the bias can be minimized by using a trial wavefunction which accurately captures
the physics of the target system.
It is, therefore, important to check that candidate trial wavefunctions have the correct ordering,
symmetries, etc. before running an AFQMC calculation.
In addition to this, some methods for computing a trial wavefunction will lead to better results than
others for specific problems.
For example, both free-electron trial wavefunctions and Hartree-Fock (HF) trial wavefunctions
are reasonable for low-, to intermediate-U values.
Assuming that all candidate trial wavefunctions have the desired properties,
it may be necessary to test which wavefunction is better by running afqmc with each,
and comparing the results.
In this walk through, we will compare free-electron trial wavefunctions
to unrestricted Hartree-Fock trial wavefunctions.

+++ {"id": "Clkg0_nE5xcn"}

### Free-electron Trial Wavefunction

`afqmctools` provides a convenience function, `free_electron()` ,for calculating free-electron trial wavefunctions.
Internally, `free_electron()` applies a small twist in order to break k-space degeneracy for open-shell systems.
It also accepts a user-supplied twist angle to used instead.
`free_electron()` will print the actual twist used in units of radians as, for example,

```
Generating free-electron trial wavefunction with twist = [4.10803005e-06 4.58334805e-04]
```


We note that, because of the twist angle, the Hamiltonian will typically need to be rebuilt.
All of this is done internally if `free_electron()` is given the original lattice, and hamiltonian parameters
as shown below.

... do we want to have a diagram here of k-space, and the expected one-body

```{code-cell} ipython3
---
id: 4MY5lScg5xco
colab:
  base_uri: https://localhost:8080/
outputId: e6000516-1e4f-4ae3-db34-af0b0c4d03f7
---
# First, let's compute the Free-Electron trial wavefunction
from afqmctools.wavefunction.free_electron import free_electron
from afqmctools.wavefunction.common import write_wfn

input_params = dict(
    lattice = lattice_params,                           # from step 1. above
    hamiltonian = hamiltonian_params["hamiltonian"]     # from step 2. above
)


wfn,spin_symm,autohf_results = free_electron(
    source=input_params,
    nelec=nelec,
    twist=None,                          # (optional) using the default small twist
    return_autohf = True,
    filling_strategy="balanced"
)

# and write the wavefunction for use in SAFIRE
write_wfn(
    filename=fe_scratch_dir/"free_electron.h5",
    wfn=wfn,
    walker_type=spin_symm,
    nelec=nelec,
    norb=lattice.N_sites
)
```

+++ {"id": "-xW_O9f35xco"}

Now that we have computed a free-electron wavefunction, we can inspect the results to validate it.
Internally, `free_electron()` uses the `autohf` HF solver in order to evaluate the energy of the free-electron trial wavefunction
for the interacting Hamiltonian.
In this case, for $N_{\uparrow} = N_{\downarrow} = 8$, we see that the total energy is
Etotal = Etotal=-5.18522319306247 $t$ with a one-body contribution of EK = -22.69026695983466 $t$,
and a Hubbard U contribution of EU=17.50504376677219.

Next, let's take a look at the charge and spin densities of the trial wavefunction.

```{code-cell} ipython3
---
id: Ctx8QQM65xco
colab:
  base_uri: https://localhost:8080/
  height: 701
outputId: a868b2dd-398f-40d5-90b7-9a8036f6659b
---
state_results, functions = autohf_results

# get the density from autohf
makeRDMs = functions['makeRDMs']
state = state_results['state']
rdm = makeRDMs(state)

# get lattice dimensions from the Lattice instance
L = lattice.L

rho_up = rdm[0].diagonal().reshape(*L)
rho_down = rdm[1].diagonal().reshape(*L)

rho_charge = rho_up + rho_down

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    cmap = 'bwr',
    vmin = 0.0,
    vmax = 2.0
)
```

```{code-cell} ipython3
---
id: O40aHv5A5xco
colab:
  base_uri: https://localhost:8080/
  height: 690
outputId: 76c9590a-d19b-42dc-d78f-08d0ed538197
---
from afqmctools.observables.spin import local_spin

rho_spin = 0.5*(rho_up - rho_down)

vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    cmap = 'bwr',
    vmin = -0.5,
    vmax = 0.5
)
```

+++ {"id": "R9Br4_CK5xco"}

As expected for a free-electron trial wavefunction with an even number of electrons,
the charge- and spin-densities are uniform.
For the charge-density, we see that the uniform charge density is 1.0 electron/site,
consistent with half-filling.
For the spin-density, we see that the uniform spin density is 0.0 spin/site,
which is expected for an even number of up and down electrons.

+++ {"id": "uV3v2_Ou5xco"}

### Running AFQMC

Next, we will run an AFQMC calculation.
SAFIRE uses an input file in json format to define AFQMC run parameters.
`afqmctools` comes with a convenience function to generate a json input file as demonstrated below.
We have included reasonable AFQMC run parameters for this calculation below.

```{code-cell} ipython3
:id: FCd6cFpd5xco

from afqmctools.inputs.from_hdf import write_json

# make an input file
afqmc_params = {
    "timestep": 0.01,
    "steps": 12000,
    "n_walkers_per_mpi_task": 100,
    "population_control_interval": 5,
    "walker_ortho_interval": 5,
    "measure_interval_multiplier": 1,
    "seed" : 42,                          # just for reproducibility
    "propagator": {
      "use_cp_constraint": True,
      "use_real_vbias" : True
    }
}

write_json(
    fe_scratch_dir/"afqmc.json",
    fwfn0=fe_scratch_dir/"free_electron.h5",
    fham0=fe_scratch_dir/"afqmc.h5",
    exec_opts=afqmc_params
)
```

```{code-cell} ipython3
---
id: 5YJevKw85xco
colab:
  base_uri: https://localhost:8080/
outputId: 601aa3a3-9001-40d2-84fe-7cc725c17dff
---
run_afqmc(
    run_dir=fe_scratch_dir,
    np=16,
    output_file=None # this will put output here!
)
```

```{code-cell} ipython3
---
id: 9_qLNbRM5xco
colab:
  base_uri: https://localhost:8080/
  height: 921
outputId: 6de71736-72b7-4efa-bc41-8e0985c0759d
---
# analyze
from stats.scalar_dat import analyze_scalar_data

nequil = 5.0

analysis_settings = dict(
    fname = fe_scratch_dir /"qmc.s000.scalar.dat",
    xaxis = "time",     # use units of imaginary time for equilibration
    nequil = nequil,    # length of equilibration phase (in units of imaginary time)
    trace = True,       # plots a trace of the scalar data
)

E,dE = analyze_scalar_data(analysis_settings)

# we will use this value in Exercise 1.1.2 as well
ref_stoch_uncertainty = dE
```

+++ {"id": "LRmDOsA-S8og"}

The Hubbard model at half-filling is sign-free, so we would expect numerically exact results; For U/t =4, the exact energy is -13.62185, so why do we disagree?

Even if the underlying Hamiltonian is sign-free (i.e. $\langle \Psi_\mathrm{T}|\Phi\rangle \ge 0$ for all walkers $|\Phi\rangle$), in the importance sampled AFQMC random walk, an improperly chosen trial wavefunction can still induce a bias if its nodes $\langle \Psi_\mathrm{T}|\Phi\rangle = 0$ do not coincide with those of the ground state. Especially free-electron trial wavefunctions in open-shell geometries are susceptible to this problem.

There are different approaches to dealing with this problem. One can either

1. move the nodes to the right position by choosing a trial wavefunction with the correct symmetry properties such as particle-hole symmetry or
2. allow the walkers to move around the nodes ergodically by using a noncollinear wavefunction that is not aligned with the spin axis of the Hubbard-Stratonovich decoupling.

Next, we will use a noncollinear Hartree-Fock trial wavefunction.

+++ {"id": "tsMzr8175xcp"}

### Hartree-Fock Trial Wavefunction

Now, we will compute and characterize an unrestricted Hartree-Fock (UHF) wavefunction.
We will use autoHF to accomplish this.

```{code-cell} ipython3
---
id: fAmcxlnp5xcp
colab:
  base_uri: https://localhost:8080/
outputId: 88d99bd1-dd65-4ca9-a2d1-4943aef286af
---
from autohf import lattice_hf,AutoHFHamiltonian
from afqmctools.inputs.from_autohf import autohf_to_afqmc

hf_settings = dict(
    ansatz = 'SD',
    steps = 200,
    nelec = nelec,
    batch_size = 50,
    seed = 42,
    measure_spin = True,
    random_initial_phi0 = False,
    noncollinear = True,
    verbose = False
)

results = lattice_hf(
    hamiltonian=AutoHFHamiltonian(source=hamiltonian),
    settings=hf_settings
)

autohf_to_afqmc(
    results,
    output_fname = fe_scratch_dir/"uhf_wfn.h5"
)
```

```{code-cell} ipython3
---
id: SbsGsL0h5xcp
colab:
  base_uri: https://localhost:8080/
  height: 701
outputId: 4e4d58d4-8505-4316-dee8-3f61b4b57a12
---
# get the density from autohf
state_results, functions = results

makeRDMs = functions['makeRDMs']
state = state_results['state']

rdm = makeRDMs(state)

# get lattice dimensions from the Lattice instance
L = lattice.L

# noncollinear
M = lattice.N_sites
rho_up = rdm[:M,:M].diagonal().reshape(*L)
rho_down = rdm[M:,M:].diagonal().reshape(*L)

# For collinear:
#rho_up = rdm[0].diagonal().reshape(*L)
#rho_down = rdm[1].diagonal().reshape(*L)

rho_charge = rho_up + rho_down
rho_spin = rho_up - rho_down

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    vmin=0.0,
    vmax=2.0
)
```

```{code-cell} ipython3
---
id: BjfjakIz5xcp
colab:
  base_uri: https://localhost:8080/
  height: 690
outputId: 335549e0-8890-4318-a08e-d7468c4fac09
---
from afqmctools.observables.spin import local_spin

vis.plot_lattice(
    lattice,
    density=rho_spin,
    density_label="spin density",
    cmap="bwr",
    vmin=-0.5,
    vmax=0.5
)
```

```{code-cell} ipython3
:id: _9GEgLIt5xcp

from afqmctools.inputs.from_hdf import write_json

# make an input file
afqmc_params = {
    "timestep": 0.01,
    "steps": 12000,
    "n_walkers_per_mpi_task": 100,
    "population_control_interval": 5,
    "walker_ortho_interval": 5,
    "measure_interval_multiplier": 1,
    "seed" : 42,                          # just for reproducibility
    "propagator": {
      "use_cp_constraint": True,
      "use_real_vbias" : True
    }
}

write_json(
    fe_scratch_dir/"afqmc.json",
    fwfn0=fe_scratch_dir/"uhf_wfn.h5",
    fham0=fe_scratch_dir/"afqmc.h5",
    exec_opts=afqmc_params
)
```

```{code-cell} ipython3
---
id: DIYwZYvz5xcp
colab:
  base_uri: https://localhost:8080/
outputId: 5414bf9b-a425-4e44-c469-c1b29b5f34d1
---
run_afqmc(
    run_dir=fe_scratch_dir,
    np=16,
    output_file=None # this will put output here!
)
```

```{code-cell} ipython3
---
id: nn3u7u8k5xcp
colab:
  base_uri: https://localhost:8080/
  height: 921
outputId: 78be3a24-0f0c-4577-cb48-6ad867cb1f1b
---
# analyze
from stats.scalar_dat import analyze_scalar_data

nequil = 20.0

analysis_settings = dict(
    fname = fe_scratch_dir /"qmc.s000.scalar.dat",
    xaxis = "time",     # use units of imaginary time for equilibration
    nequil = nequil,    # length of equilibration phase (in units of imaginary time)
    trace = True,       # plots a trace of the scalar data
)

E,dE = analyze_scalar_data(analysis_settings)

print(f"The AFQMC energy is {E:.6f} +/- {dE:.6f}")
```

+++ {"id": "0vF3gnwWSgir"}

If you used all of our settings, you should get see

```
The AFQMC energy is -13.614404 +/- 0.004229
```

Again, we expect to achieve very close to the exact energy for the Hubbard model at half filling which is -13.62185 for U/t=4.
We see much better agreement now than we did with the free-electron trial wavefunction before; however, we're still off by about 0.007 t, or about twice
the stochastic uncertainty of our result. Why do we not quite agree?

We have residual systematic errors to remove. Specifically, we will want
to remove the trotter error.
We can achieve this either by using a very small step size, or by extrapolating to $\tau = 0$.

Let's rerun with a smaller $\tau = 0.005$

+++ {"id": "Q-t-66BmVVn5"}

## Rerun AFQMC with a smaller τ

When we decrease the step size, we will
also need to increase the number of steps, and
each interval involved in order to maintain the same
amount of statistics.

```{code-cell} ipython3
:id: 8IfICCwnVU8m

from afqmctools.inputs.from_hdf import write_json

# make an input file
afqmc_params = {
    "timestep": 0.005,
    "steps": 24000,
    "n_walkers_per_mpi_task": 100,
    "population_control_interval": 10,
    "walker_ortho_interval": 10,
    "measure_interval_multiplier": 1,
    "seed" : 42,                          # just for reproducibility
    "propagator": {
      "use_cp_constraint": True,
      "use_real_vbias" : True
    }
}

write_json(
    fe_scratch_dir/"afqmc.json",
    fwfn0=fe_scratch_dir/"uhf_wfn.h5",
    fham0=fe_scratch_dir/"afqmc.h5",
    exec_opts=afqmc_params
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3g8uLM4_V5Vo
outputId: fa772c06-bf39-46ac-e81d-e57e14cc7449
---
run_afqmc(
    run_dir=fe_scratch_dir,
    np=16,
    output_file=None # this will put output here!
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 920
id: 8L0WKycOV9L4
outputId: 1d4d7374-5c21-43a7-eae5-46bafc88d52d
---
# analyze
from stats.scalar_dat import analyze_scalar_data

analysis_settings = dict(
    fname = fe_scratch_dir /"qmc.s000.scalar.dat",
    xaxis = "time",     # use units of imaginary time for equilibration
    nequil = nequil,    # length of equilibration phase (in units of imaginary time)
    trace = True,       # plots a trace of the scalar data
)

E,dE = analyze_scalar_data(analysis_settings)

print(f"The AFQMC energy is {E:.6f} +/- {dE:.6f}")
```

+++ {"id": "SVS5nLE7YvBi"}

Now, you should see

```
-13.618391
```

which is off by about 0.0035 t compared with the exact energy, -13.62185. We can continue to smaller step sizes to remove the remaining disagreement.

+++ {"id": "ntTQjKcjU3_p"}

## Computing Observables with Back-Propagation

General second-quantized observables can be computed using AFQMC via the back-propagation algorithm.
See the [SAFIRE documentation](https://users.flatironinstitute.org/~beskridge/safire/user_manual/estimators.html#back-propagation-bp-estimators) for more information on Back-propagated observables.
To "turn on" back-propagation, add an `"estimator"` block with `"name": "back_propagation"` to an `"execute"`
block in the input file.
Here is a sample input file where an ellipsis has been used to hide the rest of the input.

```json
{
  "afqmc": {
    
    /* ... */
      
    "execute": {

      /* ... */

      "population_control_interval" : 10,

      /* ... */

      "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "extra_path_restoration": true,
        "bp_walker_ortho_interval": 5,
        "measure_interval_multiplier": [10,15,20],
        "equil_multiplier": 50,
        "onerdm": {
            "name": "one_rdm"
        }
      }
    }
  }
}
```

For back-propagation (BP), we will need to provide a BP time interval.
We will need to check for convergence in this parameter.
Notice that, just like the default estimator which measures and prints the energy, the number of steps is determined indirectly using the "measure_interval_multiplier" parameter;
however, unlike the default estimator, the BP estimator accepts multiple integr values for "measure_interval_multiplier" as shown in the example above.
In this case, BP will be run with each of the BP lengths provided and saved in an output HDF5 file separately.
This simplifies checking for convergence in the BP length by allowing us to re-use the "forward" propagation run for each BP length.

+++ {"id": "ZjxOUIHUQeuD"}

### Run AFQMC with Back-propagation

Run SAFIRE using the following input file to compute the one-rdm using back-propagation.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: CxlJYdWoREbB
outputId: 819142c2-1650-4107-9ab3-f2bd8927f689
---
from afqmctools.inputs.from_hdf import write_json

# make an input file
afqmc_params = {
    "timestep": 0.005,
    "steps": 24000,
    "n_walkers_per_mpi_task": 100,
    "population_control_interval": 10,
    "walker_ortho_interval": 10,
    "measure_interval_multiplier": 1,
    "seed" : 42,                          # just for reproducibility
    "propagator": {
      "use_cp_constraint": True,
      "use_real_vbias" : True
    },
    "estimator": {
          "name": "back_propagation",
          "path_restoration": True,
          "extra_path_restoration": True,
          "bp_walker_ortho_interval": 10,
          "measure_interval_multiplier": [20,40,60], # 🚧 Under construction 🚧 : Kyle is working on a good set of values here.
          "equil_multiplier": 420,
          "onerdm": {
              "name": "one_rdm"
        }
    }
}

print(afqmc_params)

write_json(
    fe_scratch_dir/"afqmc_3.json",
    fwfn0=fe_scratch_dir/"uhf_wfn.h5",
    fham0=fe_scratch_dir/"afqmc.h5",
    exec_opts=afqmc_params,
    series=3
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: IIFG3vyw4WLr
outputId: fb551b42-39fd-4dd1-905f-015d1237d024
---
run_afqmc(
    input_file="afqmc_3.json",
    run_dir=fe_scratch_dir,
    np=16,
    output_file=None # this will put output here!
)
```

+++ {"id": "WFUPHBHt5xcp"}

### Analyze the AFQMC output

The AFQMC code saves "scalar" data, such at the energy, in a file called `qmc.s000.scalar.dat` (a text file)
and non-scalar data, such as the one-rdm and other observables, in a file called `qmc.s000.stat.h5` (HDF5 file).
The AFQMC code includes tools to analyze both.
We have already analyzed the energy in the `qmc.s000.scalar.dat` file using `analyze_scalar_data()`.
We will now focus on analyzing observables saved in the HDF5 output file, `qmc.s000.stat.h5`.

+++ {"id": "-Y7bGG0xwFOh"}

## Average Observables

afqmctools provides tools to directly average observables from the HDF5 output
in the `afqmctools.analysis.average` module.
At the core of the module is the `average_observable()` function which extracts samples of a user specified observable from the HDF5 output file, and averages the samples while accounting for autocorrelation automatically.
Additionally, it users may provide one or more "transform" functions which are applied to the data samples immediately after reading from HDF5.
afqmctools provides a few transform function "factories"for generating common transforms.
We will use the `hermitize_factory()` to hermitize the one-rdm samples since SAFIRE measures only the upper-triangular part of the one-rdm.

In the code block below, we will plot the diagonal of the one-rdm for each BP average (i.e. for each BP length) in order to check for convergence in BP length.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 517
id: fqLt304CwKks
outputId: 4593e5a7-2e66-446b-e2b1-483f5193b453
---
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.analysis.transform import hermitize_factory
from afqmctools.analysis.average import WALKER_TYPE,get_metadata,average_observable

SAFIRE_hdf5_output = fe_scratch_dir / "qmc.s003.stat.h5"
nspins = 1 # this is a noncollinear calculation, so one (combined) spin sector!
M = lattice.N_sites # get the number of sites to help with plotting

walker_type = WALKER_TYPE[get_metadata(SAFIRE_hdf5_output)["WalkerType"]]

# set up a transform the hermitize the one-rdm
hermitize = hermitize_factory(
    walker_type=walker_type
)

# get the metadata for plotting
metadata = get_metadata(SAFIRE_hdf5_output,"/Observables/BackPropagated/")
print(metadata)

bp_steps = metadata["BackPropSteps"]
naverages = metadata["NumAverages"]

print("BP steps", bp_steps)

# array to save one-rdm for each average
one_rdms = np.zeros((naverages,nspins,4*(M**2)),dtype=np.complex128)
one_rdms_err = np.zeros((naverages,nspins,4*(M**2)),dtype=np.complex128)

rho_ups = np.zeros((naverages,*lattice.L),dtype=np.complex128)
rho_downs = np.zeros((naverages,*lattice.L),dtype=np.complex128)

for avg in range(naverages):
    # Extract and Transform data
    one_rdms[avg],one_rdms_err[avg] = average_observable(
        SAFIRE_hdf5_output,
        eqlb=5,
        estimator='back_propagated',
        name="one_rdm",
        transform=[hermitize],
        ix=avg
    )

    # handle spin!!
    print(one_rdms[avg].shape)

    lattice_dims = lattice.L
    rho_diag = one_rdms[avg].reshape((2*M,2*M)).diagonal()
    rho_up = rho_diag[:M]#.reshape(lattice_dims)
    rho_down = rho_diag[M:]#.reshape(lattice_dims)

    rho_diag_err = one_rdms_err[avg].reshape((2*M,2*M)).diagonal()
    rho_up_err = rho_diag_err[:M]#.reshape(lattice_dims)
    rho_down_err = rho_diag_err[M:]#.reshape(lattice_dims)

    sites = np.arange(M)

    # add in constant energy for consistency with SAFIRE convention
    plt.errorbar(x=sites,y=rho_up.real,yerr=rho_up_err.real,fmt='-o',color=f"C{avg}",label=r"diagonal of $\rho_\uparrow$" + f"{avg}")
    plt.errorbar(x=sites,y=rho_down.real,yerr=rho_down_err.real,fmt='--',color=f"C{avg}",label=r"diagonal of $\rho_\downarrow$" + f"{avg}")

    # save density for visualizations below!
    rho_ups[avg] = rho_up.reshape(lattice_dims)
    rho_downs[avg] = rho_down.reshape(lattice_dims)


plt.legend()
plt.show()
```

+++ {"id": "QOZdjPwg2JPg"}

## Visualizing Observables on the Lattice

We used the `afqmctools.utils.visualize` module's `plot_lattice()` function earlier in order to visualize the lattice itself.
The same function allows us to colorize each lattice site based on some observables.
Below, we plot the charge density, then the spin density on the lattice.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 701
id: 8I7F7yFX22Ya
outputId: 3eb88439-52df-4ab6-989f-edb3fc5d0aad
---
import afqmctools.utils.visualize as vis

avg_to_plot = 2

rho_up = rho_ups[avg_to_plot]
rho_down = rho_downs[avg_to_plot]

rho_charge = rho_up + rho_down

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    vmin=0.0,
    vmax=2.0
)
```

```{code-cell} ipython3
import afqmctools.utils.visualize as vis

avg_to_plot = 1

rho_up = rho_ups[avg_to_plot]
rho_down = rho_downs[avg_to_plot]

rho_spin = 0.5*(rho_up - rho_down)

vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    vmin=-0.5,
    vmax=0.5
)
```
