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

# 4×4 Square Lattice Hubbard Model

Author: Kyle Eskridge

In this tutorial, we will perform AFQMC calculations for the Hubbard model on
a 4×4 lattice.
The Hamiltonian is given by,
$$
\hat{H} = -t \sum_{\langle i,j\rangle} \hat{c}^\dagger_i \hat{c}_j + U \sum_{i} \hat{n}^{\uparrow}_i \hat{n}^{\downarrow}_i,
$$
where $i,j$ are lattice site indices, angle brackets indicate nearest-neighbors,
$\hat{c}^\dagger_i$/$\hat{c}_i$ are electronic creation/annihilation operators,
and $\hat{n}^{\sigma}_i$ are spin-resolved number operators corresponding to site $i$.
The 4×4 lattice is small enough to make exact diagonalization possible.
We will use exact results for this system as reference data throughout
this tutorial.

The tutorial is structured as follows.
We will begin with an explanation of how to build lattice model Hamiltonians
using the `safiretools` Python package.
Next, we will walk through computing the ground state energy of this Hamiltonian for a particular filling and value of U,
comparing a couple of options for trial wavefunctions.
Next, we will continue the walk through by computing the ground state charge,
and spin density for the same filling and value of U.
Next, we will demonstrated how to obtain exact results using AFQMC
for the special case of half filling for which, if care is taken,
there is no sign problem.

+++ {"id": "dfgyQ5OC5xcn"}

## Building a Lattice Model Hamiltonian

`safiretools` provides a convenience lattice model Hamiltonian builder and writer which
will allow us to construct the standard Hubbard model with nearest-neighbor hopping and on-site U, among
other Hamiltonians.
The basic steps for using the lattice model Hamiltonian builder are:
1. build the lattice
2. build the lattice model Hamiltonian from the lattice and the Hamiltonian parameters

The lattice instance will also facilitate visualizing the charge and spin densities later on.

```{code-cell} ipython3
---
id: tNji__B25xcn
colab:
  base_uri: https://localhost:8080/
  height: 855
outputId: c0edb976-6ddc-4ada-eb2f-66c8587a93b7
---
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

from safiretools import Lattice
from safiretools import vis

# Step 1. build the lattice
lattice = Lattice.from_type(
    'square',
    (4, 4),                               # lattice dimensions along a1 and a2
    boundaries=('periodic', 'periodic'),  # boundary conditions along a1 and a2
    twists=(0.0, 0.0),                    # twist angles in radians along a1 and a2
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
from safiretools import LatticeHamiltonian

# set the number of electrons - we'll use this again
nelec = (8,8)

# Step 2. build the lattice model Hamiltonian
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t=1.0,
    U=4.0,
)
```

+++ {"id": "MqK7sb3Z5xcn"}

We now have a standard Hubbard model with nearest-neighbor hopping and on-site U.
Next, we'll walk through an AFQMC calculation where we'll compute the ground state energy.

+++ {"id": "9KnqEEIS5xcn"}

## AFQMC Walkthrough: Ground State Energy

Here, we will compute the ground state energy of the standard Hubbard model
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
it may be necessary to test which wavefunction is better by running AFQMC calculations with each,
and comparing the results.
In this walk through, we will compare free-electron trial wavefunctions
to unrestricted Hartree-Fock trial wavefunctions.

+++ {"id": "Clkg0_nE5xcn"}

### Free-Electron Trial Wavefunction

`safiretools` provides a convenience classmethod, `Wavefunction.from_free_electron()`,
for calculating free-electron trial wavefunctions.

At half filling this 4×4 lattice is open-shell: the highest occupied level is
six-fold degenerate and holds only three electrons per spin, so *which* three you
occupy is arbitrary. A small twist on the
lattice breaks that degeneracy and fixes the choice.

The twist is a property of the **lattice**, so it has to be applied when the
Hamiltonian is built. `from_free_electron()` uses whatever Hamiltonian it is given. 
We therefore build a *second*, Hamiltonian with a twist 
for the trial wavefunction alone; the untwisted one from step 2. is what AFQMC
actually runs, and is also what we hand the Hartree-Fock solver further down.
`from_free_electron()` warns if you hand it a Hamiltonian whose shells are still
degenerate at the fill level.

```{code-cell} ipython3
---
id: 4MY5lScg5xco
colab:
  base_uri: https://localhost:8080/
outputId: e6000516-1e4f-4ae3-db34-af0b0c4d03f7
---
# First, let's compute the Free-Electron trial wavefunction
from safiretools import Wavefunction

# a small irrational twist, in radians per axis, breaks the k-space degeneracy
twist = (1/3252409**0.5, 1/105976**0.5)

fe_lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'), twists=twist)
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, t=1.0, U=4.0)

wfn = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=nelec
)
```

+++ {"id": "-xW_O9f35xco"}

Now that we have computed a free-electron wavefunction, we can inspect the results to validate it.
Building a wavefunction and evaluating its energy are separate calls, so we compute the energy of the
free-electron trial wavefunction for the interacting Hamiltonian by hand.
For a single Slater determinant $\Phi$, the one-rdm is $\rho^\sigma_{ij} = \sum_n \Phi^{\sigma*}_{in} \Phi^\sigma_{jn}$,
which gives the one-body energy $E_\mathrm{K} = \sum_{\sigma ij} t^\sigma_{ij} \rho^\sigma_{ij}$
and the Hubbard energy $E_U = U \sum_i \rho^\uparrow_{ii} \rho^\downarrow_{ii}$.
For $N_{\uparrow} = N_{\downarrow} = 8$, the total energy is
$E = -8.006\,t$ with a one-body contribution of $E_\mathrm{K} = -24.006\,t$,
and a Hubbard U contribution of $E_U = 16.000\,t$.

Next, let's take a look at the charge and spin densities of the trial wavefunction.

```{code-cell} ipython3
---
id: Ctx8QQM65xco
colab:
  base_uri: https://localhost:8080/
  height: 701
outputId: a868b2dd-398f-40d5-90b7-9a8036f6659b
---
import numpy as np

# one-rdm of the free-electron determinant
alpha, beta = wfn.determinant(0)
rdm = np.array([alpha.conj() @ alpha.T, beta.conj() @ beta.T])

# the one-body term holds the spin-up and spin-down blocks one above the other
M = fe_lattice.N_sites
t_ij = fe_hamiltonian.get_one_body().toarray().reshape(2, M, M)
U_i = fe_hamiltonian.get_U().diagonal()

E_K = np.sum(t_ij * rdm).real
E_U = np.sum(U_i * rdm[0].diagonal() * rdm[1].diagonal()).real
print(f"E = {E_K + E_U:.6f}, E_K = {E_K:.6f}, E_U = {E_U:.6f}")

# get lattice dimensions from the Lattice instance
L = lattice.L

rho_up = rdm[0].diagonal().reshape(*L)
rho_down = rdm[1].diagonal().reshape(*L)

rho_charge = rho_up + rho_down

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
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
rho_spin = 0.5*(rho_up - rho_down)

vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
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
`safiretools` comes with a convenience function to generate a JSON job file as demonstrated below.
We have included reasonable AFQMC run parameters for this calculation below.

```{code-cell} ipython3
:id: FCd6cFpd5xco

from safiretools import write_jobfile

# make an input file, which also writes the wavefunction and the Hamiltonian to HDF5
write_jobfile("afqmc.json", {
    "seed": 42,                          # just for reproducibility
    "execute": {
        "wavefunction": wfn,
        "hamiltonian": hamiltonian,
        "timestep": 0.01,
        "steps": 12000,
        "equilibration_steps": 500,
        "num_walkers": 1600,
        "population_control_interval": 5,
        "walker_ortho_interval": 5,
        "measure_interval": 5,
        "propagator": {
            "use_cp_constraint": True,
            "project_force_bias": True
        }
    }
})
```

```{code-cell} ipython3
---
id: 5YJevKw85xco
colab:
  base_uri: https://localhost:8080/
outputId: 601aa3a3-9001-40d2-84fe-7cc725c17dff
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
---
id: 9_qLNbRM5xco
colab:
  base_uri: https://localhost:8080/
  height: 921
outputId: 6de71736-72b7-4efa-bc41-8e0985c0759d
---
import matplotlib.pyplot as plt
from safiretools import Results

E_exact = -13.62185

results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
plt.axhline(E_exact, color="crimson")
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
from autohf import AutoHFHamiltonian, solve_hf

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

results = solve_hf(
    hamiltonian=AutoHFHamiltonian(source=hamiltonian),
    settings=hf_settings
)

# the noncollinear orbitals are spinors, spin-up components first
orbitals, = results[0]["orbitals"]
uhf_wfn = Wavefunction.from_single_determinant(
    np.asarray(orbitals).reshape(2, lattice.N_sites, sum(nelec))
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
vis.plot_lattice(
    lattice,
    density=rho_spin,
    density_label="spin density",
    vmin=-0.5,
    vmax=0.5
)
```

```{code-cell} ipython3
:id: _9GEgLIt5xcp

# make an input file
write_jobfile("afqmc.json", {
    "seed": 42,                          # just for reproducibility
    "execute": {
        "wavefunction": uhf_wfn,
        "hamiltonian": hamiltonian,
        "timestep": 0.01,
        "steps": 12000,
        "equilibration_steps": 2000,
        "num_walkers": 1600,
        "population_control_interval": 5,
        "walker_ortho_interval": 5,
        "measure_interval": 5,
        "propagator": {
            "use_cp_constraint": True,
            "project_force_bias": True
        }
    }
})
```

```{code-cell} ipython3
---
id: DIYwZYvz5xcp
colab:
  base_uri: https://localhost:8080/
outputId: 5414bf9b-a425-4e44-c469-c1b29b5f34d1
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
---
id: nn3u7u8k5xcp
colab:
  base_uri: https://localhost:8080/
  height: 921
outputId: 78be3a24-0f0c-4577-cb48-6ad867cb1f1b
---
results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
plt.axhline(E_exact, color="crimson")
```

+++ {"id": "0vF3gnwWSgir"}

If you used all of our settings, you should get an output compatible with

```
Energy = -13.6144 ± 0.0042
```

Again, we expect to achieve very close to the exact energy for the Hubbard model at half filling which is -13.62185 for U/t=4.
We see much better agreement now than we did with the free-electron trial wavefunction before; however, we're still off by about 0.007 t, or about twice
the stochastic uncertainty of our result. Why do we not quite agree?

We have residual systematic errors to remove. Specifically, we will want
to remove the trotter error.
We can achieve this either by using a very small step size, or by extrapolating to $\tau = 0$.

Let's rerun with a smaller $\tau = 0.005$

+++ {"id": "Q-t-66BmVVn5"}

## Rerun AFQMC with a Smaller τ

When we decrease the step size, we will
also need to increase the number of steps, and
each interval involved in order to maintain the same
amount of statistics.

```{code-cell} ipython3
:id: 8IfICCwnVU8m

# make an input file
write_jobfile("afqmc.json", {
    "seed": 42,                          # just for reproducibility
    "execute": {
        "wavefunction": uhf_wfn,
        "hamiltonian": hamiltonian,
        "timestep": 0.005,
        "steps": 24000,
        "equilibration_steps": 4000,
        "num_walkers": 1600,
        "population_control_interval": 10,
        "walker_ortho_interval": 10,
        "measure_interval": 10,
        "propagator": {
            "use_cp_constraint": True,
            "project_force_bias": True
        }
    }
})
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3g8uLM4_V5Vo
outputId: fa772c06-bf39-46ac-e81d-e57e14cc7449
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 920
id: 8L0WKycOV9L4
outputId: 1d4d7374-5c21-43a7-eae5-46bafc88d52d
---
results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
plt.axhline(E_exact, color="crimson")
```

+++ {"id": "SVS5nLE7YvBi"}

Now, you should see an energy of about

```
Energy = -13.6184
```

which is off by about 0.0035 t compared with the exact energy, -13.62185. We can continue to smaller step sizes to remove the remaining disagreement.

+++ {"id": "ntTQjKcjU3_p"}

## Computing Observables with Back-Propagation

General second-quantized observables can be computed using AFQMC via the back-propagation algorithm.
See {ref}`backprop_estimator` for more information on back-propagated observables.
To turn on back-propagation, add a `backprop` block to the `estimators` of an `execute`
block in the input file.
Here is a sample input file where an ellipsis has been used to hide the rest of the input.

```json
{
  /* ... */

  "execute": {

    /* ... */

    "population_control_interval": 10,
    "equilibration_steps": 500,

    /* ... */

    "estimators": {
      "backprop": {
        "walker_ortho_interval": 5,
        "propagation_steps": [100, 150, 200],
        "onerdm": {}
      }
    }
  }
}
```

For back-propagation (BP), we will need to provide a BP length.
We will need to check for convergence in this parameter.
Notice that a BP estimator has no `measure_interval`: the back-propagation lengths, in steps, are
given by `propagation_steps`, and the longest of them doubles as the measurement interval. Specifying multiple length at once is useful to check the convergence within a single run.
Every BP length is saved as its own observable, `BackPropEstimator/Steps=<n>`, in the results file.

+++ {"id": "ZjxOUIHUQeuD"}

### Run AFQMC with Back-Propagation

Run SAFIRE using the following input file to compute the one-rdm using back-propagation.

```{code-cell} ipython3

bp_lengths = [200, 400, 600]

# make an input file
write_jobfile("afqmc_3.json", {
    "seed": 42,                          # just for reproducibility
    "execute": {
        "wavefunction": uhf_wfn,
        "hamiltonian": hamiltonian,
        "timestep": 0.005,
        "steps": 24000,
        "num_walkers": 1600,
        "population_control_interval": 10,
        "walker_ortho_interval": 10,
        "measure_interval": 10,
        "equilibration_steps": 4200,
        "propagator": {
            "use_cp_constraint": True,
            "project_force_bias": True
        },
        "estimators": {
            "backprop": {
                "walker_ortho_interval": 10,
                "propagation_steps": bp_lengths,
                "onerdm": {}
            }
        }
    }
})
```

```{code-cell} ipython3
---
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc_3.json
```

+++ {"id": "WFUPHBHt5xcp"}

### Analyze the AFQMC Output

SAFIRE saves all of its results in the `afqmc_3.results.h5` file.
We have already analyzed the energy using `safiretools.Results`.
We will now focus on analyzing the back-propagated observables.

+++ {"id": "-Y7bGG0xwFOh"}

## Average Observables

`safiretools.Results` reads the back-propagated one-rdm of each BP length from the results file.
SAFIRE writes the full matrix, with the spin-up block first for a noncollinear calculation.
The charge and spin densities follow from its diagonal, and `results.evaluate()` computes them
with a jackknife error that accounts for the autocorrelation of the samples.

In the code block below, we will plot the charge and spin densities for each BP length in order to check for convergence in BP length.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 517
id: fqLt304CwKks
outputId: 4593e5a7-2e66-446b-e2b1-483f5193b453
---
M = lattice.N_sites # get the number of sites to help with plotting

def densities(rdm):
    """Charge and spin density on each site, from the noncollinear one-rdm."""
    n = rdm[0].diagonal().real
    n_up, n_down = n[:M], n[M:]
    return np.stack([n_up + n_down, 0.5*(n_up - n_down)])

results = Results("afqmc_3.results.h5")

# densities and their errors for each BP length, saved for the visualizations below
rho, rho_err = {}, {}
sites = np.arange(M)
fig, (ax_charge, ax_spin) = plt.subplots(1, 2, figsize=(12, 4))
for steps in bp_lengths:
    rho[steps], rho_err[steps] = results.evaluate(densities, [f"BackPropEstimator/Steps={steps}/OneRDM"])

    ax_charge.errorbar(x=sites,y=rho[steps][0],yerr=rho_err[steps][0],fmt='-o',label=f"{steps} steps")
    ax_spin.errorbar(x=sites,y=rho[steps][1],yerr=rho_err[steps][1],fmt='-o',label=f"{steps} steps")

ax_charge.set_xlabel("site")
ax_charge.set_ylabel("charge density")
ax_spin.set_xlabel("site")
ax_spin.set_ylabel("spin density")
ax_charge.legend()
plt.show()
```

+++ {"id": "QOZdjPwg2JPg"}

## Visualizing Observables on the Lattice

We used `vis.plot_lattice()` earlier in order to visualize the lattice itself.
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
bp_to_plot = bp_lengths[-1]

rho_charge, rho_spin = rho[bp_to_plot]

vis.plot_lattice(
    lattice,
    density=rho_charge,
    density_label="charge density",
    vmin=0.0,
    vmax=2.0
)
```

```{code-cell} ipython3
vis.plot_lattice(
    lattice,
    density=rho_spin,
    density_label="spin density",
    vmin=-0.5,
    vmax=0.5
)
```
