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

+++ {"id": "82e12143-bd3a-4c82-9a51-bcf96abd6950"}

# Computing general observables

<b>Goal:</b>
Become acquainted with how to compute general observables with SAFIRE.

## What you will learn

1. What an "estimator" is and how to set one up in the json input file
2. What a mixed estimator is and when to use one to compute an observable
3. What a back-propagated estimator is and when to use one to compute an observable
4. How to compute observables in post-processing with afqmctools

## Introduction

AuxiliaryFiels uses "estimators" to compute observables.
Each estimator corresponds to a formal method for computing an observable - i.e. mixed estimators, back-propagated estimators, etc. - We will explain each of the these in the next section.
As we mentioned in {doc}`../02_understanding_the_input_file/02_understanding_the_input_file`,

you can add estimators to an AFQMC calculation using an "estimator" input block.
By default, an "energy" estimator is included which is a specialized mixed estimator.
Other estimators allow you to specify one or more additional observables to compute.

## Formal Background

Although we have not seen this explicitly, we have, so far, computed the ground state energy with AFQMC using a so-called
mixed estimator of the form

$$
\langle \hat{O} \rangle_\mathrm{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_\mathrm{T} | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_\mathrm{T} | \Phi_{n,k} \rangle}
$$

where $n$ is the projection step index,
$| \Phi_{n,k} \rangle$ are Slater determinant random walkers with weight $W_{n,k}$ (from importance sampling),
and $| \Psi_\mathrm{T} \rangle$ is the trial wavefunction.
The mixed estimator provides an unbiased estimate for any observable which commutes with the Hamiltonian.
For the AFQMC energy, $\hat{O} = \hat{H}$, and trivially $[ \hat{O}, \hat{H} ] = 0$;
therefore, the mixed estimator is an unbiased choice for the total ground state energy.

On the other hand, it is often interesting to compute observables which do not commute with the Hamiltonian.
For example, the one-body reduced density matrix, from which many properties can be computed, does not commute
with the Hamiltonian.
In these cases, a back-propagated estimator is necessary to mitigate the bias that would be incurred
by using a mixed-estimator.
The back-propagated estimator has the form,
$$
\langle \hat{O} \rangle_\mathrm{BP} = \frac{1}{\sum_k W_{n+m,k}} \sum_k W_{n+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{n,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{n,k}\rangle}
$$

where $| \Phi_{n,k} \rangle$ are the usual forward-projected Slater determinant random walkers,
and $| \tilde{\Phi}_{m,k} \rangle$ are the back-propagated walkers given by,
$$
| \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{n,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{n+m-1,k} ) | \Psi_\mathrm{T} \rangle.
$$
The index $n$ corresponds to the current forward projection step,
and $m$ is the back-propagated step index.
We note that each random walker has a corresponding back-propagated partner
which share the same path in auxiliary-field space.
It is important to note that the two Slater determinants are
distinct from each other since the propagator $B(x - \bar{x})$ is applied differently to each.
i.e. the propagator is applied to different Slater determinants when moving in the forward-, and backward-directions.

### ▶️ Run the cell below to setup the tutorial

```{code-cell} ipython3
:id: 616ca764-f070-434a-8146-e1511f273342

# Run me (shift+enter or click the play button) to setup the tutorial!
from pathlib import Path

# simple setup
from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "4d4d8135-bc2d-4caa-bbaf-c88ab045176c"}

## Setup the Test System

In this tutorial, we will use PySCF for convenience; however, you can use the knowledge that you've gained in the previous tutorials to use a different quantum chemistry code if desired.

Instead of the minimal basis Hydrogen dimer, we will use the nitrogen dimer at its equilibrium bond length.
We will be using the set of canonical RHF orbitals as the orthonormal basis for AFQMC.

The code block below will:

*   setup and run RHF using PySCF
*   generate and save the RHF Slater determinant to the SAFIRE HDF5 format
*   generate and save the Hamiltonian to the SAFIRE HDF5 format

These were all covered in previous tutorials.

![](files/NitrogenDimer.png)

### ▶️ Run the code block below to perform all of the steps above

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c844616d-45c5-498a-b6f9-c995027965e6
outputId: 973c839f-58d4-486f-9f5c-21f2cd93fec7
---
from pathlib import Path

import h5py as h5
import numpy as np
from pyscf import gto,scf,mcscf

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from safiretools import Hamiltonian
from afqmctools.inputs.from_hdf import write_json

from stats.scalar_dat import analyze_scalar_data

# Here, we'll generate the mol object

delta = 2.3 #2.118 # 3.0 # Bohr

atom = f'''
N 0. 0. {delta/2}
N 0. 0. -{delta/2}
'''
mol = gto.M(
    atom = atom,
    basis = '631g',
    unit = 'Bohr',
    verbose = 5,
)

rhf_chkfile = 'rhf_chkfile.h5'

rhf = scf.RHF(mol)
rhf.chkfile = scratch_dir / rhf_chkfile
rhf.run()


import numpy as np
from safiretools import NOMSDWavefunction

number_of_electrons = mol.nelec
number_of_orbitals = mol.nao_nr()

print("Number of orbitals = ", number_of_orbitals)

# Save the Hamiltonian. .to_hdf5() replaces only the Hamiltonian in the file,
#   so the Hamiltonian and the wavefunction can be written in either order.
basis_scf_data = load_from_pyscf_chk_mol(
    chkfile = scratch_dir / rhf_chkfile,
)

Hamiltonian.from_pyscf(
    basis_scf_data,
    chol_cut = 1e-5,
    verbose=True
).to_hdf5(scratch_dir / "afqmc.h5")

# now the trial wavefunction, appended to the same file
orbitals = np.eye(number_of_orbitals)

phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "afqmc.h5")
```

+++ {"id": "dSJ9e4tRBxxx"}

## The "Energy" Estimator

By default, SAFIRE will include an "energy estimator" in the calculation.
This is formally a mixed estimator where $\hat{O} = \hat{H}$.
The Hamiltonian trivially commutes with itself, so a mixed estimator of the energy is unbiased.

You can add an energy estimator input block to your json input file in order to customize its setting.
For example, in the input block,

```json
"estimator": {
  "name": "energy",
  "print_components": true,
  "overwrite": true
}
```

the "name" keyword tells us what kind of estimator we want.
In this case, we want an "energy" estimator.
We are setting the "print_components" and "overwrite" flags to true.
"print_components" tells SAFIRE to print the one-body, two-body coulomb, and two-body exchange contributions to the energy separately.
"overwrite" tells SAFIRE to overwrite the default energy estimator
with the one that we have defined.

### Most Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
|<b>setting</b>|<b>default</b>|<b>description</b>|
|--:|:-:|:--|
| <b>        print_components</b> |  False |  If True, print the one-body, two-body coulomb, and two-body exchange contributions to the energy in the `*.scalar.dat` file  |
| <b> equil_multiplier </b> | 0 |  The multiplier that determines the length of the equilibration phase. The equilibration phase length is computed as $(equilibration\_length) = (equil\_multiplier) * (population\_control\_inveral)$ where the population control interval is defined in the execute block. During the equilibration phase, observables are not computed. |
| <b>        overwrite</b> | False |   If True, overwrite the default energy estimator with the explicitly defined energy estimator |

### Less Common Settings

<style>
td, th {
   border: none!important;
}
</style>
  
|<b>setting</b>|<b>default</b>|<b>description</b>|
|--:|:-:|:--|
| <b>        print_sign</b> | False |  If True, print various quantities related to the sign/phase of random walkers in the `*.scalar.dat` file  |
| <b>        remove</b> |  False |   If True, remove the default energy estimator. **Note that the explicitly defined energy estimator will not be used in the calculation!** |
| <b>        truncate</b> |  False |   If True, truncate the energy such that data that fall outside of the range $[-3 var, 3 var]$, with $var$ being the variance, are set to either $-3 var$ or $3 var$, whichever is closer |

+++

### ▶️ Run the code block below to write an input file

```{code-cell} ipython3
:id: wOOIRUtfCIfh

from afqmctools.inputs.from_hdf import write_json

# some shared parameters for the AFQMC run
execute_options = {
    "walker_set" : {
        "walker_type" : "COLLINEAR"
    },
    "timestep": 0.01,
    "steps": 10000,
    "population_control_interval" : 10,
    "measure_interval_multiplier": 1,
    "walker_ortho_interval" : 10 ,
    "n_walkers_per_mpi_task": 20,
    "seed" : 42,
    "estimator" : {
        "name" : "energy",
        "print_components" : True,
        "overwrite" : True
    }
}

write_json(
    scratch_dir / "afqmc.json",
    fwfn0=scratch_dir / "afqmc.h5",
    walker_type="COLLINEAR",
    exec_opts=execute_options
)
```

+++ {"id": "Yr39jGz1DCyk"}

### 📝 Your Turn : Run SAFIRE with an explicit energy estimator block

Now you can run SAFIRE, and you should see the following column headers in the `*scalar.dat` file.

```
 block  time  nWalkers weight PseudoEloc LogOvlpFactor EnergyEstim__nume_real  EnergyEstim__nume_imag EnergyEstim__deno_real  EnergyEstim__deno_imag EnergyEstim__timer OneBodyEnergyEstim__nume_real EXXEnergyEstim__nume_real ECoulEnergyEstim__nume_real MixedEstim_timer Eshift freeMemory
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: k_zbz82RGF0K
outputId: ce74d680-635b-456e-a468-0e7ada8b10f7
---
run_afqmc(
    run_dir=scratch_dir,                # directory to run SAFIRE in - output files will be there as well
    input_file="afqmc.json",
    np=16,                              # number of MPI tasks
    output_file=None
)
```

+++ {"id": "h6GBStnB0mVH"}

You can check the `*.scalar.dat` file to verify that the extra columns that we requested were printed.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: rJfeETOmyKKU
outputId: acb22087-0747-4430-8dc9-03ef1ff115a2
---
!head -n 10 $scratch_dir/qmc.s000.scalar.dat
```

+++ {"id": "rrop2GLPGDmQ"}

## Extracting and Analyzing the AFQMC energy.

We learned about basic energy analysis using the CLI and the `analyze_scalar_data()` Python function in the Hello SAFIRE tutorial.
We can use the same tools here to get the average AFQMC energy and the stochastic uncertainty.

For pedagogical reasons, we will instead read the entire scalar data file so that we can plot several of the data columns as a function of total projection time.
In the code block below, the total energy (solid blue line) is plotted against as well as the sum of the three individual columns (dotted orange line) corresponding to the one-body energy, the Coulomb energy, and the exchange energy.
The two curves agree with each other to machine precision.

### ▶️ Run the code block below to extract and plot energy samples

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 472
id: HxCV-iXdDXP7
outputId: a6731663-ba05-44a0-bb1f-3e0474f02351
---
from types import SimpleNamespace
from stats.scalar_dat import analyze_scalar_data,read_scalar_table

# read scalar table and return as a pandas DataFrame
args = SimpleNamespace(
    fname=scratch_dir / "qmc.s000.scalar.dat",
    append=None,
    mark_header="#"
)
df = read_scalar_table(args)


# read the energy components
one_body_energy = df["OneBodyEnergyEstim__nume_real"].values
direct_coulomb_energy = df["ECoulEnergyEstim__nume_real"].values
exchange_energy = df["EXXEnergyEstim__nume_real"].values

two_body_energy = direct_coulomb_energy + exchange_energy

# read the total energy
total_energy = df["EnergyEstim__nume_real"].values

# plot the energy both ways
import matplotlib.pyplot as plt

plt.plot(total_energy,label="Total Energy from energy Estimator")
plt.plot(one_body_energy + direct_coulomb_energy + exchange_energy, ":", label=r"$E_1 + E_{Coul} + E_{x}$")

plt.legend()
plt.xlabel("Sample")
plt.ylabel(r"$\Delta E \equiv E_\mathrm{MixedEstimator} - E_{EnergyEstimator}$")
plt.title(r"Total energy vs. sample")

plt.show()
```

+++ {"id": "yzpB-AJrAZ24"}

## Computing Observables using a "Mixed" Estimator

SAFIRE also includes a mixed estimator that periodically evaluates,

$$
\langle \hat{O} ⟩_\mathrm{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_\mathrm{T} | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_\mathrm{T} | \Phi_{n,k} \rangle},
$$
where $\hat{O}$ is an operator.
The one-body reduced density matrix (1rdm), given by

$$\hat{O} = \hat{\rho}^1 = \sum_{ij}\sum_{\sigma \sigma'} \hat{c}^\dagger_{i \sigma}\hat{c}_{j \sigma'},$$

and the two-body reduced density matrix (2rdm), given by,

$$\hat{O} = \hat{\rho}^2 = \sum_{ijkl}\sum_{\sigma \sigma'}\hat{c}^\dagger_{i\sigma}\hat{c}^\dagger_{j\sigma'}\hat{c}_{k\sigma'}\hat{c}_{l\sigma}$$

are common observables.

<div class="alert alert-block alert-info">
<b>Note:</b>
    computing the 2rdm is very time and memory intensive; this should only be done for small systems.
</div>

### Specifying Observables to Measure

You can specify which observables you want to compute by
adding a corresponding observable block to an estimator
block.
For example, the estimator block below includes both the 1rdm, and the 2rdm.

```json
"estimator" : {
    "name" : "mixed",
    "onerdm" : {
        "name" : "my_onerdm"
    },
    "twordm" : {
        "name" : "my_twordm"
    }
}
```

As we will see later, this is exactly how observables are specified for the BP estimator as well.
See the [User Manual](https://users.flatironinstitute.org/~beskridge/safire/user_manual/observables.html#observables) for a complete list of currently implemented observables.

### Pedagogical Demonstration

Below, we will use a mixed estimator to compute the 1-rdm, and the 2-rdm.
We can then directly compute each contribution to the AFQMC energy,

$$E_{one-body} = \sum_{ij} H^1_{ij} \rho^1_{ji} $$

$$E_{two-body} = \sum_{ijkl} H^2_{ijkl} \rho^2_{ijkl} $$

and compare with the results from the energy estimator.
**We should find exact agreement between the two if we use the same AFQMC run parameters.**

We note that computing the 2rdm is very memory intensive since many samples of the 2rdm will be saved.
It is not typically computed unless absolutely necessary.

### ▶️ Run the code block below to write an input file with mixed estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QcoDuWPjAZDx
outputId: 2e894444-7a35-4787-c654-cc4f84f6ffd6
---
from afqmctools.inputs.from_hdf import write_json

# some shared parameters for the AFQMC run
execute_options = {
    "walker_set" : {
        "walker_type" : "COLLINEAR"
    },
    "timestep": 0.01,
    "steps": 10000,
    "measure_interval_multiplier": 1,
    "population_control_interval" : 10,
    "walker_ortho_interval" : 10 ,
    "n_walkers_per_mpi_task": 20,
    "seed" : 42,
    "estimator" : {
        "name" : "energy",
        "print_components" : True,
        "overwrite" : True
    },
    "estimator" : {
        "name" : "mixed",
        "onerdm" : {
            "name" : "my_onerdm"
        },
        "twordm" : {
            "name" : "my_twordm"
        }
    }
}

write_json(
    scratch_dir / "afqmc_1.json",
    fwfn0=scratch_dir / "afqmc.h5",
    exec_opts=execute_options,
    series=1
)
```

+++ {"id": "TrVqnMi4AxuT"}

Now, we can run AFQMC with a Mixed Estimator in order to compute the 1-rdm.
**This will probably take a while!** To finish the tutorial more quickly, remove the "twordm" block from the input and just compute the one-body energy.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 6fnE-_O_HUgs
outputId: 47d9ccce-e999-47e9-f36c-2a4005cd87a0
---
run_afqmc(
    run_dir=scratch_dir,                # directory to run SAFIRE in - output files will be there as well
    input_file="afqmc_1.json",
    np=16,                              # number of MPI tasks
    output_file=None
)
```

+++ {"id": "fgJ_Pp2EA7oA"}

### Data Extraction and Analysis

afqmctools provides several tools for extracting data - i.e. retrieving data from the SAFIRE output files - and analyzing data - arriving at a final average with stochastic uncertainty.

Since some observables require a large amount of memory to store, `afqmctools` also provides "transforms" which are applied to the raw observables as they are read from disk.
For example, in the cell below, a energy evaluation transform is used to compute the one-body energy, a scalar, on the fly as one-rdm samples are read from the *.h5 data file.
This dramatically reduces the memory needed to analyze the AFQMC results from SAFIRE.

### ▶️ Run the code below to evaluate the one-body energy from the one-rdm, and compare with the energy estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 544
id: c9LgOOxOA_xI
outputId: 0305b057-ca8b-4c5d-e70c-7cd43158dc5a
---
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from afqmctools.analysis.extraction import extract_observable,get_metadata
from afqmctools.analysis.transform import hermitize_factory,eval_one_body_obs_factory,eval_two_body_obs_factory

from afqmctools.analysis.average import WALKER_TYPE

# read the Hamiltonian
H = Hamiltonian.from_hdf5(scratch_dir / "afqmc.h5")
M = H.nmo
Econst = H.enuc

walker_type = WALKER_TYPE[get_metadata(scratch_dir/"qmc.s001.stat.h5")["WalkerType"]]

# set up a transform to compute the one-body energy from the one-rdm
eval_one_body_energy = eval_one_body_obs_factory(
    one_body_operator=H.hcore,
    walker_type=walker_type
)
# set up a transform the hermitize the one-rdm
hermitize = hermitize_factory(
    walker_type=walker_type
)


# Extract and Transform data
Energy_1body = extract_observable(
    scratch_dir/"qmc.s001.stat.h5",    # filename
    estimator='mixed',
    name="one_rdm",                    # observable type
    ix=0,                              # index of the "average" - we'll return to this below!
    transform=[hermitize,eval_one_body_energy],  # list of transforms to apply *FROM LEFT TO RIGHT*
)
# add in constant energy for consistency with SAFIRE convention
Energy_1body += Econst


plt.plot(one_body_energy,label="1-body energy from EnergyEstimator")
plt.plot(Energy_1body,'--',label="1-body energy (Mixed estimator)")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("Total energy")
plt.title("Total energy vs. sample")
plt.show()
```

### ▶️ Run the code block below to plot the difference between the mixed, and energy estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 598
id: snk1U2F7JtDc
outputId: 0b701347-7f3e-41da-f6d6-3f038105d649
---
# now, look at the difference!
delta_E = Energy_1body[:,0] - one_body_energy

# plot the difference between the two energies
plt.plot(delta_E,label="1-body energy difference")
plt.legend()
plt.xlabel("Sample")
plt.ylabel(r"$\Delta E \equiv E_\mathrm{MixedEstimator} - E_{EnergyEstimator}$")
plt.title(r"$\Delta E_{1-body}$ vs. sample")

plt.show()
```

+++ {"id": "wDMvF0aT04_x"}

As we can see, the energy computed using the one-body reduced density matrix with a mixed estimator yields the exact same one-body energies as the energy estimator to very high precision.
This is no surprise since the energy estimator is simply a specialized mixed estimator.

### 📝 Your Turn : Extract the 2rdm and compute the 2-body energy

Note: You can skip this section if you chose to not compute the "twordm"
observable above.

Repeat the same exercise for the two-body energy.
As we saw in the energy estimator section, the exchange and direct coulomb energies are printed to file separately by the energy estimator.

To compute the two-body energy, you will need to measure the "twordm" observable during AFQMC, and extract and transform it.

We provide code for setting up the appropriate transform functions in the code block below.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: afFNaJpQdymb
outputId: 56bc5ba2-1416-4468-ec0e-d7e74a7e87fc
---
import numpy as np
import matplotlib.pyplot as plt

from afqmctools.analysis.transform import hermitize_factory,eval_one_body_obs_factory,eval_two_body_obs_factory

ncv = H.chol.shape[-1]
cholesky_vectors = np.transpose(H.chol.reshape((M,M,ncv)),(2,0,1))
eri = np.einsum("gil,gjk->iljk",cholesky_vectors,cholesky_vectors).flatten()

eval_two_body_energy = eval_two_body_obs_factory(
    two_body_operator=eri,  # 📝 YOUR TURN: Supply the correct two-body operator
    walker_type=walker_type
)

Energy_2body_mixed = extract_observable(
    scratch_dir/"qmc.s001.stat.h5",
    estimator='mixed',
    name="two_rdm",
    ix=0,
    transform=eval_two_body_energy, # 📝 YOUR TURN: Supply the transform function to compute the two-body energy,
    batch_size=1
)

plt.plot(two_body_energy,label="Two-body energy from energy estimator")
plt.plot(Energy_2body_mixed,":",label="Two-body energy from mixed estimator")
plt.legend()
plt.xlabel("Sample")
plt.ylabel(r"Energy $E_{Ha}$")
plt.title("Two-body energy vs. sample")
plt.show()


# plot the difference as well!
delta_E = Energy_2body_mixed[:,0] - two_body_energy

# plot the difference between the two energies
plt.plot(delta_E,label="two-body energy difference")
plt.legend()
plt.xlabel("Sample")
plt.ylabel(r"$\Delta E \equiv E_\mathrm{MixedEstimator} - E_{EnergyEstimator}$")
plt.title(r"$\Delta E_{2-body}$ vs. sample")

plt.show()
```

+++ {"id": "NjutJDKY4B_m"}

Once again, we see that the two body energy computed from the two-body reduced density matrix samples is the same as the two body energy from the energy estimator (after adding the exchange and direct Coulomb contributions together, of course).

+++ {"id": "9bKC3Q6G5Mx9"}

## Averaging Observables

Averaging observables computed using AFQMC requires some care.

First, one must make sure that the data is equilibrated.

Second, in AFQMC, successive samples are not independent since they depend on the recent history of each walker.
This is known as "autocorrelation".
Care must be take to remove autocorrelation effects or the stochastic uncertainty will be underestimated.

The former can be achieved by visualizing the observable over imaginary time, and finding the equilibration time by inspection.
We saw this in {doc}`../01_hello_safire/01_hello_safire`.
afqmctools provides tools for automatically computing the auto-correlation time - i.e. the time interval at which measurements are no longer autocorrelated - and adjusting the number of effective samples accordingly.

In the code cell below, we demonstrate how to use the provided tools to automatically compute averages.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: ca0ef2b4
outputId: 260beb99-16d6-4e60-e2b9-16665ef099af
---
from afqmctools.analysis.average import average_observable

E1body = average_observable(
    filename=scratch_dir/"qmc.s001.stat.h5",
    name = "one_rdm",
    eqlb=20, # EXERCISE: Set this to an appropriate value based on the energy vs projection time curve above.
    estimator='mixed',
    ix=0,
    blocksize=1,
    transform=[hermitize,eval_one_body_energy],
    remove_auto_correlation=True # you can turn this off if you prefer to remove autocorrelation yourself.
)

## compare with the analyze_scalar_stats result:
from stats.scalar_dat import analyze_scalar_data

analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    series_column = "time",
    nequil = 5,
    trace = True
))
```

+++ {"id": "6a1baf12-288e-4f8f-afaf-6f9d83e899d4"}

## Back-Propagation (BP) in SAFIRE

Mixed estimators are biased for observables $\hat{O}$ such that $[\hat{O},\hat{H}] \neq 0$, which requires the use of pure estimators. The Back-Propagation (BP) algorithm is used to compute pure estimators efficiently in SAFIRE.

### Basic BP Configuration

BP is invoked in SAFIRE by including a "back_propagation" estimator in the input file:

```json
{
  "estimator": {
    "name": "back_propagation",
    "path_restoration": true,
    "bp_walker_ortho_interval": 5,
    "measure_interval_multiplier": 20,
    "equil_multiplier": 100,
    "onerdm": {
      "name": "one_rdm"
    }
  }
}
```

Observables are added to the BP block in the same way as for mixed estimators.

### Measurement Intervals

Since BP involves an additional projection, the measurement interval controls the number of back-propagated steps.
Just as in the execute block or a mixed estimator block, the input file includes a measurement interval "multiplier", and the actual measurement interval, $m$, is determined as:

$$
m = (\text{measure\_interval\_multiplier}) \times (\text{population\_control\_interval})
$$

### Multiple BP Lengths

SAFIRE implements the capability of running BP with multiple BP measurement lengths.
We call each measurement with a different BP length an "average".
This allows the forward projection to be reused while checking for convergence in the BP length.
To define multiple BP lengths/averages, simply provide a JSON array for measure_interval_multiplier corresponding to all of the BP lengths that you would like to use:

```json
{
  "estimator": {
    "name": "back_propagation",
    "path_restoration": true,
    "bp_walker_ortho_interval": 10,
    "measure_interval_multiplier": [60, 70, 80],
    "equil_multiplier": 80,
    "onerdm": {
      "name": "one_rdm"
    }
  }
}
```

### Equilibration Phase

The BP estimator implements an equilibration phase at the beginning of the AFQMC calculation where no BP is performed.
This is recommended since computing observables is expensive and since AFQMC needs to equilibrate before samples can meaningfully contribute to the average.
Similarly to the measure_interval_multiplier, the equilibration time is specified as an equilibration multiplier (`"equil_multiplier"`) and the actual equilibration time is given by:

$$
\textrm{equil\_time} = (\textrm{equil\_multiplier}) \times (\textrm{population\_control\_interval})
$$

### All Settings

<style>
td, th {
   border: none!important;
}
</style>
  
|<b>setting</b>|<b>default</b>|<b>description</b>|
|--:|:-:|:--|
| <b>        measure_interval_multiplier</b> |   1  |  controls the number of back propagation steps which are determined as $(\textrm{number of BP steps}) = (\textrm{measure\_interval\_multiplier}) \times (\textrm{population\_control\_interval})$. Can be either a single value or multiple values. If multiple values are provided, BP will be performed with each corresponding number of BP steps and observables are accumulated and saved for each distinct BP length.  |
| <b> equil_multiplier </b> | 0 |  The multiplier that determines the length of the equilibration phase. The equilibration phase length is computed as $(\textrm{equilibration\_length}) = (\textrm{equil\_multiplier}) * (\textrm{population\_control\_inveral})$ where the population control interval is defined in the execute block. During the equilibration phase, observables are not computed. |
| <b>        bp_walker_ortho_interval</b> |   1  |   interval for performing walker orthonormalization during back-propagation (i.e. for the left-hand side Walkers)  |
| <b>        path_restoration</b> |   false  |  If true, perform path restoration.  |
| <b>        extra_path_restoration</b> |   false  |   If true, perform an extra path restoration.  |

### Demonstration of BP

For the sake of execution time, we will compute only the one-rdm using the back-propagation estimator.
We will find that the one-body energy computed using the BP one-rdm will *not* agree with the one computed using a mixed estimator.
The BP-estimated one-body energy is unbiased, and represents a better estimate of the one-body energy than that provided by the mixed estimator.

+++

### ▶️ Run the code block below to write an input file with a BP Estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 41e0454a
outputId: fbf1e8a4-bc84-40c5-9135-0d6a6571b8e8
---
# Example: Setting up a BP calculation with multiple BP lengths
from afqmctools.inputs.from_hdf import write_json

# Create a scratch directory for BP calculations
scratch_dir_bp = scratch_dir / "bp"
scratch_dir_bp.mkdir(parents=True, exist_ok=True)

# Copy the Hamiltonian and wavefunction files
import shutil
shutil.copy(scratch_dir / "afqmc.h5", scratch_dir_bp / "afqmc.h5")

# Set up AFQMC parameters with back-propagation
execute_options_bp = {
    "walker_set": {
        "walker_type": "COLLINEAR"
    },
    "timestep": 0.005,
    "steps": 20000,
    "measure_interval_multiplier": 2,
    "population_control_interval": 5,
    "walker_ortho_interval": 10,
    "n_walkers_per_mpi_task": 100,
    "seed": 42,
    "estimator": {
        "name": "back_propagation",
        "path_restoration": True,
        "bp_walker_ortho_interval": 5,
        "measure_interval_multiplier": [1, 2, 3],  # Multiple BP lengths
        "equil_multiplier": 30,  # Equilibration phase
        "onerdm": {
            "name": "one_rdm"
        }
    }
}

write_json(
    scratch_dir_bp / "afqmc_bp.json",
    fwfn0=scratch_dir_bp / "afqmc.h5",
    walker_type="COLLINEAR",
    exec_opts=execute_options_bp,
    series=2
)

print("Back-propagation input file created with the following BP parameters:")
print(f"- Multiple BP lengths: {execute_options_bp['estimator']['measure_interval_multiplier']}")
print(f"- Equilibration multiplier: {execute_options_bp['estimator']['equil_multiplier']}")
print(f"- Population control interval: {execute_options_bp['population_control_interval']}")
print(f"- Actual BP lengths: {[m * execute_options_bp['population_control_interval'] for m in execute_options_bp['estimator']['measure_interval_multiplier']]}")
print(f"- Equilibration time: {execute_options_bp['estimator']['equil_multiplier'] * execute_options_bp['population_control_interval']}")
```

### ▶️ Run SAFIRE with BP

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 918df5d4
outputId: cb9daf3b-71f6-46c9-a100-bfd1ae9323f4
---
# Run AFQMC with back-propagation
print("Running AFQMC with back-propagation estimator...")
print("This will take longer than mixed estimators due to the additional back-propagation steps.")

run_afqmc(
    run_dir=scratch_dir_bp,
    input_file="afqmc_bp.json",
    np=16,
    output_file=None
)
```

### ▶️ Run the code below to evaluate the one-body energy from the one-rdm, and compare with the BP estimator

```{code-cell} ipython3
:id: 39c23cf1

# Extract and analyze BP results
from afqmctools.analysis.extraction import extract_observable, get_metadata
from afqmctools.analysis.average import WALKER_TYPE
from afqmctools.analysis.transform import hermitize_factory

# Check the metadata to understand the walker type
metadata = get_metadata(scratch_dir_bp / "qmc.s002.stat.h5")
bp_metadata = get_metadata(scratch_dir_bp / "qmc.s002.stat.h5", path="/Observables/BackPropagated/")

walker_type = WALKER_TYPE[metadata["WalkerType"]]
print(f"Walker type: {walker_type}")

# set up a transform the hermitize the one-rdm
hermitize = hermitize_factory(
    walker_type=walker_type
)

# Extract one-RDM data from all BP averages
print("\nExtracting one-RDM data from back-propagation averages...")

# We specified 3 BP lengths:
bp_lengths = bp_metadata["BackPropSteps"]
bp_onerdm_data = {}

for iav, bp_length in enumerate(bp_lengths):
    print(f"Extracting average {iav} (BP length = {bp_length} * {execute_options_bp['population_control_interval']} = {bp_length * execute_options_bp['population_control_interval']} steps)")

    # Extract the one-RDM for this average
    onerdm_bp = extract_observable(
        scratch_dir_bp / "qmc.s002.stat.h5",
        estimator='back_propagated',
        name="one_rdm",
        ix=iav,  # Index of the BP average
        transform=[hermitize],  # Apply hermitization
        batch_size=10
    )
    bp_onerdm_data[iav] = onerdm_bp
    print(f"  Shape of one-RDM data: {onerdm_bp.shape}")

print(f"\nSuccessfully extracted one-RDM data from {len(bp_onerdm_data)} BP averages")
```

```{code-cell} ipython3
# Extract and analyze BP results
from afqmctools.analysis.extraction import extract_observable, get_metadata
from afqmctools.analysis.average import WALKER_TYPE
from afqmctools.analysis.transform import hermitize_factory
# Check the metadata to understand the walker type
metadata = get_metadata(scratch_dir_bp / "qmc.s002.stat.h5")
bp_metadata = get_metadata(scratch_dir_bp / "qmc.s002.stat.h5", path="/Observables/BackPropagated/")

walker_type = WALKER_TYPE[metadata["WalkerType"]]
print(f"Walker type: {walker_type}")

# set up a transform the hermitize the one-rdm
hermitize = hermitize_factory(
    walker_type=walker_type
)

bp_lengths = bp_metadata["BackPropSteps"]
naverages = len(bp_lengths)
bp_energy = np.zeros((naverages),dtype=np.complex128)
bp_energy_err = np.zeros((naverages),dtype=np.complex128)

for iav, bp_length in enumerate(bp_lengths):
    print(f"Extracting average {iav} (BP length = {bp_length} steps)")

    # Extract the one-RDM for this average
    energy_bp, denergy_bp = average_observable(
        scratch_dir_bp / "qmc.s002.stat.h5",
        estimator='back_propagated',
        name="one_rdm",
        eqlb=0, # equilibrated for beta = 4.0 during SAFIRE run!!
        ix=iav,  # Index of the BP average
        transform=[hermitize,eval_one_body_energy],  # Apply hermitization
        batch_size=10
    )
    bp_energy[iav] = energy_bp[0] + Econst
    bp_energy_err[iav] = denergy_bp[0]
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
from stats.scalar_dat import analyze_scalar_data

E_mixed, dE_mixed = analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    series_column = "time",
    column = "OneBodyEnergyEstim__nume_real",
    nequil = 10
))

bp_steps = bp_lengths * execute_options_bp['population_control_interval']
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# plot reference value from mixed
ax.fill_between([min(bp_steps)-1, max(bp_steps)+1],
                 E_mixed - dE_mixed, E_mixed + dE_mixed,
                 alpha=0.2, color='red')
ax.axhline(y=E_mixed, color='red', linestyle='--', alpha=0.7, label=f'Mixed Estimator')

# Plot convergence
ax.errorbar(bp_steps, bp_energy, yerr=bp_energy_err, marker='o', capsize=5, capthick=2, label='BP Estimator')

ax.set_xlabel('BP Length (steps)')
ax.set_ylabel('Mean One-body Energy (Ha)')
ax.set_title('BP Convergence with BP Length')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

+++ {"id": "d0f6abfa"}

### Key Takeaways from Back-Propagation Analysis

**When to Use BP vs Mixed Estimators:**

1. **Mixed Estimators**: Use for observables that commute with the Hamiltonian (like energy). They are computationally cheaper and provide unbiased estimates.

2. **Back-Propagation**: Required for observables that do not commute with the Hamiltonian (like the one-body reduced density matrix). BP provides pure estimators that are unbiased.

**Important BP Configuration Points:**

- **Equilibration**: Always include an equilibration phase (`equil_multiplier`) to allow AFQMC to reach equilibrium before BP measurements begin.

- **Multiple BP Lengths**: Use multiple BP lengths to check convergence. Longer BP lengths generally provide more accurate results but are more expensive. However, BP is unstable for long back propagation times. This is reflected in the large stochastic uncertainty for large BP length.

- **Computational Cost**: BP calculations are significantly more expensive than mixed estimators due to the additional back-propagation steps.

**Convergence Analysis:**

The plots above show how the BP estimator converges with increasing BP length. For observables that don't commute with the Hamiltonian, you should see:
- Different results between mixed and BP estimators (showing the bias in mixed estimators)
- Dependence of BP estimators on back-propagation length. It is important to note that BP does not necessarily converge and is unstable for large total imaginary time.

This demonstrates why BP is essential for computing unbiased estimates of observables like charge densities, spin densities, and other quantities derived from the one-body reduced density matrix.

+++ {"id": "PginJzpCI28-"}

## Summary

In this tutorial, you were acquainted with how to compute general observables with SAFIRE.

## What you learned

1. What a mixed estimator is and when to use one to compute an observable
2. What a back-propagated estimator is and when to use one to compute an observable
3. How to compute observables in post-processing with afqmctools
