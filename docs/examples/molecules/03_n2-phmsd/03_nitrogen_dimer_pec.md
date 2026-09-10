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

+++ {"id": "W94B8DFIE6lu"}

# N₂ molecule: bond stretching and multi-Slater trial wavefunctions

<b>Goal:</b>
This example demonstrates how to compute the potential energy curve (PEC) of a diatomic molecule
using an appropriate trial wavefunction.

## What you will learn

1. How to select the number of Slater determinants to include in the trial wavefunction, balancing computational cost and accuracy
2. How to automate the computation of a potential energy curve using afqmctools within Python

## Run the code block below to set up the example

```{code-cell} ipython3
:id: ljFeTy3vE6ls

from pathlib import Path

import h5py as h5
import numpy as np
from pyscf import gto,scf,mcscf,lib

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from safiretools import MolecularHamiltonian
from safiretools import Wavefunction
from afqmctools.inputs.from_hdf import write_json

from stats.scalar_dat import analyze_scalar_data

from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

num_mpi_tasks = 16 # adjust based on your resources
```

+++ {"id": "6ZBVQf-6E6lv"}

## The Nitrogen dimer

Diatomic molecules display strong static electron correlation effects (i.e. correlation due to many energetically proximate electron configurations)
in the intermediate bond length regime.
The $N_2$ molecule is a notorious example and has historically been used to benchmark the accuracy of quantum chemistry approaches [1].
Below is a plot of the error in the energy, relative to exact results from DRMG,
from reference [1].

<!---
<div>
<img src="./figs/NitrogenMolecule.png" width="500"/>
</div>
--->

![](files/ReferenceFigure.png)

We have include both the quantum chemistry "gold standard" CCSD(T) results, in orange, as well as CCSDT for reference.
There are two AFQMC curves, one using a UHF trial wavefunction (in pink), and
one using a truncated CASSCF(12o,6e) trial wavefunction (in violet).
For AFQMC, the stochastic uncertainty is smaller than the symbol size.

For bond lengths close to the equilibrium bond length of 2.118 Bohr,
AFQMC agrees with exact results to well within chemical accuracy (  $ 1 kcal/mol ≈ 1.6 mE_{Ha}$) regardless of the trial wavefunction used.
For intermediate and stretched bond lengths, the trial wavefunction from MCSCF (in this case, from CASSCF(12o,6e)) is necessary to chemical accuracy using AFQMC across the PEC.
Still, for all nearly all bond lengths, AFQMC with a UHF trial wavefunction produces more accurate results,
in terms of absolute error, than either the quantum chemistry "gold standard" method CCSD(T) or CCSDT.
AFQMC with a CASSCF(12o,6e) trial wavefunction outperforms both CCSD(T) and CCSDT in terms of accuracy at all bond lengths.

Next, we will reproduce the PEC of the Nitrogen dimer using AFQMC / CASSCF(12o,6e) trial wavefunctions.

### References:

[1] W. A. Al-Saidi, S. Zhang, H. Krakauer. "Bond breaking with auxiliary-field quantum Monte Carlo." *J. Chem. Phys.* **127**, 144101 (2007).

+++ {"id": "ESezCWGWE6lv"}

## Converging AFQMC in the quality of the trial wavefunction

As we saw in {doc}`../02_B_atom_SHCI_trial_wfn/06_SHCI_trial_wavefunction`, AFQMC trial wavefunctions are typically multi-Slater determinant expansions of the form form

$$
| \Psi_\mathrm{T} \rangle = \sum_n^{N_\mathrm{det}} C_{n} | \Phi_n \rangle,
$$

where $ C_{n} $ are CI coefficients,
and $| \Phi_n \rangle$ are Slater determinants.
Typically, CI trial wavefunctions are truncated CASSCF or selected CI (SCI) wavefunctions.

For quantum chemistry, AFQMC uses the phaseless approximation in order to control the fermionic sign / phase problem at the cost of introducing a removable bias.
This bias can be systematically removed by using a better trial wavefunction.
If we start with a decent CASSCF or SCI wavefunction, then we can characterize the "quality" of a trial wavefunction by the number of Slater determinants that are included in the trial wavefunction, $N_\mathrm{det}$.

<b>A critical step in achieving high-accuracy results, therefore, is to converge the AFQMC energy in $N_\mathrm{det}$</b>

SAFIRE provides a keyword in the "wavefunction" json input block to limit the number of Slater determinants, "ndets_to_read", that are actually read from the HDF5 wavefunction file. Here is a sample json input block.

```json
{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0
    },
    "execute": {
      "walker_set": {
        "walker_type": "COLLINEAR"
      },
      "wavefunction": {
        "filename": "your_wfn_file.h5",
        "ndets_to_read": 500
      },
      "timestep": 0.01,
      "steps": 10000,
      "n_walkers_per_mpi_task": 100,
      "seed": 42
    }
  }
}
```

In this case, a maximum of 500 Slater determinants will be read from "your_wfn_file.h5" regardless of how many Slater determinants were
written to the file.
We will save significantly more Slater determinants to the wavefunction HDF5 file than we expect to need.
This allows us to reuse the same wavefunction file while controlling the trial wavefunction quality via the "ndets_to_read" parameter.

+++ {"id": "IQBeUJTghyNm"}

### Getting an orthonormal basis and a trial wavefunction

We will now setup an AFQMC calculation. As you should now be familiar with from previous tutorials, we will need to choose an orthonormal orbital basis to work within.
Since we will be using CASSCF trial wavefunctions, it is convenient to use the set of optimized CASSCF orbitals,
$$
\phi_i (\vec{r}) = \sum_μ C_{\mu i} g_{\mu} (\vec{r}),
$$
as a basis.

<b>Run the code block below in order to run CASSCF in PySCF</b>
(alternatively, run your favorite Quantum Chemistry code, and save the CASSCF wavefunction in the wavefunction HDF5 file).

```{code-cell} ipython3
# Here, we'll generate the mol object

delta = 3.0 # Bohr

atom = f'''
N 0. 0. {delta/2}
N 0. 0. -{delta/2}
'''
mol = gto.M(
    atom = atom,
    basis = 'ccpvdz',
    unit = 'Bohr'
)

# run CASSCF to get an orbital basis AND a CAS wavefunction
casscf_chkfile = scratch_dir / 'rhf_chkfile.h5'
casscf_chkfile.unlink(missing_ok=True) # delete if already exists

rhf = scf.RHF(mol)
rhf.chkfile = casscf_chkfile
rhf.run()
mc = mcscf.CASSCF(rhf, 12, 6).run() # (12o,6e)
# save the ci to the chkfile too
lib.chkfile.save(mc.chkfile, 'mcscf/ci', mc.ci)

# write the CAS wavefunction to a file
Wavefunction.from_pyscf_cas(
    mol=mol,
    cas_chkfile= casscf_chkfile,
    tol=1.0e-3, # use a small value to save *many* determinants.
    max_det=2000
).to_hdf5(scratch_dir / 'afqmc.h5')
```

+++ {"id": "iIgGy3gjiv0l"}

### Checkpoint

At this point, you should have a CASSCF trial wavefunction saved in an SAFIRE HDF5 file (we used "afqmc.h5" as our filename, but you are free to use whatever name you would like).

You will also need the CASSCF orbitals in order to represent the Hamiltonian in that basis.
In PySCF, these are simply saved within the Checkpoint (*.chk) file.

+++ {"id": "90fQdnFHHAWP"}

## Save the Hamiltonian

We have already chosen the set of CASSCF orbitals as a trial wavefunction.
We will now build the Hamiltonian in the basis of CASSCF orbitals, and save it to an SAFIRE HDF5 Hamiltonian file.

afqmctools provides a helper function, `load_from_pyscf_chk_mol()`,  to read all of the necessary information from a PySCF checkpoint file.
`MolecularHamiltonian.from_pyscf()` then builds the Hamiltonian from it, and `.to_hdf5()` writes it out.
`.to_hdf5()` replaces only the Hamiltonian in the file it is given, so it can be added to the
file that already holds the trial wavefunction.

<div class="alert alert-block alert-info">
<b>Note:</b>
    By default, load_from_pyscf_chk_mol() will try to read data from an "scf" group in a PySCF checkpoint file. Set the "base" parameter to "mcscf" to read a CASSCF wavefunction.
</div>

```{code-cell} ipython3
# Save the Hamiltonian
basis_scf_data = load_from_pyscf_chk_mol(
    chkfile = casscf_chkfile,
    base = 'mcscf'
)

# write Hamiltonian
MolecularHamiltonian.from_pyscf(
    basis_scf_data,
    chol_cut = 1e-5,
    verbose=True
).to_hdf5(scratch_dir / 'afqmc.h5')
```

+++ {"id": "I9k2Vt5g2ZiT"}

### Checkpoint

In addition to the trial wavefunction HDF5 file from before, you should also now have a Hamiltonian HDF5 file. We saved both the trial wavefunction and the Hamiltonian into a file called "afqmc.h5", but you can use whatever file name(s) you would like. Simply update the code blocks based on your filenames.

+++ {"id": "hDAoo29kE6lv"}

### Run SAFIRE

Before we move on to computing the PEC, we need to converge the AFQMC energy in $N_\mathrm{det}$.
To avoid doing this for every point on the PEC, we will simply choose a point on the PEC where we expect the most static correlation, converge the AFQMC energy in $N_\mathrm{det}$ at that point, and use that converged $N_\mathrm{det}$ value uniformly across the entire PEC.
This avoids running many calculations at each point on the PEC to check convergence at the cost of perhaps too many Slater determinants at some points
on the PEC.

<b>We will automate this using Python and the tutorial helper function, run_afqmc().</b>

Again, we will use the "ndets_to_read" keyword in the "wavefunction" input block to limit the number of Slater determinants.
In the code block below, you will notice that we are setting,

```python
execute_options["wavefunction"] = { "ndets_to_read" : ndet }
```

in the "execute_options" dictionary.
`write_json()` will ***append*** settings within an existing input block when they are passed in this way.
For example,

```
write_json("afqmc.json","wfn.h5",exec_opts={ "ndets_to_read" : 500 })
```

will generate a "wavefunction" block that points to the wavefunction file, with the following contents:

```json
{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0
    },
    "execute": {
      "walker_set": {
        "walker_type": "COLLINEAR"
      },
      "wavefunction": {
        "filename": "afqmc.h5",
        "ndets_to_read": 500
      },
      "timestep": 0.01,
      "steps": 10000,
      "n_walkers_per_mpi_task": 10
    }
  }
}
```

<b>Run the code block below in order to compute $E_\mathrm{AFQMC}$ vs $N_\mathrm{det}$</b>
Note: This might take some time. You might want to run this on a computing cluster instead.

```{code-cell} ipython3
# some shared parameters for the AFQMC run
execute_options = {
    "timestep": 0.01,
    "steps": 10000,
    "measure_interval_multiplier": 1,
    "population_control_interval" : 10,
    "walker_ortho_interval" : 10 ,
    "n_walkers_per_mpi_task": 20,
    "seed" : 42
}


ndets = [50,250,500,750] # list of Ndet values to run
energies = []
stochastic_uncertainties = []
# compute E(N_det) for the CAS trial wavefunction
for ndet in ndets:

    # make a sub-scratch dir for better book keeping
    scratch_dir_local = scratch_dir / f"Ndet_{ndet}"
    scratch_dir_local.mkdir(parents=True, exist_ok=True)

    execute_options["wavefunction"] = { "ndets_to_read" : ndet }

    write_json(
        scratch_dir_local / "afqmc.json",
        fwfn0=scratch_dir / "afqmc.h5",
        exec_opts=execute_options
    )
    # run AFQMC
    run_afqmc(
        run_dir=scratch_dir_local,
        run_mode="local_cpu",
        timeout_mins=100,
        input_file="afqmc.json",
        np=num_mpi_tasks,         # number of MPI tasks
        output_file="afqmc.out"   # optionally direct output to file instead of here
    )

    settings = dict(
        fname = scratch_dir_local/"qmc.s000.scalar.dat",
        xaxis = "time",
        nequil = 5,
        trace = True
    )

    E,dE = analyze_scalar_data(settings)

    energies.append(E)
    stochastic_uncertainties.append(dE)

    print(f" [Result] {E} {dE}")
```

+++ {"id": "1Fvs5Vbm1Ko0"}

### Plot $E_\mathrm{AFQMC}$ vs $N_\mathrm{det}$

```{code-cell} ipython3
# plot the results
import matplotlib.pyplot as plt

plt.errorbar(ndets, energies, yerr=stochastic_uncertainties, fmt='o:')
plt.xlabel("Number of Slater determinants")
plt.ylabel("Energy (Ha)")
plt.title("Convergence of AFQMC in the number of Slater determinants")
plt.show()
```

+++ {"id": "WNfHDpRIE6lw"}

### Result

From the plot of energy vs $N_\mathrm{det}$, we have converged the AFQMC energy to within our target stochastic uncertainty of less than 1.6 m$E_\text{Ha}$ at about $N_\mathrm{det} = 500$.
Since we performed this test at a point on the PEC where there are strong static correlation effects, we expect this
value to provide converged results across the PEC.
Now, we can automate the calculation of the PEC using this value for $N_\mathrm{det}$.

+++ {"id": "c4MDXZD3E6lw"}

## Automating the computation of the PEC using Python / afqmctools

Now that we have converged $E_\mathrm{AFQMC}$ in $N_\mathrm{det}$, we are ready to compute the PEC.
In the code block below, we have included a function that performs all of the steps that we have already performed (excluding converging $E_\mathrm{AFQMC}$ in $N_\mathrm{det}$) given a value of the $N_2$ bond length, $\delta_{N-N}$.

Specifically, given $\delta_{N-N}$ and `ndets_to_read`, the function automates:

1. running CASSCF using PySCF (for convenience), and saving a truncated CASSCF wavefunction to the SAFIRE HDF5 format.
2. building and saving the Hamiltonian in the basis of CASSCF orbitals to the SAFIRE HDF5 format.
3. writing the json input file, with our chosen value of `ndets_to_read`, and running SAFIRE.
4. Extracting the AFQMC energy using the stats module.

<b>
Run the cell block below in order to load the "run_afqmc_on_dimer()" function into memory.
</b>

```{code-cell} ipython3
:id: 8u68vNTFE6lw

def run_afqmc_on_dimer(delta,ndets_to_read=None, num_mpi_tasks=16):

    scratch_dir = Path("data") / f"N2_bondlength_{delta:.4f}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running AFQMC on dimer with bond length {delta:.4f} Angstroms")

    # 1. run CASSCF to get a basis and a trial wavefunction
    atom = f'''
    N 0. 0. {delta/2}
    N 0. 0. -{delta/2}
    '''
    mol = gto.M(
        atom = atom,
        basis = 'ccpvdz',
        unit = 'Bohr'
    )


    casscf_chkfile = 'rhf_chkfile.h5'

    rhf = scf.RHF(mol)
    rhf.chkfile = scratch_dir / casscf_chkfile
    rhf.run()

    mc = mcscf.CASSCF(rhf, 12, 6).run() # (12o,6e)
    lib.chkfile.save(mc.chkfile, 'mcscf/ci', mc.ci)

    # write the CAS wavefunction to a file
    fout = scratch_dir / 'afqmc.h5'

    Wavefunction.from_pyscf_cas(
        mol=mol,
        cas_chkfile=scratch_dir / casscf_chkfile,
        tol= 1.0e-4,
        max_det=10000
    ).to_hdf5(fout)

    # 2. write the Hamiltonian, into the same file as the wavefunction
    basis_scf_data = load_from_pyscf_chk_mol(scratch_dir / casscf_chkfile, 'scf')

    chol_tol = 1e-5

    # write Hamiltonian
    MolecularHamiltonian.from_pyscf(
        basis_scf_data,
        chol_cut=chol_tol,
        verbose=True
    ).to_hdf5(fout)

    # write json file
    afqmc_execution_options = {
        "timestep": 0.01,
        "steps": 3200,
        "measure_interval_multiplier": 1,
        "population_control_interval" : 10,
        "walker_ortho_interval" : 10 ,
        "n_walkers_per_mpi_task": 20,
        "seed" : 42,
        "wavefunction" : { "ndets_to_read" : ndets_to_read }
    }

    write_json(
        scratch_dir/ "afqmc.json",
        fwfn0=scratch_dir / "afqmc.h5",
        exec_opts=afqmc_execution_options
    )


    print("Running AFQMC",flush=True)

    # run afqmc
    run_afqmc(
        run_dir=scratch_dir,
        run_mode="local_cpu",
        timeout_mins=100,
        input_file="afqmc.json",
        np=num_mpi_tasks,             # number of MPI tasks
        output_file="afqmc.out"   # optionally direct output to file instead of here
    )

    settings = dict(
        fname = scratch_dir/"qmc.s000.scalar.dat",
        xaxis = "time",
        nequil = 5,
        trace = True
    )

    E,dE = analyze_scalar_data(settings)

    print(f" [Result] {delta}  {E} {dE}")

    return delta, E, dE
```

+++ {"id": "cF6ZNhWBfkHE"}

### Compute the PEC

Now we can use the `run_afqmc_on_dimer()` function to compute the PEC of the $N_2$ molecule.

```{code-cell} ipython3
:id: HAXNMGv0fmdE

# we can simply use the automation function above to compute the PEC

num_mpi_tasks = 16 # set this based on your resources!

# full dataset from the paper
# bondlengths = [2.118,2.4,2.7,3.0,3.6,4.2]
# reduced dataset for the example
bondlengths = [2.118,2.7,3.6]

ndets_to_read = 500

energies = []
stochastic_uncertainties = []
for delta in bondlengths:
  _, E, dE = run_afqmc_on_dimer(delta,ndets_to_read, num_mpi_tasks=num_mpi_tasks)
  energies.append(E)
  stochastic_uncertainties.append(dE)
```

+++ {"id": "xVSpP-qtgNv_"}

### Plot the PEC

Now, we are ready to plot the PEC to see our results.

```{code-cell} ipython3
:id: XdQyhYsTgRV6

import matplotlib.pyplot as plt

plt.errorbar(bondlengths, energies, yerr=stochastic_uncertainties, fmt='o:', label="AFQMC/CASSCF")
plt.xlabel("Bond length (Bohr)")
plt.ylabel("Energy (Ha)")
plt.title(r"$N_2$ molecule PEC")
plt.grid(True)
plt.tight_layout()
plt.show()
```

+++ {"id": "2013EJrv-npD"}

Compare this with the PEC generated using the data from the original paper.

![](files/NitrogenPecAbs.png)

+++ {"id": "89Ml1OB9E6lx"}

## Summary

You have computed the potential energy curve (PEC) of the $N_2$ molecule, which is a standard benchmarking system in quantum chemistry.

### What you learned

1. How to select the number of Slater determinants to include in the trial wavefunction, balancing computational cost and accuracy
2. How to automate the computation of a potential energy curve using afqmctools within Python
