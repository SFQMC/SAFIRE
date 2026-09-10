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

+++ {"id": "eglSrlxk2Ip8"}

# Local Embedding

Following the methodology presented in:

https://pubs.acs.org/doi/10.1021/acs.jctc.8b01244

## Additional Software

This example uses the freely available [embeddingAFQMC](https://github.com/bkesk/embeddingAFQMC) Python package
which, among other things, provides tools to analyze the locality of a set of orbitals.

## O adsorption on $H_{20}$ chain

```{code-cell} ipython3
:id: v08YqIkR2r02

# Some setup
from pathlib import Path
from tutorial_utils import run_afqmc

# from embeddingAFQMC
from embedding.orbital.assign import euclid_nd, gen_orbital_stats, print_stats

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

Hartree = 27.211407953 # eV
Bohr = 0.529177        # Angstrom

MPI_TASKS = 16
```

+++ {"id": "vDrFukfq5u0t"}

### Automate the AFQMC

We can automate the local embedding workflow, for a given O atom height, $h$, and localization radii $R_o$, and $R_v$, using a Python function. The workflow consists of the following steps.

1. define the atomic positions
2. run ROHF to obtain both a trial wavefunction, and a basis
3. unitarily localize the set of ROHF orbitals using so-called "split localization". This can be done by partitioning the ROHF orbitals into the following sets, and localizing each set separately via Foster-Boys localization for example).
    - energetic core orbitals (no need to localize, we will freeze these no mater what)
    - occupied orbitals (including singly-occupied orbitals here; however, it is also possible to partition the singly-occupied orbitals from the doubly-occupied orbitals)
    - virtual orbitals
4. Select active orbitals using the $R_o$ and $R_v$ localization radii which apply to the occupied and virtual orbitals, respectively. An orbital is included in the active space if its centroid is localized within the applicable localization radius. Here, we use the 2-dimensional projection of the orbital centroid positions onto the $xy$-plane instead of the full 3-dimensional position.
5. perform a frozen core transformation on the many-body Hamiltonian based on the chosen active / inactive orbitals, and save the effective Hamiltonian to a SAFIRE HDF5 input file.
6. Write the ROHF determinant to the SAFIRE HDF5 format to use as a trial wavefunction.
7. Generate the json input file for SAFIRE
8. run AFQMC using SAFIRE
9. Compute the average AFQMC energy using afqmctools's analysis tooling

See the codeblock below.

```{code-cell} ipython3
:id: b7gcg4ZA6-kD

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from pyscf import gto, scf, lo

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from safiretools import MolecularHamiltonian, NOMSDWavefunction
from afqmctools.inputs.from_hdf import write_json
from stats.scalar_dat import analyze_scalar_data

from tutorial_utils import run_afqmc


# custom metric to take the projection of the orbital position onto the xy plane
def euclid_2d(r1, r2):
    """
    Return the 2D Euclidean distance between two points r1 and r2, ignoring coordinates beyond the first two dimensions.
    """
    return euclid_nd(r1[:2], r2[:2])

def get_afqmc_energy(h, Ro=np.inf, Rv=np.inf, N_energetic_core=0, rhf_guess_rdm=None):
    # 1. generate underlying CMO basis
    # run initial HF calculation to generate basis

    local_scratch = scratch_dir / f"h={h:5.5f}"
    local_scratch.mkdir(parents=True, exist_ok=True)
    atom_chkfile = local_scratch / f"rhf_h={h:5.5f}.chk"

    def get_atomic_positions(h):
        """
        h is input in Bohr, but needs to be converted to angstrom
        """

        return f"""O 0.000000 0.0 {h*Bohr}
                H -0.4704240745 0.0 0.0
                H 0.4704240745 0.0 0.0
                H 1.4112722235 0.0 0.0
                H 2.3521203725 0.0 0.0
                H 3.2929685215 0.0 0.0
                H 4.2338166705 0.0 0.0
                H 5.1746648195 0.0 0.0
                H 6.1155129685 0.0 0.0
                H 7.0563611175 0.0 0.0
                H 7.9972092665 0.0 0.0
                H 8.9380574155 0.0 0.0
                H 9.8789055645 0.0 0.0
                H 10.8197537135 0.0 0.0
                H 11.7606018625 0.0 0.0
                H 12.7014500115 0.0 0.0
                H 13.6422981605 0.0 0.0
                H 14.5831463095 0.0 0.0
                H 15.5239944585 0.0 0.0
                H 16.4648426075 0.0 0.0
                H 17.4056907565 0.0 0.0"""

    mol = gto.M(
        basis = 'ccpvdz',
        atom=get_atomic_positions(h),
        spin=0,
        verbose=3
    )

    # run HF
    mf = scf.RHF(mol).newton()
    mf.chkfile = atom_chkfile
    # it turns out to be helpful to initialize each HF calculation from the final solution of the
    # previous geometry
    dm0 = rhf_guess_rdm if rhf_guess_rdm is not None else None
    mf.kernel(dm0)

    # perform stability analysis - see PySCF example:
    # https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/17-stability.py
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability()
    Erhf = mf.kernel()

    rhf_rdm = mf.make_rdm1()

    # 2. generate a local basis
    # remove energetic core first - we will typically freeze these no matter what!
    core_orbitals =  mf.mo_coeff[:,:N_energetic_core]
    fb_core = lo.Boys(mol, core_orbitals)
    core_orbitals = fb_core.kernel()

    # foster-boys localization - occupied
    occ_orbitals = mf.mo_coeff[:,N_energetic_core:][:,mf.mo_occ[N_energetic_core:]>0]
    fb_occ = lo.Boys(mol, occ_orbitals)
    fb_occ_orbitals = fb_occ.kernel()

    # foster-boys localization - virtual
    virt_orbitals = mf.mo_coeff[:,N_energetic_core:][:,mf.mo_occ[N_energetic_core:]==0]
    fb_virt = lo.Boys(mol, virt_orbitals)
    fb_virt_orbitals = fb_virt.kernel()

    # 3. select active local orbitals
    core_stats = gen_orbital_stats(mol, core_orbitals, metric=euclid_2d)
    print_stats(core_stats)

    occ_stats_gen = gen_orbital_stats(mol, fb_occ_orbitals, metric=euclid_2d)
    occ_stats = sorted(occ_stats_gen, reverse=True)
    print_stats(occ_stats)

    virt_stats_gen = gen_orbital_stats(mol, fb_virt_orbitals, metric=euclid_2d)
    virt_stats = sorted(virt_stats_gen)
    print_stats(virt_stats)

    occ_sorted_indices = [ o.index for o in occ_stats ] # occ_stats is already sorted by distance!
    virt_sorted_indices = [ o.index for o in virt_stats ]
    print(occ_sorted_indices)
    print(virt_sorted_indices)

    local_basis = np.concatenate((core_orbitals, fb_occ_orbitals[:,occ_sorted_indices], fb_virt_orbitals[:,virt_sorted_indices]), axis=1)

    # select active orbitals

    # convert to orbital counts
    Ncore = sum( [1 for o in occ_stats if o.distance > Ro] ) + N_energetic_core
    N_active_occ = max(mol.nelec) - Ncore
    N_active_virt = sum( [1 for o in virt_stats if o.distance <= Rv] ) # this is the number of *active* virtuals

    ncas = N_active_occ + N_active_virt

    print(f"Ncore = {Ncore}")
    print(f"N_active_occ = {N_active_occ}")
    print(f"N_active_virt = {N_active_virt}")
    print(f"ncas = {ncas}")

    # 4. build and save the Hamiltonian
    # generate the Frozen Core Hamiltonian + save for AFQMC

    afqmc_hamil_file = local_scratch / "local_embedding_H.h5"

    scf_data = load_from_pyscf_chk_mol(
        chkfile = atom_chkfile
    )

    # edit the orbitals!
    scf_data["mo_coeff"] = local_basis

    MolecularHamiltonian.from_pyscf(
        scf_data,
        cas=(2*(N_active_occ),ncas),
        chol_cut = 1e-5,
        verbose=False
    ).to_hdf5(afqmc_hamil_file)

    # 5. Write the trial wavefunction

    # write the trial wavefunction
    afqmc_wfn_file = local_scratch / "active_space_rhf_wfn.h5"

    active_det_up = np.eye(ncas,N_active_occ)
    nelec_active = (N_active_occ,N_active_occ)

    NOMSDWavefunction(
        coeffs=np.array([1.0]),
        dets=np.array([active_det_up]),
        spin_symm='closed',
        nelec=nelec_active,
        nmo=ncas
    ).to_hdf5(afqmc_wfn_file)

    # 6. write json input file
    execute_options = {
        "timestep": 0.01,
        "steps": 6000,
        "population_control_interval" : 10,
        "measure_interval_multiplier": 1,
        "walker_ortho_interval" : 10,
        "n_walkers_per_mpi_task": 20,
        "seed" : 42,
        "walker_set": {
            "walker_type": "COLLINEAR"
        },
    }
    write_json(
        local_scratch / "afqmc.json",
        fwfn0=afqmc_wfn_file,
        fham0=afqmc_hamil_file,
        exec_opts=execute_options
    )

    # 7. run AFQMC
    # run the AFQMC calculation...
    run_afqmc(
        run_dir=local_scratch,
        run_mode="local_cpu",
        timeout_mins=100,
        input_file="afqmc.json",
        np=MPI_TASKS,             # number of MPI tasks
        output_file="afqmc.out"   # optionally include output here by setting to None
    )

    # 8. analyze results
    settings = dict(
        fname = local_scratch/"qmc.s000.scalar.dat",
        xaxis = "time",
        nequil = 10,
        trace = True
    )

    return *analyze_scalar_data(settings),Erhf,rhf_rdm
```

+++ {"id": "siRKdtkN9GVZ"}

## AFQMC Potential Energy Curve

Let's use the function that we defined to compute $E[h]$ for $(R_o,R_v) = (3.8,6.2)$ Bohr.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: AsrchZED9FfD
outputId: d46cbef5-7b91-4c67-9ae3-c5217e07894c
---
o_atom_heights = [ 1.247, 1.436, 1.814, 2.381, 3.326, 3.893]

Ro = 3.8 # Bohr
Rv = 6.2 # Bohr

def save_results(f,label,data):
    data = np.array(data)
    if label in f:
        del f[label]
    dset = f.create_dataset(label, data=data)

rhf_rdm = None
E_rhf_list = []
rhf_rdm_list = []
E_list = []
dE_list = []
for h in o_atom_heights:
    E,dE,Erhf,rhf_rdm = get_afqmc_energy(h=h,Ro=Ro,Rv=Rv,N_energetic_core=1,rhf_guess_rdm=rhf_rdm)
    E_list.append(E)
    dE_list.append(dE)
    E_rhf_list.append(Erhf)
    rhf_rdm_list.append(rhf_rdm)
    print(f"h = {h}")
    print(f"E = {E*Hartree} eV")
    print(f"dE = {dE*Hartree} eV")
    print(f"E_rhf = {Erhf*Hartree} eV")
    print(f"correlation E = {(E - Erhf)*Hartree}")

    with h5.File(scratch_dir / "afqmc_results.h5", "a") as f:
        save_results(f, "h", data=o_atom_heights)
        save_results(f, "E", data=E_list)
        save_results(f, "dE", data=dE_list)
        save_results(f, "E_rhf", data=E_rhf_list)
        save_results(f, "rhf_rdms", data=rhf_rdm_list)
```

+++ {"id": "_LSxHrlU8l7s"}

Now, let's plot the resulting energies

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 472
id: d74PFj8g8pAR
outputId: 91e568eb-80d6-4e00-8e5e-0596bf3bfa65
---
# read results from file to allow saving / re-analysis.
with h5.File(scratch_dir / "afqmc_results.h5", "r") as f:
    o_atom_heights = f["h"][:]
    E_list = f["E"][:]
    dE_list = f["dE"][:]
    E_rhf_list = f["E_rhf"][:]
    rhf_rdm_list = f["rhf_rdms"][:]

# simple plot with no fit
plt.errorbar(o_atom_heights, np.array(E_list)*Hartree, yerr=np.array(dE_list)*Hartree, fmt='o', color='green', label='AFQMC(SAFIRE)')

plt.legend()
plt.xlabel('h (Bohr)')
plt.ylabel('E (eV)')
plt.title('Energy vs O atom height')
plt.savefig(scratch_dir / "energy_vs_height.png")
plt.show()
```

+++ {"id": "yTVtvwYR8qz0"}

We can also perform a Morse fit to the AFQMC results using standard Python libraries and some simple functions.
In the code block below, we use bootstrapping to determine the stochastic
uncertainty of the Morse fit parameters.

```{code-cell} ipython3
:id: uXlB1k1h8zpy


from scipy.optimize import curve_fit

def morse_potential(r, D, a, r0, D0):
    """
    Morse potential function.

    .. math::

        V(r) = D(1 - e^{-a(r - r0)})^2 + D0

    where

    D: well depth
    a: width of the potential
    r0: equilibrium bond length
    D0: offset of the potential

    Parameters
    ----------
    r : array_like
        Distance from the equilibrium position.
    D : float
        Well depth of the potential.
    a : float
        Width of the potential.
    r0 : float
        Equilibrium bond length.
    D0 : float
        Offset of the potential.

    Returns
    -------
    array_like
        The Morse potential evaluated at distance `r`.
    """
    return D * (1 - np.exp(-a * (r - r0)))**2 + D0

def boostrap_curve_fit(f, x, y, dy, nboostrap_samples=100, **kwargs):
    """
    This function uses bootstrap resampling to estimate the uncertainty in the fit parameters.
    """

    opt_params_list = []

    for i in range(nboostrap_samples):
        # get random data from a guassian distribution with mean y and standard deviation dy
        y_sample = np.random.normal(loc=y, scale=dy)

        # fit the morse potential to the sampled data
        opt_params, _ = curve_fit(f, x, y_sample, **kwargs)
        opt_params_list.append(opt_params)

    mean_opt_params = np.mean(opt_params_list, axis=0)
    std_dev_opt_params = np.std(opt_params_list, axis=0)
    std_error_opt_params = std_dev_opt_params / np.sqrt(nboostrap_samples)


    return mean_opt_params, std_dev_opt_params, std_error_opt_params
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 668
id: 9QDuUutS9BwO
outputId: f5afcd28-81bc-4dcc-927c-0bdc3e332daf
---
# read results from file to allow saving / re-analysis.
with h5.File(scratch_dir / "afqmc_results.h5", "r") as f:
    o_atom_heights = f["h"][:]
    E_list = f["E"][:]
    dE_list = f["dE"][:]
    E_rhf_list = f["E_rhf"][:]
    rhf_rdm_list = f["rhf_rdms"][:]

o_atom_heights_ref = np.array([1.247,1.436,1.814,2.381,3.326,3.893]) # Bohr
E_ref = np.array([-2333.238,-2334.615,-2334.563,-2332.685,-2330.862,-2330.586]) # eV
dE_ref = np.array([0.011,0.011,0.014,0.017,0.019,0.019]) # eV

# curve fit the reference data
popt_mean, popt_stddev, popt_stderr  = boostrap_curve_fit(
    morse_potential,
    o_atom_heights_ref,
    E_ref,
    dE_ref,
    p0=[4.5, 1.0, 1.6, np.min(E_ref)]
)

print(f"AFQMC (literature) Fitted parameters: \nD={popt_mean[0]} +/-  {popt_stderr[0]}")
print(
    f"a={popt_mean[1]} +/- {popt_stderr[1]}"
    f", r0={popt_mean[2]} +/- {popt_stderr[2]}"
    f", D0={popt_mean[3]} +/- {popt_stderr[3]}"
)

# plot the reference data and the fitted curve
x_fit = np.linspace(1.0, 5.0, 100)
y_fit = morse_potential(x_fit, *popt_mean)
plt.figure(figsize=(10, 6))
plt.errorbar(o_atom_heights_ref, E_ref, yerr=dE_ref, fmt='o', color='red', label='AFQMC(Literature)')
#plt.plot(x_fit, y_fit, label='Fitted Morse Potential', color='red')
plt.title('Morse Potential Fit to O Atom Height Data')


# plot the 1-sigma uncertainty band
y_fit_upper = morse_potential(x_fit, *(popt_mean + popt_stderr))
y_fit_lower = morse_potential(x_fit, *(popt_mean - popt_stderr))

plt.errorbar(o_atom_heights, np.array(E_list)*Hartree, yerr=np.array(dE_list)*Hartree, fmt='o', color='green', label='AFQMC(SAFIRE)')

# fit and plot the AFQMC data
popt_afqmc, pcov_afqmc = curve_fit(
    morse_potential,
    o_atom_heights,
    np.array(E_list)*Hartree,
    sigma=np.array(dE_list)*Hartree,
    p0=[4.5, 1.0, 1.6, np.min(E_list)*Hartree]
)
print(f"AFQMC fitted parameters: \nD={popt_afqmc[0]} +/- {np.sqrt(pcov_afqmc[0,0])}")
print(
    f"a={popt_afqmc[1]} +/- {np.sqrt(pcov_afqmc[1,1])}"
    f", r0={popt_afqmc[2]} +/- {np.sqrt(pcov_afqmc[2,2])}"
    f", D0={popt_afqmc[3]} +/- {np.sqrt(pcov_afqmc[3,3])}"
)
y_fit_afqmc = morse_potential(x_fit, *popt_afqmc)
plt.plot(x_fit, y_fit_afqmc, label='AFQMC Fit', color='green')
# plot the 1-sigma uncertainty band for AFQMC
y_fit_afqmc_upper = morse_potential(x_fit, *(popt_afqmc + np.sqrt(np.diag(pcov_afqmc))))
y_fit_afqmc_lower = morse_potential(x_fit, *(popt_afqmc - np.sqrt(np.diag(pcov_afqmc))))
plt.fill_between(x_fit, y_fit_afqmc_lower, y_fit_afqmc_upper, color='green', alpha=0.2, label='AFQMC 1-sigma Uncertainty Band')

plt.plot(x_fit, y_fit_afqmc_lower, ':', color='black', alpha=0.5, linewidth=1)
plt.plot(x_fit, y_fit_afqmc_upper, ':', color='black', alpha=0.5, linewidth=1)

plt.legend()
plt.xlabel('h (Bohr)')
plt.ylabel('E (eV)')
plt.title('Energy vs O atom height')
plt.savefig(scratch_dir / "energy_vs_height.png")
plt.show()
```
