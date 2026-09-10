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

# $H_2O$ Molecule: Charge Density

<b>Goal:</b>
In this example, we compute the electronic charge density of
the water molecule at its equilibrium geometry.

## In Development!!

- need to figure out how to get the unit volume from PySCF.

## **Run the code block below to set up the example**

```{code-cell} ipython3
:id: ljFeTy3vE6ls

# Run me (shift+enter or click the play button) to setup the tutorial!
from pathlib import Path

import h5py as h5
import numpy as np
from pyscf import gto,scf,mcscf

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from safiretools import Wavefunction
from afqmctools.inputs.from_hdf import write_json

from stats.scalar_dat import analyze_scalar_data

from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "6ZBVQf-6E6lv"}

## The Water Molecule

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ACWiQaTbLK6S
outputId: c0680e89-bb37-4f76-fa41-1d7e4d5d6e38
---
"""
minimal example for understanding how PySCF computes the charge density and dipole moment.

The sum of rho does not equal
the total number of electrons...

"""

from pathlib import Path

import numpy as np
from pyscf import gto,scf

from pyscf.scf.hf import dip_moment
from pyscf.tools import cubegen

scratch = scratch_dir
chkfile = scratch / "uhf.chk"

"""
Equilibrium geometry of $H_2O$:

a0 = 0.9572
theta = 104.52 degrees
"""

a0 = 0.9572
theta = 104.52
ay = a0*np.cos(np.radians(theta/2))
ax = a0*np.sin(np.radians(theta/2))

atoms = f"""
O 0.000 0.000 0.000
H {ax} {-ay} 0.0
H {-ax} {-ay} 0.0
"""

mol = gto.M(
    atom = atoms,
    spin = 0,
    basis = 'cc-pvdz',
    verbose = 4
)

mf = scf.UHF(mol).newton()
mf.chkfile = chkfile
mf.kernel()

rdm = mf.make_rdm1()

nx = ny = nz = 3

rho = cubegen.density(mol,'h2o_den.cube', rdm, nx=nx, ny=ny, nz=nz)


mycube = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz)
mycube.read('h2o_den.cube')
volume_element = mycube.get_volume_element() # NOTE: this is the volume in fractional coords!!! Need actual volume element!!

ngrids = rho.shape[0] * rho.shape[1] * rho.shape[2]
print(f"Integrated charge = {np.sum(rho)*volume_element}")
print(f"grid size = {nx} x {ny} x {nz}")
print(f"volume element = {volume_element}")


#print(f"Integrated charge = {np.sum(rho)}")
#print(f"Number of grid points = {ngrids}")

#print(f"Charge per volume element = {np.sum(rho)*volume_element}")

```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ZQFIPYmPOdrJ
outputId: b07b428b-b187-4019-c8e3-be2605a49dd4
---
dir(rho)
#print(rdm.shape)
#print(rho.shape)
```

+++ {"id": "0Y-83DT3L-r9"}

# analyze with afqmctools

```{code-cell} ipython3
:id: YtU1MnKiL-22

from pathlib import Path

from pyscf.tools import cubegen
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from pyscf import gto

from pyscf.scf.hf import dip_moment

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

scratch = Path("./scratch")
scratch.mkdir(exist_ok=True)

reference = Path("./ref_results")
reference.mkdir(exist_ok=True)

afqmc_run_dir = Path("./run_afqmc")
afqmc_run_dir.mkdir(exist_ok=True)


uhf_results = reference / "uhf_rdm.h5"
dft_results = reference / "dft_pbe.h5"
cc_results = reference / "cc.h5"

scf_data_basis = load_from_pyscf_chk_mol(scratch / "rhf.chk")

mo_coeff = scf_data_basis["mo_coeff"]

a0 = 0.9572
theta = 104.52
ay = a0*np.cos(np.radians(theta))
ax = a0*np.sin(np.radians(theta))

atoms = f"""
O 0.000 0.000 0.000
H {ax} {ay} 0.0
H {-ax} {ay} 0.0
"""

x_min = -1.2
x_max = 1.2
y_min = -1.2
y_max = 1.2
x_points = 200
y_points = 200


def rdm_mo2ao(mol,rdm_mo,C):
    r"""
    Convert RDM from MO basis to AO basis.

    .. math::

        \gamma_{\mu\nu} = S_{\mu\mu'} C_{\mu' i}  \gamma_{ij} C^*_{j \nu'} S_{\nu' \nu}

    """
    S = mol.intor('int1e_ovlp')
    C = np.array(C)
    X = S @ C
    rdm_ao = X @ rdm_mo @ X.conj().T
    return rdm_ao

def plot_xy_density(spin_summed_rho,label=""):
    # make 2d density plot in the x-y plane at z=0
    x = np.linspace(x_min, x_max, x_points)
    y = np.linspace(y_min, y_max, y_points)

    if spin_summed_rho.shape[0] == 2:
        spin_summed_rho = np.sum(spin_summed_rho, axis=0)

    # Create coordinate matrices
    X, Y = np.meshgrid(x, y)

    #import pdb; pdb.set_trace()

    # Example function to evaluate on the grid (e.g., Gaussian function)
    rho = cubegen.density(mol, f"density{"_"+label if label != "" else ""}.cube", spin_summed_rho, nx=200, ny=200, nz=2)

    # Plot the 2D density plot
    plt.figure(figsize=(8, 6))
    plt.matshow(
        100*rho[:,:,0],
        cmap="hot",#"bone",#"Purples" #"viridis",
        interpolation="bicubic"
    )
    plt.colorbar(label="Density")

    # Set tick positions and labels
    tick_positions_x = np.linspace(0, x_points - 1, 5, dtype=int)
    tick_labels_x = np.round(np.linspace(x_min, x_max, 5), 2)
    tick_positions_y = np.linspace(0, y_points - 1, 5, dtype=int)
    tick_labels_y = np.round(np.linspace(y_min, y_max, 5), 2)

    plt.gca().set_xticks(tick_positions_x)
    plt.gca().set_xticklabels(tick_labels_x)
    plt.gca().set_yticks(tick_positions_y)
    plt.gca().set_yticklabels(tick_labels_y)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Charge Density of $H_2O$ in X-Y plane")
    plt.show()


print("Input Molecule")
print(atoms)

mol = gto.M(
    atom = atoms,
    spin = 0,
    basis = 'cc-pvdz',
    verbose = 4
)

with h5.File(uhf_results,"r") as f:
    rdm_uhf = f["rdm_uhf"][:]

print("UHF Density")
rho = cubegen.density(mol, "UHF_density.cube", rdm_uhf, nx=200, ny=200, nz=1)

plot_xy_density(rdm_uhf,label="UHF")

xcut_idx = 12 #np.argmax(np.sum(rho_uhf, axis=(1,2)))
print(f"Integrated charge = {np.sum(rho)}")

if False:
    plt.matshow(
        100*rho[xcut_idx],
        cmap="bone",#"Purples" #"viridis",
        interpolation="bicubic"
    )
    plt.show()

mu_vec =  dip_moment(mol,rdm_uhf,unit="Ha")
print(f"mu vector = {mu_vec}")
print(f"Dipole moment = {np.sqrt(np.sum(np.power(mu_vec,2)))}")


with h5.File(dft_results,"r") as f:
    rdm_dft = f["rdm"][:]

print("DFT-PBE Density")
rho = cubegen.density(mol, "DFT_density.cube", rdm_dft, nx=200, ny=200, nz=1)


plot_xy_density(rdm_dft,"DFT")

xcut_idx = 12 #np.argmax(np.sum(rho_uhf, axis=(1,2)))
print(f"Integrated charge = {np.sum(rho)}")

if False:
    plt.matshow(
        100*rho[xcut_idx],
        cmap="bone",#"Purples" #"viridis",
        interpolation="bicubic"
    )
    plt.show()

mu_vec =  dip_moment(mol,rdm_dft,unit="Ha")
print(f"mu vector = {mu_vec}")
print(f"Dipole moment = {np.sqrt(np.sum(np.power(mu_vec,2)))}")

with h5.File(cc_results,"r") as f:
    rdm_cc = f["rdm_cc"][:]

print("CCSD Density")
#rho = cubegen.density(mol, "CC_density.cube", rdm_cc, nx=200, ny=200, nz=1)
#plot_xy_density(rho,"CCSD")

xcut_idx = 12 #np.argmax(np.sum(rho_uhf, axis=(1,2)))
print(f"Integrated charge = {np.sum(rho)}")

if False:
    plt.matshow(
        100*rho[xcut_idx],
        cmap="bone",#"Purples" #"viridis",
        interpolation="bicubic"
    )
    plt.show()

mu_vec =  dip_moment(mol,rdm_cc,unit="Ha")
print(f"mu vector = {mu_vec}")
print(f"Dipole moment = {np.sqrt(np.sum(np.power(mu_vec,2)))}")

from afqmctools.analysis.rdm import average_afqmc_rdm

rho_avg, delta_rho = average_afqmc_rdm(
    rdm_file=afqmc_run_dir/"qmc.s000.stat.h5"
)

naverages = rho_avg.shape[0]
nspins = rho_avg.shape[1]
nmo = rho_avg.shape[2]


print("AFQMC Density")
for average in range(naverages):

    spin_summed_rho = np.sum(rho_avg[average],axis=0)

    # TODO: convert to ao basis!!!
    rdm_ao = rdm_mo2ao(mol,spin_summed_rho,mo_coeff)

    print(f"Average {average}")
    rho = cubegen.density(mol, f"AFQMC_density_avg{average}.cube", rdm_ao, nx=21, ny=100, nz=100)
    xcut_idx = 12
    print(f"Integrated charge = {np.sum(rho[average])}")
    if False:
        plt.matshow(
            100*rho[xcut_idx],
            cmap="bone",#"Purples" #"viridis",
            interpolation="bicubic"
        )
        plt.show()

    mu_vec =  dip_moment(mol,rdm_ao,unit="Ha")

    print(f"mu vector = {mu_vec}")
    print(f"Dipole moment = {np.sqrt(np.sum(np.power(mu_vec,2)))}")

    # make 2d density plot in the x-y plane at z=0
    x = np.linspace(x_min, x_max, x_points)
    y = np.linspace(y_min, y_max, y_points)

    # Create coordinate matrices
    X, Y = np.meshgrid(x, y)

    # Example function to evaluate on the grid (e.g., Gaussian function)
    #rho = cubegen.density(mol, "AFQMC_density.cube", rdm_ao, nx=200, ny=200, nz=1)

    # Plot the 2D density plot
    #plt.figure(figsize=(8, 6))
    #plt.matshow(
    #    100*rho[xcut_idx],
    #    cmap="bone",#"Purples" #"viridis",
    #    interpolation="bicubic"
    #)
    #plt.colorbar(label="Density")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.title("Charge Density of $H_2O$ in X-Y plane")
    #plt.show()

print("Done")
```
