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

# Diamond Ground State Energy and Charge Density from PySCF

In this example, we compute the ground state energy and the real-space charge density of diamond,
starting from a periodic Hartree-Fock calculation in [PySCF](https://pyscf.org) instead of Quantum ESPRESSO and CoQuí.
safiretools converts the PySCF solution into a k-point Hamiltonian and a trial wavefunction for SAFIRE,
and PySCF evaluates the Gaussian basis functions to put the density on a real-space grid.

Set `AFQMC_EXEC` to the SAFIRE executable.
All calculations run in a scratch directory `data`.

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

import matplotlib.pyplot as plt
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.dft.numint import KNumInt
from safiretools import NOMSDWavefunction, PeriodicHamiltonian, Results, vis, write_jobfile
```

## Hartree-Fock with PySCF

We use the 2-atom primitive cell of diamond at the experimental lattice constant, with GTH pseudopotentials and the minimal
`gth-szv` basis of 4 orbitals per atom, on a $2\times2\times2$ k-point mesh.
The `mesh` of the cell sets the plane-wave grid on which the two-electron integrals are evaluated.

```{code-cell} ipython3
a = 3.567  # lattice constant in Angstrom
cell = gto.Cell(
    a=a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    atom=f"C 0 0 0; C {a / 4} {a / 4} {a / 4}",
    basis="gth-szv",
    pseudo="gth-pade",
    mesh=[25] * 3,
).build()

kmf = scf.KRHF(cell, kpts=cell.make_kpts([2, 2, 2]))
kmf.kernel()
```

## Hamiltonian and Trial Wavefunction

`PeriodicHamiltonian.from_pyscf` Cholesky-decomposes the two-electron integrals for each momentum transfer,
in the basis of the Hartree-Fock orbitals at each k-point.
`NOMSDWavefunction.from_pbc_scf` takes the Hartree-Fock determinant as the trial wavefunction.
The conversion runs serially in Python, so it takes a little while.
For production-sized solids, CoQuí is the supported route.

```{code-cell} ipython3
ham = PeriodicHamiltonian.from_pyscf(kmf)
wfn = NOMSDWavefunction.from_pbc_scf(kmf)
print(f"{ham.nkpts} k-points, {ham.nmo} orbitals per k-point")
```

## Run AFQMC

The energy estimator is on by default.
The `backprop` block measures the back-propagated one-body density matrix at each back-propagation length in `bp_lengths`, given in steps
(see {doc}`../../../tutorials/solids/04_computing_observables/04_computing_observables`).
`write_jobfile` writes the Hamiltonian and the wavefunction to HDF5 files next to the job file.

```{code-cell} ipython3
dt = 0.01
bp_lengths = [100, 200]
write_jobfile("afqmc.json", {
    "seed": 7,
    "execute": {
        "hamiltonian": ham,
        "wavefunction": wfn,
        "timestep": dt,
        "steps": 10000,
        "equilibration_steps": 1000,
        "measure_interval": 10,
        "population_control_interval": 10,
        "walker_ortho_interval": 10,
        "num_walkers": 320,
        "estimators": {
            "backprop": {"propagation_steps": bp_lengths, "onerdm": {}},
        },
    },
})
```

```{code-cell} ipython3
:tags: [scroll-output]

!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

## Ground State Energy

```{code-cell} ipython3
results = Results("afqmc.results.h5")

energy, energy_err = results.average("Energy")
print(f"E_HF    = {kmf.e_tot * len(kmf.kpts):.4f} Ha")
print(f"E_AFQMC = {energy.real:.4f} ± {energy_err.real:.4f} Ha")

vis.plot_timeseries(results)
plt.show()
```

The energies are those of the supercell spanned by the k-point mesh, 8 primitive cells.

## Charge Density

SAFIRE measures the density matrix $\rho_{ij} = \langle c^\dagger_i c_j \rangle$ in the working basis,
whose orbital $i$ at k-point $\mathbf{k}$ has index $k M + i$, with $M$ orbitals per k-point.
`ham.basis_rotation` holds the transformation $X_\mathbf{k}$ from the Bloch atomic orbitals $\chi_{\mu\mathbf{k}}$ at each k-point to the working basis,
$\phi_{i\mathbf{k}} = \sum_\mu \chi_{\mu\mathbf{k}} (X_\mathbf{k})_{\mu i}$.
The density matrix is diagonal in $\mathbf{k}$, so the density is

$$
n(\mathbf{r}) = \frac{1}{N_\mathbf{k}} \sum_\mathbf{k} \sum_{\mu\nu} \chi_{\mu\mathbf{k}}(\mathbf{r})\, D^\mathbf{k}_{\mu\nu}\, \chi^*_{\nu\mathbf{k}}(\mathbf{r}),
\qquad D^\mathbf{k} = X_\mathbf{k}\, (\rho^\mathbf{k})^T X_\mathbf{k}^\dagger,
$$

where $D^\mathbf{k}$ is in the convention of PySCF's `make_rdm1`, which lets PySCF's `KNumInt` evaluate the density.
The walkers are closed shell, so SAFIRE stores one spin channel, which we double.

```{code-cell} ipython3
M, X = ham.nmo, ham.basis_rotation


def ao_density_matrix(rdm):
    """Spin-summed density matrix in the Bloch atomic orbitals at each k-point."""
    rho = rdm.sum(axis=0) * (2 / rdm.shape[0])
    return np.array([X_k @ rho[k * M:(k + 1) * M, k * M:(k + 1) * M].T @ X_k.conj().T for k, X_k in enumerate(X)])
```

We evaluate the density in the $(1\bar{1}0)$ plane, which contains the zigzag chains of C-C bonds,
over a $\sqrt{2}a \times a$ section spanned by the $[110]$ and $[001]$ directions.

```{code-cell} ipython3
a_bohr = a / lib.param.BOHR
s, t = np.meshgrid(np.linspace(0, np.sqrt(2) * a_bohr, 141), np.linspace(0, a_bohr, 101), indexing="ij")
coords = s[..., None] * np.array([1, 1, 0]) / np.sqrt(2) + t[..., None] * np.array([0, 0, 1])

ni = KNumInt()
ao = ni.eval_ao(cell, coords.reshape(-1, 3), kpts=kmf.kpts)


def density(rdm):
    return ni.eval_rho(cell, ao, ao_density_matrix(rdm)).real.reshape(s.shape)
```

For reference, the Hartree-Fock density follows from the density matrix of the trial determinant $\Phi$,
$\rho_{ij} = \sum_n \Phi^*_{in} \Phi_{jn}$.
We take the average back-propagated density matrix at each length from `Results.average`.
Each sample is separated by a window, so for long windows we can assume they are weakly correlated and set `rebinsize=1`.

```{code-cell} ipython3
phi = wfn.determinant(0)
n_hf = density((phi.conj() @ phi.T)[None])

dn = {}
for L in bp_lengths:
    rdm, _ = results.average(f"BackPropEstimator/Steps={L}/OneRDM", rebinsize=1)
    dn[L] = density(rdm) - n_hf
dn_max = max(abs(d).max() for d in dn.values())

fig, axes = plt.subplots(1, 1 + len(bp_lengths), figsize=(4.5 * (1 + len(bp_lengths)), 3.5), sharey=True)
im = axes[0].pcolormesh(s, t, n_hf, shading="gouraud")
fig.colorbar(im, ax=axes[0])
axes[0].set_title("Hartree-Fock")
for ax, L in zip(axes[1:], bp_lengths):
    im = ax.pcolormesh(s, t, dn[L], shading="gouraud", cmap="RdBu_r", vmin=-dn_max, vmax=dn_max)
    fig.colorbar(im, ax=ax)
    ax.set_title(rf"AFQMC $-$ HF, $\tau_\mathrm{{BP}} = {L * dt:g}$ Ha$^{{-1}}$")
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlabel("$[110]$ (Bohr)")
axes[0].set_ylabel("$[001]$ (Bohr)")
fig.tight_layout()
```

The pseudopotential has no core electrons, so the valence density has a hole at each nucleus and piles up on the bonds.
The difference shows how correlation redistributes the charge.
It should agree between the back-propagation lengths within the noise; otherwise, increase `bp_lengths`.

## Converging the Calculation

The minimal `gth-szv` basis captures little of the correlation energy.
For quantitative results, converge in the basis set (e.g. `gth-dzvp`, `gth-tzv2p`), in the k-point mesh,
in the `timestep`, and in the Cholesky threshold `chol_cut` of `PeriodicHamiltonian.from_pyscf`.
