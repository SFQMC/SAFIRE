"""Unfold a supercell AFQMC one-body RDM into the momentum distribution n(q).

n(k+G) = sum_ij c*_ik(G) rho_ij(k) c_jk(G), with c_ik(G) the QE plane-wave
coefficients of the Bloch orbitals used to build the CoQui Hamiltonian.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata


def read_qe_orbitals(save_dir, kpoints, nband):
    """Read q = k+G (1/Bohr) and the lowest `nband` coefficients from wfc{k+1}.hdf5.

    Only the first len(kpoints) files are read, and each file's xk must match
    `kpoints` (hamiltonian.h5:/System/BZ/kpoints), which fixes the k ordering.

    Returns
    -------
    qvecs : (nq, 3) array, all k concatenated in k order.
    coeffs : list of (nband, ngw_k) complex arrays, one per k.
    """
    qvecs, coeffs = [], []
    for ik, k in enumerate(kpoints):
        with h5py.File(Path(save_dir) / f"wfc{ik + 1}.hdf5", "r") as f:
            if not np.allclose(f.attrs["xk"], k, atol=1e-8):
                raise ValueError(f"wfc{ik + 1}.hdf5 has xk = {f.attrs['xk']}, expected {k}")
            mill = f["MillerIndices"]
            bg = np.array([mill.attrs[f"bg{i}"] for i in (1, 2, 3)])
            qvecs.append(mill[...] @ bg + k)
            evc = f["evc"][:nband]
        coeffs.append(evc[:, 0::2] + 1j * evc[:, 1::2])
    return np.concatenate(qvecs), coeffs


def unfold_nq(onerdm, coeffs):
    """Spin-summed n(q) for all q, concatenated in the order of `read_qe_orbitals`.

    `onerdm` is SAFIRE's (nspin, nk*M, nk*M) OneRDM with orbital index k*M + i; only its
    diagonal k-blocks enter, and a single spin channel (closed-shell walkers) is doubled.
    The result is linear in `onerdm`, so this can go inside `Results.evaluate`.
    """
    rho = onerdm.sum(axis=0) * (2 / onerdm.shape[0])
    M = coeffs[0].shape[0]
    nq = [np.einsum("ig,ij,jg->g", c.conj(), rho[k * M:(k + 1) * M, k * M:(k + 1) * M], c).real
          for k, c in enumerate(coeffs)]
    return np.concatenate(nq)


def radial_average(qmag, nq, edges):
    """Mean |q| and mean n(q) in each non-empty |q| bin."""
    idx = np.digitize(qmag, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    counts = np.bincount(idx[keep], minlength=len(edges) - 1)
    full = counts > 0
    q = np.bincount(idx[keep], qmag[keep], len(edges) - 1)[full] / counts[full]
    n = np.bincount(idx[keep], nq[keep], len(edges) - 1)[full] / counts[full]
    return q, n


def fermi_wavevector(volume, n_electrons):
    """Free-electron k_F = (3 pi^2 n)^(1/3) in 1/Bohr."""
    return (3 * np.pi**2 * n_electrons / volume) ** (1 / 3)


def plot_slab(ax, qvecs, nq, k_F):
    """Per-spin n(q) in the |q_z| < 0.1 k_F slab, interpolated, with the q-points and the k_F circle."""
    qz_tol = 0.1 * k_F
    slab = (np.abs(qvecs[:, 2]) < qz_tol) & (np.linalg.norm(qvecs, axis=1) < 2 * k_F)
    qx, qy, ns = qvecs[slab, 0], qvecs[slab, 1], nq[slab]
    xi, yi = np.meshgrid(np.linspace(qx.min(), qx.max(), 300),
                         np.linspace(qy.min(), qy.max(), 300))
    grid = np.nan_to_num(griddata((qx, qy), ns, (xi, yi), method="cubic"))
    pc = ax.pcolormesh(xi, yi, grid, shading="auto", cmap="magma", vmin=0, vmax=1)
    ax.scatter(qx, qy, c=ns, cmap="magma", vmin=0, vmax=1, s=12, edgecolors="white", alpha=0.6)
    ax.figure.colorbar(pc, ax=ax, label=r"$n(\mathbf{q})$ per spin")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(k_F * np.cos(t), k_F * np.sin(t), "c--", lw=1.5, label=r"$k_\mathrm{F}$")
    ax.set(xlabel=r"$q_x$ (Bohr$^{-1}$)", ylabel=r"$q_y$ (Bohr$^{-1}$)", aspect="equal",
           title=rf"$n(\mathbf{{q}})$, $|q_z| < {qz_tol:.3f}$ Bohr$^{{-1}}$")
    ax.legend(loc="upper right", fontsize=8)
