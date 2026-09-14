# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

r"""
`PeriodicHamiltonian` — a Cholesky-factorized Hamiltonian for a periodic system,
generated from a PySCF ``pbc`` SCF calculation.

There is one on-disk format — ``Hamiltonian/KPFactorized``, which the
executable's ``KPFactorizedHamiltonian`` reads — and one Cholesky solver,
`PeriodicCholesky`, with a ``kp_sym`` flag choosing how the two-body integrals
are factorized:

``kp_sym=True``
    Factorize each momentum transfer *Q* separately, exploiting
    :math:`k_1 - k_2 + G = Q`. Gives one ``L_Q`` per momentum transfer over the
    original k-point mesh.

``kp_sym=False``
    Factorize the full :math:`(k_1, k_2)` matrix at once, giving a *supercell*
    Hamiltonian in one combined orbital basis.

**A supercell Hamiltonian is just the Γ point of the supercell**, so it is
written in the same k-point format with a single k-point: ``nkpts = 1``,
``nmo_pk = [nmo_tot]``, ``QKTok2 = [[0]]``, ``MinusK = [0]``, and one ``L0`` of
shape ``(1, nmo_tot**2 * nchol)``. Nothing downstream has to special-case it,
and the Cholesky vectors stay complex, as the format and the executable require.

The two solver modes differ only in how the k-point pairs are enumerated and how
the already-visited pivots are indexed; the factorization loop itself is shared.

.. note:: The factorization runs serially and is written by
          `PeriodicHamiltonian.to_hdf5`. **CoQuí is the supported route for
          production-sized solids**; see DESIGN.md.
"""

import logging
import math
import time
import warnings

import numpy as np
import h5py as h5

from safiretools.hamiltonian.base import (
    Hamiltonian,
    open_for_hamiltonian,
    write_hamiltonian_format,
)
from safiretools.hamiltonian.fcidump import write_fcidump_kpoint
from safiretools.hdf5 import from_complex, to_complex
from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

KPOINT_GROUP = 'Hamiltonian/KPFactorized'
"""Group the k-point-symmetric factorization is written under."""

_NUM_GRID_SHIFTS = 27
"""Number of reciprocal-lattice shifts searched: all of (-1, 0, 1)^3."""


# ----------------------------------------------------------------------
# momentum bookkeeping
# ----------------------------------------------------------------------

def construct_qk_maps(cell, kpts):
    r"""
    Build the momentum-transfer maps used by the k-point-symmetric solver.

    For a given :math:`Q` (any vector in the first Brillouin zone) there are
    ``nkpts`` k-point pairs mapping to it, such that :math:`(k_1 - k_2) + G = Q`
    with :math:`G` in the reciprocal lattice. The mapping is taken as
    :math:`K = k_1` and :math:`Q = (k_1 - k_2) + G`.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        PySCF cell.
    kpts : numpy.ndarray
        k-points, shape ``(nkpts, 3)``.

    Returns
    -------
    QKToK2 : numpy.ndarray
        ``QKToK2[Q, k1] = k2``, shape ``(nkpts, nkpts)``.
    kminus : numpy.ndarray
        ``kminus[Q]`` is the index of :math:`-Q`, shape ``(nkpts,)``.

    Raises
    ------
    ValueError
        If a mapping is missing or ambiguous, which means the k-point mesh is
        not closed under the reciprocal lattice.
    """
    nkpts = len(kpts)
    Qpts = kpts - kpts[0]

    QKToK2 = np.zeros((nkpts, nkpts), dtype=np.int32) - 1
    kminus = np.zeros((nkpts,), dtype=np.int32) - 1

    kvecs = cell.reciprocal_vectors()
    shifts = [np.dot(np.array([i, j, k]), kvecs)
              for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]

    for iq, Q in enumerate(Qpts):
        for ia, ka in enumerate(kpts):
            for ic, kc in enumerate(kpts):
                candidates = [ka - kc] + [ka - kc + G for G in shifts]
                for q in candidates:
                    if np.abs(np.dot(q - Q, q - Q)) < 1e-10:
                        if QKToK2[iq, ia] >= 0:
                            raise ValueError(
                                "More than one solution while constructing the QK map"
                            )
                        QKToK2[iq, ia] = ic
                        break
            if QKToK2[iq, ia] < 0:
                raise ValueError(f"Could not construct the QK mapping for Q={iq}, k={ia}")

    for iq, Q in enumerate(Qpts):
        for iqm, Qm in enumerate(Qpts):
            if any(np.abs(np.dot(Q + Qm - G, Q + Qm - G)) < 1e-10 for G in shifts):
                if kminus[iq] >= 0:
                    raise ValueError("More than one solution to Q + Qm = G")
                kminus[iq] = iqm
        if kminus[iq] < 0:
            raise ValueError(f"Could not solve Q + Qm = G for Q={iq}")

    return QKToK2, kminus


def generate_grid_shifts(cell):
    r"""
    Build the FFT-grid index maps for every shift by a reciprocal-lattice vector
    :math:`Q = k_k - k_i + k_l - k_j`.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        PySCF cell.

    Returns
    -------
    gmap : numpy.ndarray
        ``(27, ngs)`` index map, one row per shift.
    Qi : numpy.ndarray
        ``(27, 3)`` shift vectors, in the same order.
    ngs : int
        Number of grid points.
    """
    mesh = cell.mesh
    ngs = np.prod(mesh)

    g1 = np.arange(ngs, dtype=np.int32).reshape(mesh, order='C')
    gmap = np.zeros((_NUM_GRID_SHIFTS, ngs), np.int32)
    Qi = np.zeros((_NUM_GRID_SHIFTS, 3), dtype=np.float64)

    kvecs = cell.reciprocal_vectors()
    for ii, (nx, ny, nz) in enumerate(
            (nx, ny, nz)
            for nx in range(-1, 2) for ny in range(-1, 2) for nz in range(-1, 2)):
        Qi[ii, :] = np.dot(np.array([nx, ny, nz]), kvecs)
        gmap[ii, :] = np.roll(g1, (-nx, -ny, -nz), axis=(0, 1, 2)).reshape(-1, order='C')

    return gmap, Qi, ngs


def get_ortho_ao(cell, kpts, lindep_cutoff=0.0):
    """
    Canonical orthogonalization transformation for a periodic cell.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        PySCF cell.
    kpts : numpy.ndarray
        k-points.
    lindep_cutoff : float, optional
        Basis functions whose overlap eigenvalues fall below this are dropped.
        Should match the value used in ``pyscf.scf.addons.remove_linear_dep``.

    Returns
    -------
    X : numpy.ndarray
        ``(nkpts, nao, nao)`` transformation matrix.
    nmo_per_kpt : numpy.ndarray
        Number of orthogonalized orbitals kept at each k-point.
    """
    from pyscf import lib as pyscf_lib

    kpts = np.reshape(kpts, (-1, 3))
    nkpts = len(kpts)
    nao = cell.nao_nr()

    s1e = pyscf_lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    X = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    nmo_per_kpt = np.zeros(nkpts, dtype=np.int32)

    for k in range(nkpts):
        sdiag, Us = np.linalg.eigh(s1e[k])
        keep = sdiag > lindep_cutoff
        nmo_per_kpt[k] = keep.sum()
        X[k, :, 0:nmo_per_kpt[k]] = Us[:, keep] / np.sqrt(sdiag[keep])

    return X, nmo_per_kpt


def setup_basis_map(nmo_pk, nkpts):
    """
    Map each ``(orbital, k-point)`` pair onto its index in the combined
    supercell basis.

    Parameters
    ----------
    nmo_pk : sequence of int
        Number of orbitals at each k-point.
    nkpts : int
        Number of k-points.

    Returns
    -------
    ik2n : numpy.ndarray
        ``(nmo_max, nkpts)`` index map; -1 where a k-point has no such orbital.
    nmo_tot : int
        Size of the combined basis.
    """
    nmo_max = int(np.max(nmo_pk))
    ik2n = -1 * np.ones((nmo_max, nkpts), dtype=np.int32)

    count = 0
    for ki in range(nkpts):
        for i in range(nmo_pk[ki]):
            ik2n[i, ki] = count
            count += 1

    return ik2n, count


def _zero_electron_energy(cell, kpts, nelectron, exxdiv='ewald') -> float:
    """
    The constant energy: the nuclear repulsion of every cell, plus the Madelung
    correction when the exchange divergence is treated by Ewald summation.
    """
    from pyscf.pbc import tools

    e0 = len(kpts) * cell.energy_nuc()
    if exxdiv == 'ewald':
        emad = -0.5 * nelectron * tools.pbc.madelung(cell, kpts)
        logger.info("adding ewald correction to the energy: %s", emad)
        e0 += emad
    return e0


def _default_nelec(cell, nkpts):
    """``(nup, ndown)`` for the whole supercell, from the cell's own counts."""
    nup = nkpts * (cell.nelectron + cell.spin) // 2
    return nup, nkpts * cell.nelectron - nup


# ----------------------------------------------------------------------
# the Cholesky solver
# ----------------------------------------------------------------------

class PeriodicCholesky:
    """
    Modified Cholesky factorization of the two-body integrals of a periodic
    system.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        PySCF cell.
    kpts : numpy.ndarray
        k-points, shape ``(nkpts, 3)``.
    nmo_pk : sequence of int
        Number of orbitals at each k-point.
    kp_sym : bool
        Factorize each momentum transfer separately (True) or the full k-point
        pair matrix at once (False). See the module docstring.
    maxvecs : int, optional
        Multiplier bounding the number of Cholesky vectors: the loop stops at
        ``maxvecs * nmo_max`` per momentum transfer when `kp_sym`, and at
        ``maxvecs * nmo_tot`` otherwise. Default 20.
    gtol_chol : float, optional
        Stop once the largest residual falls below this. Default 1e-5.
    verbose : bool, optional
        Log per-iteration progress.

    Notes
    -----
    The Cholesky buffer is bounded by the loop that fills it —
    ``maxvecs * nmo_max`` per momentum transfer when `kp_sym`, and
    ``maxvecs * nmo_tot`` otherwise.
    """

    def __init__(self, cell, kpts, nmo_pk, *, kp_sym, maxvecs=20,
                 gtol_chol=1e-5, verbose=True) -> None:
        from pyscf.pbc import df, tools

        self.cell = cell
        self.kpts = kpts
        self.nkpts = len(kpts)
        self.nmo_pk = np.asarray(nmo_pk)
        self.nmo_max = int(np.max(self.nmo_pk))
        self.nmo_tot = int(np.sum(self.nmo_pk))
        self.kp_sym = kp_sym
        self.gtol_chol = gtol_chol
        self.verbose = verbose

        self.max_cholesky_vectors = maxvecs * (self.nmo_max if kp_sym else self.nmo_tot)

        # the k-point(-pair) axis and the orbital-pair axis of the work arrays
        self.nkk = self.nkpts if kp_sym else self.nkpts * self.nkpts
        self.nij = self.nmo_max * self.nmo_max
        logger.info("total number of orbitals: %d", self.nmo_tot)

        if kp_sym:
            self.QKToK2, self.kminus = construct_qk_maps(cell, kpts)
            self.kconserv = None
        else:
            self.QKToK2 = self.kminus = None
            self.kconserv = tools.get_kconserv(cell, kpts)

        self.gmap, self.Qi, self.ngs = generate_grid_shifts(cell)
        self.df = df.FFTDF(cell, kpts)

    # -- mode-dependent bookkeeping ------------------------------------

    def momentum_blocks(self):
        """
        The momentum transfers to factorize, in order.

        One entry per :math:`Q` not related to an earlier one by
        :math:`Q \\rightarrow -Q` when `kp_sym`, and a single ``None`` block
        covering all k-point pairs otherwise.
        """
        if not self.kp_sym:
            return [None]
        return [Q for Q in range(self.nkpts) if Q <= self.kminus[Q]]

    def k_pairs(self, block):
        """The ``(k1, k2)`` pairs of momentum block `block`."""
        if self.kp_sym:
            return [(k1, self.QKToK2[block][k1]) for k1 in range(self.nkpts)]
        return [(k1, k2) for k1 in range(self.nkpts)
                for k2 in range(self.nkpts)]

    def _row(self, k1, k2) -> int:
        """Index of the ``(k1, k2)`` pair along the work arrays' k axis."""
        return k1 if self.kp_sym else k1 * self.nkpts + k2

    def _conserves_momentum(self, k1, k2, k3, k4) -> bool:
        """
        Whether ``(k1, k2)`` couples to the pivot pair ``(k3, k4)``. Every pair
        within one momentum block does when `kp_sym`; otherwise k-point
        conservation has to be checked explicitly.
        """
        return True if self.kp_sym else k3 == self.kconserv[k1, k2, k4]

    def _new_done(self):
        """
        Array marking pivots already used, so a repeat can be caught.

        Only ``k3`` is needed when `kp_sym`, since ``k4`` follows from it within
        a momentum block.
        """
        shape = ((self.nkpts, self.nmo_max, self.nmo_max) if self.kp_sym
                 else (self.nkpts, self.nkpts, self.nmo_max, self.nmo_max))
        return np.zeros(shape, dtype=np.int32)

    def _done_index(self, k3, k4, i3, i4):
        return (k3, i3, i4) if self.kp_sym else (k3, k4, i3, i4)

    # -- shared numerics -----------------------------------------------

    def generate_orbital_products(self, pairs, X, Xaoik, Xaolj) -> None:
        r"""
        Fill the left and right pair densities for a set of k-point pairs.

        ``Xaolj`` holds :math:`\rho_{lj}(G)`; ``Xaoik`` holds the same products
        multiplied by the Coulomb kernel.

        Parameters
        ----------
        pairs : list of tuple(int, int)
            The ``(k1, k2)`` pairs to fill.
        X : sequence of numpy.ndarray
            Per-k-point transformation into the working basis.
        Xaoik, Xaolj : numpy.ndarray
            Output buffers, ``(nkk, ngs, nij)``.
        """
        from pyscf.pbc import tools

        for k, (k1, k2) in enumerate(pairs):
            npairs = self.nmo_pk[k1] * self.nmo_pk[k2]

            Xaoik[k, :, 0:npairs] = self.df.get_mo_pairs_G(
                (X[k1].copy(), X[k2].copy()),
                (self.kpts[k1], self.kpts[k2]),
                (self.kpts[k2] - self.kpts[k1]),
                compact=False)

            Xaolj[k, :, :] = Xaoik[k, :, :]
            coulG = tools.get_coulG(self.cell, self.kpts[k2] - self.kpts[k1],
                                    mesh=self.df.mesh)
            Xaoik[k, :, :] *= (coulG * self.cell.vol / self.ngs**2).reshape(-1, 1)

    def generate_diagonal(self, pairs, Xaoik, Xaolj):
        """
        The diagonal of the two-body matrix, which is the initial Cholesky
        residual, together with its largest element.

        Returns
        -------
        residual : numpy.ndarray
            ``(nkk, nij)`` residual.
        pivot : tuple(int, int, int, int, float)
            ``(k1, k2, i1, i2, magnitude)`` of the largest element.
        """
        residual = np.zeros((self.nkk, self.nij), dtype=np.float64)

        for k, (k1, k2) in enumerate(pairs):
            for ij in range(min(self.nij, self.nmo_pk[k1] * self.nmo_pk[k2])):
                intg = np.dot(Xaoik[k, :, ij], Xaolj[k, :, ij].conj())
                if (intg.real < 0) or (abs(intg.imag) > 1e-9):
                    logger.error("negative or complex diagonal term: "
                                 "%d %d %d %d %13.8e", k1,
                                 ij // self.nmo_pk[k2], k2,
                                 ij % self.nmo_pk[k2], intg)

                residual[k, ij] = intg.real

        return residual, self._largest_residual(pairs, residual)

    def _largest_residual(self, pairs, residual):
        """
        The pivot for the next Cholesky vector: the largest remaining residual,
        as ``(k1, k2, i1, i2, magnitude)``.
        """
        maxv = 0.0
        pivot = (-1, -1, -1, -1)

        for k, (k1, k2) in enumerate(pairs):
            for ij in range(min(self.nij, self.nmo_pk[k1] * self.nmo_pk[k2])):
                if abs(residual[k, ij]) > maxv:
                    maxv = abs(residual[k, ij])
                    pivot = (k1, k2, ij // self.nmo_pk[k2],
                             ij % self.nmo_pk[k2])

        return (*pivot, maxv)

    def _shifted_pivot_column(self, k1, k2, k3, k4, Xkl0, Xkl):
        r"""
        The pivot's pair density, shifted onto this pair's FFT grid.

        A nonzero :math:`Q = k_2 - k_1 + k_3 - k_4` means the two pairs use
        grids offset by a reciprocal-lattice vector, so the pivot column has to
        be re-indexed through `generate_grid_shifts`.
        """
        q1 = self.kpts[k2] - self.kpts[k1] + self.kpts[k3] - self.kpts[k4]
        if np.sum(abs(q1)) <= 1e-9:
            return Xkl0[0:self.ngs]

        for ii in range(_NUM_GRID_SHIFTS):
            if np.sum(np.linalg.norm(q1 - self.Qi[ii, :])) < 1e-12:
                Xkl[:] = Xkl0[self.gmap[ii, :]]
                return Xkl

        raise RuntimeError(f"Could not find the reciprocal lattice shift for Q = {q1}")

    def run(self, X):
        """
        Factorize the two-body integrals, one momentum block at a time.

        Parameters
        ----------
        X : sequence of numpy.ndarray
            Per-k-point transformation into the working basis.

        Yields
        ------
        block : int or None
            The momentum transfer just factorized, or None in supercell mode.
        cholvecs : numpy.ndarray
            The block's Cholesky vectors, ``(nkk, nij, numv)``. The buffer is
            reused between blocks, so consume each one before asking for the
            next.
        """
        ngs = self.ngs
        maxvecs = self.max_cholesky_vectors

        logger.info("approx total memory for orbital products: %.2e GB",
                    2 * 16 * self.nkk * ngs * self.nij / 1024**3)

        Xaoik = np.zeros((self.nkk, ngs, self.nij), dtype=np.complex128)
        Xaolj = np.zeros((self.nkk, ngs, self.nij), dtype=np.complex128)
        cholvecs = np.zeros((self.nkk, self.nij, maxvecs), dtype=np.complex128)
        Xkl = np.zeros(ngs, dtype=np.complex128)
        Xkl0 = np.zeros(ngs, dtype=np.complex128)
        done = self._new_done()

        for block in self.momentum_blocks():
            pairs = self.k_pairs(block)
            logger.info("calculating factorization for momentum block %s", block)

            cholvecs[:] = 0
            done[:] = 0

            start = time.time()
            self.generate_orbital_products(pairs, X, Xaoik, Xaolj)
            logger.info("time to generate orbital products: %13.8e", time.time() - start)

            residual, (k3, k4, i3, i4, vmax) = self.generate_diagonal(
                pairs, Xaoik, Xaolj)
            done[self._done_index(k3, k4, i3, i4)] = 1

            numv = 0
            while True:
                if numv >= maxvecs:
                    warnings.warn(
                        "Too many vectors needed to converge the Cholesky "
                        f"factorization ({maxvecs}). Increase maxvecs.",
                        RuntimeWarning
                    )
                    break

                # the pivot's pair density, and its Cholesky history
                row = self._row(k3, k4)
                column = i3 * self.nmo_pk[k4] + i4
                Xkl0[:] = Xaolj[row, 0:ngs, column]
                history = cholvecs[row, column, 0:numv].copy()

                # 1. evaluate the new column (ik|i_max k_max)
                for k, (k1, k2) in enumerate(pairs):
                    if not self._conserves_momentum(k1, k2, k3, k4):
                        continue

                    pivot_column = self._shifted_pivot_column(k1, k2, k3, k4, Xkl0, Xkl)
                    npairs = self.nmo_pk[k1] * self.nmo_pk[k2]
                    cholvecs[k, 0:npairs, numv] = np.dot(
                        Xaoik[k, :, 0:npairs].T, pivot_column.conj())

                # 2. subtract the projection along the previous components
                cholvecs[:, :, numv] -= np.dot(cholvecs[:, :, 0:numv], history.conj())
                cholvecs[:, :, numv] /= math.sqrt(vmax)

                residual -= (cholvecs[:, :, numv] * cholvecs[:, :, numv].conj()).real

                k3, k4, i3, i4, vmax = self._largest_residual(pairs, residual)

                if self.verbose:
                    logger.info("cholesky iteration %d: max residual %13.8e", numv, vmax)

                numv += 1
                if vmax < self.gtol_chol:
                    break

                if done[self._done_index(k3, k4, i3, i4)] > 0:
                    raise RuntimeError(
                        "Error in Cholesky decomposition: pivot "
                        f"({k3}, {k4}, {i3}, {i4}) was already used, residual {vmax}"
                    )
                done[self._done_index(k3, k4, i3, i4)] = 1

            yield block, cholvecs[:, :, :numv]


class PeriodicHamiltonian(Hamiltonian):
    r"""
    A k-point-factorized Hamiltonian for a periodic system.

    Parameters
    ----------
    hcore : list of numpy.ndarray
        One-body Hamiltonian, one ``(nmo_pk[k], nmo_pk[k])`` block per k-point.
    chol : dict
        Cholesky data as ``{Q: L_Q}``, with ``L_Q`` of shape
        ``(nkpts, nmo_max**2 * nchol_Q)``. Only momentum transfers with
        ``Q <= minus_k[Q]`` are stored; the rest follow by symmetry.
    kpts : numpy.ndarray
        k-points, shape ``(nkpts, 3)``.
    nmo_pk : sequence of int
        Number of orbitals at each k-point.
    qk_to_k2 : numpy.ndarray
        ``qk_to_k2[Q, k1] = k2``, the momentum-transfer map.
    minus_k : numpy.ndarray
        ``minus_k[Q]`` is the index of :math:`-Q`.
    nchol_pk : numpy.ndarray
        Number of Cholesky vectors per momentum transfer.
    enuc : float, optional
        Constant energy, including the Madelung correction. Default 0.0.
    nelec : tuple(int, int), optional
        ``(nup, ndown)`` for the whole supercell. Default ``(0, 0)``.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry. Default `SpinSymm.CLOSED`.

    Notes
    -----
    A supercell Hamiltonian is this same class with ``nkpts == 1`` — the Γ point
    of the supercell — so there is no separate representation to branch on.
    """

    def __init__(self, hcore, chol, kpts, nmo_pk, qk_to_k2, minus_k, nchol_pk,
                 enuc=0.0, nelec=(0, 0), spin_symm=SpinSymm.CLOSED) -> None:
        super().__init__(spin_symm=spin_symm)

        self.hcore = hcore
        self.chol = chol
        self.kpts = np.asarray(kpts)
        self.nmo_pk = np.asarray(nmo_pk)
        self.qk_to_k2 = np.asarray(qk_to_k2)
        self.minus_k = np.asarray(minus_k)
        self.nchol_pk = np.asarray(nchol_pk)
        self.enuc = float(np.real(enuc))
        self.nelec = tuple(nelec)

        nkpts = len(self.kpts)
        for name, array, shape in (('nmo_pk', self.nmo_pk, (nkpts,)),
                                   ('qk_to_k2', self.qk_to_k2, (nkpts, nkpts)),
                                   ('minus_k', self.minus_k, (nkpts,)),
                                   ('nchol_pk', self.nchol_pk, (nkpts,))):
            if array.shape != shape:
                raise ValueError(
                    f"{name} has shape {array.shape}, expected {shape} for "
                    f"{nkpts} k-points"
                )

    @property
    def nkpts(self) -> int:
        """Number of k-points. 1 for a supercell (Γ-point) Hamiltonian."""
        return len(self.kpts)

    @property
    def nmo_tot(self) -> int:
        """Total number of orbitals across all k-points."""
        return int(np.sum(self.nmo_pk))

    @property
    def nmo_max(self) -> int:
        """Largest per-k-point orbital count."""
        return int(np.max(self.nmo_pk))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pyscf(cls, source, kpoint_symmetry=True, chol_cut=1e-5,
                   maxvecs=20, exxdiv='ewald', nelec=None,
                   verbose=False) -> "PeriodicHamiltonian":
        """
        Generate a periodic Hamiltonian from a PySCF ``pbc`` SCF calculation.

        Parameters
        ----------
        source : str or pathlib.Path or dict
            A PySCF checkpoint file, or an already-loaded ``scf_data`` mapping
            from `safiretools.convert.pyscf.load_pyscf_chk`. Uses the keys
            ``'hcore'``, ``'X'``, ``'cell'``, ``'kpts'`` and ``'nmo_pk'``.
        kpoint_symmetry : bool, optional
            Factorize per momentum transfer (True) rather than as one supercell
            (False). Default True. A supercell result comes back as a Γ-point
            Hamiltonian, i.e. with ``nkpts == 1``.
        chol_cut : float, optional
            Cholesky convergence threshold. Default 1e-5.
        maxvecs : int, optional
            Cholesky-vector bound multiplier. Default 20.
        exxdiv : str, optional
            ``'ewald'`` adds the Madelung correction to the constant energy.
        nelec : tuple(int, int), optional
            Overrides the electron count taken from the cell.
        verbose : bool, optional
            Log per-iteration Cholesky progress.

        Returns
        -------
        PeriodicHamiltonian

        Notes
        -----
        The whole factorization is built in memory and written by `to_hdf5`,
        serially. CoQuí is the supported route for production-sized solids.
        """
        from safiretools.convert.pyscf import as_scf_data

        scf_data = as_scf_data(source, periodic=True)

        cell, kpts, X = scf_data['cell'], scf_data['kpts'], scf_data['X']
        nmo_pk = np.asarray(scf_data['nmo_pk'])
        nelec = nelec if nelec is not None else _default_nelec(cell, len(kpts))

        solver = PeriodicCholesky(cell, kpts, nmo_pk, kp_sym=kpoint_symmetry,
                                  maxvecs=maxvecs, gtol_chol=chol_cut,
                                  verbose=verbose)

        hcore_pk = _transform_hcore(scf_data['hcore'], X, nmo_pk)
        enuc = _zero_electron_energy(cell, kpts, sum(nelec), exxdiv)

        if not kpoint_symmetry:
            (_, cholvecs), = solver.run(X)
            return cls(**_supercell_layout(hcore_pk, cholvecs, solver),
                       enuc=enuc, nelec=nelec)

        chol = {}
        nchol_pk = np.zeros(len(kpts), dtype=np.int32)
        for Q, block in solver.run(X):
            chol[Q] = _kpoint_block(block, solver)
            nchol_pk[Q] = block.shape[-1]

        return cls(hcore=hcore_pk, chol=chol, kpts=kpts, nmo_pk=nmo_pk,
                   qk_to_k2=solver.QKToK2, minus_k=solver.kminus,
                   nchol_pk=nchol_pk, enuc=enuc, nelec=nelec)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_hdf5(self, path) -> None:
        """
        Write this Hamiltonian in the ``Hamiltonian/KPFactorized`` format.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist. A Hamiltonian
            already in the file is replaced; everything else — notably a
            ``Wavefunction`` — is left alone, so a Hamiltonian and a
            wavefunction can share one file in either order.
        """
        with open_for_hamiltonian(path) as fh5:
            write_hamiltonian_format(fh5, 'kpoint')
            group = fh5['Hamiltonian']
            _write_kpoint_descriptors(group, self.kpts, self.nmo_pk, self.qk_to_k2,
                                      self.minus_k, self.nelec, self.enuc)
            for ki in range(self.nkpts):
                _write_kpoint_h1(group, ki, self.nmo_pk[ki], self.hcore[ki])

            group.create_dataset("NCholPerKP", data=self.nchol_pk)

            kp_group = fh5.create_group(KPOINT_GROUP)
            for Q, L in self.chol.items():
                kp_group.create_dataset(f"L{Q}", data=to_complex(L))

    def to_fcidump(self, path, tol=1e-8, ctol=1e-12, sym=1, cplx=True,
                   paren=False, use_spinor=False) -> None:
        """
        Write this Hamiltonian as a plain-text FCIDUMP file, over the combined
        basis of every k-point.

        The two-electron integrals are reconstructed from the Cholesky vectors
        and written out in full, so the file is far larger than the HDF5
        `to_hdf5` writes and this is only practical for small cells and meshes.

        Parameters
        ----------
        path : str or pathlib.Path
            FCIDUMP file to write. Overwritten if it exists.
        tol : float, optional
            Only write integrals above this magnitude. Default 1e-8.
        ctol : float, optional
            Largest imaginary part tolerated when `cplx` is False.
            Default 1e-12.
        sym : int, optional
            Write only the symmetry-inequivalent integrals of this
            permutational symmetry, 1, 4 or 8. Default 1, i.e. everything.
        cplx : bool, optional
            Write in the complex format. Default True, and what a k-point
            factorization normally needs.
        paren : bool, optional
            Write complex values parenthesized as ``(real,imag)`` rather than
            in two columns.
        use_spinor : bool, optional
            Not implemented for a k-point Hamiltonian; see below.

        Raises
        ------
        ValueError
            If the k-points do not all carry the same number of orbitals, or if
            `cplx` is False and the integrals have imaginary parts above `ctol`.
        NotImplementedError
            If `use_spinor` is set.

        Notes
        -----
        The orbital index of the FCIDUMP is the combined
        ``k * nmo_pk + i``, which the writer assumes to be laid out uniformly,
        so a mesh whose k-points carry different orbital counts is rejected
        rather than written wrongly. That happens when linear dependencies are
        removed per k-point, i.e. when ``get_ortho_ao`` is given a nonzero
        ``lindep_cutoff``.
        """
        if len(set(int(nmo) for nmo in self.nmo_pk)) > 1:
            raise ValueError(
                f"the k-points carry different orbital counts ({list(self.nmo_pk)}), "
                "which the combined FCIDUMP orbital index cannot express"
            )

        chol, nchol_pk = self._chol_all_momenta()

        write_fcidump_kpoint(path, self.hcore, chol, self.enuc, self.nmo_tot,
                             self.nelec, self.nmo_pk, nchol_pk, self.qk_to_k2,
                             tol=tol, sym=sym, paren=paren, cplx=cplx, ctol=ctol,
                             use_spinor=use_spinor)

    def _chol_all_momenta(self):
        r"""
        Every momentum transfer's Cholesky block in :math:`Q` order, and the
        Cholesky-vector count that goes with each.

        `chol` stores only the momentum transfers with ``Q <= minus_k[Q]``; the
        partner of each is recovered by remapping the k-points and transposing
        the orbital pair,

        .. math:: L^{-Q}_{k_1}[i, j, n] = \big(L^{Q}_{k_2}[j, i, n]\big)^*,
                  \quad k_2 = \mathrm{qk\_to\_k2}[-Q, k_1]

        which is what any reader of the on-disk format has to do as well.
        `nchol_pk` is zero at a reconstructed momentum transfer, since nothing
        was factorized there, so it comes back filled in from the partner.

        Returns
        -------
        chol : list of numpy.ndarray
            One ``(nkpts, nmo_max**2 * nchol_Q)`` block per momentum transfer.
        nchol_pk : numpy.ndarray
            Cholesky-vector count per momentum transfer.

        Raises
        ------
        ValueError
            If neither a momentum transfer nor its partner has a block.
        """
        nmo = self.nmo_max
        nchol_pk = np.array(self.nchol_pk, dtype=np.int32)
        blocks = []

        for Q in range(self.nkpts):
            if Q in self.chol:
                blocks.append(np.asarray(self.chol[Q]))
                continue

            partner = int(self.minus_k[Q])
            if partner not in self.chol:
                raise ValueError(
                    f"no Cholesky vectors for momentum transfer {Q}, nor for its "
                    f"-Q partner {partner}"
                )

            nchol = int(nchol_pk[partner])
            nchol_pk[Q] = nchol

            stored = np.asarray(self.chol[partner]).reshape(self.nkpts, nmo, nmo,
                                                            nchol)
            block = np.empty_like(stored)
            for k1 in range(self.nkpts):
                block[k1] = stored[self.qk_to_k2[Q][k1]].transpose(1, 0, 2).conj()
            blocks.append(block.reshape(self.nkpts, nmo * nmo * nchol))

        return blocks, nchol_pk

    @classmethod
    def _read_hdf5(cls, path, fmt: str) -> "PeriodicHamiltonian":
        """Read a periodic Hamiltonian written by `to_hdf5`."""
        with h5.File(path, 'r') as fh5:
            group = fh5['Hamiltonian']
            dims = group['dims'][...]
            nkpts = int(dims[2])
            nelec = (int(dims[4]), int(dims[5]))
            enuc = float(group['Energies'][...][0])

            kpts = group['KPoints'][...]
            nmo_pk = group['NMOPerKP'][...]
            qk_to_k2 = group['QKTok2'][...]
            minus_k = group['MinusK'][...]
            nchol_pk = group['NCholPerKP'][...]

            hcore = [from_complex(group[f'H1_kp{ki}'][...]) for ki in range(nkpts)]
            chol = {Q: from_complex(fh5[f'{KPOINT_GROUP}/L{Q}'][...])
                    for Q in range(nkpts) if f'L{Q}' in fh5[KPOINT_GROUP]}

        return cls(hcore=hcore, chol=chol, kpts=kpts, nmo_pk=nmo_pk,
                   qk_to_k2=qk_to_k2, minus_k=minus_k, nchol_pk=nchol_pk,
                   enuc=enuc, nelec=nelec)


def write_rhoG(scf_data, path, gcut, kpoint_symmetry=True,
               verbose=False) -> None:
    r"""
    Write the real-space density :math:`\rho(G)` on the FFT grid.

    .. warning:: Not implemented.

    Parameters
    ----------
    scf_data : dict
        Unpacked PySCF checkpoint; see `PeriodicHamiltonian.from_pyscf`.
    path : str or pathlib.Path
        HDF5 file to write.
    gcut : float
        Plane-wave cutoff for the density.
    kpoint_symmetry : bool, optional
        Which factorization the density would follow.
    verbose : bool, optional
        Log progress.

    Raises
    ------
    NotImplementedError
        Always.

    Notes
    -----
    The signature is kept so that a working implementation has an obvious place
    to land.
    """
    raise NotImplementedError("write_rhoG is not implemented.")


# ----------------------------------------------------------------------
# assembling and writing
# ----------------------------------------------------------------------

def _transform_hcore(hcore, X, nmo_pk):
    """The one-body Hamiltonian at each k-point, in the working basis."""
    return [np.dot(X[ki][:, 0:nmo_pk[ki]].T.conj(),
                   np.dot(hcore[ki], X[ki][:, 0:nmo_pk[ki]]))
            for ki in range(len(nmo_pk))]


def _supercell_layout(hcore_pk, cholvecs, solver) -> dict:
    r"""
    Recast a supercell factorization as a Γ-point k-point Hamiltonian.

    A supercell Hamiltonian *is* the Γ point of the supercell, so it needs no
    representation of its own: collapse the k-point axis into one combined basis
    of size ``nmo_tot`` and hand back the same constructor arguments any k-point
    Hamiltonian takes, with ``nkpts == 1``.

    Parameters
    ----------
    hcore_pk : list of numpy.ndarray
        Per-k-point one-body blocks in the working basis.
    cholvecs : numpy.ndarray
        The serial solver's ``(nkk, nij, nchol)`` Cholesky block.
    solver : PeriodicCholesky
        The solver that produced `cholvecs`, for its orbital counts.

    Returns
    -------
    dict
        ``hcore``, ``chol``, ``kpts``, ``nmo_pk``, ``qk_to_k2``, ``minus_k`` and
        ``nchol_pk``, ready to pass to `PeriodicHamiltonian`.

    Notes
    -----
    The one-body part is block diagonal in k — the one-body operator conserves
    crystal momentum — while the Cholesky vectors couple every ``(k1, k2)``
    pair, which is exactly what makes the combined basis necessary.

    The :math:`1/\sqrt{N_k}` factor is the same one the k-point format carries,
    with :math:`N_k` the *original* k-point count rather than the single Γ point
    the result is expressed in.
    """
    nmo_pk = solver.nmo_pk
    nkpts = len(solver.kpts)
    nchol = cholvecs.shape[-1]

    ik2n, nmo_tot = setup_basis_map(nmo_pk, nkpts)
    factor = 1.0 / math.sqrt(nkpts)

    hcore = np.zeros((nmo_tot, nmo_tot), dtype=np.complex128)
    for ki, block in enumerate(hcore_pk):
        for i in range(nmo_pk[ki]):
            for j in range(nmo_pk[ki]):
                hcore[ik2n[i, ki], ik2n[j, ki]] = block[i, j]

    L = np.zeros((nmo_tot * nmo_tot, nchol), dtype=np.complex128)
    for k, (k1, k2) in enumerate(solver.k_pairs(None)):
        for ij in range(nmo_pk[k1] * nmo_pk[k2]):
            i, j = divmod(ij, nmo_pk[k2])
            L[ik2n[i, k1] * nmo_tot + ik2n[j, k2], :] = cholvecs[k, ij, :] * factor

    return {
        'hcore': [hcore],
        'chol': {0: L.reshape(1, nmo_tot * nmo_tot * nchol)},
        'kpts': np.zeros((1, 3)),
        'nmo_pk': np.array([nmo_tot], dtype=np.int32),
        'qk_to_k2': np.zeros((1, 1), dtype=np.int32),
        'minus_k': np.zeros(1, dtype=np.int32),
        'nchol_pk': np.array([nchol], dtype=np.int32),
    }


def _kpoint_block(cholvecs, solver):
    r"""
    Reshape a Cholesky block into the ``KPFactorized`` layout,
    ``(nkpts, nmo_max**2 * nchol)``.

    Carries the same :math:`1/\sqrt{N_k}` normalization the on-disk format
    expects to find already applied.
    """
    nkpts = len(solver.kpts)
    nchol = cholvecs.shape[-1]
    factor = 1.0 / math.sqrt(nkpts)

    return cholvecs.reshape(nkpts, solver.nmo_max**2 * nchol) * factor


def _write_kpoint_descriptors(group, kpts, nmo_pk, qk_to_k2, minus_k, nelec,
                              enuc) -> None:
    """Write the k-point Hamiltonian's descriptor datasets into `group`."""
    nkpts = len(kpts)

    group.create_dataset(
        "dims",
        data=np.array([0, 0, nkpts, int(np.sum(nmo_pk)), nelec[0], nelec[1], 0, 0],
                      dtype=np.int32))
    group.create_dataset("ComplexIntegrals", data=np.array([1], dtype=np.int32))
    group.create_dataset("Energies", data=np.array([enuc, 0.], dtype=np.float64))
    group.create_dataset("KPoints", data=np.asarray(kpts, dtype=np.float64))
    group.create_dataset("NMOPerKP", data=np.asarray(nmo_pk, dtype=np.int32))
    group.create_dataset("QKTok2", data=np.asarray(qk_to_k2, dtype=np.int32))
    group.create_dataset("MinusK", data=np.asarray(minus_k, dtype=np.int32))


def _write_kpoint_h1(group, ki, nmo, h1) -> None:
    """Write the one-body block at k-point `ki`, interleaved complex."""
    if h1.shape != (nmo, nmo):
        raise ValueError(f"H1 at kpoint {ki} has shape {h1.shape}, expected ({nmo}, {nmo})")

    group.create_dataset(f"H1_kp{ki}", data=to_complex(h1))
