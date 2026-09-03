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

.. note:: The sparse ``Hamiltonian/Factorized`` format the supercell path used to
          write has been removed from the AFQMC executable, which is why the
          supercell path emits the k-point format instead.
"""

import logging
import math
import os
import time
import warnings

import numpy as np
import h5py as h5

from safiretools.hamiltonian.base import (
    Hamiltonian,
    clear_hamiltonian,
    open_for_hamiltonian,
)
from safiretools.hdf5 import from_complex, to_complex
from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

KPOINT_GROUP = 'Hamiltonian/KPFactorized'
"""Group the k-point-symmetric factorization is written under."""

_NUM_GRID_SHIFTS = 27
"""Number of reciprocal-lattice shifts searched: all of (-1, 0, 1)^3."""


# ----------------------------------------------------------------------
# work partitioning
# ----------------------------------------------------------------------

def fair_share(N: int, npr: int, rk: int):
    """
    Split `N` items over `npr` ranks and return rank `rk`'s ``[start, end)``.

    The first ``N % npr`` ranks take one extra item each.
    """
    npp, nxtra = N // npr, N % npr
    if rk < nxtra:
        i0 = rk * (npp + 1)
        return i0, i0 + npp + 1
    i0 = rk * npp + nxtra
    return i0, i0 + npp


def bisect(a, x, lo=0, hi=None) -> int:
    """
    The index at which `x` would be inserted into the sorted sequence `a`,
    after any equal entries.

    Equivalent to `bisect.bisect_right`, kept here so the partition logic has no
    dependency on element type beyond ``<``.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class Partition:
    """
    Distribution of the Cholesky work over MPI ranks.

    Each rank owns a slice of the k-point (or k-point-pair) axis and a slice of
    the orbital-pair axis. When there are more ranks than k-points, the ranks
    are grouped ``nproc_pk`` to a k-point and split along orbital pairs instead.

    Parameters
    ----------
    comm : mpi4py communicator
        Communicator the work is spread over.
    maxvecs : int
        Multiplier for the Cholesky-vector buffer; see `PeriodicCholesky`.
    nmo_tot : int
        Total number of orbitals across all k-points.
    nmo_max : int
        Largest per-k-point orbital count.
    nkpts : int
        Number of k-points.
    kp_sym : bool, optional
        Partition over single k-points (True) rather than k-point pairs
        (False). Default False.

    Attributes
    ----------
    kk0, kkN, nkk : int
        This rank's slice of the k-point (pair) axis.
    ij0, ijN, nij : int
        This rank's slice of the orbital-pair axis.
    nproc_pk : int
        Ranks per k-point.
    n2k1, n2k2 : numpy.ndarray
        For ``kp_sym=False``, the ``(k1, k2)`` pair each local row stands for.
    """

    def __init__(self, comm, maxvecs, nmo_tot, nmo_max, nkpts, kp_sym=False) -> None:
        self.maxvecs = maxvecs * nmo_tot
        self.rank = comm.rank
        self.size = comm.size

        if comm.size <= nkpts:
            work = nkpts if kp_sym else nkpts * nkpts
            self.kkbounds = np.zeros(comm.size + 1, dtype=np.int32)
            for i in range(comm.size):
                self.kkbounds[i], self.kkbounds[i + 1] = fair_share(work, comm.size, i)
            self.kk0, self.kkN = fair_share(work, comm.size, comm.rank)
            self.nproc_pk = 1
        else:
            if comm.size % nkpts != 0:
                raise ValueError(
                    "If nproc > nkpts, nproc must evenly divide the number of "
                    f"k-points; got nproc={comm.size}, nkpts={nkpts}"
                )
            self.nproc_pk = comm.size // nkpts
            if kp_sym:
                self.kk0 = comm.rank // self.nproc_pk
                self.kkN = self.kk0 + 1
            else:
                self.mykpt = comm.rank // self.nproc_pk
                self.kk0 = self.mykpt * nkpts
                self.kkN = self.kk0 + nkpts

        self.ij0, self.ijN = fair_share(nmo_max * nmo_max, self.nproc_pk,
                                        comm.rank % self.nproc_pk)
        self.nij = self.ijN - self.ij0
        self.nkk = self.kkN - self.kk0

        if not kp_sym:
            self.n2k1 = np.array([k // nkpts for k in range(self.kk0, self.kkN)],
                                 dtype=np.int32)
            self.n2k2 = np.array([k % nkpts for k in range(self.kk0, self.kkN)],
                                 dtype=np.int32)


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
    system, in parallel over MPI.

    Parameters
    ----------
    comm : mpi4py communicator
        Communicator the work is spread over.
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
    afqmctools allocated the Cholesky buffer at ``maxvecs * nmo_tot`` in both
    modes while the k-point-symmetric loop could never exceed
    ``maxvecs * nmo_max``, over-allocating by a factor of ``nkpts``. The buffer
    now matches the loop bound; the factorization is unchanged.
    """

    def __init__(self, comm, cell, kpts, nmo_pk, *, kp_sym, maxvecs=20,
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

        self.part = Partition(comm, maxvecs, self.nmo_tot, self.nmo_max,
                              self.nkpts, kp_sym=kp_sym)
        logger.info("each kpoint is distributed across %d mpi tasks", self.part.nproc_pk)
        logger.info("total number of orbitals: %d", self.nmo_tot)

        if kp_sym:
            self.QKToK2, self.kminus = construct_qk_maps(cell, kpts)
            self.kconserv = None
        else:
            self.QKToK2 = self.kminus = None
            self.kconserv = tools.get_kconserv(cell, kpts)

        self.gmap, self.Qi, self.ngs = generate_grid_shifts(cell)
        self.df = df.FFTDF(cell, kpts)
        self.maxres_buff = np.zeros(5 * comm.size, dtype=np.float64)

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
        """This rank's local ``(k1, k2)`` pairs for momentum block `block`."""
        if self.kp_sym:
            return [(k1, self.QKToK2[block][k1])
                    for k1 in range(self.part.kk0, self.part.kkN)]
        return list(zip(self.part.n2k1, self.part.n2k2))

    def _global_row(self, k1, k2) -> int:
        """
        Index of the ``(k1, k2)`` pair along the distributed k axis. Local
        storage is at ``_global_row(...) - part.kk0``.
        """
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
        Fill the left and right pair densities for this rank's k-point pairs.

        ``Xaolj`` holds :math:`\rho_{lj}(G)`; ``Xaoik`` holds the same products
        multiplied by the Coulomb kernel.

        Parameters
        ----------
        pairs : list of tuple(int, int)
            This rank's ``(k1, k2)`` pairs.
        X : sequence of numpy.ndarray
            Per-k-point transformation into the working basis.
        Xaoik, Xaolj : numpy.ndarray
            Output buffers, ``(nkk, ngs, nij)``.
        """
        from pyscf.pbc import tools

        part = self.part
        for k, (k1, k2) in enumerate(pairs):
            if part.ij0 > self.nmo_pk[k1] * self.nmo_pk[k2]:
                continue

            i0 = part.ij0 // self.nmo_pk[k2]
            iN = part.ijN // self.nmo_pk[k2]
            if part.ijN % self.nmo_pk[k2] != 0:
                iN += 1
            iN = min(iN, self.nmo_pk[k1])

            pij = part.ij0 % self.nmo_pk[k2]
            n_ = min(part.ijN, self.nmo_pk[k1] * self.nmo_pk[k2]) - part.ij0

            X_t = X[k1][:, i0:iN].copy()
            Xaoik[k, :, 0:n_] = self.df.get_mo_pairs_G(
                (X_t, X[k2].copy()),
                (self.kpts[k1], self.kpts[k2]),
                (self.kpts[k2] - self.kpts[k1]),
                compact=False)[:, pij:pij + n_]

            Xaolj[k, :, :] = Xaoik[k, :, :]
            coulG = tools.get_coulG(self.cell, self.kpts[k2] - self.kpts[k1],
                                    mesh=self.df.mesh)
            Xaoik[k, :, :] *= (coulG * self.cell.vol / self.ngs**2).reshape(-1, 1)

    def generate_diagonal(self, pairs, Xaoik, Xaolj):
        """
        The diagonal of the two-body matrix, which is the initial Cholesky
        residual, together with this rank's largest element.

        Returns
        -------
        residual : numpy.ndarray
            ``(nkk, nij)`` residual.
        k1max, k2max, i1max, i2max : int
            Location of the largest element.
        maxv : float
            Its magnitude.
        """
        part = self.part
        residual = np.zeros((part.nkk, part.nij), dtype=np.float64)
        maxv = 0.0
        k1max = k2max = i1max = i2max = -1

        for k, (k1, k2) in enumerate(pairs):
            for ij in range(part.nij):
                if (ij + part.ij0) >= self.nmo_pk[k1] * self.nmo_pk[k2]:
                    break

                intg = np.dot(Xaoik[k, :, ij], Xaolj[k, :, ij].conj())
                if (intg.real < 0) or (abs(intg.imag) > 1e-9):
                    i = (ij + part.ij0) // self.nmo_pk[k2]
                    j = (ij + part.ij0) % self.nmo_pk[k2]
                    logger.error("negative or complex diagonal term: "
                                 "%d %d %d %d %13.8e", k1, i, k2, j, intg)

                residual[k, ij] = intg.real
                if abs(intg) > maxv:
                    maxv = abs(intg)
                    k1max, k2max = k1, k2
                    i1max = (ij + part.ij0) // self.nmo_pk[k2]
                    i2max = (ij + part.ij0) % self.nmo_pk[k2]

        return residual, k1max, k2max, i1max, i2max, maxv

    def _largest_residual(self, pairs, residual):
        """This rank's largest remaining residual and where it sits."""
        part = self.part
        maxv = 0.0
        k1max = k2max = i1max = i2max = -1

        for k, (k1, k2) in enumerate(pairs):
            for ij in range(part.nij):
                if (ij + part.ij0) >= self.nmo_pk[k1] * self.nmo_pk[k2]:
                    break
                if abs(residual[k, ij]) > maxv:
                    maxv = abs(residual[k, ij])
                    k1max, k2max = k1, k2
                    i1max = (ij + part.ij0) // self.nmo_pk[k2]
                    i2max = (ij + part.ij0) % self.nmo_pk[k2]

        return k1max, k2max, i1max, i2max, maxv

    def _pick_pivot(self, comm, k1max, k2max, i1max, i2max, maxv):
        """
        Agree on the global pivot: gather every rank's largest residual and take
        the largest of those.
        """
        comm.Allgather(np.array([k1max, k2max, i1max, i2max, maxv], dtype=np.float64),
                       self.maxres_buff)

        vmax = 0.0
        pivot = (0, 0, 0, 0)
        for i in range(comm.size):
            if self.maxres_buff[i * 5 + 4] > vmax:
                vmax = self.maxres_buff[i * 5 + 4]
                pivot = tuple(int(self.maxres_buff[i * 5 + j]) for j in range(4))

        return (*pivot, vmax)

    def _pivot_owner(self, comm, k3, k4, i3, i4) -> int:
        """The rank holding the pivot column."""
        part = self.part
        kkmax = self._global_row(k3, k4)

        if comm.size <= self.nkpts:
            return bisect(part.kkbounds[1:comm.size + 1], kkmax)

        i34 = i3 * self.nmo_pk[k4] + i4
        for i in range(part.nproc_pk):
            _, ijN_ = fair_share(self.nmo_max * self.nmo_max, part.nproc_pk, i)
            if i34 < ijN_:
                return k3 * part.nproc_pk + i

        raise RuntimeError(
            f"could not locate the rank owning pivot ({k3}, {k4}, {i3}, {i4})"
        )

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

    def run(self, comm, X):
        """
        Factorize the two-body integrals, one momentum block at a time.

        Parameters
        ----------
        comm : mpi4py communicator
            Communicator the work is spread over.
        X : sequence of numpy.ndarray
            Per-k-point transformation into the working basis.

        Yields
        ------
        block : int or None
            The momentum transfer just factorized, or None in supercell mode.
        cholvecs : numpy.ndarray
            This rank's Cholesky vectors for the block, ``(nkk, nij, numv)``.
            The buffer is reused between blocks, so consume each one before
            asking for the next.

        Notes
        -----
        Every rank must iterate this generator to completion in step with the
        others: the loop is collective.
        """
        part = self.part
        ngs = self.ngs
        maxvecs = self.max_cholesky_vectors

        logger.info("approx total memory for orbital products: %.2e GB",
                    2 * 16 * part.nkk * ngs * part.nij / 1024**3)

        Xaoik = np.zeros((part.nkk, ngs, part.nij), dtype=np.complex128)
        Xaolj = np.zeros((part.nkk, ngs, part.nij), dtype=np.complex128)
        cholvecs = np.zeros((part.nkk, part.nij, maxvecs), dtype=np.complex128)
        Xkl = np.zeros(ngs, dtype=np.complex128)
        Xkl0 = np.zeros(ngs + maxvecs, dtype=np.complex128)
        Vbuff = np.zeros(maxvecs, dtype=np.complex128)
        done = self._new_done()

        for block in self.momentum_blocks():
            pairs = self.k_pairs(block)
            logger.info("calculating factorization for momentum block %s", block)

            cholvecs[:] = 0
            done[:] = 0

            start = time.time()
            self.generate_orbital_products(pairs, X, Xaoik, Xaolj)
            logger.info("time to generate orbital products: %13.8e", time.time() - start)

            residual, *pivot_local = self.generate_diagonal(pairs, Xaoik, Xaolj)
            k3, k4, i3, i4, vmax = self._pick_pivot(comm, *pivot_local)
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

                # broadcast the pivot's pair density and its Cholesky history
                owner = self._pivot_owner(comm, k3, k4, i3, i4)
                column = i3 * self.nmo_pk[k4] + i4 - part.ij0
                if comm.rank == owner:
                    local_row = self._global_row(k3, k4) - part.kk0
                    Xkl0[0:ngs] = Xaolj[local_row, 0:ngs, column]
                    Xkl0[ngs:ngs + numv] = cholvecs[local_row, column, 0:numv]
                    Vbuff[0:numv] = cholvecs[local_row, column, 0:numv]
                    comm.Bcast(Xkl0[0:ngs + numv], root=owner)
                else:
                    comm.Bcast(Xkl0[0:ngs + numv], root=owner)
                    Vbuff[0:numv] = Xkl0[ngs:ngs + numv]

                # 1. evaluate the new column (ik|i_max k_max)
                for k, (k1, k2) in enumerate(pairs):
                    if not self._conserves_momentum(k1, k2, k3, k4):
                        continue
                    if part.ij0 > self.nmo_pk[k1] * self.nmo_pk[k2]:
                        continue

                    pivot_column = self._shifted_pivot_column(k1, k2, k3, k4, Xkl0, Xkl)
                    n_ = min(self.nmo_pk[k1] * self.nmo_pk[k2], part.ijN) - part.ij0
                    cholvecs[k, 0:n_, numv] = np.dot(Xaoik[k, :, 0:n_].T,
                                                     pivot_column.conj())

                # 2. subtract the projection along the previous components
                cholvecs[:, :, numv] -= np.dot(cholvecs[:, :, 0:numv], Vbuff[0:numv].conj())
                cholvecs[:, :, numv] /= math.sqrt(vmax)

                residual -= (cholvecs[:, :, numv] * cholvecs[:, :, numv].conj()).real

                pivot_local = self._largest_residual(pairs, residual)
                k3, k4, i3, i4, vmax = self._pick_pivot(comm, *pivot_local)

                if self.verbose and comm.rank == 0:
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

            comm.barrier()
            yield block, cholvecs[:, :, :numv]


# ----------------------------------------------------------------------
# parallel HDF5 handling
# ----------------------------------------------------------------------

class FileHandler:
    """
    Open an HDF5 file for a parallel Cholesky write.

    With parallel HDF5 (`phdf`) every rank opens the same file collectively.
    Without it, rank 0 opens the real file and every other rank opens its own
    ``rank<N>_<filename>``, which rank 0 merges afterwards.

    Parameters
    ----------
    comm : mpi4py communicator
        Communicator.
    filename : str or pathlib.Path
        File to open.
    mode : str, optional
        h5py file mode for the real output file. Default ``'w'``. The per-rank
        scratch files are always truncated, since they only ever hold one run's
        partial Cholesky blocks.
    phdf : bool, optional
        Use parallel HDF5. Default False.

    Examples
    --------
    >>> with FileHandler(comm, filename) as f:
    ...     f.create_dataset("test", data=data)
    """

    def __init__(self, comm, filename, mode="w", phdf=False) -> None:
        self.phdf = phdf
        self.comm = comm

        if phdf:
            self.h5f = h5.File(filename, mode, driver='mpio', comm=comm)
            self.h5f.atomic = False
        elif comm.rank == 0:
            self.h5f = h5.File(filename, mode)
        else:
            self.h5f = h5.File(rank_filename(comm.rank, filename), "w")

    def __enter__(self):
        self.h5f.phdf = self.phdf
        return self.h5f

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5f.close()
        return False


def rank_filename(rank: int, filename):
    """The per-rank scratch file name used when parallel HDF5 is unavailable."""
    filename = os.fspath(filename)
    directory, name = os.path.split(filename)
    return os.path.join(directory, f"rank{rank}_{name}")


# ----------------------------------------------------------------------
# the Hamiltonian
# ----------------------------------------------------------------------

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
    def from_pyscf(cls, scf_data, comm=None, kpoint_symmetry=True, chol_cut=1e-5,
                   maxvecs=20, exxdiv='ewald', nelec=None,
                   verbose=False) -> "PeriodicHamiltonian":
        """
        Generate a periodic Hamiltonian from a PySCF ``pbc`` SCF calculation,
        holding the result in memory.

        Parameters
        ----------
        scf_data : dict
            Unpacked PySCF checkpoint, as produced by
            ``afqmctools.utils.pyscf_utils.load_from_pyscf_chk``. Uses the keys
            ``'hcore'``, ``'X'``, ``'cell'``, ``'kpts'`` and ``'nmo_pk'``.
        comm : mpi4py communicator, optional
            Must be serial. Use `write_from_pyscf` to run in parallel.
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

        Raises
        ------
        ValueError
            If `comm` spans more than one rank — a distributed factorization is
            never assembled in memory; see `write_from_pyscf`.
        """
        comm = comm if comm is not None else _SerialComm()
        if comm.size > 1:
            raise ValueError(
                "PeriodicHamiltonian.from_pyscf builds the whole factorization in "
                "memory and is serial-only; use PeriodicHamiltonian.write_from_pyscf "
                "to generate and write in parallel."
            )

        cell, kpts, X = scf_data['cell'], scf_data['kpts'], scf_data['X']
        nmo_pk = np.asarray(scf_data['nmo_pk'])
        nelec = nelec if nelec is not None else _default_nelec(cell, len(kpts))

        solver = PeriodicCholesky(comm, cell, kpts, nmo_pk, kp_sym=kpoint_symmetry,
                                  maxvecs=maxvecs, gtol_chol=chol_cut, verbose=verbose)

        hcore_pk = _transform_hcore(scf_data['hcore'], X, nmo_pk)
        enuc = _zero_electron_energy(cell, kpts, sum(nelec), exxdiv)

        if not kpoint_symmetry:
            (_, cholvecs), = solver.run(comm, X)
            return cls(**_supercell_layout(hcore_pk, cholvecs, solver),
                       enuc=enuc, nelec=nelec)

        chol = {}
        nchol_pk = np.zeros(len(kpts), dtype=np.int32)
        for Q, block in solver.run(comm, X):
            chol[Q] = _kpoint_block(block, solver)
            nchol_pk[Q] = block.shape[-1]

        return cls(hcore=hcore_pk, chol=chol, kpts=kpts, nmo_pk=nmo_pk,
                   qk_to_k2=solver.QKToK2, minus_k=solver.kminus,
                   nchol_pk=nchol_pk, enuc=enuc, nelec=nelec)

    @classmethod
    def write_from_pyscf(cls, comm, scf_data, path, kpoint_symmetry=True,
                         chol_cut=1e-5, maxvecs=20, exxdiv='ewald', nelec=None,
                         phdf=False, verbose=False) -> None:
        """
        Generate a periodic Hamiltonian and stream it straight to `path`.

        Equivalent to ``from_pyscf(...).to_hdf5(path)`` but never holds more
        than one momentum block in memory, and works over an MPI communicator of
        any size. This is the path to use for production-sized systems.

        Parameters
        ----------
        comm : mpi4py communicator
            Communicator the work is spread over.
        scf_data : dict
            Unpacked PySCF checkpoint; see `from_pyscf`.
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist; a Hamiltonian
            already in it is replaced and anything else preserved, as for
            `to_hdf5`.
        kpoint_symmetry : bool, optional
            Factorize per momentum transfer (True) rather than as one supercell
            (False). Default True.
        chol_cut, maxvecs, exxdiv, nelec, verbose
            As for `from_pyscf`.
        phdf : bool, optional
            Use parallel HDF5 rather than per-rank scratch files.

        Raises
        ------
        NotImplementedError
            For a distributed supercell factorization. Its Cholesky vectors
            scatter into the combined basis rather than filling a contiguous
            slice of it, so the per-rank merge below does not apply.
        """
        cell, kpts, X = scf_data['cell'], scf_data['kpts'], scf_data['X']
        nmo_pk = np.asarray(scf_data['nmo_pk'])
        nelec = nelec if nelec is not None else _default_nelec(cell, len(kpts))

        if not kpoint_symmetry:
            if comm.size > 1:
                raise NotImplementedError(
                    "Writing a supercell (kpoint_symmetry=False) factorization in "
                    "parallel is not implemented: each rank's Cholesky vectors "
                    "scatter across the combined orbital basis instead of filling a "
                    "contiguous slice, so the per-rank merge used for the k-point "
                    "path does not apply. Use kpoint_symmetry=True to run in "
                    "parallel, or generate the supercell Hamiltonian serially."
                )
            cls.from_pyscf(scf_data, comm=comm, kpoint_symmetry=False,
                           chol_cut=chol_cut, maxvecs=maxvecs, exxdiv=exxdiv,
                           nelec=nelec, verbose=verbose).to_hdf5(path)
            return

        tstart = time.time()
        solver = PeriodicCholesky(comm, cell, kpts, nmo_pk, kp_sym=True,
                                  maxvecs=maxvecs, gtol_chol=chol_cut, verbose=verbose)

        with FileHandler(comm, path, "a", phdf) as h5file:
            clear_hamiltonian(h5file)
            group = h5file.create_group("Hamiltonian")
            kp_group = h5file.create_group(KPOINT_GROUP)

            _write_kpoint_basics(comm, group, cell, kpts, scf_data['hcore'], X,
                                 nmo_pk, solver.QKToK2, solver.kminus, nelec,
                                 exxdiv=exxdiv)

            logger.info("time to reach Cholesky: %13.8e s", time.time() - tstart)
            tstart = time.time()

            num_cholvecs = np.zeros(len(kpts), dtype=np.int32)
            for Q, cholvecs in solver.run(comm, X):
                num_cholvecs[Q] = cholvecs.shape[-1]
                _write_kpoint_block(comm, kp_group, solver, Q, cholvecs,
                                    phdf=h5file.phdf)
                comm.barrier()

            group.create_dataset("NCholPerKP", data=num_cholvecs)
            logger.info("time to perform Cholesky: %13.8e s", time.time() - tstart)

            comm.barrier()
            if not phdf and comm.rank == 0:
                _merge_rank_files(comm, kp_group, path, solver.kminus)

        comm.barrier()

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
            group = fh5.create_group("Hamiltonian")
            _write_kpoint_descriptors(group, self.kpts, self.nmo_pk, self.qk_to_k2,
                                      self.minus_k, self.nelec, self.enuc)
            for ki in range(self.nkpts):
                _write_kpoint_h1(group, ki, self.nmo_pk[ki], self.hcore[ki])

            group.create_dataset("NCholPerKP", data=self.nchol_pk)

            kp_group = fh5.create_group(KPOINT_GROUP)
            for Q, L in self.chol.items():
                kp_group.create_dataset(f"L{Q}", data=to_complex(L))

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


def write_rhoG(comm, scf_data, path, gcut, kpoint_symmetry=True, phdf=False,
               verbose=False) -> None:
    r"""
    Write the real-space density :math:`\rho(G)` on the FFT grid.

    .. warning:: Not implemented.

    Parameters
    ----------
    comm : mpi4py communicator
        Communicator the work would be spread over.
    scf_data : dict
        Unpacked PySCF checkpoint; see `PeriodicHamiltonian.from_pyscf`.
    path : str or pathlib.Path
        HDF5 file to write.
    gcut : float
        Plane-wave cutoff for the density.
    kpoint_symmetry : bool, optional
        Which factorization the density would follow.
    phdf : bool, optional
        Use parallel HDF5.
    verbose : bool, optional
        Log progress.

    Raises
    ------
    NotImplementedError
        Always.

    Notes
    -----
    This replaces afqmctools' ``write_rhoG_kpoints`` and
    ``write_rhoG_supercell``, neither of which was ever functional:
    ``write_rhoG_kpoints`` referenced an undefined ``nelec``, and
    ``write_rhoG_supercell`` called a bare ``quit()`` before touching three more
    undefined names. The signature is kept so that a working implementation has
    an obvious place to land.
    """
    raise NotImplementedError(
        "write_rhoG is not implemented. afqmctools' write_rhoG_kpoints and "
        "write_rhoG_supercell were both non-functional (NameError and a bare "
        "quit() respectively), so there was nothing to port."
    )


class _SerialComm:
    """
    Stand-in communicator for a serial run, so the solver needs no MPI import
    when there is nothing to distribute.
    """

    size = 1
    rank = 0

    def barrier(self):
        pass

    def Bcast(self, buffer, root=0):
        pass

    def Allgather(self, sendbuf, recvbuf):
        recvbuf[:len(sendbuf)] = sendbuf


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
        The solver that produced `cholvecs`, for its partition and orbital
        counts.

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
    part = solver.part
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
        for ij in range(part.nij):
            if (ij + part.ij0) >= nmo_pk[k1] * nmo_pk[k2]:
                break
            i = (ij + part.ij0) // nmo_pk[k2]
            j = (ij + part.ij0) % nmo_pk[k2]
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
    Reshape a serial rank's Cholesky block into the ``KPFactorized`` layout,
    ``(nkpts, nmo_max**2 * nchol)``.

    Carries the same :math:`1/\sqrt{N_k}` normalization the on-disk format
    expects to find already applied.
    """
    part = solver.part
    nkpts = len(solver.kpts)
    nchol = cholvecs.shape[-1]
    factor = 1.0 / math.sqrt(nkpts)

    L = np.zeros((nkpts, solver.nmo_max**2 * nchol), dtype=np.complex128)
    for kk in range(part.nkk):
        L[kk + part.kk0, part.ij0 * nchol:part.ijN * nchol] = (
            cholvecs[kk, :, :].ravel() * factor)
    return L


def _write_kpoint_descriptors(group, kpts, nmo_pk, qk_to_k2, minus_k, nelec,
                              enuc, fill=True) -> None:
    """
    Write the k-point Hamiltonian's descriptor datasets into `group`.

    `fill` False creates the datasets without writing to them, so that under
    parallel HDF5 every rank can take part in the collective creation while only
    one actually writes.
    """
    nkpts = len(kpts)

    dims = group.create_dataset("dims", (8,), dtype=np.int32)
    complex_integrals = group.create_dataset("ComplexIntegrals", (1,), dtype=np.int32)
    energies = group.create_dataset("Energies", (2,), dtype=np.float64)
    kpoints = group.create_dataset("KPoints", (nkpts, 3), dtype=np.float64)
    nmo_per_kp = group.create_dataset("NMOPerKP", (nkpts,), dtype=np.int32)
    qk = group.create_dataset("QKTok2", (nkpts, nkpts), dtype=np.int32)
    kminus = group.create_dataset("MinusK", (nkpts,), dtype=np.int32)

    if not fill:
        return

    dims[:] = np.array([0, 0, nkpts, int(np.sum(nmo_pk)), nelec[0], nelec[1], 0, 0])
    complex_integrals[:] = 1
    energies[:] = np.array([enuc, 0.])
    kpoints[:, :] = np.asarray(kpts, dtype=np.float64)
    nmo_per_kp[:] = np.asarray(nmo_pk, dtype=np.int32)
    qk[:, :] = np.asarray(qk_to_k2, dtype=np.int32)
    kminus[:] = np.asarray(minus_k, dtype=np.int32)


def _write_kpoint_h1(group, ki, nmo, h1) -> None:
    """Write the one-body block at k-point `ki`, interleaved complex."""
    dataset = group.create_dataset(f"H1_kp{ki}", (nmo, nmo, 2), dtype=np.float64)
    if h1 is None:
        return

    if h1.shape != (nmo, nmo):
        raise ValueError(f"H1 at kpoint {ki} has shape {h1.shape}, expected ({nmo}, {nmo})")
    dataset[:, :, 0] = np.real(h1)
    dataset[:, :, 1] = np.imag(h1)


def _write_kpoint_basics(comm, group, cell, kpts, hcore, X, nmo_pk, qk_to_k2,
                         minus_k, nelec, exxdiv='ewald') -> None:
    """
    Write the descriptor datasets and the one-body Hamiltonian collectively.

    Every rank has to take part: under parallel HDF5 dataset creation is
    collective, and without it each rank is writing its own scratch file.
    """
    nkpts = len(kpts)
    enuc = _zero_electron_energy(cell, kpts, sum(nelec), exxdiv) if comm.rank == 0 else 0.0
    comm.barrier()

    _write_kpoint_descriptors(group, kpts, nmo_pk, qk_to_k2, minus_k, nelec, enuc,
                              fill=comm.rank == 0)
    comm.barrier()

    hcore_pk = _transform_hcore(hcore, X, nmo_pk) if comm.rank == 0 else None
    for ki in range(nkpts):
        _write_kpoint_h1(group, ki, nmo_pk[ki],
                         hcore_pk[ki] if comm.rank == 0 else None)

    comm.barrier()


def _write_kpoint_block(comm, kp_group, solver, Q, cholvecs, phdf=False) -> None:
    r"""
    Write one momentum block's Cholesky vectors.

    With parallel HDF5 (or a single rank) every rank writes into its slice of
    the full ``L{Q}`` dataset. Otherwise each non-root rank writes its own block
    plus an ``Ldim{Q}`` descriptor, which `_merge_rank_files` uses to place it.
    """
    part = solver.part
    nkpts = len(solver.kpts)
    numv = cholvecs.shape[-1]
    factor = 1.0 / math.sqrt(nkpts)

    if phdf or comm.rank == 0:
        LQ = kp_group.create_dataset(
            f"L{Q}", (nkpts, solver.nmo_max * solver.nmo_max * numv, 2),
            dtype=np.float64)
        for kk in range(part.nkk):
            LQ[kk + part.kk0, part.ij0 * numv:part.ijN * numv, :] = (
                to_complex(cholvecs[kk, :, :].ravel() * factor))
    else:
        kp_group.create_dataset(
            f"Ldim{Q}",
            data=np.array([part.nkk, part.nij, part.kk0, part.ij0, part.ijN, numv],
                          dtype=np.int32))
        LQ = kp_group.create_dataset(f"L{Q}", (part.nkk, part.nij * numv, 2),
                                     dtype=np.float64)
        for kk in range(part.nkk):
            LQ[kk, :, :] = to_complex(cholvecs[kk, :, :].ravel() * factor)


def _merge_rank_files(comm, kp_group, path, minus_k) -> None:
    """
    Fold every non-root rank's scratch file into the real one and delete it.

    Only used when parallel HDF5 is unavailable; see `FileHandler`.
    """
    nkpts = len(minus_k)
    for rank in range(1, comm.size):
        scratch = rank_filename(rank, path)
        with h5.File(scratch, 'r') as fh5:
            for Q in range(nkpts):
                if Q > minus_k[Q]:
                    continue
                nkk, nij, kk0, ij0, ijN, numv = fh5[f"{KPOINT_GROUP}/Ldim{Q}"][:]
                LQ = fh5[f"{KPOINT_GROUP}/L{Q}"][:].reshape((nkk, nij * numv, 2))
                kp_group[f"L{Q}"][kk0:kk0 + nkk, ij0 * numv:ijN * numv, :] = LQ
        os.remove(scratch)
