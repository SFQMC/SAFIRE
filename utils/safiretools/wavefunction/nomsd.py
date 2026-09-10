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
`NOMSDWavefunction` — a non-orthogonal multi-Slater-determinant trial
wavefunction,

.. math:: |\Psi_T\rangle = \sum_i c_i\, |\Phi_i\rangle

stored as the coefficients :math:`c_i` plus one orbital matrix per determinant.

Every domain that produces this representation reaches it through a classmethod
factory — `NOMSDWavefunction.from_free_electron`,
`~NOMSDWavefunction.from_pyscf`, `~NOMSDWavefunction.from_pbc_scf` — which
delegate to the implementation modules alongside this one.
"""

from warnings import warn

import numpy as np

from safiretools.types import SpinSymm
from safiretools.wavefunction import io
from safiretools.wavefunction.base import Wavefunction
from safiretools.wavefunction.slater import (
    ORTHONORMAL_TOL,
    is_orthonormal,
    modified_gram_schmidt,
    spin_blocks,
)


class NOMSDWavefunction(Wavefunction):
    r"""
    A non-orthogonal multi-Slater-determinant wavefunction.

    Parameters
    ----------
    coeffs : array_like
        Determinant coefficients :math:`c_i`, ``(ndets,)``.
    dets : array_like
        Orbital matrices, ``(ndets, npol*nmo, ncols)``, where ``ncols`` is
        ``sum(nelec_per_spin)``: the spin channels occupy consecutive column
        blocks. That is ``nup`` columns for a closed-shell wavefunction (the
        beta channel repeats the alpha one), ``nup + ndown`` when collinear, and
        ``nup + ndown`` over ``2*nmo`` rows when noncollinear.
    nelec : tuple(int, int)
        Physical electron counts ``(nup, ndown)``.
    spin_symm : SpinSymm or str or int
        Spin symmetry, coerced through `SpinSymm.from_input`.
    psi0 : sequence of numpy.ndarray, optional
        Initial Slater determinant for the AFQMC walkers, one block per spin
        channel. Taken from ``dets[0]`` when omitted.
    nmo : int, optional
        Number of orbitals. Inferred from `dets` and `spin_symm` when omitted.

    Raises
    ------
    ValueError
        If `dets` has the wrong rank or does not match `nelec` and `spin_symm`.

    Examples
    --------
    >>> wavefunction = NOMSDWavefunction.from_free_electron(hamiltonian,
    ...                                                     nelec=(8, 8))
    >>> wavefunction.to_hdf5('afqmc.h5')
    """

    _HDF5_GROUP = 'NOMSD'

    def __init__(self, coeffs, dets, nelec, spin_symm, psi0=None,
                 nmo: int = None) -> None:
        dets = np.asarray(dets, dtype=np.complex128)
        if dets.ndim != 3:
            raise ValueError(
                "dets must have shape (ndets, npol*nmo, ncols), got shape "
                f"{dets.shape}; a single determinant still needs its leading axis"
            )

        if nmo is None:
            npol = 2 if SpinSymm.from_input(spin_symm) is SpinSymm.NONCOLLINEAR else 1
            nmo = dets.shape[1] // npol

        self.dets = dets

        super().__init__(coeffs=coeffs, nelec=nelec, nmo=nmo,
                         spin_symm=spin_symm, psi0=psi0)

        if self.dets.shape != (self.ndets, self.nrows, sum(self.nelec_per_spin)):
            raise ValueError(
                f"dets has shape {self.dets.shape}, expected "
                f"({self.ndets}, {self.nrows}, {sum(self.nelec_per_spin)}) for "
                f"{self.ndets} determinant(s) of a {self.spin_symm.label} "
                f"wavefunction with nelec={self.nelec} and nmo={self.nmo}"
            )

    # ------------------------------------------------------------------
    # views of the determinants
    # ------------------------------------------------------------------

    def spin_blocks(self, idet: int = 0):
        """
        The per-spin column blocks of determinant `idet`, as a tuple of length
        `Wavefunction.nspin`.
        """
        return tuple(spin_blocks(self.dets[idet], self.nelec_per_spin))

    def _default_psi0(self) -> tuple:
        """The leading determinant's spin blocks."""
        return tuple(block.copy() for block in self.spin_blocks(0))

    def _warn_about_default_psi0(self) -> None:
        """
        Warn when a collinear trial's own determinant is being reused as the
        initial walker, which equilibrates slowly.
        """
        if self.nspin == 2:
            warn(
                "Using this wavefunction's own Slater determinant for the "
                "initial walkers of a collinear trial wavefunction. This can "
                "lead to very slow equilibration in AFQMC calculations; ROHF "
                "Slater determinants are recommended (pass psi0=)."
            )

    def _slater_matrices(self):
        for idet in range(self.ndets):
            for ispin, block in enumerate(self.spin_blocks(idet)):
                yield f'dets[{idet}] spin {ispin}', block

        if self._psi0 is not None:
            for ispin, block in enumerate(self._psi0):
                yield f'psi0 spin {ispin}', block

    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "NOMSDWavefunction":
        """
        Return a copy whose determinants — and explicit `psi0`, if any — have
        orthonormal columns. See `Wavefunction.orthonormalize`.
        """
        dets = self.dets.copy()
        for idet in range(self.ndets):
            start = 0
            for nelec in self.nelec_per_spin:
                block = dets[idet, :, start:start + nelec]
                if not is_orthonormal(block, tol=tol):
                    dets[idet, :, start:start + nelec] = modified_gram_schmidt(block)
                start += nelec

        psi0 = self._psi0
        if psi0 is not None:
            psi0 = tuple(block if is_orthonormal(block, tol=tol)
                         else modified_gram_schmidt(block) for block in psi0)

        return type(self)(coeffs=self.coeffs.copy(), dets=dets, nelec=self.nelec,
                          spin_symm=self.spin_symm, psi0=psi0, nmo=self.nmo)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def _write_payload(self, group) -> None:
        io.write_nomsd(group, self.dets, self.nelec_per_spin)

    @classmethod
    def _read_payload(cls, group, header: dict) -> "NOMSDWavefunction":
        spin_symm = header['spin_symm']
        nelec = header['nelec']

        nelec_per_spin = _nelec_per_spin(spin_symm, nelec)
        dets = io.read_nomsd(group, header['ndets'], nelec_per_spin)

        return cls(coeffs=header['coeffs'], dets=dets, nelec=nelec,
                   spin_symm=spin_symm, psi0=header['psi0'],
                   nmo=header['nmo'])

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------


    @classmethod
    def from_pbc_scf(cls, scf_data, ortho_ao=True, rediag=True, low=0.1,
                     high=0.95, orthonormalize=True) -> "NOMSDWavefunction":
        """
        Build a single-determinant trial wavefunction from a periodic PySCF SCF
        calculation.

        One determinant is used even when bands are partially occupied, by
        occupying the leading configuration; use
        `PHMSDWavefunction.from_pbc_scf` for the multi-determinant expansion
        over those bands.

        See Also
        --------
        safiretools.wavefunction.pbc.from_pbc_scf : full parameter documentation.
        """
        from safiretools.wavefunction.pbc import from_pbc_scf

        return from_pbc_scf(
            scf_data, ortho_ao=ortho_ao, rediag=rediag, ndet_max=1, low=low,
            high=high, orthonormalize=orthonormalize,
        )


def _nelec_per_spin(spin_symm, nelec) -> tuple:
    """
    `Wavefunction.nelec_per_spin` for a spin symmetry and electron-count pair,
    before an instance exists to ask.
    """
    if spin_symm is SpinSymm.COLLINEAR:
        return tuple(nelec)
    if spin_symm is SpinSymm.NONCOLLINEAR:
        return (sum(nelec),)
    return (nelec[0],)


def infer_spin_symm(orbitals, nelec, nmo: int) -> SpinSymm:
    """
    Infer the spin symmetry an orbital matrix is expressed in.

    Parameters
    ----------
    orbitals : numpy.ndarray
        A single determinant's orbital matrix, ``(npol*nmo, ncols)``.
    nelec : tuple(int, int)
        Physical electron counts ``(nup, ndown)``.
    nmo : int
        Number of spatial orbitals.

    Returns
    -------
    SpinSymm

    Raises
    ------
    ValueError
        If the shape matches no spin symmetry.

    Notes
    -----
    Decided by shape: ``2*nmo`` rows means noncollinear; otherwise the column
    count separates a closed-shell matrix (one ``nup``-wide block, requiring
    ``nup == ndown``) from a collinear one (``nup + ndown`` columns).

    afqmctools' ``_get_slater_type`` tested ``nup == ndown`` alone and so read
    any equal-population collinear matrix as closed-shell, halving it. A matrix
    with no beta electrons is `SpinSymm.COLLINEAR` with ``ndown == 0``.
    """
    orbitals = np.asarray(orbitals)
    if orbitals.ndim != 2:
        raise ValueError(
            f"expected a single determinant's 2-dimensional orbital matrix, "
            f"got shape {orbitals.shape}"
        )

    nup, ndown = nelec
    nrows, ncols = orbitals.shape

    if nrows == 2 * nmo and ncols == nup + ndown:
        return SpinSymm.NONCOLLINEAR

    if nrows != nmo:
        raise ValueError(
            f"orbital matrix has {nrows} rows, expected {nmo} (collinear or "
            f"closed shell) or {2 * nmo} (noncollinear)"
        )

    if ncols == nup and nup == ndown:
        return SpinSymm.CLOSED
    if ncols == nup + ndown:
        return SpinSymm.COLLINEAR

    raise ValueError(
        f"orbital matrix has {ncols} columns, which matches neither a closed "
        f"shell ({nup}, and only when nup == ndown) nor a collinear "
        f"({nup + ndown}) wavefunction with nelec={tuple(nelec)}"
    )
