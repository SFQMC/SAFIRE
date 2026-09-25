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
    orthonormalize,
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
            nmo = dets.shape[1] // SpinSymm.from_input(spin_symm).npol

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

    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "NOMSDWavefunction":
        """
        Return a copy whose determinants — and explicit `psi0`, if any — have
        orthonormal columns. See `Wavefunction.orthonormalize`.
        """
        dets = self.dets.copy()
        for det in dets:
            for block in spin_blocks(det, self.nelec_per_spin):
                block[...] = orthonormalize(block, tol=tol)

        psi0 = None if self._psi0 is None else tuple(
            orthonormalize(block, tol=tol) for block in self._psi0)

        return type(self)(coeffs=self.coeffs.copy(), dets=dets, nelec=self.nelec,
                          spin_symm=self.spin_symm, psi0=psi0, nmo=self.nmo)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def _write_payload(self, group) -> None:
        if self._psi0 is None and self.nspin == 2:
            warn(
                "Using this wavefunction's own Slater determinant for the "
                "initial walkers of a collinear trial wavefunction. This can "
                "lead to very slow equilibration in AFQMC calculations; ROHF "
                "Slater determinants are recommended (pass psi0=)."
            )

        io.write_nomsd(group, self.dets, self.nelec_per_spin)

    @classmethod
    def _read_payload(cls, group, header: dict) -> "NOMSDWavefunction":
        spin_symm = header['spin_symm']
        nelec = header['nelec']

        dets = io.read_nomsd(group, header['ndets'], spin_symm.nelec_per_spin(nelec))

        return cls(coeffs=header['coeffs'], dets=dets, nelec=nelec,
                   spin_symm=spin_symm, psi0=header['psi0'],
                   nmo=header['nmo'])
