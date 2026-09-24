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
`PHMSDWavefunction` — a particle-hole multi-Slater-determinant trial
wavefunction,

.. math:: |\Psi_T\rangle = \sum_i c_i\, |D_i\rangle

where each :math:`|D_i\rangle` is an occupation-number string over a shared
orbital basis rather than its own orbital matrix. This is the representation a
selected-CI or CASSCF expansion produces.

The determinants are stored as the coefficients :math:`c_i` plus the occupied
orbital indices ``occa``/``occb``, optionally over an explicit orbital
reference.
"""

import numpy as np

from safiretools.types import SpinSymm
from safiretools.wavefunction import io
from safiretools.wavefunction.base import Wavefunction
from safiretools.wavefunction.slater import (
    ORTHONORMAL_TOL,
    is_orthonormal,
    modified_gram_schmidt,
)


class PHMSDWavefunction(Wavefunction):
    """
    A particle-hole multi-Slater-determinant wavefunction.

    Parameters
    ----------
    coeffs : array_like
        Determinant coefficients, ``(ndets,)``.
    occa, occb : array_like
        Occupied-orbital indices per determinant, one array per independent
        spin channel: ``(ndets, nup)`` and ``(ndets, ndown)`` when collinear,
        and `occb` of zero width otherwise, since the two polarizations then
        share a single channel. They index ``0`` to ``npol*nmo - 1``: spatial
        orbitals, except when noncollinear, where they are spinors and already
        carry the beta offset. A collinear `occb` does not — that offset is
        applied on write.
    nmo : int
        Number of orbitals the occupation numbers index, spatial unless
        noncollinear.
    nelec : tuple(int, int), optional
        Physical electron counts. Taken from `occa` and `occb`'s widths when
        omitted, and checked against them when given.
    orbitals : sequence of numpy.ndarray, optional
        Orbital matrices the occupation numbers refer to — one for a
        closed-shell-like reference, two for a spin-resolved one. Omitted, the
        occupation numbers index the Hamiltonian's own basis.
    psi0 : sequence of numpy.ndarray, optional
        Initial Slater determinant for the AFQMC walkers. Built from the leading
        determinant's occupations when omitted.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry.

    Raises
    ------
    ValueError
        If `occa`/`occb` disagree with `nelec`, if an orbital index falls
        outside the basis, or if more than two references are given.

    """

    _HDF5_GROUP = 'PHMSD'

    def __init__(self, coeffs, occa, occb, nmo: int, nelec=None, orbitals=None,
                 psi0=None, spin_symm=SpinSymm.COLLINEAR) -> None:
        self.occa = _occupations(occa, 'occa')
        self.occb = _occupations(occb, 'occb')

        if self.occa.shape[0] != self.occb.shape[0]:
            raise ValueError(
                f"occa and occb describe different numbers of determinants "
                f"({self.occa.shape[0]} and {self.occb.shape[0]})"
            )

        if nelec is None:
            nelec = (self.occa.shape[1], self.occb.shape[1])

        self.orbitals = None if orbitals is None else tuple(
            np.asarray(matrix, dtype=np.complex128)
            for matrix in orbitals if matrix is not None)

        if self.orbitals is not None and len(self.orbitals) > 2:
            raise ValueError(
                f"a particle-hole wavefunction takes at most two orbital "
                f"references, got {len(self.orbitals)}"
            )
        if self.orbitals is not None and not self.orbitals:
            self.orbitals = None

        super().__init__(coeffs=coeffs, nelec=nelec, nmo=nmo,
                         spin_symm=spin_symm, psi0=psi0)

        self._validate_occupations()

    def _validate_occupations(self) -> None:
        # one width per independent spin channel, so the beta array has zero
        #   width whenever the two polarizations share a single channel
        widths = self.nelec_per_spin + (0,) * (2 - self.nspin)

        for name, occ, width in (('occa', self.occa, widths[0]),
                                 ('occb', self.occb, widths[1])):
            if occ.shape != (self.ndets, width):
                raise ValueError(
                    f"{name} has shape {occ.shape}, expected "
                    f"({self.ndets}, {width}) for {self.ndets} determinant(s) of a "
                    f"{self.spin_symm.label} wavefunction with nelec={self.nelec}"
                )
            if occ.size and (occ.min() < 0 or occ.max() >= self.nrows):
                raise ValueError(
                    f"{name} holds orbital indices outside [0, {self.nrows}): "
                    f"[{occ.min()}, {occ.max()}]"
                )

    @property
    def nreferences(self) -> int:
        """
        Number of explicit orbital references, which ``type`` records on disk:
        0 when the occupation numbers index the Hamiltonian's basis directly.
        """
        return 0 if self.orbitals is None else len(self.orbitals)

    def _default_psi0(self) -> tuple:
        """
        The leading determinant, as columns of the identity selected by its own
        occupation numbers — one block per independent spin channel.
        """
        identity = np.eye(self.nrows, dtype=np.complex128)
        return tuple(identity[:, occ[0]].copy()
                     for occ in (self.occa, self.occb)[:self.nspin])

    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "PHMSDWavefunction":
        """
        Return a copy whose orbital references — and explicit `psi0`, if any —
        have orthonormal columns. See `Wavefunction.orthonormalize`.
        """
        def fix(matrix):
            return matrix if is_orthonormal(matrix, tol=tol) \
                else modified_gram_schmidt(matrix)

        orbitals = None if self.orbitals is None else tuple(
            fix(matrix) for matrix in self.orbitals)
        psi0 = None if self._psi0 is None else tuple(
            fix(block) for block in self._psi0)

        return type(self)(coeffs=self.coeffs.copy(), occa=self.occa.copy(),
                          occb=self.occb.copy(), nmo=self.nmo, nelec=self.nelec,
                          orbitals=orbitals, psi0=psi0,
                          spin_symm=self.spin_symm)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def _write_payload(self, group) -> None:
        io.write_phmsd(group, self.occa, self.occb, nmo=self.nmo,
                       orbitals=self.orbitals)

    @classmethod
    def _read_payload(cls, group, header: dict) -> "PHMSDWavefunction":
        occa, occb, orbitals = io.read_phmsd(
            group, header['ndets'], header['nelec'], header['nmo'],
            spin_symm=header['spin_symm'])

        return cls(coeffs=header['coeffs'], occa=occa, occb=occb,
                   nmo=header['nmo'], nelec=header['nelec'],
                   orbitals=orbitals, psi0=header['psi0'],
                   spin_symm=header['spin_symm'])


def _occupations(occ, name: str):
    """Coerce an occupation-number array to a 2-D ``(ndets, nelec)`` int array."""
    occ = np.asarray(occ)

    if occ.size == 0:
        # a spin channel with no electrons still needs its determinant axis
        return occ.reshape(occ.shape[0] if occ.ndim else 0, 0).astype(np.int64)

    occ = np.atleast_2d(occ)
    if occ.ndim != 2:
        raise ValueError(
            f"{name} must have shape (ndets, nelec), got shape {occ.shape}"
        )

    return occ.astype(np.int64, copy=False)
