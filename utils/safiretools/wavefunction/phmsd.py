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
from safiretools.wavefunction.base import (
    ORTHONORMAL_TOL,
    Wavefunction,
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
        Occupied-orbital indices per determinant, ``(ndets, nup)`` and
        ``(ndets, ndown)``. Both index the same orbital basis, ``0`` to
        ``nmo - 1``; the beta offset the file format uses is applied on write.
    nmo : int
        Number of orbitals the occupation numbers index.
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
        Spin symmetry. Only `SpinSymm.COLLINEAR` is supported — see Notes.

    Raises
    ------
    ValueError
        If `occa`/`occb` disagree with `nelec`, if an orbital index falls
        outside the basis, if more than two references are given, or if
        `spin_symm` is not collinear.

    Notes
    -----
    The AFQMC executable reads particle-hole wavefunctions only with collinear
    walkers (``read_ph_wavefunction_hdf`` rejects both closed-shell and
    noncollinear), so `spin_symm` is always `SpinSymm.COLLINEAR`. A fully
    spin-polarized expansion is therefore collinear with ``ndown == 0``, and its
    beta blocks go to disk with zero width.
    """

    _HDF5_GROUP = 'PHMSD'

    def __init__(self, coeffs, occa, occb, nmo: int, nelec=None, orbitals=None,
                 psi0=None, spin_symm=SpinSymm.COLLINEAR) -> None:
        if SpinSymm.from_input(spin_symm) is not SpinSymm.COLLINEAR:
            raise ValueError(
                "a particle-hole wavefunction is always collinear; the AFQMC "
                "executable reads no other spin symmetry for this "
                f"representation, and {SpinSymm.from_input(spin_symm).label} "
                "was requested"
            )

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
                         spin_symm=SpinSymm.COLLINEAR, psi0=psi0)

        self._validate_occupations()

    def _validate_occupations(self) -> None:
        for name, occ, nelec in (('occa', self.occa, self.nelec[0]),
                                 ('occb', self.occb, self.nelec[1])):
            if occ.shape != (self.ndets, nelec):
                raise ValueError(
                    f"{name} has shape {occ.shape}, expected "
                    f"({self.ndets}, {nelec}) for {self.ndets} determinant(s) "
                    f"with nelec={self.nelec}"
                )
            if occ.size and (occ.min() < 0 or occ.max() >= self.nmo):
                raise ValueError(
                    f"{name} holds orbital indices outside [0, {self.nmo}): "
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
        occupation numbers.
        """
        identity = np.eye(self.nmo, dtype=np.complex128)
        return (identity[:, self.occa[0]].copy(),
                identity[:, self.occb[0]].copy())

    def _slater_matrices(self):
        for ispin, block in enumerate(self.psi0):
            yield f'psi0 spin {ispin}', block

        for index, matrix in enumerate(self.orbitals or ()):
            yield f'orbitals[{index}]', matrix

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
                          orbitals=orbitals, psi0=psi0)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def _write_payload(self, group) -> None:
        io.write_phmsd(group, self.occa, self.occb, nmo=self.nmo,
                       orbitals=self.orbitals)

    @classmethod
    def _read_payload(cls, group, header: dict) -> "PHMSDWavefunction":
        occa, occb, orbitals = io.read_phmsd(
            group, header['ndets'], header['nelec'], header['nmo'])

        return cls(coeffs=header['coeffs'], occa=occa, occb=occb,
                   nmo=header['nmo'], nelec=header['nelec'],
                   orbitals=orbitals, psi0=header['psi0'])

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pbc_scf(cls, scf_data, ortho_ao=True, rediag=True, ndet_max=None,
                     low=0.1, high=0.95,
                     orthonormalize=True) -> "PHMSDWavefunction":
        """
        Build a multi-determinant trial wavefunction from a periodic PySCF SCF
        calculation with partially occupied degenerate bands.

        Parameters
        ----------
        ndet_max : int, optional
            Largest number of determinants to keep. Every determinant the
            degeneracy allows when omitted.

        Raises
        ------
        ValueError
            If the SCF data has no partial occupancies, so that a single
            determinant already describes it — use
            `NOMSDWavefunction.from_pbc_scf` for that.

        See Also
        --------
        safiretools.wavefunction.pbc.from_pbc_scf : full parameter documentation.
        """
        from safiretools.wavefunction.pbc import from_pbc_scf

        wavefunction = from_pbc_scf(
            scf_data, ortho_ao=ortho_ao, rediag=rediag, ndet_max=ndet_max,
            low=low, high=high, orthonormalize=orthonormalize,
        )

        if not isinstance(wavefunction, cls):
            raise ValueError(
                "this SCF calculation has no partially occupied degenerate "
                "bands, so a single determinant describes it exactly; use "
                "NOMSDWavefunction.from_pbc_scf"
            )

        return wavefunction


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
