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
factory — `NOMSDWavefunction.from_single_determinant`,
`~NOMSDWavefunction.from_free_electron`, `~NOMSDWavefunction.from_pyscf`,
`~NOMSDWavefunction.from_pbc_scf` — which delegate to the implementation
modules alongside this one.
"""

import numpy as np

from safiretools.types import SpinSymm
from safiretools.wavefunction import io
from safiretools.wavefunction.base import Wavefunction
from safiretools.wavefunction.slater import (
    ORTHONORMAL_TOL,
    format_spin_layout,
    orthonormalize,
    parse_spin_layout,
)


class NOMSDWavefunction(Wavefunction):
    r"""
    A non-orthogonal multi-Slater-determinant wavefunction.

    Parameters
    ----------
    coeffs : array_like
        Determinant coefficients :math:`c_i`, ``(ndets,)``.
    dets : numpy.ndarray or tuple of numpy.ndarray
        The occupied orbitals of every determinant, as columns. The layout
        decides the spin symmetry, and the shape the electron counts and the
        number of orbitals:

        - closed shell: an array ``(ndets, nmo, nup)``, shared by both spins;
        - collinear: a *tuple* of arrays ``(ndets, nmo, nup)`` and
          ``(ndets, nmo, ndown)``;
        - noncollinear: an array ``(ndets, 2, nmo, nelec)`` of spinor
          orbitals, spin-up components first. `nelec` is then
          ``(nelec, 0)``, as the file format records it.

    Raises
    ------
    TypeError
        If `dets` is none of the layouts.
    ValueError
        If `dets` holds a different number of determinants than `coeffs`.

    Examples
    --------
    >>> wavefunction = NOMSDWavefunction.from_free_electron(hamiltonian,
    ...                                                     nelec=(8, 8))
    >>> wavefunction.to_hdf5('afqmc.h5')
    """

    _HDF5_GROUP = 'NOMSD'

    def __init__(self, coeffs, dets) -> None:
        spin_symm, blocks = parse_spin_layout(dets, stacked=True, name='dets')

        widths = tuple(block.shape[-1] for block in blocks)
        if spin_symm is SpinSymm.COLLINEAR:
            nelec = widths
        elif spin_symm is SpinSymm.NONCOLLINEAR:
            nelec = (widths[0], 0)
        else:
            nelec = widths * 2

        # one (ndets, npol*nmo, n) stack per spin channel, copied so the
        #   caller's arrays are not aliased
        self._dets = tuple(block.copy() for block in blocks)

        super().__init__(coeffs=coeffs, nelec=nelec,
                         nmo=self._dets[0].shape[1] // spin_symm.npol,
                         spin_symm=spin_symm)

        if self._dets[0].shape[0] != self.ndets:
            raise ValueError(
                f"dets holds {self._dets[0].shape[0]} determinant(s) but coeffs "
                f"has {self.ndets}"
            )

    # ------------------------------------------------------------------
    # views of the determinants
    # ------------------------------------------------------------------

    @property
    def dets(self):
        """Every determinant, in the layout the constructor takes."""
        return format_spin_layout(self._dets, self.spin_symm)

    def determinant(self, idet: int = 0):
        """
        Determinant `idet`, in the layout `Wavefunction.from_single_determinant`
        takes.
        """
        return format_spin_layout(tuple(channel[idet] for channel in self._dets),
                                  self.spin_symm)

    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "NOMSDWavefunction":
        """
        Return a copy whose determinants have orthonormal columns. See
        `Wavefunction.orthonormalize`.
        """
        dets = tuple(np.array([orthonormalize(block, tol=tol) for block in channel],
                              dtype=np.complex128)
                     for channel in self._dets)

        return type(self)(coeffs=self.coeffs.copy(),
                          dets=format_spin_layout(dets, self.spin_symm))

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def _write_payload(self, group) -> None:
        io.write_nomsd(group, self._dets)

    @classmethod
    def _read_payload(cls, group, header: dict) -> "NOMSDWavefunction":
        spin_symm = header['spin_symm']
        dets = io.read_nomsd(group, header['ndets'], spin_symm.nspin)

        return cls(coeffs=header['coeffs'],
                   dets=format_spin_layout(dets, spin_symm))
