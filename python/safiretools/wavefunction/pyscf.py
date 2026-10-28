# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Trial wavefunctions from molecular PySCF calculations.

Two constructions live here, reached through
`NOMSDWavefunction.from_pyscf` and `PHMSDWavefunction.from_pyscf_cas`: a single
Slater determinant built from an SCF object's occupied orbitals, and a
multi-determinant expansion read out of a CASSCF/CASCI object.

Both express the wavefunction in the same working basis the Hamiltonian uses, so
the ``basis`` and ``active_space`` arguments must match the ones
`safiretools.MolecularHamiltonian.from_pyscf` was given.
"""

import logging

import numpy as np

from safiretools.convert.pyscf import determine_spin_symm, orbitals, working_basis
from safiretools.types import SpinSymm
from safiretools.wavefunction.slater import (
    format_spin_layout,
    make_slater,
    transform_slater,
)

logger = logging.getLogger(__name__)


def from_pyscf(mf, basis=None, active_space=None):
    """
    Build a single-determinant trial wavefunction from a molecular PySCF SCF
    object.

    Parameters
    ----------
    mf
        The converged SCF solution the wavefunction is built *from*. Its spin
        symmetry is that of the solution: noncollinear for GHF, collinear for
        UHF or an ROHF solution with singly occupied orbitals, closed for RHF.
    basis : None or 'ortho_ao' or object or numpy.ndarray, optional
        The orbitals the wavefunction is expressed *in*, which must be the ones
        the Hamiltonian was built in; takes the same values as
        `safiretools.MolecularHamiltonian.from_pyscf`'s `basis`. Defaults to
        `mf`'s own.
    active_space : tuple(int, int), optional
        ``(nelecas, ncas)`` active space, which must match the Hamiltonian's.
        Occupied orbitals are trimmed to the active window and reindexed into
        it. Incompatible with ``basis='ortho_ao'``.

    Returns
    -------
    NOMSDWavefunction
        A single-determinant wavefunction.

    Raises
    ------
    ValueError
        If `mf` holds no orbitals, if `basis` cannot be used, or if ``mo_occ``
        does not describe the expected number of occupied orbitals.
    """
    from safiretools.wavefunction.nomsd import NOMSDWavefunction

    mol = mf.mol
    mo_coeff, mo_occ = orbitals(mf)
    spin_symm = determine_spin_symm(mo_coeff, mo_occ, mol.nao_nr())

    X, (nfzc, nfzv) = working_basis(mf, basis, active_space)

    nelec = tuple(n - nfzc for n in mol.nelec)
    norb = X.shape[-1] - nfzc - nfzv

    occa, occb = _occupied_indices(mo_occ, spin_symm, nfzc=nfzc, nfzv=nfzv)
    _check_occupations(occa, occb, nelec, spin_symm)

    overlap = mol.intor('int1e_ovlp')
    transform = overlap @ X[:, nfzc:X.shape[-1] - nfzv]
    channels = tuple(transform_slater(occupied, transform) + 0j for occupied
                     in make_slater(spin_symm, mo_coeff, (occa, occb)))

    logger.info("built a %s single-determinant trial wavefunction: "
                "nelec=%s, nmo=%d", spin_symm.label, nelec, norb)

    return NOMSDWavefunction.from_single_determinant(
        format_spin_layout(channels, spin_symm))


def from_pyscf_cas(mc, tol=1e-4, max_det=None):
    """
    Read a CASSCF/CASCI expansion as a particle-hole multi-determinant
    wavefunction.

    The determinants are occupations of `mc`'s own orbitals, so the Hamiltonian
    has to be built with ``basis=mc``.

    Parameters
    ----------
    mc
        A PySCF ``mcscf`` object that has been run, holding ``ci``, ``ncore``,
        ``ncas``, ``nelecas`` and ``fcisolver``.
    tol : float, optional
        Keep determinants whose coefficient exceeds this in magnitude. Default
        1e-4.
    max_det : int, optional
        Keep at most this many determinants, the largest coefficients first. All
        determinants above `tol` are kept when omitted.

    Returns
    -------
    PHMSDWavefunction
        The truncated expansion, over the full orbital basis: the frozen core is
        reinserted into every determinant's occupations.
    """
    from safiretools.wavefunction.phmsd import PHMSDWavefunction

    coeffs, occa, occb = ci_expansion(mc, tol=tol, max_det=max_det)

    logger.info("read %d determinant(s) from the CI expansion", len(coeffs))

    return PHMSDWavefunction(
        coeffs=np.array(coeffs, dtype=np.complex128),
        occa=occa,
        occb=occb,
        nmo=np.shape(mc.mo_coeff)[-1],
        nelec=mc.mol.nelec,
    )


# ----------------------------------------------------------------------
# reading a reference's occupations
# ----------------------------------------------------------------------

def _occupied_indices(mo_occ, spin_symm: SpinSymm, nfzc=0, nfzv=0):
    """
    Occupied orbital indices per spin channel, in the basis the default AFQMC
    initial state is built in.

    With an active space requested, the indices are trimmed to the active window
    and shifted so that they are local to it.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray or None)
        Alpha and beta indices. A closed-shell reference repeats alpha as beta;
        a noncollinear one has None for beta.
    """
    mo_occ = np.asarray(mo_occ)

    def trim(indices):
        return indices[(indices >= nfzc) & (indices < mo_occ.shape[-1] - nfzv)]

    if spin_symm is SpinSymm.CLOSED:
        occ = trim(np.flatnonzero(mo_occ >= 1))
        return occ, occ

    if spin_symm is SpinSymm.NONCOLLINEAR:
        return trim(np.flatnonzero(mo_occ >= 1)), None

    if mo_occ.ndim == 1:
        # an ROHF reference records both channels in one occupancy vector
        occa = np.flatnonzero(mo_occ > 0)
        occb = np.flatnonzero(mo_occ > 1)
    elif mo_occ.ndim == 2:
        occa = np.flatnonzero(mo_occ[0] > 0)
        occb = np.flatnonzero(mo_occ[1] > 0)
    else:
        raise ValueError(
            f"mo_occ has shape {mo_occ.shape}, which describes no collinear "
            "reference"
        )

    return trim(occa), trim(occb)


def _check_occupations(occa, occb, nelec, spin_symm: SpinSymm) -> None:
    """Check that the occupancies account for exactly `nelec` electrons."""
    if spin_symm is SpinSymm.NONCOLLINEAR:
        if len(occa) != sum(nelec):
            raise ValueError(
                f"mo_occ defines {len(occa)} occupied spinor orbitals, expected "
                f"{sum(nelec)}"
            )
        return

    if len(occa) != nelec[0]:
        raise ValueError(
            f"mo_occ defines {len(occa)} alpha occupied orbitals, expected "
            f"{nelec[0]}"
        )
    if len(occb) != nelec[1]:
        raise ValueError(
            f"mo_occ defines {len(occb)} beta occupied orbitals, expected "
            f"{nelec[1]}"
        )


# ----------------------------------------------------------------------
# reading a CI expansion
# ----------------------------------------------------------------------

def ci_expansion(mc, tol=1e-4, max_det=None):
    r"""
    Truncate a PySCF CI expansion into an explicit determinant list.

    For a wavefunction

    .. math::
        |\Psi\rangle = \sum_{ij} c_{ij}\,
                       |\Phi^{\uparrow}_i \Phi^{\downarrow}_j\rangle

    return the coefficients with :math:`|c_{ij}| > \mathrm{tol}` and the
    occupied orbitals of each determinant, largest coefficient first.

    Parameters
    ----------
    mc
        A PySCF ``mcscf`` object that has been run. Its CI vector ``ci`` over
        ``ncas`` active orbitals and ``nelecas`` electrons is read through
        ``fcisolver.large_ci``, and its ``ncore`` core orbitals, taken to be
        the lowest, are reinserted into every determinant's occupations, so the
        result indexes the full orbital basis.
    tol : float, optional
        Keep determinants whose coefficient exceeds this in magnitude. Default
        1e-4.
    max_det : int, optional
        Keep at most this many determinants. All of them when omitted.

    Returns
    -------
    coeffs : numpy.ndarray
        Kept coefficients, ``(ndets,)``.
    occa, occb : numpy.ndarray
        Occupied orbital indices, ``(ndets, nup)`` and ``(ndets, ndown)``.
    """
    ncore = mc.ncore
    coeffs, occa, occb = zip(*mc.fcisolver.large_ci(
        mc.ci, mc.ncas, tuple(mc.nelecas), tol=tol, return_strs=False))

    order = np.argsort(np.abs(coeffs))[::-1]
    if max_det is not None:
        order = order[:max_det]

    core = list(range(ncore))
    return (
        np.asarray(coeffs)[order],
        np.array([core + [orbital + ncore for orbital in occa[i]] for i in order]),
        np.array([core + [orbital + ncore for orbital in occb[i]] for i in order]),
    )
