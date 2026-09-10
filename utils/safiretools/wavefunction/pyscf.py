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
Slater determinant built from an SCF solution's occupied orbitals, and a
multi-determinant expansion read out of a CASSCF/CASCI checkpoint.

Both express the wavefunction in the same working basis the Hamiltonian uses, so
the ``ortho_ao`` and ``cas`` arguments must match the ones
`safiretools.MolecularHamiltonian.from_pyscf` was given.
"""

import logging

import numpy as np
import h5py as h5

from safiretools.hamiltonian.molecular import _get_transform_from_scf_data
from safiretools.types import SpinSymm
from safiretools.wavefunction.slater import make_slater, transform_slater

logger = logging.getLogger(__name__)


def from_pyscf(scf_data, basis_scf_data=None, ortho_ao=False, cas=None,
               spin_symm=None, orthonormalize=True):
    """
    Build a single-determinant trial wavefunction from a molecular PySCF SCF
    calculation.

    Parameters
    ----------
    scf_data : dict
        Unpacked PySCF checkpoint, as produced by
        ``afqmctools.utils.pyscf_utils.load_from_pyscf_chk_mol``. Uses the keys
        ``'mol'``, ``'mo_coeff'``, ``'mo_occ'``, ``'nelec'``, ``'norb'``,
        ``'X'`` and ``'walker_type'``.
    basis_scf_data : dict, optional
        A second checkpoint whose orbitals define the basis to express the
        wavefunction in. Defaults to `scf_data` itself.
    ortho_ao : bool, optional
        Work in the Löwdin-orthogonalized AO basis rather than the MO basis.
        Must match the Hamiltonian. Default False.
    cas : tuple(int, int), optional
        ``(nelecas, ncas)`` active space, which must match the Hamiltonian's.
        Occupied orbitals are trimmed to the active window and reindexed into
        it. Incompatible with `ortho_ao`.
    spin_symm : SpinSymm or str or int, optional
        Overrides the spin symmetry inferred from the orbital matrix's shape.
    orthonormalize : bool, optional
        Orthonormalize the resulting Slater matrix. Default True.

    Returns
    -------
    NOMSDWavefunction
        A single-determinant wavefunction.

    Raises
    ------
    ValueError
        If ``mo_occ`` does not describe the expected number of occupied
        orbitals, or the orbital matrix's shape matches no spin symmetry.

    Notes
    -----
    afqmctools' ``write_wfn_mol`` also took ``wfn=`` (a caller-supplied Slater
    matrix, documented there as not fully supported) and ``init=`` (the initial
    walker determinant). Neither is needed now: construct
    `safiretools.NOMSDWavefunction` directly for the former, and assign to
    ``wavefunction.psi0`` for the latter.

    A reference with no beta electrons — which a large enough frozen core can
    produce from an open-shell one — is `SpinSymm.COLLINEAR` with
    ``ndown == 0``, and its beta blocks go to disk with zero width.
    """
    from safiretools.wavefunction.nomsd import NOMSDWavefunction, infer_spin_symm

    if basis_scf_data is None:
        basis_scf_data = scf_data

    nelec = scf_data['nelec']
    norb = scf_data['norb']
    reference_symm = SpinSymm.from_input(scf_data['walker_type'])

    X, (nfzc, nfzv) = _get_transform_from_scf_data(basis_scf_data, ortho_ao, cas)

    nelec = tuple(n - nfzc for n in nelec)
    norb -= (nfzc + nfzv)

    occa, occb = _occupied_indices(scf_data['mo_occ'], reference_symm,
                                   nfzc=nfzc, nfzv=nfzv)
    _check_occupations(occa, occb, nelec, reference_symm)

    orbitals = make_slater(reference_symm, scf_data['mo_coeff'],
                           (occa, occb), nelec)

    overlap = scf_data['mol'].intor('int1e_ovlp')
    orbitals = transform_slater(
        orbitals, overlap @ X[:, nfzc:X.shape[-1] - nfzv]) + 0j

    if spin_symm is None:
        spin_symm = infer_spin_symm(orbitals, nelec, norb)
    else:
        spin_symm = SpinSymm.from_input(spin_symm)

    logger.info("built a %s single-determinant trial wavefunction: "
                "nelec=%s, nmo=%d", spin_symm.label, nelec, norb)

    wavefunction = NOMSDWavefunction(
        coeffs=np.array([1.0 + 0j]),
        dets=orbitals[np.newaxis, ...],
        nelec=nelec,
        spin_symm=spin_symm,
        nmo=norb,
    )

    return wavefunction.orthonormalize() if orthonormalize else wavefunction


def from_pyscf_cas(mol, cas_chkfile, tol=1e-4, max_det=None):
    """
    Read a CASSCF/CASCI expansion from a PySCF checkpoint as a particle-hole
    multi-determinant wavefunction.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule the expansion was computed for.
    cas_chkfile : str or pathlib.Path
        PySCF checkpoint holding an ``mcscf`` group with ``ci``, ``ncore`` and
        ``ncas``.
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

    cas_meta = read_cas_meta(cas_chkfile)
    ncore = int(cas_meta['ncore'])
    ncas = int(cas_meta['ncas'])
    nactive = tuple(n - ncore for n in mol.nelec)

    coeffs, occa, occb = ci_expansion(cas_meta['ci'], ncas, nactive, ncore,
                                      tol=tol, max_det=max_det)

    logger.info("read %d determinant(s) from %s", len(coeffs), cas_chkfile)

    return PHMSDWavefunction(
        coeffs=np.array(coeffs, dtype=np.complex128),
        occa=occa,
        occb=occb,
        nmo=mol.nao_nr(),
        nelec=mol.nelec,
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
        Alpha and beta indices; the beta entry is None for the single-channel
        spin symmetries.
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
    if occb is not None and len(occb) != nelec[1]:
        raise ValueError(
            f"mo_occ defines {len(occb)} beta occupied orbitals, expected "
            f"{nelec[1]}"
        )


# ----------------------------------------------------------------------
# reading a CI expansion
# ----------------------------------------------------------------------

def read_cas_meta(chkfile, group='mcscf') -> dict:
    """
    Read the ``ci``, ``ncore`` and ``ncas`` fields of a PySCF CAS checkpoint.

    Parameters
    ----------
    chkfile : str or pathlib.Path
        Checkpoint file to read.
    group : str, optional
        HDF5 group holding the CAS results. Default ``'mcscf'``.

    Returns
    -------
    dict
        Keys ``ci``, ``ncore`` and ``ncas``.
    """
    with h5.File(chkfile, 'r') as fh5:
        return {key: fh5[group][key][()] for key in ('ci', 'ncore', 'ncas')}


def ci_expansion(ciab, norb: int, nelec, ncore: int, tol=1e-4, max_det=None):
    r"""
    Truncate a PySCF CI coefficient matrix into an explicit determinant list.

    For a wavefunction

    .. math::
        |\Psi\rangle = \sum_{ij} c_{ij}\,
                       |\Phi^{\uparrow}_i \Phi^{\downarrow}_j\rangle

    return the coefficients with :math:`|c_{ij}| > \mathrm{tol}` and the
    occupied orbitals of each determinant, largest coefficient first.

    Parameters
    ----------
    ciab : numpy.ndarray
        CI coefficient matrix, ``(n_alpha_dets, n_beta_dets)``.
    norb : int
        Number of active orbitals.
    nelec : tuple(int, int)
        Active-space electron counts.
    ncore : int
        Number of core orbitals, taken to be the lowest `ncore`. They are
        reinserted into every determinant's occupations, so the result indexes
        the full orbital basis.
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
    from pyscf.fci.addons import large_ci

    coeffs, occa, occb = zip(*large_ci(ciab, norb, tuple(nelec), tol=tol,
                                       return_strs=False))

    order = np.argsort(np.abs(coeffs))[::-1]
    if max_det is not None:
        order = order[:max_det]

    core = list(range(ncore))
    return (
        np.asarray(coeffs)[order],
        np.array([core + [orbital + ncore for orbital in occa[i]] for i in order]),
        np.array([core + [orbital + ncore for orbital in occb[i]] for i in order]),
    )
