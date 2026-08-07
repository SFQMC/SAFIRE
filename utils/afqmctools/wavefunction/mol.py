# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from warnings import warn
import logging
logger = logging.getLogger(__name__)

import numpy as np
import h5py
import scipy.linalg

from collections.abc import Iterable

try: # PySCF is now considered an optional dependency!
    from afqmctools.utils.pyscf_utils import (
        ci_wavefunction,
        read_cas_meta
    )
except ModuleNotFoundError as e:
    logger.dev(f"Could not load pyscf: {e}")

from afqmctools.utils.io import (
    to_complex, 
    add_group, 
)

from afqmctools.utils.slater_types import (
    _SlaterType,
    _get_slater_type,
    _slater2dims,
    _slater_enum_map
)

from afqmctools.wavefunction.common import (
    write_wfn,
    write_nomsd,
    write_nomsd_ghf,
    write_phmsd
) 

from afqmctools.hamiltonian.mol import _transform_from_scf_data


def _make_slater_closed(mo_coeffs: np.ndarray, nocc: Iterable[int], nelec: int) -> np.ndarray:
    wfn = np.zeros((mo_coeffs.shape[1], nelec))
    wfn[nocc, np.arange(nelec)] = 1

    return mo_coeffs @ wfn + 0j


def _make_slater_collinear(
    mo_coeffs: np.ndarray, nocc: Iterable[Iterable[int]], nelec: Iterable[int]
) -> np.ndarray:
    if len(nocc) != len(nelec):
        raise ValueError(
            f"Lengths of nocc ({len(nocc)}) and nelec ({len(nelec)}) are not the same."
        )

    if len(mo_coeffs.shape) == 3:
        return np.concatenate(
            [
                _make_slater_closed(spin_mo_coeffs, spin_nocc, spin_nelec)
                for spin_mo_coeffs, spin_nocc, spin_nelec in zip(mo_coeffs, nocc, nelec)
            ],
            axis=1,
        )
    return np.concatenate(
        [
            _make_slater_closed(mo_coeffs, spin_nocc, spin_nelec)
            for spin_nocc, spin_nelec in zip(nocc, nelec)
        ],
        axis=1,
    )


def _make_slater(walker_type: _SlaterType, mo_coeffs: np.ndarray, nocc: Iterable[Iterable[int]], nelec: Iterable[int]) -> np.ndarray:
    if walker_type == _SlaterType.CLOSED:
        return _make_slater_closed(mo_coeffs, nocc[0], nelec[0])
    elif walker_type == _SlaterType.COLLINEAR:
        return _make_slater_collinear(mo_coeffs, nocc, nelec)
    elif walker_type == _SlaterType.NONCOLLINEAR:
        return _make_slater_closed(mo_coeffs, nocc[0], sum(nelec))


def _transform_slater(wfn: np.ndarray, X: np.ndarray):
    if X.shape[0] != wfn.shape[0]:
        X = np.kron(np.eye(2), X)
    return X.conj().T @ wfn


def _get_occupied_indices(mo_occ, walker_type, nfzc=0, nfzv=0):
    """Return occupied orbital indices for each spin sector.

    Indices are returned in the basis used to build the default AFQMC init
    state. When a CAS is requested, occupied orbitals are trimmed to the active
    window and shifted so they are local to that active-space basis.
    """
    mo_occ = np.asarray(mo_occ)
    walker_type = _slater_enum_map(walker_type)

    def _trim_active_space(indices):
        return indices[(indices >= nfzc) & (indices < mo_occ.shape[-1] - nfzv)]

    if walker_type == _SlaterType.CLOSED:
        occ = np.flatnonzero(mo_occ >= 1)
        occ = _trim_active_space(occ)
        return occ, occ

    if walker_type == _SlaterType.COLLINEAR:
        if mo_occ.ndim == 1:
            occa = np.flatnonzero(mo_occ > 0)
            occb = np.flatnonzero(mo_occ > 1)
        elif mo_occ.ndim == 2:
            occa = np.flatnonzero(mo_occ[0] > 0)
            occb = np.flatnonzero(mo_occ[1] > 0)
        else:
            raise ValueError("Invalid mo_occ shape for collinear walkers")

        return _trim_active_space(occa), _trim_active_space(occb)

    if walker_type == _SlaterType.NONCOLLINEAR:
        occ = np.flatnonzero(mo_occ >= 1)
        return _trim_active_space(occ), None

    raise ValueError("Invalid Slater determinant type")


def generate_wavefunction(scf_data, basis_scf_data=None, ortho_ao=False, cas=None, verbose=False):
    if basis_scf_data is None:
        basis_scf_data = scf_data

    nelec = scf_data['nelec']
    norb = scf_data['norb']

    walker_type = _slater_enum_map(
        scf_data['walker_type']
    )

    X, (nfzc, nfzv) = _transform_from_scf_data(basis_scf_data, ortho_ao, cas)

    nelec = tuple(n-nfzc for n in nelec)
    norb -= (nfzc + nfzv)

    occa, occb = _get_occupied_indices(
        scf_data["mo_occ"],
        walker_type,
        nfzc=nfzc,
        nfzv=nfzv
    )

    if walker_type == _SlaterType.NONCOLLINEAR:
        if len(occa) != sum(nelec):
            raise ValueError(
                f"mo_occ defines {len(occa)} alpha occupied orbitals, expected {sum(nelec)}"
            )
    else:
        if len(occa) != nelec[0]:
            raise ValueError(
                f"mo_occ defines {len(occa)} alpha occupied orbitals, expected {nelec[0]}"
            )
        if occb is not None and len(occb) != nelec[1]:
            raise ValueError(
                f"mo_occ defines {len(occb)} beta occupied orbitals, expected {nelec[1]}"
            )

    wfn = _make_slater(
        walker_type,
        scf_data["mo_coeff"],
        (occa, occb),
        nelec
    )

    overlap = scf_data["mol"].intor("int1e_ovlp")

    wfn = _transform_slater(wfn, overlap @ X[:, nfzc:X.shape[-1]-nfzv])

    return wfn + 0j, nelec, norb

def write_wfn_mol(scf_data, filename, basis_scf_data=None, wfn=None,
                  init=None, verbose=False, ortho_ao=False, cas=None):
    """Generate SAFIRE format trial wavefunction.

    Parameters
    ----------
    scf_data : dict
        Dictionary containing scf data extracted from pyscf checkpoint file.
    filename : string
        HDF5 file path to store wavefunction to.
    wfn : tuple
        User defined wavefunction. Not fully supported. Default None.
    init : optional
        Initial wavefunction to use in AFQMC. Default is None.
    basis_scf_data : dict, optional
        Dictionary containing scf data for the basis set in which to express
        the wavefunction. If None, the wavefunction will be expressed in the
        basis defined by `scf_data`.
    verbose : bool, optional
        If True, print additional information. Default is False.
    ortho_ao : bool, optional
        Work in Löwdin orthogonalized atomic orbitals instead of molecular orbitals.
        Make sure this is consistent with the Hamiltonian.
    cas : tuple, optional
        If provided, indicates that the wavefunction is to be expressed
        in a CAS active space as (# active electrons, # active orbitals).
        Cannot be combined with `ortho_ao`.

    Returns
    -------
    wfn : :class:`np.ndarray`
        Wavefunction as numpy array. Format depends on wavefunction.
    """
    if wfn is None:
        wfn, nelec, norb = generate_wavefunction(scf_data=scf_data, basis_scf_data=basis_scf_data, ortho_ao=ortho_ao, cas=cas, verbose=verbose)
    else:
        nelec = scf_data["nelec"]
        norb = scf_data["norb"]
    
    walker_type = _get_slater_type(wfn, nelec, norb)
    
    # Wavefunctions are assumed to be a 1-D iterable of wavefunctions, even for a single determinant!
    write_wfn(
        filename,
        (np.array([1.0+0j]),np.array([wfn])),
        nelec=nelec,
        norb=norb,
        walker_type=walker_type,
        verbose=verbose,
        init=init
    )
    return nelec


def write_cas_wfn(mol, cas_chkfile, outname='afqmc_wfn.h5', tol_trunc=1.0e-4, max_det=None):
    """
    Write a CAS wavefunction to an HDF5 file.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object from PySCF.
    cas_chkfile : str
        Path to the PySCF CAS checkpoint file.
    outname : str
        Output HDF5 file name.
    tol_trunc : float
        Truncation threshold for CI coefficients.
    max_det : int
        Maximum number of determinants to keep. If None, all determinants within tol_trunc are kept.
    """
    # read CAS information
    cas_meta = read_cas_meta(cas_chkfile)
    ncas = cas_meta['ncas']
    ncore = cas_meta['ncore']
    nvals = [n-ncore for n in mol.nelec]

    print(cas_meta)

    ciab = cas_meta[
        'ci'
    ]

    nmo = mol.nao_nr()
    ci, occa, occb = ci_wavefunction(
        ciab,
        ncas,
        nvals,
        ncore,
        tol_trunc,
        max_det=max_det
    )

    ndet = len(ci)
    print('number of determinants: %d' % ndet)

    ci = np.array(ci, dtype=np.complex128)
    uhf = True # UHF always true for CI expansions.
    write_wfn(
        outname,
        (ci, occa, occb),
        uhf, # this just describes the walker type!
        mol.nelec,
        nmo
    )

