#! /usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import json

from warnings import warn

import h5py as h5
import numpy
from pyscf import lib,scf
from pyscf.lib.chkfile import load_mol
from pyscf.pbc.lib.chkfile import load_cell

from afqmctools.utils.linalg import get_ortho_ao_mol
from afqmctools.utils.slater_types import (
    _SlaterType,
    )


def _walker_type_from_mo(mo_coeff, mo_occ, nao):
    """
    Determine the walker type of a PySCF SCF solution.

    The occupations are used rather than the shape of `mo_coeff` because an
    RHF and an ROHF coefficient matrix are both `(nao,nmo)` and cannot be
    told apart by shape alone. `_get_slater_type` must not be used here: it
    expects a Slater matrix over the occupied orbitals only, whereas
    `mo_coeff` holds the full set of molecular orbitals.

    Parameters
    ----------
    mo_coeff : np.ndarray
        PySCF molecular orbital coefficients: `(nao,nmo)` for RHF/ROHF,
        `(2,nao,nmo)` for UHF, `(2*nao,2*nmo)` for GHF.
    mo_occ : np.ndarray
        PySCF orbital occupations matching `mo_coeff`.
    nao : int
        the number of spatial atomic orbitals.

    Returns
    -------
    _SlaterType
        the walker type implied by the SCF solution.
    """
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)

    # UHF: spin-resolved coefficients
    if mo_coeff.ndim == 3:
        return _SlaterType.COLLINEAR

    # GHF: coefficients span both spin blocks
    if mo_coeff.ndim == 2 and mo_coeff.shape[0] == 2*nao:
        return _SlaterType.NONCOLLINEAR

    # spin-resolved occupations without spin-resolved coefficients
    if mo_occ.ndim == 2:
        return _SlaterType.COLLINEAR

    # ROHF: singly-occupied orbitals present
    if numpy.any(numpy.isclose(mo_occ, 1.0)):
        return _SlaterType.COLLINEAR

    # RHF: every occupied orbital is doubly occupied
    return _SlaterType.CLOSED


def chk_is_pbc(chkfile):
    """
    Determine if the pyscf chkfile corresponds to a PBC calculation.
        This is determined by the presence or absence of lattice vectors
        in the `mol` object. This lattice vectors are called 'a'
    """
    with h5.File(chkfile,'r') as f:
        if "a" in json.loads(f['mol'][()]).keys():
            return True
        else:
            return False


def mf_from_chkfile(chkfile, scf_class, scf):
  """Load mf object from chkfile, useful for restart and post processing

    Parameters
    ----------
    chkfile : str 
        path to PySCF chkfile
    scf_class : pyscf.scf.SCF 
        PySCF SCF class to construct e.g. scf.RHF, pbc.scf.RHF
    scf: module
        Module containing the scf_class (e.g. pyscf.scf, pyscf.pbc.scf)

    Returns
    -------

        class: mf object

    Examples
    --------

    >>> from pyscf import scf
    >>> rhf = mf_from_chkfile('rhf.chk', scf.RHF, scf)
    >>> from pyscf.pbc import scf
    >>> rhf = mf_from_chkfile('krhf.chk', scf.KRHF, scf)
  """
  cell, scf_rec = scf.chkfile.load_scf(chkfile)
  mf = scf_class(cell)
  mf.__dict__.update(scf_rec)
  return mf


def load_from_pyscf_chk(chkfile,hcore=None,orthoAO=False):
    cell = load_cell(chkfile)
    assert(cell is not None)

    kpts=None
    singleK = False
    if lib.chkfile.load(chkfile, 'scf/kpt') is not None :
        kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpt'))
        singleK = True
    else:
        kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
        assert(kpts is not None)
    kpts = numpy.reshape(kpts,(-1,3))
    nkpts = len(kpts)
    nao = cell.nao_nr()

    Xocc = lib.chkfile.load(chkfile, 'scf/mo_occ')
    mo_energy = lib.chkfile.load(chkfile, 'scf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile, 'scf/mo_coeff')
    fock = numpy.asarray(lib.chkfile.load(chkfile, 'scf/fock'))
    assert(fock is not None)
    if isinstance(Xocc,list):
        # 3 choices:
        if isinstance(Xocc[0],list):
            # KUHF
            isUHF = True
            assert(len(Xocc[0])==nkpts)
        elif singleK:
            # UHF
            isUHF = True
            assert(len(Xocc) == 2)
            Xocc = numpy.asarray(Xocc)
        else:
            # KRHF
            isUHF = False
            assert(len(Xocc) == nkpts)
    else:
        assert(singleK)
        if len(Xocc) == 2:
            isUHF = True
        else:
            # single kpoint RHF
            isUHF = False
            Xocc = ([Xocc])
            assert(len(Xocc)==nkpts)

    if hcore is None:
        hcore = numpy.asarray(lib.chkfile.load(chkfile, 'scf/hcore'))
        assert(hcore is not None)
    hcore = numpy.reshape(hcore,(-1,nao,nao))
    assert(hcore.shape[0]==nkpts)

    if(cell.spin!=0 and not isUHF):
        raise ValueError("cell.spin!=0 only allowed with UHF calculation")

    if(not orthoAO and isUHF):
        raise ValueError("orthoAO=True required with UHF calculation\n")

    if orthoAO:
        X_ = numpy.asarray(lib.chkfile.load(chkfile, 'scf/orthoAORot')).reshape(nkpts,nao,-1)
        assert(X_ is not None)
        nmo_pk = numpy.asarray(lib.chkfile.load(chkfile, 'scf/nmo_per_kpt'))
        # do this properly!!!
        if len(nmo_pk.shape) == 0:
            nmo_pk = numpy.asarray([nmo_pk])
        X = []
        for k in range(len(nmo_pk)):
            X.append(X_[k][:,0:nmo_pk[k]])
        assert(nmo_pk is not None)
    else:
        # can safely assume isUHF == False
        X = lib.chkfile.load(chkfile, 'scf/mo_coeff')
        if singleK:
            assert(len(X.shape) == 2)
            assert(X.shape[0] == nao)
            X = ([X])
        assert(len(X) == nkpts)
        nmo_pk = numpy.zeros(nkpts,dtype=numpy.int32)
        for ki in range(nkpts):
            nmo_pk[ki]=X[ki].shape[1]
            assert(nmo_pk[ki] == Xocc[ki].shape[0])

    if singleK:
        assert(nkpts==1)
        if isUHF:
            assert len(fock.shape) == 3
            assert fock.shape[0] == 2
            assert fock.shape[1] == nao
            fock = fock.reshape((2,1,fock.shape[1],fock.shape[2]))
            assert len(Xocc.shape) == 2
            Xocc = Xocc.reshape((2,1,Xocc.shape[1]))
            assert len(mo_energy.shape) == 2
            mo_energy = mo_energy.reshape((2,1,mo_energy.shape[1]))
        else:
            assert len(fock.shape) == 2
            assert fock.shape[0] == nao
            fock = fock.reshape((1,1)+fock.shape)
            mo_energy = mo_energy.reshape((1,-1))
    if len(fock.shape) == 3:
        fock = fock.reshape((1,)+fock.shape)
    scf_data = {
        'cell': cell,
        'kpts': kpts,
        'Xocc': Xocc, 
        'hcore': hcore, 
        'X': X, 
        'nmo_pk': nmo_pk,
        'mo_coeff': mo_coeff,
        'nao': nao,
        'fock': fock,
        'mo_energy': mo_energy,
        'walker_type' : _SlaterType.COLLINEAR if isUHF else _SlaterType.CLOSED
    }
    return scf_data


def load_from_pyscf_chk_mol(chkfile, base='scf', soc_type=None):
    """
    Loads data from a PysCF checkfile and returns a Python dict containing that data.

    Parameters
    ----------
    chkfile : str
        name of PySCF checkpoint file to read data from
    base : str
        name of HDF5 base group to read data from (default='scf')
    soc_type : {None, 'sfx2c', 'x2c', 'ecp'}, optional
        Include spin orbit coupling in either the spin-free exact 2-component (`'sfx2c'`), the full exact 2-component (`'x2c'`) flavor, or through the ECP (`'ecp'`).
    """
    mol = load_mol(chkfile)
    nmo = mol.nao_nr()
    mo_occ = numpy.array(lib.chkfile.load(chkfile, base+'/mo_occ'))
    mo_coeff = numpy.array(lib.chkfile.load(chkfile, base+'/mo_coeff'))

    walker_type = _walker_type_from_mo(
        mo_coeff,
        mo_occ,
        nmo
    )

    with h5.File(chkfile, 'r') as fh5:
        if '/scf/hcore' in fh5:
            if soc_type is not None:
                warn("Reading hcore from file, but it is unclear if the requested spin orbit coupling is included!")
            hcore = fh5['/scf/hcore'][:]
            if hcore.shape[-1] == 2*nmo:
                walker_type = _SlaterType.NONCOLLINEAR,
        else:
            if soc_type == "sfx2c":
                hcore = mol.RHF().sfx2c1e().get_hcore()
            elif soc_type == "x2c":
                hcore = mol.GHF().x2c1e().get_hcore()
                walker_type = _SlaterType.NONCOLLINEAR
            elif soc_type == "ecp":
                hcore = scf.GHF(mol).get_hcore() + get_ecp_soc(mol)
                walker_type = _SlaterType.NONCOLLINEAR
            elif soc_type is None:
                hcore = scf.hf.get_hcore(mol)
            else:
                raise ValueError(f"Unknown soc_type '{soc_type}'")

        if '/scf/orthoAORot' in fh5:
            X = fh5['/scf/orthoAORot'][:]
            # necessary if orbitals have been removed (i.e. due to linear dependence)
            nmo = X.shape[-1]
        else:
            s1e = mol.intor('int1e_ovlp_sph')
            X = get_ortho_ao_mol(s1e)
        
        if 'j3c' in fh5:
            df_ints = fh5['j3c'][:]
        else:
            df_ints = None

    scf_data = {
        'mol': mol,
        'nelec' : mol.nelec,
        'mo_occ': mo_occ,
        'hcore': hcore,
        'norb' : nmo,
        'X': X,
        'mo_coeff': mo_coeff,
        'df_ints': df_ints,
        'walker_type' : walker_type, # suggested walker type
        'soc_type' : soc_type # identifies that hcore has SOC
    }
    return scf_data

def ci2chk(chkfile,ci):
    with h5.File(chkfile,"a") as f:
        if 'mcscf/ci' not in f:
            f.create_dataset("mcscf/ci",data=ci)
        else:
            f['mcscf/ci'][...] = ci


def ci_wavefunction(ciab, norb, nelec, ncore, tol=1.0e-4, max_det=None, sort_by_ci=True):
    r"""
    Extract CI wavefunction from CI wavefunction in PySCF format.

    For wavefunctions with format:
    
    .. math::
        | \Psi \rangle = \sum_{ij} c_{ij} | \Phi^{\uparrow}_i \Phi^{\downarrow}_j \rangle
    
    return the CI coefficients and the corresponding alpha and beta determinants which satisfy 
    the truncation criteria,

    .. math::
        |c_{ij}| > \text{tol}

    If a maximum number of determinants is specified, the determinants with the largest
    Ci coefficients are kept.

    Parameters
    ----------
    ciab:tuple
        CI coefficient matrix with shape (number of alpha determinants, number of beta determinants)
    norb:int
        Number of orbitals.
    nelec:tuple
        Number of alpha and beta electrons.
    ncore:int
        Number of core orbitals. These are always assumed to be the lowest ncore orbitals.
    tol:float, optional
        Tolerance for truncating CI expansion. All determinants with ciab > tol are kept. Default is 1.0e-6.
    max_det:int, optional
        Maximum number of determinants to keep. If None, keep all determinants with.
    sort_by_ci:bool, optional
        If True, the CI coefficients are sorted by magnitude. Default is True.

    Returns
    -------
    ci:tuple
        CI coefficients.
    occa:tuple
        orbital occupancies of Alpha determinants.
    occb:tuple
        orbital occupancies of Beta determinants.

    Notes
    -----
    If max_det is None, the Ci coefficients are NOT sorted by magnitude.
    """
    from pyscf.fci.addons import large_ci
    nalpha, nbeta = nelec
    ci, occa, occb = zip(*large_ci(ciab, norb, (nalpha, nbeta),
                         tol=tol, return_strs=False))
    
    if sort_by_ci or max_det is not None:
        sort_idx = numpy.argsort(numpy.abs(ci))[::-1].astype(int)
        ci = tuple(numpy.array(ci)[sort_idx])
        occa = tuple(numpy.array(occa)[sort_idx])
        occb = tuple(numpy.array(occb)[sort_idx])

    if max_det is not None:
        max_det = min(max_det, len(ci))
        ci = tuple(ci[:max_det])
        occa = tuple(occa[:max_det])
        occb = tuple(occb[:max_det])

    # Reinsert frozen core to fully specify each determinant
    core = [i for i in range(ncore)]
    occa = [numpy.array(core + [o + ncore for o in oa]) for oa in occa]
    occb = [numpy.array(core + [o + ncore for o in ob]) for ob in occb]
    return ci, occa, occb


def read_cas_meta(chkfile, group='mcscf', root=None):                                                                                                                        
    meta = dict()
    keys = [
        'ci', #( root and 'ci' ) or ( f'ci_{root}' ),
        'ncore',
        'ncas'
    ]
    with h5.File(chkfile, 'r') as f:
        for key in keys:
            meta[key] = f[group][key][()]
    return meta

def get_ecp_soc(mol):
    """Compute the ECP-SOC term for `mol`. See pyscf example gto/20-soc_ecp.py."""

    s = .5 * lib.PauliMatrices
    ecpso = -1j * lib.einsum('sxy,spq->xpyq', s, mol.intor('ECPso'))
    return ecpso.reshape(ecpso.shape[0] * ecpso.shape[1], ecpso.shape[2] * ecpso.shape[3])
