# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Write mean-field trial wavefunctions to file."""
import numpy as np
import h5py
import itertools
import time
import sys
import scipy.linalg

from afqmctools.wavefunction.common import (
    write_wfn,
    _SlaterType,
    _get_slater_type,
    _slater_enum_map
) 

def write_wfn_pbc_old(scf_data, ortho_ao, filename, rediag=True,
                  verbose=False, ndet_max=1, low=0.1, high=0.95):
    """Generate trial wavefunction for PBC simulation.

    Parameters
    ----------
    scf_data : dict
        Dictionary containing scf data extracted from pyscf checkpoint file.
    ortho_ao : bool
        Whether we are working in orthogonalised AO basis or not.
    filename : string
        HDF5 file path to store wavefunction to.
    rediag : bool
        Whether to rediagonalise Fock matrix to compute MO coeffs in
        orthoagonlised AO basis. Default: True.
    verbose : bool
        Print additional information. Default: False.

    Returns
    -------
    wfn : :class:`np.ndarray'
        Wavefunction as numpy array. Format depends on wavefunction.
    """
    # Single determinant for the moment.
    cell = scf_data['cell']
    C = scf_data['mo_coeff']
    
    X = scf_data['X']
    mo_occ = np.array(scf_data['Xocc'])
    fock = scf_data['fock']
    nmo_pk = scf_data['nmo_pk']
    mo_energy = scf_data['mo_energy']
    nmo_tot = np.sum(nmo_pk)
    kpts = scf_data['kpts']
    nkpts = len(kpts)
    uhf = scf_data['isUHF']
    
    if len(fock.shape) == 3:
        fock = fock.reshape((1,)+fock.shape)
    if verbose:
        print(" # Generating trial wavefunction.")
        if uhf:
            print(" # UHF-like trial wavefunction.")
        else:
            print(" # RHF-like trial wavefunction.")
        print(" # k-points: ")
        for (i,k) in enumerate(kpts):
            print(" # {:d} {}".format(i,k))
    
    (eigs, orbs, ks, bands) = generate_orbitals(
        fock,
        X,
        nmo_pk,
        rediag,
        ortho_ao,
        mo_energy,
        uhf, 
        verbose=verbose
    )
    
    re_occ, trial, ndeg, srt, isrt = reoccupy(
                                        mo_occ,
                                        eigs,
                                        uhf,
                                        verbose,
                                        ndet_max=ndet_max,
                                        low=low,
                                        high=high
                                    )
    
    # mo_occs from pyscf is a list of numpy arrays of potentially different
    # length so can't just use np.sum.
    nalpha = int(round(sum(sum(occ) for occ in re_occ[0])))
    nbeta = int(round(sum(sum(occ) for occ in re_occ[1])))
    nelec = (nalpha, nbeta)
    
    if ndeg == 1 or ndet_max == 1:
        if verbose:
            print(" # Writing NOMSD single Slater determinant Trial.")
        wfn = create_wavefunction(orbs, re_occ, nmo_pk, nelec, uhf, verbose)
        coeff = np.array([1.0+0j])
        trial = (coeff, wfn)
   
    # For RHF only nalpha entries will be filled.
    orb_mat_a = scipy.linalg.block_diag(*orbs[0])
    print_eigenvalues(ks, bands, eigs, uhf, srt, nelec, verbose)
    if uhf:
        orb_mat_b = scipy.linalg.block_diag(*orbs[1])
    else:
        orb_mat_b = None
    
    write_wfn(
        filename,
        trial,
        uhf,
        (nalpha, nbeta),
        nmo_tot,
        orbmat=(orb_mat_a, orb_mat_b),
        verbose=verbose
    )
    return nelec


def write_wfn_pbc(
    scf_data,
    ortho_ao,
    filename,
    rediag=True,
    verbose=False,
    ndet_max=1,
    low=0.1,
    high=0.95
    ):
    """Generate trial wavefunction for PBC simulation.

    Parameters
    ----------
    scf_data : dict
        Dictionary containing scf data extracted from pyscf checkpoint file.
    ortho_ao : bool
        Whether we are working in orthogonalised AO basis or not.
    filename : string
        HDF5 file path to store wavefunction to.
    rediag : bool
        Whether to rediagonalise Fock matrix to compute MO coeffs in
        orthoagonlised AO basis. Default: True.
    verbose : bool
        Print additional information. Default: False.

    Returns
    -------
    wfn : :class:`np.ndarray'
        Wavefunction as numpy array. Format depends on wavefunction.
    """
    # Single determinant for the moment.
    X = scf_data['X']
    mo_occ = np.array(scf_data['Xocc'])
    fock = scf_data['fock']
    nmo_pk = scf_data['nmo_pk']
    mo_energy = scf_data['mo_energy']
    nmo_tot = np.sum(nmo_pk)
    kpts = scf_data['kpts']
    #uhf = scf_data['isUHF']

    # TODO: make this independent of the trial type!
    walker_type = _slater_enum_map(
        scf_data['walker_type']
    )

    uhf = (walker_type == _SlaterType.COLLINEAR)

    if len(fock.shape) == 3:
        fock = fock.reshape((1,)+fock.shape)
    
    if verbose:
        print(" # Generating trial wavefunction.")
        if walker_type == _SlaterType.COLLINEAR:
            print(" # UHF-like trial wavefunction.")
        else:
            print(" # RHF-like trial wavefunction.")
        print(" # k-points: ")
        for (i,k) in enumerate(kpts):
            print(" # {:d} {}".format(i,k))
    
    (eigs, orbs, ks, bands) = generate_orbitals(
        fock,
        X,
        nmo_pk,
        rediag,
        ortho_ao,
        mo_energy,
        walker_type == _SlaterType.COLLINEAR, 
        verbose=verbose
    )
    
    re_occ, trial, ndeg, srt, isrt = reoccupy(
                                        mo_occ,
                                        eigs,
                                        walker_type == _SlaterType.COLLINEAR,
                                        verbose,
                                        ndet_max=ndet_max,
                                        low=low,
                                        high=high
                                    )
    
    # mo_occs from pyscf is a list of numpy arrays of potentially different
    # length so can't just use np.sum.
    nalpha = int(round(sum(sum(occ) for occ in re_occ[0])))
    nbeta = int(round(sum(sum(occ) for occ in re_occ[1])))
    nelec = (nalpha, nbeta)
    
    if ndeg == 1 or ndet_max == 1:
        if verbose:
            print(" # Writing NOMSD single Slater determinant Trial.")
        wfn = create_wavefunction(orbs, re_occ, nmo_pk, nelec, uhf, verbose)
        coeff = np.array([1.0+0j])
        trial = (coeff, wfn)
   
    # For RHF only nalpha entries will be filled.
    orb_mat_a = scipy.linalg.block_diag(*orbs[0])
    print_eigenvalues(ks, bands, eigs, uhf, srt, nelec, verbose)
    if uhf:
        orb_mat_b = scipy.linalg.block_diag(*orbs[1])
    else:
        orb_mat_b = None
    
    write_wfn(
        filename,
        wfn=trial,
        walker_type=walker_type,
        nelec=(nalpha, nbeta),
        norb=nmo_tot,
        orbmat=(orb_mat_a, orb_mat_b),
        verbose=verbose
    )
    return nelec


def slater_gto2mo(
        phi,
        nelec=None,
        slater_type=None,
        transform_matrix=None,
        **kwargs
    ):
    """
    convert Slater determinant from gto basis to molecular orbital basis
    
    inputs:
    phi (np.ndarray) : a numpy ndarray which represents the Slater determinant
                        in the underlying gto basis
    *optional* nelec (iterable of length 2) : (nalpha, nbeta) where nalpha and nbeta are the
                        number of up(alpha) and down(beta) electrons expressed as ints. If provided,
                        nelec is used to help distinguish between a Closed (i.e. RHF-like) determinant
                        and a Noncollinear (i.e. GHF-like) determinant
    *optional* slater_type ( a _SlaterType instance or an int ) : the type of Slater determinant
                       provided. If provided, this will override the automatic detection of
                       the Slater determinant type
    *optional* 'transform_mat' (np.ndarray) : specifies a custom transformation matrix to use
    **kwargs: (*all optional*) key-word arguments are ignored except for:
        - 'orthAO' : presence of keyword will force the use of an orthogonalized AO basis,
                        as opposed to a molecule orbital basis.
        - 'basis' (np.ndarray) : specifies an orbital basis to use - is ignored if orthAO is set (to anything!) 
        - 'overlap' (np.ndarray) : specifies the GTO-basis overlap matrix.
        - 'mol' (pyscf.gto.Mole) : a Mole object that describes the system (used to compute the overlap matrix
                                    if it was not provided)
    """

    if 'overlap' in kwargs:
        overlap = kwargs['overlap']
    elif 'mol' in kwargs:
        overlap = kwargs['cell'].intor_pbc('int1e_ovlp')
    else:
        raise ValueError(
            "Converting from GTO basis to molecular orbital basis\n"
            "requires either an overlap matrix OR a PySCF Mole object."
        )
    
    if transform_matrix is None:
        if 'basis' in kwargs and 'orthAO' not in kwargs:
            basis = kwargs['basis']
            transform_matrix = basis.conj().T @ overlap
        else:
            # use canonical orthgonalization
            raise NotImplementedError

    nmo = transform_matrix.shape[0]

    if slater_type is None:
        slater_type = _get_slater_type(phi,nelec)
        
    na,nb = nelec

    if _SlaterType(slater_type) is _SlaterType.CLOSED:
        phi_mo = np.zeros(
            shape=(nmo,sum(nelec)),
            dtype=np.complex128
        )

        phi_mo[:,:na] = transform_matrix @ phi[:,:na]
        return phi_mo
    
    elif _SlaterType(slater_type) is _SlaterType.COLLINEAR:
        if nb == 0:
            raise ValueError(
                "Collinear walker with 0 beta electrons is not supported: "
                "use 'fully_polarized' walkers instead"
            )

        # Needs to handle both UHF and ROHF format for phi!
        phi_mo = np.zeros(
            shape=(nmo,sum(nelec)),
            dtype=np.complex128
        )
        if phi.shape[0] == 2:
            # ROHF
            phi_a = phi_b = phi
        elif phi.shape[0] == 3:
            # UHF
            phi_a = phi[0]
            phi_b = phi[1]
        else:
            raise ValueError(
                "Input orbitals ('phi') has invalid dimensions:"
                "for Collinear walkers shape is either : "
                "(2,nmo,*) for UHF -or- (nmo,*) for ROHF."
                )

        phi_mo[:,:na] = transform_matrix @ phi_a[:,:na]
        phi_mo[:,na:] = transform_matrix @ phi_b[:,:nb]
        return phi_mo
    elif _SlaterType(slater_type) is _SlaterType.NONCOLLINEAR:
        phi_mo = np.zeros(
            shape=(2*nmo,sum(nelec)),
            dtype=np.complex128
        )
        phi[:nmo,:na+nb] = transform_matrix @ phi[:nmo,:na+nb]
        phi[nmo:,:na+nb] = transform_matrix @ phi[nmo:,:na+nb]
    elif _SlaterType(slater_type) is _SlaterType.FULLYPOLARIZED:
        phi_mo = np.zeros(
            shape=(nmo,na),
            dtype=np.complex128
        )
        # Also need to handle both UHF and ROHF format for *phi*
        phi_mo[:,:na] = transform_matrix @ phi[:,:na]
        return phi_mo
    else:
        raise ValueError("invalid Slater determinant type")


def generate_orbitals(fock, X, nmo_pk, rediag, ortho_ao,
                      mo_energy, uhf, verbose=False):
    eigs_a = []
    eigs_b = []
    ks = []
    bands = []
    full_mo_a = []
    full_mo_b = []
    nk = len(X)
    for k in range(nk):
        if verbose:
            print(" # Generating trial wavefunction for kpoint: "
                  "{:d}".format(k))
        if ortho_ao:
            start = time.time()
            ea, orb_a = rediag_fock(fock[0,k], X[k][:,:nmo_pk[k]])
            full_mo_a.append(orb_a)
            eigs_a += list(ea)
            if uhf:
                eb, orb_b = rediag_fock(fock[1,k], X[k][:,:nmo_pk[k]])
                full_mo_b.append(orb_b)
                eigs_b += list(eb)
            ks.append([k]*len(ea))
            bands.append([i for i in range(len(ea))])
            if verbose:
                print(" # Time to rediagonalise fock: "
                      "  {:13.8e} s".format(time.time()-start))
        else:
            if uhf:
                print("WARNING: UHF wavefunction with ortho_ao = False not "
                      "allowed.")
                sys.exit()
            ea = mo_energy[k]
            eigs_a.append(ea)
            orb_a = np.eye(len(ea), dtype=np.complex128)
            full_mo_a.append(orb_a)
            ks.append([k]*len(ea))
            bands.append([i for i in range(len(ea))])
            eigs_b = eigs_a
            foll_mo_b = full_mo_b
    return ((eigs_a,eigs_b), (full_mo_a,full_mo_b), ks, bands)

def create_wavefunction(orbs, occs, nmo_pk, nelec, uhf, verbose):
    nalpha, nbeta = nelec
    orb_a, orb_b = orbs
    occ_a, occ_b = occs
    nmo_tot = sum(nmo_pk)
    if verbose:
        print(" # Shape of supercell wavefunction "
              "({:d},{:d})".format(nmo_tot, (nalpha+nbeta) if uhf else nalpha))
        print(" # Number of electrons (nalpha, nbeta) = ({}, {})".format(nalpha, nbeta))
    row = 0
    col = 0
    col_b = nalpha
    if uhf:
        wfn = np.zeros((1,nmo_tot,nalpha+nbeta), dtype=np.complex128)
    else:
        wfn = np.zeros((1,nmo_tot,nalpha), dtype=np.complex128)
    for k in range(len(nmo_pk)):
        row_end = row + nmo_pk[k]
        nocca = int(round(sum(occ_a[k])))
        col_end = col + nocca
        wfn[0,row:row_end,col:col_end] = orb_a[k][:,:nocca].copy()
        if uhf:
            noccb = int(round(sum(occ_b[k])))
            col_end = col_b + noccb
            wfn[0,row:row_end,col_b:col_end] = orb_b[k][:,:noccb].copy()
            col_b += noccb
        row += nmo_pk[k]
        col += nocca
    return wfn

def print_eigenvalues(ks, bands, eigs, uhf, srt, nelec, verbose):
    # Can't use ravel because arrays can be of different length
    ks = np.array([val for item in ks for val in item])
    bands = np.array([val for item in bands for val in item])
    # TODO: Dangerous should subselect from already sorted array.
    print(" # Recomputed MO energies.")
    if uhf:
        eigs_a, eigs_b = eigs
        eigs_a = np.array(eigs_a).ravel()
        eigs_b = np.array(eigs_b).ravel()
        header = ("index", "kpoint", "band", "e_mo(alpha)", "e_mo(beta)")
        fmt = " # {:5s}  {:5s}    {:5s}  {:>17s}  {:>17s}"
        print(fmt.format(*header))
        zipped = zip(ks[srt[0]], bands[srt[0]],
                     eigs_a[srt[0]], eigs_b[srt[1]])
        fmt = " {:5d}   {:5d}     {: 13.8e}    {: 13.8e}"
        for i, t in enumerate(zipped):
            tag = ''
            if i == nelec[0]-1:
                tag = ' <--- HOMO(alpha)'
            elif i == nelec[1]-1:
                tag += ' <--- HOMO(beta) '
            print(" # {:5d}  ".format(i)+fmt.format(*t)+tag)
            if verbose < 2 and i >= max(nelec[0]-1,nelec[1]-1):
                break
    else:
        eigs_a, eigs_b = eigs
        eigs_a = np.array(eigs_a).ravel()
        header = ("index", "kpoint", "band", "e_mo")
        fmt = " # {:5s}  {:5s}    {:5s}  {:>17s}"
        print(fmt.format(*header))
        zipped = zip(ks[srt], bands[srt], eigs_a[srt])
        fmt = " {:5d}   {:5d}     {: 13.8e}"
        for i, t in enumerate(zipped):
            tag = ''
            if i == nelec[0]-1:
                tag = ' <--- HOMO(alpha)'
            print(" # {:5d}  ".format(i)+fmt.format(*t)+tag)
            if verbose < 2 and i >= nelec[0]-1:
                break

def rediag_fock(fock, X):
    """Rediagonalise Fock matrix.

    Parameters
    ----------
    fock : :class:`np.ndarray'
        Fock matrix for given kpoint.
    X : :class:`np.ndarray'
        Transformation matrix.

    Returns
    -------
    eigs : :class:`np.array'
        MO eigenvalues.
    eigv : :class:`np.ndarray'
        Eigenvectors.
    """
    fock_oao = np.dot(X.conj().T, np.dot(fock, X))
    e, c = np.linalg.eigh(fock_oao)
    return e, c

def reoccupy(mo_occ, mo_energy, uhf, verbose, low=0.25,
             high=0.95, ndet_max=1):
    if uhf:
        if verbose:
            print(" # Determining occupancies for alpha electrons.")
        re_occ_a, ndeg_a, msd_a, srt_a, isrt_a, p_a = (
                determine_occupancies(mo_occ[0],
                                      mo_energy[0],
                                      False,
                                      verbose=verbose>=1,
                                      low=low, high=high)
                )
        if verbose:
            print(" # Determining occupancies for beta electrons.")
        re_occ_b, ndeg_b, msd_b, srt_b, isrt_b, p_b = (
                determine_occupancies(mo_occ[1],
                                      mo_energy[1],
                                      False,
                                      verbose=verbose>=1,
                                      low=low, high=high)
                )
        ndeg = max(ndeg_a,ndeg_b)
        nalpha = int(round(np.sum(re_occ_a)))
        nbeta = int(round(np.sum(re_occ_b)))
        re_occ = [re_occ_a, re_occ_b]
        srt = (srt_a,srt_b)
        isrt = (isrt_a,isrt_b)
        if msd_a is not None and msd_b is not None:
            if verbose:
                print(" # Maximum number of determinants: "
                      " {}".format(len(msd_a)*len(msd_b)))
            if ndet_max == 1:
                trial = (1.0, msd_a[0], msd_b[0])
            else:
                occs_a, occs_b = zip(*itertools.product(msd_a,msd_b))
                pdets = np.outer(p_a,p_b).ravel()
                srt_det = pdets.argsort()
                msd_coeff = (pdets/sum(pdets))**0.5
                nd = min(len(occs_a),ndet_max)
                sd = srt_det[:nd]
                trial = (msd_coeff[sd],
                         np.array(occs_a)[sd],
                         np.array(occs_b)[sd])
        else:
            trial = None
    else:
        re_occ, ndeg, msd, srt, isrt, p = (
                determine_occupancies(mo_occ,
                                      mo_energy[0],
                                      True,
                                      verbose=verbose>=1,
                                      low=low, high=high)
                )
        trial = None
        re_occ = [re_occ/2.0, re_occ/2.0]
    # For metallic / system with partial occupancies.

    return re_occ, trial, ndeg, srt, isrt

def determine_occupancies(mo_occ, mo_energy, rhf, low=0.1,
                          high=0.95, verbose=False, refdet=0,
                          offset=0):
    nocc = 0
    nelec = sum(sum(occ) for occ in mo_occ)
    col_sort = np.ravel(mo_energy).argsort()
    col_sort_inv = col_sort.argsort()
    nmo_pk = []
    for occ in mo_occ:
        occ = occ > high
        nmo_pk.append(len(occ))
        nocc += sum(occ)
    if rhf:
        nocc = 2*nocc
    nleft = int(round(nelec-nocc))
    if nleft == 0:
        if verbose:
            print(" # All occupancies are one or zero.")
        ndeg = 1
        msd = None
        pcomb = None
        return (mo_occ, ndeg, msd, col_sort, col_sort_inv, pcomb)
    else:
        if verbose:
            print(" # Found partially occupied bands.")
            print(" # Constructing multi determinant trial wavefunction from "
                  "degenerate orbitals.")
        mo_order = np.array(mo_occ).ravel()[col_sort]
        deg = (mo_order < high) & (mo_order > low)
        poccs = mo_order[deg]
        ndeg = sum(deg)
        if ndeg == 0:
            print(" # Warning: trying to occupy {} electrons in {} orbitals.".format(nleft, ndeg))
            low = 0.5*mo_order[(mo_order<low)&(mo_order>1e-10)][0]
            print(" # Decreasing low parameter to {:13.8e}".format(low))
            deg = (mo_order < high) & (mo_order > low)
            poccs = mo_order[deg]
            ndeg = sum(deg)
            if ndeg == 0:
                print(" # Error: trying to occupy {} electrons in {} orbitals.".format(nleft, ndeg))
                print(" # MO occupancies > 0: ")
                for i, o in enumerate(mo_order[(mo_order<low)&(mo_order>1e-10)]):
                    print(" # {:4d} {:13.8e}".format(i, o))
                sys.exit()
        # Supercell indexed.
        deg_orb = np.where(deg)[0]
        combs = [c for c in itertools.combinations(deg_orb, int(nleft))]
        pcomb = np.array([np.prod(poccs[np.array(c)-nocc]) for c in combs])
        if verbose:
            print(" # Distributing {} electrons in {} orbitals.".format(nleft,ndeg))
        core = list(np.where(mo_order > high)[0])
        core = [c for c in core]
        msd = [core + list(d) for d in combs]
        # Remap to primitive cell (kpt,band) indexing.
        reordered = []
        for i, d in enumerate(msd):
            mo_occ_new = np.zeros(len(mo_order), dtype=np.int32)
            mo_occ_new[d] = 1
            if i == refdet:
                ref = mo_occ_new[col_sort_inv]
            reordered.append(np.where(mo_occ_new[col_sort_inv])[0])
        locc = []
        s = 0
        e = nmo_pk[0]
        for ik in range(len(mo_occ)):
            locc.append(ref[s:e])
            s += nmo_pk[ik]
            try:
                e += nmo_pk[ik+1]
            except IndexError:
                e = -1
        return (locc, ndeg, reordered, col_sort, col_sort_inv, pcomb)
