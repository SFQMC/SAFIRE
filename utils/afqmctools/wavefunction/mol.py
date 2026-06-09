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

try: # PySCF is now considered an optional dependency!
    from pyscf import fci
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


def slater_gto2mo(
        phi,
        nelec=None,
        slater_type=None,
        transform_matrix=None,
        **kwargs
    ):
    """
    Convert Slater determinant from GTO basis to molecular orbital basis.

    Parameters
    ----------
    phi : np.ndarray
        A numpy ndarray which represents the Slater determinant in the
        underlying GTO basis.
    nelec : iterable of length 2, optional
        (nalpha, nbeta) where nalpha and nbeta are the number of up (alpha)
        and down (beta) electrons expressed as ints. If provided, `nelec` is
        used to help distinguish between a Closed (i.e. RHF-like) determinant
        and a Noncollinear (i.e. GHF-like) determinant.
    slater_type : _SlaterType or int, optional
        The type of Slater determinant provided. If provided, this will
        override the automatic detection of the Slater determinant type.
    transform_matrix : np.ndarray, optional
        Specifies a custom transformation matrix to use.
    **kwargs
        (*all optional*) keyword arguments are ignored except for:

        orthAO : bool
            Presence of keyword will force the use of an orthogonalized AO
            basis, as opposed to a molecule orbital basis.
        basis : np.ndarray
            Specifies an orbital basis to use - is ignored if orthAO is set
            (to anything!).
        overlap : np.ndarray
            Specifies the GTO-basis overlap matrix.
        mol : pyscf.gto.Mole
            A Mole object that describes the system (used to compute the
            overlap matrix if it was not provided).
        cas : tuple
            If provided, indicates that the wavefunction is to be expressed in
            a CAS active space as (# active electrons, # active orbitals).

    Returns
    -------
    phi_mo : np.ndarray
        The Slater determinant expressed in the molecular orbital basis.
    """
    # TODO: we are not correctly handling the spin sectors here (at least, we aren't
    # covering all cases that we can! For example, the Noncollinear implementation
    # assumes a spin-restricted input.)

    if transform_matrix is None:

        if 'overlap' in kwargs:
            overlap = kwargs['overlap']
        elif 'mol' in kwargs:
            overlap = kwargs['mol'].intor('int1e_ovlp')
        else:
            raise ValueError(
                "Converting from GTO basis to molecular orbital basis\n"
                "requires either an overlap matrix OR a PySCF Mole object."
            )

        if 'basis' in kwargs and 'orthAO' not in kwargs:
            basis = kwargs['basis'] # This implicitly assumes a correct basis!
            if _SlaterType(slater_type) == _SlaterType.COLLINEAR and basis.shape[0] == 2:
                print("Basis contains UHF orbitals. Using the majority spin sector for orbital basis.")
                warn("Using the majority spin sector for orbital basis can led to poor representation "
                     "of the of the minority spin-sector if the basis is truncated in any way.")
                basis = basis[0]

            transform_matrix = basis.conj().T @ overlap
        elif kwargs.get('orthAO',False):
            transform_matrix = kwargs['X'].T @ overlap
        else:
            raise ValueError("Can't construct a valid transformation matrix")

    # ngto is the number of underyling basis functions, regardless of gto or other
    ngto = nmo = transform_matrix.shape[0]

    if slater_type is None:
        slater_type = _get_slater_type(phi,nelec)
        
    na,nb = nelec
    if kwargs.get('cas',None) is not None:
        nactive_electrons = kwargs['cas'][0]
        nactive_orbitals = kwargs['cas'][1]
        nfzc = (sum(nelec) - nactive_electrons) // 2
        nfzv = transform_matrix.shape[0] - (nfzc + (nactive_orbitals if nactive_orbitals != -1 else
                  transform_matrix.shape[0] - nfzc))
        if nactive_orbitals == -1:
            nactive_orbitals = nmo - nfzc

        # trim the transformation matrix into the active space only (in terms of full basis)
        transform_matrix = transform_matrix[nfzc:transform_matrix.shape[0]-nfzv,:]
        if transform_matrix.shape != (nactive_orbitals,ngto):
            raise ValueError("Invalid CAS specification within slater_gto2mo(...)")

        # update nmo and nelec to reflect active space only
        nmo = nactive_orbitals if nactive_orbitals != -1 else transform_matrix.shape[0]
        na -= nfzc
        nb -= nfzc
        nelec = (na,nb)
        if nb == 0 and slater_type == _SlaterType.COLLINEAR:
            slater_type = _SlaterType.FULLYPOLARIZED
            phi = phi[0]

    else:
        nfzc = 0
        nfzv = 0


    if _SlaterType(slater_type) is _SlaterType.CLOSED:
        phi_mo = np.zeros(
            shape=(nmo,na),
            dtype=np.complex128
        )

        phi_mo = transform_matrix @ phi[:,nfzc:nfzc+na]
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
        if len(phi.shape) == 2:
            # ROHF
            phi_a = phi_b = phi
        elif len(phi.shape) == 3:
            # UHF
            phi_a = phi[0]
            phi_b = phi[1]
        else:
            raise ValueError(
                "Input orbitals ('phi') has invalid dimensions:"
                "for Collinear walkers shape is either : "
                "(2,nmo,*) for UHF -or- (nmo,*) for ROHF."
                )
        phi_mo[:,:na] = transform_matrix @ phi_a[:,nfzc:nfzc+na]
        phi_mo[:,na:] = transform_matrix @ phi_b[:,nfzc:nfzc+nb]
        return phi_mo
    elif _SlaterType(slater_type) is _SlaterType.NONCOLLINEAR:
        phi_mo = np.zeros(
            shape=(2*nmo,sum(nelec)),
            dtype=np.complex128
        )
        phi_mo[:nmo,:na+nb] = transform_matrix @ phi[:ngto,2*nfzc:2*nfzc+na+nb]
        phi_mo[nmo:,:na+nb] = transform_matrix @ phi[ngto:,2*nfzc:2*nfzc+na+nb]
        return phi_mo
    elif _SlaterType(slater_type) is _SlaterType.FULLYPOLARIZED:
        phi_mo = np.zeros(
            shape=(nmo,na),
            dtype=np.complex128
        )
        # Also need to handle both UHF and ROHF format for *phi*
        phi_mo[:,:na] = transform_matrix @ phi[:,nfzc:nfzc+na]
        return phi_mo
    else:
        raise ValueError("invalid Slater determinant type")


def make_slater(wfn_scf_data,basis_scf_data=None,cas=None):
    """
    make a single Slater determinant based on the contents of the
       the input 'scf_data' dictionary.

    Inputs:
    - wfn_scf_data:dict - defines the source of the orbitals from
                            which to build the Slater determinant
    - (optional) basis_scf_data:dict - defines the orbtial basis. If given,
                            the Slater determinant will be expressed within
                            this basis - in this case, the 'orthAO' keyword 
                            within `wfn_scf_data` will be ignored!

    Returns:
    - phi:numpy.ndarray with shape (nmo,nelec) - a single Slater determinant expressed 
        as a Slater matrix. The orbitals are taken from 'wfn_scf_data' and are expressed
        either in the basis of orbitals defined in `basis_scf_data` (if given), an orthogonalized
        AO basis (if wfn_scf_data['orthAO'] == True), or within the basis of orbitals defined in 
        `wfn_scf_data`.
    """
    walker_type = _slater_enum_map(
        wfn_scf_data.get('walker_type',_SlaterType.CLOSED)
    )

    phi_gto = _make_slater_gto(
        scf_data=wfn_scf_data,
        walker_type=walker_type
    )

    _kwargs = {
        'mol' : wfn_scf_data['mol'],
        'cas' : cas
    }

    if basis_scf_data is not None:
        _kwargs['basis'] = basis_scf_data['mo_coeff']
    elif wfn_scf_data.get('orthAO',False):
        _kwargs['orthAO'] = True
        _kwargs['X'] = wfn_scf_data['X']
    else:
        _kwargs['basis'] = wfn_scf_data['mo_coeff']

    return slater_gto2mo(
        phi=phi_gto,
        nelec=wfn_scf_data['nelec'],
        slater_type=walker_type,
        **_kwargs
    )


def _make_slater_gto(scf_data,walker_type=None,verbose=False):
    """
    Make a Slater determinant based on scf_data *in the GTO basis*

    Inputs:
    - scf_data:dict - contains data from a PySCF calculation
    - walker_type:_SlaterType - the _SlaterType corresponding to the type of wavefunction
    """
    mo_coeff = scf_data['mo_coeff']
    mo_occ = scf_data['mo_occ']
    nelec = scf_data['nelec']

    na,nb = nelec
    nmo = mo_coeff.shape[-1]

    _mo_occ_len = len(mo_occ.shape)
    _mo_coeff_len = len(mo_coeff.shape)

    if walker_type is None:
        walker_type = scf_data['walker_type']
    else:
        walker_type = _slater_enum_map(walker_type)

    if walker_type == _SlaterType.CLOSED:
        if nelec[0] != nelec[1]:
            raise ValueError("Invalid nelec for CLOSED Slater determinant")
        occ = np.array([ a for a,o in enumerate(mo_occ) if o >= 1 ])
        
        if verbose:
            print("occupying orbitals: ", *occ)

        return mo_coeff[:,occ]
    
    elif walker_type == _SlaterType.COLLINEAR:
        phi = np.zeros(
            shape=(2,nmo,max(nelec)),
            dtype=np.complex128
        )
        if _mo_occ_len == 1: # ROHF:
            occa = np.array([ a for a,o in enumerate(mo_occ) if o > 0 ])
            occb = np.array([ b for b,o in enumerate(mo_occ) if o > 1 ])
        elif _mo_occ_len == 2: #UHF
            occa = np.array([ a for a,o in enumerate(mo_occ[0]) if o > 0 ])
            occb = np.array([ b for b,o in enumerate(mo_occ[1]) if o > 0 ])
        else:
            raise ValueError("within _make_slater(...) invalid mo_occ shape")

        if verbose:
            print("occupying alpha orbitals: ", *occa)
            print("occupying beta orbitals: ", *occb)

        if _mo_coeff_len == 2: #ROHF
            phi[0,:,:na] = mo_coeff[:,occa]
            phi[1,:,:nb] = mo_coeff[:,occb]
        elif _mo_coeff_len == 3: #UHF
            phi[0,:,:na] = mo_coeff[0][:,occa]
            phi[1,:,:nb] = mo_coeff[1][:,occb]
        else:
            raise ValueError("within _make_slater(...) invalid mo_coeff shape")
        
        return phi
    
    elif walker_type == _SlaterType.NONCOLLINEAR:
        occ = np.array([ a for a,o in enumerate(mo_occ) if o >= 1 ])
        if verbose:
            print("occupying orbitals: ", *occ)
        return mo_coeff[:,occ]
    
    elif walker_type == _SlaterType.FULLYPOLARIZED:
        if nb != 0:
            raise ValueError("Attempted to build fully polarized Slater determinant with non-zero N_beta")
        
        if _mo_occ_len == 1: # ROHF:
            occa = np.array([ a for a,o in enumerate(mo_occ) if o > 0 ])
        elif _mo_occ_len == 2: #UHF
            occa = np.array([ a for a,o in enumerate(mo_occ[0]) if o > 0 ])
        else:
            raise ValueError("within _make_slater(...) invalid mo_occ shape")

        if verbose:
            print("occupying alpha orbitals: ", *occa)

        if _mo_coeff_len == 2: #ROHF
            return mo_coeff[:,occa]
        elif _mo_coeff_len == 3: #UHF
            return mo_coeff[0][:,occa]
        else:
            raise ValueError("within _make_slater(...) invalid mo_coeff shape")
    
    else:
        raise ValueError("Invalid Slater determinant type")


def write_wfn_mol(scf_data, filename, basis_scf_data=None, wfn=None,
                  init=None, verbose=False, cas=None):
    """Generate SAFIRE format trial wavefunction.

    Parameters
    ----------
    scf_data : dict
        Dictionary containing scf data extracted from pyscf checkpoint file.
    ortho_ao : bool
        Whether we are working in orthogonalised AO basis or not.
    filename : string
        HDF5 file path to store wavefunction to.
    wfn : tuple
        User defined wavefunction. Not fully supported. Default None.
    init : optional
        Initial wavefunction to use in AFQMC. Default is None.
    basis_scf_data : dict, optional
        Dictionary containing scf data for the basis set in which to express
        the wavefunction. If None, the wavefunction will be expressed in the
        basis defined by `scf_data`. Default is None.
    verbose : bool, optional
        If True, print additional information. Default is False.
    cas : tuple, optional
        If provided, indicates that the wavefunction is to be expressed
        in a CAS active space as (# active electrons, # active orbitals).
        Default is None.

    Returns
    -------
    wfn : :class:`np.ndarray`
        Wavefunction as numpy array. Format depends on wavefunction.
    """

    nelec = scf_data['nelec']        
    # the basis size
    ngto = norb = scf_data['norb']
    # ensure valid walkers up-front
    walker_type = _slater_enum_map(
        scf_data['walker_type']
    )
    if cas is not None:
        nfzc = (sum(nelec) - cas[0]) // 2
        nfzv = norb - nfzc - (cas[1] if cas[1] != -1 else
                  norb - nfzc)
        if nfzc < 0 or nfzv < 0:
            raise ValueError(
                f"Invalid CAS specification cas={cas} for system with "
                f"nelec={nelec} and norb={norb}"
            )
        if verbose:
            print(f"Freezing {nfzc} core orbitals and {nfzv} virtual orbitals.")
        norb -= (nfzc + nfzv)
        nelec = (nelec[0]-nfzc, nelec[1]-nfzc)
    
    # Catch spin-contaminated initial wavefunctions
    if walker_type == _SlaterType.COLLINEAR and init is None:
        print("Walker type is UHF/collinear; using RHF/ROHF-like initial wavefunction "
              "for a pure spin state. See J. Chem. Phys. 128, 114309 (2008) DOI: /10.1063/1.2838983")
        init = [
            np.zeros((norb,nelec[0]),dtype=np.complex128),
            np.zeros((norb,nelec[1]),dtype=np.complex128)
        ]
        init[0][:nelec[0]] = np.eye(nelec[0])
        init[1][:nelec[1]] = np.eye(nelec[1])
    elif walker_type == _SlaterType.NONCOLLINEAR and init is None:
        print("Walker type is GHF/noncollinear; using RHF-like initial wavefunction "
              "for a pure spin state. See J. Chem. Phys. 128, 114309 (2008) DOI: /10.1063/1.2838983")
        init = np.zeros((1,2*norb,nelec[0]+nelec[1]),dtype=np.complex128)
        init[0,:norb,:nelec[0]] = np.eye(norb, nelec[0])
        init[0,norb:,nelec[0]:] = np.eye(norb, nelec[1])
    elif walker_type == _SlaterType.FULLYPOLARIZED and init is None:
        print("Walker type is fully polarized; using RHF-like initial wavefunction "
              "for a pure spin state. See J. Chem. Phys. 128, 114309 (2008) DOI: /10.1063/1.2838983")
        init = np.zeros((1,norb,nelec[0]),dtype=np.complex128)
        init[0] = np.eye(norb,nelec[0])

    if wfn is None:

        if scf_data.get('orthAO',False):
            wfn = make_slater(
                wfn_scf_data=scf_data,
                basis_scf_data=None
            )
        elif basis_scf_data is not None:
            basis_type = _slater_enum_map(
                basis_scf_data['walker_type']
            )
            if basis_type == _SlaterType.COLLINEAR:
                mo_coeff = basis_scf_data['mo_coeff']
                # check that it's ROHF-like
                if mo_coeff.shape == (2,ngto,ngto) and not np.allclose(mo_coeff[0],mo_coeff[1]):
                    raise ValueError(
                        "Currently, only RHF/ROHF orbital basis sets "
                        "are supported. Alternatively, orthogonalized atomic orbital basis sets are supported"
                    )
                elif not mo_coeff.shape == (ngto,ngto):
                    raise ValueError(
                        f"mo_coeff shape {mo_coeff.shape} is not valid for collinear basis set. Check scf_data dictionary."
                    )
            elif basis_type is not _SlaterType.CLOSED:
                raise ValueError(
                    "Currently, only RHF/ROHF orbital basis sets "
                    "are supported. Alternatively, orthogonalized atomic orbital basis sets are supported"
                )
        wfn = make_slater(
            wfn_scf_data=scf_data,
            basis_scf_data=basis_scf_data,
            cas=cas
        )

    # Wavefunctions are assumed to be a 1-D iterable of wavefunctions, even for a single determinant!
    write_wfn(
        filename,
        (np.array([1.0+0j]),np.array([wfn],dtype=wfn.dtype)),
        walker_type,
        nelec,
        norb,
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

