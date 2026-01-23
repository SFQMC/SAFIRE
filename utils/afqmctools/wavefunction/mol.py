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

    nmo = transform_matrix.shape[0]

    if slater_type is None:
        slater_type = _get_slater_type(phi,nelec)
        
    na,nb = nelec

    if _SlaterType(slater_type) is _SlaterType.CLOSED:
        phi_mo = np.zeros(
            shape=(nmo,na),
            dtype=np.complex128
        )

        phi_mo = transform_matrix @ phi
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
        phi_mo[:,:na] = transform_matrix @ phi_a[:,:na]
        phi_mo[:,na:] = transform_matrix @ phi_b[:,:nb]
        return phi_mo
    elif _SlaterType(slater_type) is _SlaterType.NONCOLLINEAR:
        phi_mo = np.zeros(
            shape=(2*nmo,sum(nelec)),
            dtype=np.complex128
        )
        phi_mo[:nmo,:na+nb] = transform_matrix @ phi[:nmo,:na+nb]
        phi_mo[nmo:,:na+nb] = transform_matrix @ phi[nmo:,:na+nb]
        return phi_mo
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


def freeze_core():
    """
    Frozen core transformation for wavefunctions
    """
    raise NotImplementedError


def make_slater(wfn_scf_data,basis_scf_data=None):
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
        'mol' : wfn_scf_data['mol']
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
                  init=None, verbose=False):
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

    Returns
    -------
    wfn : :class:`np.ndarray`
        Wavefunction as numpy array. Format depends on wavefunction.
    """
    mol = scf_data['mol']
    nelec = mol.nelec
 
    norb = scf_data['norb']
    
    # Check if frozen core Hamiltonian exists and read active space info
    import os
    if os.path.exists(filename):
        try:
            with h5py.File(filename, 'r') as f:
                if 'Hamiltonian/dims' in f:
                    ham_dims = f['Hamiltonian/dims'][:]
                    # dims structure: [0, 0, 0, nmo, nelec[0], nelec[1], 0, nchol]
                    ham_nmo = ham_dims[3]
                    ham_nelec = (int(ham_dims[4]), int(ham_dims[5]))
                    
                    # If Hamiltonian has different dimensions (frozen core), use those
                    if ham_nmo != norb or ham_nelec != nelec:
                        if verbose:
                            print(f" # Using frozen core dimensions from Hamiltonian:")
                            print(f" #   norb: {norb} -> {ham_nmo}")
                            print(f" #   nelec: {nelec} -> {ham_nelec}")
                        norb = ham_nmo
                        nelec = ham_nelec
                        
                        # Read active space orbitals from Hamiltonian
                        if 'Hamiltonian/X' in f:
                            X = f['Hamiltonian/X'][:]
                            if verbose:
                                print(f" #   Using active space orbitals from Hamiltonian: {X.shape}")
                            
                            # Update scf_data to use active space info
                            scf_data = scf_data.copy()
                            scf_data['norb'] = norb
                            scf_data['nelec'] = nelec
                            scf_data['mo_coeff'] = X
                            # Update mo_occ to match active space (all active orbitals are doubly occupied)
                            scf_data['mo_occ'] = np.array([2.0] * norb)
                            
                            # Also update basis_scf_data
                            if basis_scf_data is not None:
                                basis_scf_data = basis_scf_data.copy()
                                basis_scf_data['mo_coeff'] = X
                                basis_scf_data['norb'] = norb
                                basis_scf_data['nelec'] = nelec
                                basis_scf_data['mo_occ'] = np.array([2.0] * norb)
        except Exception as e:
            if verbose:
                print(f" # Warning: Could not read Hamiltonian dimensions: {e}")

    # ensure valid walkers up-front
    walker_type = _slater_enum_map(
        scf_data['walker_type']
    )

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
                if mo_coeff.shape == (2,norb,norb) and not np.allclose(mo_coeff[0],mo_coeff[1]):
                    raise ValueError(
                        "Currently, only RHF/ROHF orbital basis sets "
                        "are supported. Alternatively, orthogonalized atomic orbital basis sets are supported"
                    )
                elif not mo_coeff.shape == (norb,norb):
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
            basis_scf_data=basis_scf_data
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


def write_wfn(
    filename,
    wfn,
    walker_type,
    nelec,
    norb,
    init=None,
    orbmat=None,
    verbose=False
    ):
    """
    Write a wavefunction to an HDF5 file. Will overwrite existing wavefunction data.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file where the wavefunction will be written.
    wfn : tuple
        Wavefunction data. For PHMSD, it is a tuple of (coeffs, occa, occb).
        For NOMSD, it is a tuple of (coeffs, wfn).
    walker_type : str
        Type of walker to use. Must be one of the types defined in _slater_enum_map.
    nelec : tuple
        Number of electrons as a tuple (nalpha, nbeta).
    norb : int
        Number of orbitals.
    init : optional
        Initial wavefunction to use in AFQMC. Default is None.
    orbmat : optional
        Orbital matrix. Default is None.
    verbose : bool, optional
        If True, print additional information. Default is False.
    
    Raises
    ------
    ValueError
        If an unknown wavefunction type is passed.
    
    Notes
    -----
    The function supports two types of wavefunctions: PHMSD and NOMSD.
        It writes the wavefunction data to the specified HDF5 file and handles
        different walker types, including corrections for user input.    
    """
    walker_type = _slater_enum_map(walker_type)

    if len(wfn) == 3:
        coeffs, occa, occb = wfn
        wfn_type = 'PHMSD'
    elif len(wfn) == 2:
        coeffs, wfn = wfn
        wfn_type = 'NOMSD'
    else:
        raise ValueError("Unknown wavefunction type passed.")

    with h5py.File(filename, 'a') as fh5:
        nalpha, nbeta = nelec

        walker_type = _slater_enum_map(walker_type)

        # Apply some user input corrections
        if walker_type == _SlaterType.COLLINEAR and nelec[1] == 0:
            print(
                "collinear/uhf/rohf walker type requested with no beta electrons: "
                "using fully spin-polarized walkers"
            )
            walker_type = _SlaterType.FULLYPOLARIZED
        
        if "Wavefunction" in fh5:
            del fh5['Wavefunction']

        if wfn_type == 'PHMSD':
            print("PHMSD trial wavfunction -> using uhf-like walkers")
            walker_type = _SlaterType.COLLINEAR

            wfn_group = add_group(fh5, 'Wavefunction/PHMSD')
            write_phmsd(wfn_group, occa, occb, nelec, norb,
                        init=init, orbmat=orbmat)
        
        elif wfn_type == 'NOMSD':
            wfn_group = add_group(fh5, 'Wavefunction/NOMSD')
            
            if (
                walker_type == _SlaterType.CLOSED or 
                walker_type == _SlaterType.COLLINEAR
            ):
                write_nomsd(
                    fh5=wfn_group, 
                    wfn=wfn,
                    uhf=walker_type==_SlaterType.COLLINEAR,
                    nelec=nelec,
                    init=init
                )
            elif (
                walker_type == _SlaterType.NONCOLLINEAR or 
                walker_type == _SlaterType.FULLYPOLARIZED
            ):
                nelec = (sum(nelec),0)
                write_nomsd_ghf(
                    fh5=wfn_group,
                    wfn=wfn,
                    nelec=nelec,
                    init=init
                )
        
        if coeffs.dtype == float:
            if verbose:
                print(" # Found real MSD coefficients. Converting to complex.")
            coeffs = np.array(coeffs, dtype=np.complex128)
        wfn_group['ci_coeffs'] = to_complex(coeffs)
        
        if walker_type == _SlaterType.NONCOLLINEAR:
            dims = [norb, nalpha+nbeta, 0, _slater2dims(walker_type), len(coeffs)]
        else:
            dims = [norb, nalpha, nbeta, _slater2dims(walker_type), len(coeffs)]

        wfn_group['dims'] = np.array(dims, dtype=np.int32)


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

