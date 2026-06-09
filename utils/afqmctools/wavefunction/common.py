# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import logging
logger = logging.getLogger(__name__)
from warnings import warn

import numpy as np
import h5py
import scipy

from afqmctools.utils.io import (
    to_complex, 
    add_group, 
    add_dataset
)

logger.dev(
    "[Warning] (for Devs) afqmctools.utils.slater_types is indirectly imported "
    "from afqmctools.wavefunction.common; Refactor to direct imports and remove " 
    "the import here."
)
from afqmctools.utils.slater_types import (
    _SlaterType,
    _slater2dims,
    _slater_enum_map,
    _get_slater_type
)


def modified_gram_schmidt(mat, tol=1.0e-12):
  """Orthonormalize the columns of ``mat`` using modified Gram-Schmidt.

  Parameters
  ----------
  mat : ndarray (n, m)
    Matrix whose columns will be orthonormalized in place order.
  tol : float, optional
    Minimum norm allowed before considering vectors linearly dependent.

  Returns
  -------
  ndarray
    Matrix with orthonormal columns spanning the same column space.

  Raises
  ------
  ValueError
    If a column is (near) linearly dependent on the previous ones.
  """
  print("Orthonormalizing Slater matrix using modified Gram-Schmidt...")
  q = np.zeros_like(mat, dtype=np.complex128)

  for j in range(mat.shape[1]):
    v = np.array(mat[:, j], dtype=np.complex128, copy=True)
    for i in range(j):
      proj = np.vdot(q[:, i], v)
      v -= proj * q[:, i]

    norm = np.linalg.norm(v)
    if norm < tol:
      raise ValueError("Linearly dependent vectors encountered during Gram-Schmidt orthogonalization")

    q[:, j] = v / norm

  return q


def check_orthonormality(wfn, nelec, wfn_type='collinear'):
  """Check orthonormality of wavefunction columns.

  Parameters
  ----------
  wfn : ndarray (1, n, m)
    Wavefunction array with shape (1, n_orbitals, n_electrons).
  nelec : tuple or int
    Number of electrons. For collinear: (n_up, n_down), for noncollinear/closed: int.
  wfn_type : str, optional
    Type of wavefunction: 'noncollinear', 'collinear', or 'closed'. Default is 'collinear'.

  Returns
  -------
  norm : float
    Dictionary containing normalization determinants and messages.
  """
  if wfn_type == 'noncollinear' or wfn_type == 'closed':
    # For noncollinear, check all columns together
    norm = np.linalg.det(wfn[0,:,:].conj().T @ wfn[0,:,:])
    print(f"Wavefunction normalization ({wfn_type}): {norm}")
  elif wfn_type == 'collinear':
    # For collinear, check spin-up and spin-down separately
    norm_up = np.linalg.det(wfn[0,:,:nelec[0]].conj().T @ wfn[0,:,:nelec[0]])
    norm_down = np.linalg.det(wfn[0,:,nelec[0]:].conj().T @ wfn[0,:,nelec[0]:])
    norm = norm_up * norm_down
    print(f"Wavefunction normalization (collinear):")
    print(f"  Norm up: {norm_up}, Norm down: {norm_down}")
    print(f"  Total norm: {norm}")
  else:
    raise ValueError(f"Unknown wavefunction type: {wfn_type}")
  return norm


def check_slater_matrix_orthonormality(slater_matrix, nelec=None, matrix_type='collinear', verbose=True, tol=1.0e-10):
  """Check orthonormality of a Slater matrix (n_orbitals, n_electrons).
  
  Parameters
  ----------
  slater_matrix : ndarray (n, m)
    Slater matrix with shape (n_orbitals, n_electrons).
  nelec : int, optional
    Number of electrons in this spin block. If None, uses shape[1].
  matrix_type : str, optional
    Type of matrix: 'spin_block' (default) or 'full'. Only affects messaging.
  verbose : bool, optional
    Print warnings if not orthonormal. Default: True.
  tol : float, optional
    Tolerance for checking orthonormality. Default: 1.0e-10.
    
  Returns
  -------
  norm : complex
    Determinant of the overlap matrix (product for all columns).
  is_orthonormal : bool
    True if the overlap matrix is close to identity (within tolerance).
  """
  if nelec is None:
    nelec = slater_matrix.shape[1]
    
  # Check orthonormality: M^H M should be identity
  overlap = slater_matrix.conj().T @ slater_matrix
  norm = np.linalg.det(overlap)
  
  # Check if overlap matrix is close to identity
  identity = np.eye(slater_matrix.shape[1], dtype=overlap.dtype)
  max_deviation = np.max(np.abs(overlap - identity))
  is_orthonormal = max_deviation < tol
  
  if not is_orthonormal and verbose:
    warn(f"Slater matrix ({matrix_type}) is not orthonormal: max deviation from identity = {max_deviation:.2e}, det(overlap) = {norm}. "
         f"Consider orthonormalizing using modified Gram-Schmidt.")
  
  return norm, is_orthonormal


def write_wfn(
    filename,
    wfn,
    walker_type,
    nelec,
    norb,
    init=None,
    orbmat=None,
    verbose=False,
    orthonormalize=True
    ):
    """
    Write wavefunction to HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file to write wavefunction to.
    wfn : tuple
        Wavefunction description. Either a list containing an array coefficients and occupation numbers (PHMSD)
        or a tuple containing an array of coefficients and an array of determinants (NOMSD).
    walker_type : str
        Type of walker to write. Options are 'closed', 'collinear', 'noncollinear', 'fullypolarized'.
    nelec : tuple
        Number of alpha and beta electrons.
    norb : int
        Number of orbitals.
    init : tuple
        Initial Slater determinants for AFQMC.
    orbmat : tuple
        Orbital matrices for PHMSD wavefunctions.
    verbose : bool
        Print additional information.
    orthonormalize : bool, optional
        If True (default), orthonormalize Slater matrices if not properly normalized.

    Raises
    ------
    ValueError
        If an unknown wavefunction type is passed.
    
    Notes
    -----
    The function supports two types of wavefunctions: PHMSD and NOMSD.
        It writes the wavefunction data to the specified HDF5 file and handles
        different walker types, including corrections for user input.    
    For user-defined wavefunctions:
        # PHMSD is a list of tuple of (ci, occa, occb).
        # NOMSD is a tuple of (list, np.ndarray).
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
            if nelec[1] > 0:
                walker_type = _SlaterType.COLLINEAR
            else:
                walker_type = _SlaterType.FULLYPOLARIZED

            wfn_group = add_group(fh5, 'Wavefunction/PHMSD')
            write_phmsd(wfn_group, occa, occb, nelec, norb,
                        init=init, orbmat=orbmat, orthonormalize=orthonormalize)
        
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
                    init=init,
                    orthonormalize=orthonormalize
                )
            elif (
                walker_type == _SlaterType.NONCOLLINEAR or 
                walker_type == _SlaterType.FULLYPOLARIZED
            ):
                if walker_type == _SlaterType.NONCOLLINEAR:
                    nelec = (nalpha+nbeta,0)
                write_nomsd_ghf(
                    fh5=wfn_group,
                    wfn=wfn,
                    nelec=nelec,
                    init=init,
                    orthonormalize=orthonormalize
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


def write_nomsd_old(fh5, wfn, uhf, nelec, thresh=1e-8, init=None, orthonormalize=True):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`np.ndarray`
        NOMSD trial wavefunctions.
    uhf : bool
        UHF style wavefunction.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    init : tuple, optional
        Initial Slater determinants [alpha_block, beta_block]. Will be orthonormality-checked.
    orthonormalize : bool, optional
        If True (default), orthonormalize Slater matrices if not properly normalized.
    """
    nalpha, nbeta = nelec

    wfn[abs(wfn) < thresh] = 0.0
    
    # Check and optionally orthonormalize wfn Slater matrices
    for idet, w in enumerate(wfn):
        # Check alpha block
        norm_a, is_ortho_a = check_slater_matrix_orthonormality(
            w[:,:nalpha], nelec=nalpha, matrix_type=f'wfn[{idet}] alpha'
        )
        if not is_ortho_a and orthonormalize:
            wfn[idet, :, :nalpha] = modified_gram_schmidt(w[:,:nalpha])
        
        # Check beta block if present
        if uhf and nbeta > 0:
            norm_b, is_ortho_b = check_slater_matrix_orthonormality(
                w[:,nalpha:nalpha+nbeta], nelec=nbeta, matrix_type=f'wfn[{idet}] beta'
            )
            if not is_ortho_b and orthonormalize:
                wfn[idet, :, nalpha:nalpha+nbeta] = modified_gram_schmidt(w[:,nalpha:nalpha+nbeta])
    
    if init is not None:
        # Check and optionally orthonormalize init parameter
        if isinstance(init, (tuple, list)) and len(init) == 2:
            init_alpha, init_beta = init
            norm_a, is_ortho_a = check_slater_matrix_orthonormality(
                init_alpha, nelec=nalpha, matrix_type='init alpha'
            )
            if not is_ortho_a and orthonormalize:
                init[0] = modified_gram_schmidt(init_alpha)
            
            if uhf and nbeta > 0:
                norm_b, is_ortho_b = check_slater_matrix_orthonormality(
                    init_beta, nelec=nbeta, matrix_type='init beta'
                )
                if not is_ortho_b and orthonormalize:
                    init[1] = modified_gram_schmidt(init_beta)
        
        add_dataset(fh5, 'Psi0_alpha', to_complex(init[0]))
        add_dataset(fh5, 'Psi0_beta', to_complex(init[1]))
    else:
        # Check and optionally orthonormalize wfn[0] for use as init
        norm_a, is_ortho_a = check_slater_matrix_orthonormality(
            wfn[0,:,:nalpha], nelec=nalpha, matrix_type='wfn[0] alpha (used as init)'
        )
        if not is_ortho_a and orthonormalize:
            wfn[0, :, :nalpha] = modified_gram_schmidt(wfn[0,:,:nalpha])
        
        add_dataset(fh5, 'Psi0_alpha',
                    to_complex(wfn[0,:,:nalpha].copy()))  
        if uhf:
            warn(
                "Using UHF Slater determinant for initial Walkers with a Collinear Trial Wavefunction. " 
                "This can lead to very slow equilibration in AFQMC calculations. " 
                "using ROHF Slater determinants for intiail Walkers is recommended."
            )
            norm_b, is_ortho_b = check_slater_matrix_orthonormality(
                wfn[0,:,nalpha:nalpha+nbeta], nelec=nbeta, matrix_type='wfn[0] beta (used as init)'
            )
            if not is_ortho_b and orthonormalize:
                wfn[0, :, nalpha:nalpha+nbeta] = modified_gram_schmidt(wfn[0,:,nalpha:nalpha+nbeta])
            
            add_dataset(
                fh5, 'Psi0_beta',
                to_complex(
                    wfn[0,:,nalpha:nalpha+nbeta].copy()
                )
            )
    
    for idet, w in enumerate(wfn):
        ix = 2*idet if uhf else idet
        psia = scipy.sparse.csr_array(w[:,:nalpha].conj().T)
        write_nomsd_single(fh5, psia, ix)
        if uhf and nbeta > 0:
            ix = 2*idet + 1
            psib = scipy.sparse.csr_array(w[:,nalpha:nalpha+nbeta].conj().T)
            write_nomsd_single(fh5, psib, ix)


def write_nomsd_ghf(fh5, wfn, nelec, thresh=1e-8, init=None, orthonormalize=True):
    """Write 'GHF'/'G. S. DET' NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`np.ndarray`
        NOMSD trial wavefunctions.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    init : ndarray, optional
        Initial Slater determinant. Will be orthonormality-checked.
    orthonormalize : bool, optional
        If True (default), orthonormalize Slater matrices if not properly normalized.
    """
    nalpha, nbeta = nelec

    if nbeta != 0:
        raise ValueError(
            "nbeta is non-zero while attempting to write NOMSD for either"
            " NONCOLLINER or FULLYPOLARIZED Slater determinants."
        )

    wfn[abs(wfn) < thresh] = 0.0
    
    wfn = np.array(wfn,dtype=np.complex128) 
    
    # Check and optionally orthonormalize wfn Slater matrices
    for idet, w in enumerate(wfn):
        norm, is_ortho = check_slater_matrix_orthonormality(
            w[:,:nalpha], nelec=nalpha, matrix_type=f'wfn[{idet}] noncollinear'
        )
        if not is_ortho and orthonormalize:
            wfn[idet, :, :nalpha] = modified_gram_schmidt(w[:,:nalpha])

    if init is not None:
        # Check and optionally orthonormalize init parameter
        norm, is_ortho = check_slater_matrix_orthonormality(
            init[0], nelec=nalpha, matrix_type='init (noncollinear)'
        )
        if not is_ortho and orthonormalize:
            init[0] = modified_gram_schmidt(init[0])
        
        add_dataset(fh5, 'Psi0_alpha', to_complex(init[0]))
    else:
        # Check and optionally orthonormalize wfn[0] for use as init
        norm, is_ortho = check_slater_matrix_orthonormality(
            wfn[0,:,:nalpha], nelec=nalpha, matrix_type='wfn[0] (used as init)'
        )
        if not is_ortho and orthonormalize:
            wfn[0, :, :nalpha] = modified_gram_schmidt(wfn[0,:,:nalpha])
        
        add_dataset(fh5, 'Psi0_alpha',to_complex(wfn[0,:,:nalpha]))
    
    for idet, w in enumerate(wfn):
        psia = scipy.sparse.csr_array(w[:,:nalpha].conj().T)
        write_nomsd_single(fh5, psia, idet)


write_nomsd = write_nomsd_old

def write_nomsd_single(fh5, psi, idet):
    """Write single component of NOMSD to hdf.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    psi : :class:`scipy.sparse.csr_array`
        Sparse representation of trial wavefunction.
    idet : int
        Determinant number.
    """
    base = 'PsiT_{:d}/'.format(idet)
    dims = [psi.shape[0], psi.shape[1], psi.nnz]
    fh5[base+'dims'] = np.array(dims, dtype=np.int32)
    fh5[base+'data_'] = to_complex(psi.data)
    fh5[base+'jdata_'] = psi.indices
    fh5[base+'pointers_begin_'] = psi.indptr[:-1]
    fh5[base+'pointers_end_'] = psi.indptr[1:]

def write_phmsd(fh5, occa, occb, nelec, norb, init=None, orbmat=None, orthonormalize=True):
    """Write PHMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    nelec : tuple
        Number of alpha and beta electrons.
    init : tuple, optional
        Initial Slater determinant [alpha_block, beta_block]. Will be orthonormality-checked.
    orbmat : tuple, optional
        Orbital transformation matrices.
    orthonormalize : bool, optional
        If True (default), orthonormalize Slater matrices if not properly normalized.
    """
    # TODO: Update if we ever wanted "mixed" phmsd type wavefunctions.
    na, nb = nelec
    if init is None:
        init = [
            np.eye(norb,dtype=np.complex128),
            np.eye(norb,dtype=np.complex128)
        ]
    
    # Check and optionally orthonormalize init parameter
    init_alpha = init[0][:,occa[0]].copy()
    norm_a, is_ortho_a = check_slater_matrix_orthonormality(
        init_alpha, nelec=na, matrix_type='init alpha (PHMSD)'
    )
    if not is_ortho_a and orthonormalize:
        init_alpha = modified_gram_schmidt(init_alpha)
    
    add_dataset(fh5, 'Psi0_alpha', to_complex(init_alpha))
    
    if nb > 0:
        init_beta = init[1][:,occb[0]].copy()
        norm_b, is_ortho_b = check_slater_matrix_orthonormality(
            init_beta, nelec=nb, matrix_type='init beta (PHMSD)'
        )
        if not is_ortho_b and orthonormalize:
            init_beta = modified_gram_schmidt(init_beta)
        
        add_dataset(fh5, 'Psi0_beta', to_complex(init_beta))
    
    if orbmat is not None:
        fh5['type'] = 1
        # Check and optionally orthonormalize orbital matrices
        oa = orbmat[0]
        norm_oa, is_ortho_oa = check_slater_matrix_orthonormality(
            oa, nelec=oa.shape[1], matrix_type='orbmat alpha (PHMSD)'
        )
        if not is_ortho_oa and orthonormalize:
            oa = modified_gram_schmidt(oa)
        
        # Expects conjugate transpose.
        oa_sparse = scipy.sparse.csr_array(oa.conj().T)
        write_nomsd_single(fh5, oa_sparse, 0)
        
        if nb > 0:
            ob = orbmat[1]
            norm_ob, is_ortho_ob = check_slater_matrix_orthonormality(
                ob, nelec=ob.shape[1], matrix_type='orbmat beta (PHMSD)'
            )
            if not is_ortho_ob and orthonormalize:
                ob = modified_gram_schmidt(ob)
            
            ob_sparse = scipy.sparse.csr_array(ob.conj().T)
            write_nomsd_single(fh5, ob_sparse, 1)
    else:
        fh5['type'] = 0
    occs = np.zeros((len(occa), na+nb), dtype=np.int32)
    occs[:,:na] = np.array(occa)
    if nb > 0:
        occs[:,na:] = norb+np.array(occb)
    fh5['occs'] = occs.ravel()
