# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Generate AFQMC data from PYSCF (molecular) simulation."""
import numpy as np
import scipy.sparse
import time
from afqmctools.utils.io import (
        format_fixed_width_floats,
        format_fixed_width_strings,
        )
from afqmctools.hamiltonian.io import (
        write_sparse,
        write_dense
        )
from afqmctools.utils.slater_types import (
    _SlaterType,
    _slater_enum_map,
    _get_slater_type
)
from afqmctools.utils.linalg import modified_cholesky_direct

def write_hamil_mol(
    scf_data, 
    hamil_file, 
    chol_cut,
    verbose=True,
    cas=None,
    ortho_ao=False,
    nelec=None,
    real_chol=True,
    dense=True,
    df=False,
    walker_type=_SlaterType.CLOSED,
    with_soc=False
    ):
    """
    Write hamiltonian from pyscf scf calculation on mol object.
    """
    hcore, chol_vecs, nelec, enuc, X = generate_hamiltonian(
        scf_data,
        verbose=verbose,
        chol_cut=chol_cut,
        cas=cas,
        ortho_ao=ortho_ao,
        nelec=nelec,
        df=df,
        walker_type=walker_type,
        with_soc=with_soc
    )

    # Want L_{(ik),n}
    chol_vecs = chol_vecs.T

    if walker_type is None:
        walker_type = _get_slater_type(
            phi=scf_data['mo_coeff'],
            nelec=scf_data['nelec'],
            M=scf_data['norb']
        )
    else:
        walker_type = _slater_enum_map(walker_type)

    if walker_type in (
        _SlaterType.CLOSED,
        _SlaterType.COLLINEAR,
        _SlaterType.FULLYPOLARIZED
    ):
        nbasis = hcore.shape[-1]
    elif walker_type == _SlaterType.NONCOLLINEAR:
        nbasis = hcore.shape[-1] // 2   
    
    if dense:
        write_dense(
            hcore,
            chol_vecs,
            nelec,
            nbasis,
            enuc,
            real_chol=real_chol,
            filename=hamil_file,
            ortho=X
            )
    else:
        write_sparse(
            hcore,
            chol_vecs,
            nelec,
            nbasis,
            enuc,
            filename=hamil_file,
            real_chol=real_chol,
            verbose=verbose,
            ortho=X
        )

def generate_hamiltonian(
        scf_data, 
        chol_cut=1e-5, 
        verbose=False, 
        cas=None,
        ortho_ao=False,
        nelec=None, 
        df=False, 
        walker_type=None,
        with_soc=False
    ):

    walker_type = _slater_enum_map(walker_type)

    # Unpack SCF data.
    # 1. core (1-body) Hamiltonian.
    hcore = scf_data['hcore']

    if with_soc:

        if walker_type == _SlaterType.NONCOLLINEAR:
            print("Building Hamiltonian with SOC")
        else:
            raise ValueError(
                "Attempted to use spin-orbit coupling "
                "without ghf walker type"
            )

    # 2. Rotation matrix to orthogonalised basis.
    if ortho_ao:
        X = scf_data['X']
    else:
        if scf_data['isUHF']:
            raise ValueError(" # UHF integrals are not allowed. Use ortho AO option (-a/--ao).")
        X = scf_data['mo_coeff']
    
    df_ints = scf_data.get('df_ints', None)
    C = scf_data['mo_coeff']
    
    # 3. Pyscf mol object.
    mol = scf_data['mol']
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        if ortho_ao:
            print(" # Transforming hcore and eri to ortho AO basis.")
        else:
            print(" # Transforming hcore and eri to MO basis.")
    # TODO: need to work out interaction of 'with_soc' and 'with_x2c'
    if 'with_x2c' in scf_data.keys() and scf_data['with_x2c']: # not x2c with SOC
        X_spin_orbs = np.block(
            [[X,np.zeros_like(X)],
             [np.zeros_like(X),X]]
             )
        h1e = np.dot(X_spin_orbs.T, np.dot(hcore, X_spin_orbs))
    else:
        h1e = np.dot(X.T, np.dot(hcore, X))
    nbasis = X.shape[-1]
    if verbose:
        print(" # Number of basis functions: {}.".format(nbasis))
    # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
    if df_ints is not None and df:
        chol_vecs = df_ints
        if verbose:
            print(" # Using DF integrals from checkpoint file.")
        assert chol_vecs.shape[1] == nbasis*nbasis
    else:
        if verbose:
            print (" # Performing modified Cholesky decomposition on ERI tensor.")
        chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print (" # Orthogonalising Cholesky vectors.")
    start = time.time()
    # Step 2.a Orthogonalise Cholesky vectors.
    chol_trans = ao2mo_chol(chol_vecs, X)
    if verbose:
        print(f" # Time to orthogonalise: {(time.time() - start)}")
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    if cas is not None:
        nfzc = (sum(mol.nelec)-cas[0])//2
        ncas = cas[1]
        if ncas == -1:
            ncas = nbasis - nfzc
        nfzv = nbasis - ncas - nfzc
        h1e, chol_trans, enuc = freeze_core(h1e, chol_trans, enuc, nfzc, ncas,
                                           verbose)
        h1e = h1e[0]
        nelec = (mol.nelec[0]-nfzc, mol.nelec[1]-nfzc)
        orbs = np.identity(h1e.shape[-1])
        orbs = orbs[nfzc:nbasis-nfzv,nfzc:nbasis-nfzv]
        X = C[:,nfzc:nbasis-nfzv]
    
    if walker_type == _SlaterType.NONCOLLINEAR and not scf_data['with_x2c']:
        h1e = np.block(
            [[h1e, np.zeros_like(h1e)],
             [np.zeros_like(h1e),h1e]]
            )

        nelec = (sum(nelec),0)

        if with_soc:
            #TODO: handle using 'cas' from above!
            h1e_soc = soc(
                mol=mol,
                mo_basis=X #TODO: double check that orthao works here!
                )
            
            h1e = h1e_soc + h1e

    return h1e, chol_trans, nelec, enuc, X

def process_generic_hamiltonian(
        H_one_body:np.array,
        cholesky_vectors:np.array=None,
        coulomb_repuslion_tensor:np.array=None,
        E0:float=0.0,
        cholesky_delta:float=1e-6,
        spin_orbital_integrals:np.array=None,
        verbose:bool=True
    ):
    r"""
    Process generic hamiltonian.

    Parameters
    ----------
    H_one_body : :class:`np.ndarray`
        One-body Hamiltonian.
    cholesky_vectors : :class:`np.ndarray`, optional
        Cholesky vectors with shape (number of Cholesky vectors,number_of_orbitals\*\*2).
        Must provide either this or coulomb_repuslion_tensor.
    coulomb_repuslion_tensor : :class:`np.ndarray`
        Coulomb repulsion tensor using Chemist's conventions, (ij|kl). Must provide exactly one of this or cholesky_vectors.
    E0 : float
        Nuclear repulsion energy.
    cholesky_delta : float, optional
        Tolerance for Cholesky decomposition. Default is 1e-6.
    spin_orbital_integrals : :class:`np.ndarray`, not implemented
        Spin orbital integrals. If provided, will be added to the Hamiltonian. See notes for conventions.
    verbose : bool
        If true print out verbose details.

    Returns
    -------
    Kij : :class:`np.ndarray`
        One-body Hamiltonian.
    L_ij_gamma : :class:`np.ndarray`
        Cholesky vectors.
    E0 : float
        Constant energy contribution.

    Notes
    -----
    - The interaction must be provided as either the Coulomb repulsion tensor or the Cholesky vectors.
    """

    # NOTE: we might need `nelec` for silly practical reasons. Try just (0,0) for now.

    if cholesky_vectors is None and coulomb_repuslion_tensor is None:
        raise ValueError("Must provide either cholesky_vectors or coulomb_repuslion_tensor.")
    
    if cholesky_vectors is not None and coulomb_repuslion_tensor is not None:
        raise ValueError("Must provide exactly one of cholesky_vectors or coulomb_repuslion_tensor.")

    if spin_orbital_integrals:
        raise NotImplementedError("Spin orbital integrals not yet implemented. Please contact the developers.")

    nmo = H_one_body.shape[-1]

    if coulomb_repuslion_tensor is not None:
        if coulomb_repuslion_tensor.shape != (nmo,nmo,nmo,nmo):
            raise ValueError("Coulomb repulsion tensor must have shape (nmo,nmo,nmo,nmo).")

        if verbose:
            print(" # Generating Cholesky vectors from Coulomb repulsion tensor.")
        cholesky_vectors = modified_cholesky_direct(
        coulomb_repuslion_tensor.reshape(nmo**2,nmo**2), 
        tol=cholesky_delta, 
        verbose=verbose, 
        cmax=4
        )
    elif cholesky_vectors is not None:
        if cholesky_vectors.shape[1] != nmo**2:
            raise ValueError("Cholesky vectors must have shape (number of Cholesky vectors,number_of_orbitals**2).")


    # go from L_{n,(ik)} to L_{(ik),n}
    cholesky_vectors = cholesky_vectors.T

    return H_one_body, cholesky_vectors, E0

    
def write_hamiltonian_generic(*arg,filename:str="afqmc.h",**kwargs):
    r"""
    Write generic hamiltonian to HDF5 file

    Parameters
    ----------
    filename : str, optional
        Name of HDF5 Filename to write to. Default is 'afqmc.h'.
    H_one_body : :class:`np.ndarray`
        One-body Hamiltonian.
    cholesky_vectors : :class:`np.ndarray`, optional
        Cholesky vectors with shape (number of Cholesky vectors,number_of_orbitals\*\*2).
        Must provide either this or coulomb_repuslion_tensor.
    coulomb_repuslion_tensor : :class:`np.ndarray`
        Coulomb repulsion tensor using Chemist's conventions, (ij|kl). Must provide exactly one of this or cholesky_vectors.
    E0 : float
        Nuclear repulsion energy.
    spin_orbital_integrals : :class:`np.ndarray`, not implemented
        Spin orbital integrals. If provided, will be added to the Hamiltonian. See notes for conventions.
    verbose : bool
        If true print out verbose details.

    Returns
    -------
    Kij : :class:`np.ndarray`
        One-body Hamiltonian.
    L_ij_gamma : :class:`np.ndarray`
        Cholesky vectors.
    E0 : float
        Constant energy contribution.

    Notes
    -----
    - The interaction must be provided as either the Coulomb repulsion tensor or the Cholesky vectors.

    """
    Kij, L_ij_gamma, E0 = process_generic_hamiltonian(*arg,**kwargs)

    write_dense(
        enuc=E0,
        hcore=Kij,
        chol=L_ij_gamma, 
        nelec=(0,0), # not used, here for backwards compatibility
        nmo=Kij.shape[-1],
        filename=filename
    )

def freeze_core(h1e, chol, ecore, nc, ncas, verbose=True):
    # 1. Construct one-body hamiltonian
    nbasis = h1e.shape[-1]

    if nbasis-nc-ncas < 0:
        raise ValueError(
            f"freeze_core: ncore = {nc}, nactive = {ncas}, nbasis = {nbasis}:\n"
            "Can't freeze more orbitals than available basis set functions"
        )

    chol = chol.reshape((-1,nbasis,nbasis))
    psi = np.identity(nbasis)[:,:nc]
    Gcore = gab(psi,psi)
    efzc = local_energy_generic_cholesky(h1e, chol, [Gcore,Gcore], ecore)
    (hc_a, hc_b) = core_contribution_cholesky(chol, [Gcore,Gcore])
    h1e = np.array([h1e,h1e])
    h1e[0] = h1e[0] + 2*hc_a
    h1e[1] = h1e[1] + 2*hc_b
    h1e = h1e[:,nc:nc+ncas,nc:nc+ncas]
    nchol = chol.shape[0]
    chol = chol[:,nc:nc+ncas,nc:nc+ncas]
    chol = chol.reshape((nchol,-1))
    # 4. Subtract one-body term from writing H2 as sum of squares.
    if verbose:
        print(f" # Number of active orbitals: {ncas}")
        print(f" # Freezing {2*nc} core electrons and {nbasis-nc-ncas} virtuals.")
        print(f" # Total Frozen core energy:{efzc[0]}")
        print(f'   # E0 (input): {ecore:13.8e}')
        print(f'   # Frozen 1-body contribution: {efzc[1] - ecore}')
        print(f'   # Frozen 2-body contribution: {efzc[2]}')
    return h1e, chol, efzc[0]

def ao2mo_chol(eri, C):
    nao = C.shape[0]
    nmo = C.shape[1]
    nik = nmo*nmo
    nchol = eri.shape[0]
    eri_ = eri.ravel()
    for i in range(nchol):
        cv = eri[i].reshape(nao,nao)
        half = np.dot(cv, C)
        # if nao < nmo we overwrite the data
        eri_[i*nik:(i+1)*nik] = np.dot(C.T, half).ravel()
    return eri_[:nchol*nik].reshape((nchol,nik))

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`np.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao*nao))
    # eri = np.zeros((nao,nao,nao,nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    start = time.time()
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print(" # Generating Cholesky decomposition of ERIs.")
        print(f" # max number of cholesky vectors = {nchol_max}")
        header = ['iteration', 'max_residual', 'time']
        print(format_fixed_width_strings(header))
        init = [delta_max, time.time()-start]
        print('{:17d} '.format(0)+format_fixed_width_floats(init))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = \sum_x L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start

            out = [delta_max, step_time]
            print('{:17d} '.format(nchol)+format_fixed_width_floats(out))

    return chol_vecs[:nchol]


def local_energy_generic_cholesky(h1e, chol_vecs, G, ecore):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`np.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    e1b = np.sum(h1e*G[0]) + np.sum(h1e*G[1])
    cv = chol_vecs
    ecoul_uu = 0
    ecoul_dd = 0
    ecoul_ud = 0
    ecoul_du = 0
    exx_uu = 0
    exx_dd = 0
    # Below to compute exx_uu/dd we do
    # t1 = np.einsum('nik,il->nkl', cv, G[0])
    # t2 = np.einsum('nlj,jk->nlk', cv.conj(), G[0])
    # exx_uu = np.einsum('nkl,nlk->', t1, t2)
    exx_uu = 0
    for c in cv:
        ecoul_uu += np.sum(c*G[0]) * np.sum(c.conj().T*G[0])
        ecoul_dd += np.sum(c*G[1]) * np.sum(c.conj().T*G[1])
        ecoul_ud += np.sum(c*G[0]) * np.sum(c.conj().T*G[1])
        ecoul_du += np.sum(c*G[1]) * np.sum(c.conj().T*G[0])
        t1 = np.dot(c.T, G[0])
        # print(t1.sum())
        t2 = np.dot(c.conj(), G[0])
        # print(t2.sum())
        exx_uu += np.einsum('ij,ji->',t1,t2)
        # print("sum:", exx_uu)
        t1 = np.dot(c.T, G[1])
        t2 = np.dot(c.conj(), G[1])
        exx_dd += np.einsum('ij,ji->',t1,t2)
    euu = 0.5*(ecoul_uu-exx_uu)
    edd = 0.5*(ecoul_dd-exx_dd)
    eud = 0.5 * ecoul_ud
    edu = 0.5 * ecoul_du
    e2b = euu + edd + eud + edu
    return (e1b+e2b+ecore, e1b+ecore, e2b)

def core_contribution_cholesky(chol_vecs, G):
    cv = chol_vecs
    hca_j = np.einsum('l,lij->ij', np.sum(cv*G[0], axis=(1,2)), cv)
    ta_k = np.einsum('lpr,pq->lrq', cv, G[0])
    hca_k = 0.5*np.einsum('lrq,lsq->rs', ta_k, cv)
    hca = hca_j - hca_k
    hcb_j = np.einsum('l,lij->ij', np.sum(cv*G[1], axis=(1,2)), cv)
    tb_k = np.einsum('lpr,pq->lrq', cv, G[1])
    hcb_k = 0.5*np.einsum('lrq,lsq->rs', tb_k, cv)
    hcb = hcb_j - hcb_k
    return (hca, hcb)

def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`np.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`np.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`np.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB


def soc(
        mol,
        mo_basis,
        ncore=0,
        nactive=None
    ):
    '''
    Produces the 1-Body SOC Hamiltonian term based on an SO-ECP for a PySCF mol object

    TODO:
    - add in SO-ECP for cell objects as well

    Inputs:
    - mol: PySCF molecule object defining the system
    - mo: molecular orbital coefficient matrix (assumed to be of ROHF type here)
    - ncore (int): number of core electrons
    - nactive (optioninal : int) : number of active orbitals, defaults to total number of orbitals - ncore

    Returns:
    
    - h_soc: the SOC Hamiltonian term in GHF form
    '''

    soc_PP = 0.5*mol.intor('ECPso')

    soc_PP_mo = np.zeros((3,mo_basis.shape[1],mo_basis.shape[1]),dtype='complex128')

    for i in range(3):
        soc_PP_mo[i,:,:] = mo_basis.conj().T @ soc_PP[i,:,:] @ mo_basis

    K_SOC_full = make_soc_one_body(soc_PP_mo)

    if ncore == 0 and nactive is None:
        return K_SOC_full
    else:
        Mfull = mo_basis.shape[1]

        offset = Mfull + ncore

        if nactive is None:
            nactive = Mfull - ncore # number of active orbitals

        K_SOC = np.zeros((2*nactive,2*nactive), dtype='complex128')
        
        K_SOC[:nactive,:nactive] = K_SOC_full[ncore:ncore+nactive, ncore:ncore+nactive] # up-up
        K_SOC[nactive:,:nactive] = K_SOC_full[offset:, ncore:ncore+nactive] # up-down
        K_SOC[:nactive,nactive:] = K_SOC_full[ncore:ncore+nactive,offset:] # down-up
        K_SOC[nactive:,nactive:] = K_SOC_full[offset:,offset:] # down-down

        return K_SOC


def make_soc_one_body(
        V_SO,
        h1_spin_free=None,
        scale=None
    ):
    r'''
    Make the SOC Hamiltonian in full spin-orbital basis from the spin-orbit
    matrix elements. The SOC Hamiltonian is given by the following equation

    .. math::

        K_{soc} = - i \frac{\alpha^2} {2} [V_{SO}]^l_{pq} * [\sigma]^l_{pq}


    Parameters
    ----------
    V_SO : np.array
        spin-orbit matrix elements, with l being a spatial index. The final
            Hamiltonian term is generated by contracting along 'l' with the pauli
            operators.
    h1_spin_free : np.array (optional) 
        spin-free 1-body Hamiltonian
    scale : float (optional)
         a scalar which the soc hamiltonian will be scaled by.

    Returns
    -------
    h1_soc : np.ndarry
       spin-orbit coupling Hamiltonian in full spin-orbital basis

       
    Notes
    -----

    The final one-body Hamiltonian will have form:

    .. math::

        h^1_{soc} = \begin{pmatrix}
        h^1_{aa} & h^1_{ab} \\
        h^1_{ba} & h^1_{bb} 
        \end{pmatrix}

    following: J. Chem. Theory Comput. 2018, 14, 154−165, the current form of the SOC
    Hamiltonian, in the GHF formulation is 
        
    .. math::

        K_{soc} = -i \frac{\alpha^2}{2} \begin{pmatrix}
        [V_{SO}]^z_{pq} & [V_{SO}]^x_{pq} - i[V_{SO}]^y_{pq} \\
        [V_{SO}]^x_{pq} + i[V_{SO}]^y_{pq} & [V_{SO}]^z_{pq}
        \end{pmatrix}
    
    
    where p, q refer to spatial orbitals - :math:`S^l_pq` are all real-valued;
    we have absorbed the factor math:`-i \frac{\alpha^2}{2}` into 
    :math:`[V_{SO}]^l_pq`.

    '''

    # alias
    if scale is None:
        SOC = V_SO
    else:
        SOC = V_SO*scale

    if h1_spin_free is None:
        M = V_SO.shape[1]
        h1_spin_free = np.zeros((M,M),dtype=np.complex128)
    else:
        M = h1_spin_free.shape[0]

    h1_soc = np.zeros((2*M,2*M),dtype=np.complex128)

    ## a-a sector
    h1_soc[:M,:M] = h1_spin_free - (1j)*SOC[2,:,:]
    
    ## b-b sector
    h1_soc[M:,M:] = h1_spin_free + (1j)*SOC[2,:,:]

    ## a-b sector
    h1_soc[:M,M:] = -(1j)*SOC[0,:,:] - SOC[1,:,:]
    
    ## b-a sector
    h1_soc[M:,:M] = -(1j)*SOC[0,:,:] + SOC[1,:,:]

    return h1_soc
