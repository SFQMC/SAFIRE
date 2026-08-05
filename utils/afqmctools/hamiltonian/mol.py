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
from afqmctools.hamiltonian.io import write_dense
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
    df=False,
    ):
    """Write hamiltonian from pyscf scf calculation on mol object."""
    hcore, chol_vecs, nelec, enuc, X = generate_hamiltonian(
        scf_data,
        verbose=verbose,
        chol_cut=chol_cut,
        cas=cas,
        ortho_ao=ortho_ao,
        nelec=nelec,
        df=df,
    )

    # Want L_{(ik),n}
    chol_vecs = chol_vecs.T
    norb = X.shape[-1]

    write_dense(
        hcore,
        chol_vecs,
        nelec,
        nmo=norb,
        enuc=enuc,
        real_chol=real_chol,
        filename=hamil_file,
        ortho=X
    )

def _transform_from_scf_data(scf_data, ortho_ao, cas = None):
    C = scf_data['mo_coeff']

    if ortho_ao:
        if cas is not None:
            raise ValueError("cas and ortho_ao cannot be used at the same time")
        return scf_data['X'], (0, 0)
    else:
        if C.ndim == 3 or C.shape[0] == 2*scf_data["norb"]:
            raise ValueError("UHF or GHF molecular orbital bases are not supported. Use ortho_ao.")
        if cas is not None:
            nfzc = (sum(scf_data["nelec"])-cas[0])//2
            ncas = cas[1]
            nmo = C.shape[-1]

            if ncas == -1:
                ncas = nmo - nfzc

            nfzv = nmo - ncas - nfzc
        else:
            nfzc = nfzv = 0

        return C, (nfzc, nfzv)


def generate_hamiltonian(
        scf_data, 
        chol_cut=1e-5, 
        verbose=False, 
        cas=None,
        ortho_ao=False,
        nelec=None, 
        df=False, 
        walker_type=None,
    ):

    if walker_type is None:
        walker_type = scf_data["walker_type"]
    else:
        walker_type = _slater_enum_map(walker_type)

    # Unpack SCF data.
    # 1. core (1-body) Hamiltonian.
    hcore = scf_data['hcore']

    # 2. Rotation matrix to orthogonalised basis.
    df_ints = scf_data.get('df_ints', None)

    X, (nfzc, nfzv) = _transform_from_scf_data(scf_data, ortho_ao, cas)

    # 3. Pyscf mol object.
    mol = scf_data['mol']

    nbasis = X.shape[-1]

    # noncollinear Hamiltonian
    if hcore.shape == (2 * X.shape[0], 2 * X.shape[0]):
        if walker_type != _SlaterType.NONCOLLINEAR:
            raise ValueError(f"Hamiltonian is noncollinear but walker_type {walker_type} is not")
 
        Xspin = np.kron(np.eye(2), X)
        h1e = Xspin.conj().T @ hcore @ Xspin
    else:
        h1e = X.conj().T @ hcore @ X

    if verbose:
        print(" # Number of basis functions: {}.".format(nbasis))
    # Step 2. Generate Cholesky decomposed ERIs in non-orthogonal AO basis.
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
    # Step 2.a Transform Cholesky vectors to orthogonal basis
    chol_trans = transform_cholesky(chol_vecs, X)
    if verbose:
        print(f" # Time to orthogonalise: {(time.time() - start)}")
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec

    if (nfzc, nfzv) != (0, 0):
        h1e, chol_trans, enuc = freeze_core(
            h1e, chol_trans, enuc, nfzc, nbasis - nfzv - nfzc, verbose
        )
        h1e = h1e[0]

    return h1e, chol_trans, nelec, enuc, X[:, nfzc:nbasis - nfzv]


def process_generic_hamiltonian(
        H_one_body:np.array,
        cholesky_vectors:np.array=None,
        coulomb_repulsion_tensor:np.array=None,
        E0:float=0.0,
        cholesky_delta:float=1e-6,
        spin_orbital_integrals:np.array=None,
        verbose:bool=True,
        cmax:int=20
    ):
    r"""
    Process generic hamiltonian.

    Parameters
    ----------
    H_one_body : :class:`np.ndarray`
        One-body Hamiltonian.
    cholesky_vectors : :class:`np.ndarray`, optional
        Cholesky vectors with shape (number of Cholesky vectors,number_of_orbitals\*\*2).
        Must provide either this or coulomb_repulsion_tensor.
    coulomb_repulsion_tensor : :class:`np.ndarray`
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

    if cholesky_vectors is None and coulomb_repulsion_tensor is None:
        raise ValueError("Must provide either cholesky_vectors or coulomb_repulsion_tensor.")
    
    if cholesky_vectors is not None and coulomb_repulsion_tensor is not None:
        raise ValueError("Must provide exactly one of cholesky_vectors or coulomb_repulsion_tensor.")

    if spin_orbital_integrals:
        raise NotImplementedError("Spin orbital integrals not yet implemented. Please contact the developers.")

    nmo = H_one_body.shape[-1]

    if coulomb_repulsion_tensor is not None:
        if coulomb_repulsion_tensor.shape != (nmo,nmo,nmo,nmo):
            raise ValueError("Coulomb repulsion tensor must have shape (nmo,nmo,nmo,nmo).")

        if verbose:
            print(" # Generating Cholesky vectors from Coulomb repulsion tensor.")
        # cmax caps Cholesky vectors at nchol_max = min(int(cmax*nmo), nmo**2). The old default cmax=4
        # silently truncated large active spaces (e.g. Fe2S2 CAS(30e,20o): 79/193 vectors, ~+22 mHa bias).
        # Default is now 20; pass cmax=None to remove the cap. Raises if the cap is hit before tol.
        cholesky_vectors = modified_cholesky_direct(
        coulomb_repulsion_tensor.reshape(nmo**2,nmo**2),
        tol=cholesky_delta,
        verbose=verbose,
        cmax=(nmo if cmax is None else cmax)
        )
        # Fail loudly rather than hand back a silently-capped factorization.
        nchol_max = min(int((nmo if cmax is None else cmax) * nmo), nmo**2)
        if cholesky_vectors.shape[0] >= nchol_max - 1:
            raise RuntimeError(
                f"modified_cholesky_direct hit its vector cap: produced "
                f"{cholesky_vectors.shape[0]} vectors with nchol_max={nchol_max} "
                f"(cmax={cmax}, nmo={nmo}), so it exited on the CAP and never reached "
                f"tol={cholesky_delta}. The Hamiltonian would be silently truncated. "
                f"Raise cmax (or pass cmax=None to remove the cap)."
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
        Must provide either this or coulomb_repulsion_tensor.
    coulomb_repulsion_tensor : :class:`np.ndarray`
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

def transform_cholesky(chol, C):
    """Apply the basis rotation `C` to the cholesky vectors `chol`""" 
    nao = C.shape[0]
    nmo = C.shape[1]
    nik = nmo*nmo
    nchol = chol.shape[0]
    chol_ = chol.ravel()
    for i in range(nchol):
        cv = chol[i].reshape(nao,nao)
        half = np.dot(cv, C)
        # if nao < nmo we overwrite the data
        chol_[i*nik:(i+1)*nik] = np.dot(C.T, half).ravel()
    return chol_[:nchol*nik].reshape((nchol,nik))

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. :cite:`motta_initio_2018`

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    max_error : float
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
