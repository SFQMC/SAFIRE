# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Simple extraction of afqmc rdms."""
import h5py
import numpy
from math import sqrt
import scipy.stats
import scipy.integrate
from afqmctools.analysis.extraction import (
        get_metadata,
        extract_observable
        )
from stats.stat_h5 import me2d
from .common import reblock

# enumish # TODO: KE replace with actual enum
WALKER_TYPE = ['undefined', 'closed', 'collinear', 'non_collinear']


def average_one_rdm(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None, kappa=None):
    r"""Get average AFQMC 1RDM.

    Returns P_{sij} = <c_{is}^+ c_{js}^> as a (nspin, M, M) dimensional array.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).

    Returns
    -------
    one_rdm : :class:`numpy.ndarray`
        Averaged 1RDM.
    one_rdm_err : :class:`numpy.ndarray`
        Error bars for 1RDM elements.
    """
    md = get_metadata(filename)
    mean, err = average_observable(filename, 'one_rdm', eqlb=eqlb, blocksize=blocksize,
                                   estimator=estimator, ix=ix, kappa=kappa)
    nbasis = md['NMO']
    wt = md['WalkerType']
    try:
        walker = WALKER_TYPE[wt]
    except IndexError:
        print('Unknown walker type {}'.format(wt))
        return None

    if walker == 'closed':
        return mean.reshape(1,nbasis,nbasis), err.reshape(1,nbasis, nbasis)
    elif walker == 'collinear':
        return mean.reshape((2,nbasis,nbasis)), err.reshape((2, nbasis, nbasis))
    elif walker == 'non_collinear':
        return mean.reshape((1,2*nbasis,2*nbasis)), err.reshape((1,2*nbasis, 2*nbasis))


def average_two_rdm(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None, kappa=None):
    r"""Average AFQMC 2RDM.

    Returns a list of 2RDMS, where 
      2RDM[s1s2,i,k,j,l] = <c_{i}^+ c_{j}^+ c_{l} c_{k}>.
      For closed shell systems, returns [(a,a,a,a),(a,a,b,b)] 
      For collinear systems, returns [(a,a,a,a),(a,a,b,b),(b,b,b,b)] 

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).

    Returns
    -------
    two_rdm : :class:`numpy.ndarray`
        List of averaged 2RDM.
    two_rdm_err : :class:`numpy.ndarray`
        List of error bars for 2RDM elements.
    """
    md = get_metadata(filename)
    mean, err = average_observable(filename, 'two_rdm', eqlb=eqlb, blocksize=blocksize,
                                   estimator=estimator, ix=ix, kappa=kappa)
    nbasis = md['NMO']
    wt = md['WalkerType']
    try:
        walker = WALKER_TYPE[wt]
    except IndexError:
        print('Unknown walker type {}'.format(wt))
        return None

    if walker == 'closed':
        return mean.reshape(2,nbasis,nbasis,nbasis,nbasis), err.reshape(2,nbasis,nbasis,nbasis,nbasis)
    elif walker == 'collinear':
        return mean.reshape(3,nbasis,nbasis,nbasis,nbasis), err.reshape(3,nbasis,nbasis,nbasis,nbasis)
    elif walker == 'non_collinear':
        return mean.reshape(2*nbasis,2*nbasis,2*nbasis,2*nbasis), err.reshape(2*nbasis,2*nbasis,2*nbasis, 2*nbasis)
        

def average_diag_two_rdm(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None, kappa=None):
    r"""Average diagonal part of 2RDM.

    Returns <c_{is}^+ c_{jt}^+ c_{jt} c_{is}> as a (2M,2M) dimensional array.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).

    Returns
    -------
    two_rdm : :class:`numpy.ndarray`
        Averaged diagonal 2RDM.
    two_rdm_err : :class:`numpy.ndarray`
        Error bars for diagonal 2RDM elements.
    """
    md = get_metadata(filename)
    mean, err = average_observable(filename, 'diag_two_rdm', eqlb=eqlb, blocksize=blocksize,
                                   estimator=estimator, ix=ix, kappa=kappa)
    nbasis = md['NMO']
    wt = md['WalkerType']
    try:
        walker = WALKER_TYPE[wt]
    except IndexError:
        print('Unknown walker type {}'.format(wt))
        return None

    if walker == 'closed':
        dm_size = nbasis*(2*nbasis-1) - nbasis*(nbasis-1) // 2
        assert mean.shape == dm_size
        two_rdm = numpy.zeros((2*nbasis, 2*nbasis), dtype=mean.dtype)
        two_rdm_err = numpy.zeros((2*nbasis, 2*nbasis), dtype=mean.dtype)
        ij = 0
        for i in range(nbasis):
            for j in range(i+1, 2*nbasis):
                two_rdm[i,j] = mean[ij]
                two_rdm_err[i,j] = err[ij]
                two_rdm[j,i] = mean[ij].conj()
                two_rdm_err[j,i] = err[ij].conj()
                ij += 1
        two_rdm[nbasis:,nbasis:] = two_rdm[:nbasis,:nbasis].copy()
    elif walker == 'collinear':
        dm_size = nbasis*(2*nbasis-1)
        assert mean.shape == dm_size
        two_rdm = numpy.zeros((2*nbasis, 2*nbasis), dtype=mean.dtype)
        two_rdm_err = numpy.zeros((2*nbasis, 2*nbasis), dtype=mean.dtype)
        ij = 0
        for i in range(2*nbasis):
            for j in range(i+1, 2*nbasis):
                two_rdm[i,j] = mean[ij]
                two_rdm_err[i,j] = err[ij]
                two_rdm[j,i] = mean[ij].conj()
                two_rdm_err[j,i] = err[ij].conj()
                ij += 1
    elif walker == 'non_collinear':
        # For non-collinear, work in the spinor basis (2*nbasis)
        spinor_basis = 2 * nbasis
        dm_size = spinor_basis * (spinor_basis - 1) // 2
        assert mean.shape[0] == dm_size, f"Expected {dm_size} elements, got {mean.shape[0]}"
        two_rdm = numpy.zeros((spinor_basis, spinor_basis), dtype=mean.dtype)
        two_rdm_err = numpy.zeros((spinor_basis, spinor_basis), dtype=mean.dtype)
        ij = 0
        for i in range(spinor_basis):
            for j in range(i+1, spinor_basis):
                two_rdm[i,j] = mean[ij]
                two_rdm_err[i,j] = err[ij]
                two_rdm[j,i] = mean[ij].conj()
                two_rdm_err[j,i] = err[ij].conj()
                ij += 1
        
    # Diagonal is zero
    return two_rdm, two_rdm_err


def average_observable(filename, name, eqlb=1, estimator='back_propagated',
                       ix=None, blocksize=1, dataset_name=None, transform=None, batch_size=None,
                       remove_auto_correlation=True, kappa=None):
    r"""Compute mean and error bar for AFQMC HDF5 observable.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    name : string
        Name of observable (see estimates.py for list).
    eqlb : int
        Number of blocks for equilibration. Default 1.
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    blocksize : int
        the number of blocks to average over while reblocking. Reblocking is faster
        if the number of samples is evenly divisible by the blocksize.
        Default 1 (i.e. don't reblock). 
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).
    transform : callable
        Function to transform the data before averaging. Optional. See Notes for details.
    batch_size : int
        Size of batch to load data. Optional. Default None (load all data at once).
    dataset_name : string
        Name of dataset to extract. Optional. Default None (uses name).
    remove_auto_correlation : bool
        If True, remove auto-correlation from the data. Default True.
    kappa : float
        Auto-correlation time. Optional. Default None (recalculate).

    Returns
    -------
    mean : :class:`numpy.ndarray`
        Averaged quantity.
    err : :class:`numpy.ndarray`
        Error bars for quantity.

    Notes
    -----

    **Transform Function Guidelines**

    - Takes a single argument (data of shape (nsamples,...)) and return transformed data
    - assume the leading dimension is nsamples

    +------------------------------+--------------------------------------+
    | Do                           | Don't                                |
    +==============================+======================================+
    | Return a new array           | Modify the data in place.            |
    +------------------------------+--------------------------------------+
    | Return the transformed data. | Return a tuple — return a single     |
    |                              | array instead.                       |
    +------------------------------+--------------------------------------+
    | Return None if data is       | Raise an exception                   |
    | invalid.                     |                                      |
    +------------------------------+--------------------------------------+

    """
    md = get_metadata(filename)
    free_proj = md['FreeProjection']
    if free_proj:
        mean = None
        err = None
        print("# Error analysis for free projection not implemented.")
    else:
        data = extract_observable(filename, name=name, estimator=estimator, 
                                  ix=ix, dataset_name=dataset_name, 
                                  transform=transform, batch_size=batch_size)
        data = reblock(data, blocksize=blocksize) if blocksize > 1 else data
        if remove_auto_correlation:
            # remove auto-correlation
            mean, err = me2d(data[eqlb:len(data)],kappa=kappa)
        else:
            mean = numpy.mean(data[eqlb:len(data)], axis=0)
            err = scipy.stats.sem(data[eqlb:len(data)].real, axis=0)
    return mean, err

def average_spinspin(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None, kappa=None):
    r"""Get average AFQMC SpinSpin correlation.

    Returns (<(X_i X_j)+(Y_i Y_j)> , <Z_i Z_j>) as a (2,M,M) dimensional array, resolving XX+YY and ZZ separately

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).

    Returns
    -------
    SS : :class:`numpy.ndarray`
        Averaged Spin-Spin correlator.
    SS_err : :class:`numpy.ndarray`
        Error bars for the SS elements.
    """
    md = get_metadata(filename)
    mean, err = average_observable(filename, 'spinspin', eqlb=eqlb, blocksize=blocksize,
                                   estimator=estimator, ix=ix, kappa=kappa)
    nbasis = md['NMO']
    wt = md['WalkerType']
    try:
        walker = WALKER_TYPE[wt]
    except IndexError:
        print('Unknown walker type {}'.format(wt))
        return None

    # reshape upper triangular data to full matrix
    SS = numpy.zeros((2,nbasis,nbasis),dtype=mean.dtype)
    triu_idx = numpy.triu_indices(nbasis)
    SS[:, triu_idx[0], triu_idx[1]] = mean.reshape(2,-1)
    SS_err = numpy.zeros_like(SS)
    SS_err[:, triu_idx[0], triu_idx[1]] = err.reshape(2,-1)
    for i in range(2):
      SS[i]  += numpy.triu(SS[i],k=1).T.conj()
      SS_err[i]  += numpy.triu(SS_err[i],k=1).T.conj()
    return SS,SS_err

def average_pair_correlation(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None, kappa=None):
    r"""Get average AFQMC pair correlation functions.

    Returns the average of the pair correlation function defined as:

    .. math::

        P^{\alpha \beta}_{ij} ≡ \frac{1}{2} ⟨ \left(c^{\dagger}_{i⇑} c^{†}_{\bar{i}_α ⇓} -  c^{\dagger}_{i⇓} c^{†}_{\bar{i}_α ⇑} \right)  \left( c_{\bar{j}_\beta⇓} c_{j ⇑}  - c_{\bar{j}_\beta ⇑} c_{j ⇓} \right) ⟩.

    where :math:`\bar{i}_α` and :math:`\bar{j}_\beta` are indices corresponding to spatial offsets (i.e. s, +x, -x, etc,)

    The mapping from the :math:`\alpha` and :math:`\beta` indices to the spin indices is also returned as a list.
    
    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).

    Returns
    -------
    P_ab_ij : :class:`numpy.ndarray`
        Averaged pair correlation function.
    dP_ab_ij : :class:`numpy.ndarray`
        Stochastic uncertainty for the pair correlation function.
    correlator_names : list
        List of correlator names corresponding to the indices of the pair correlation function.
    """
    md = get_metadata(filename)
    nbasis = md['NMO']
    wt = md['WalkerType']
    try:
        walker = WALKER_TYPE[wt]
    except IndexError:
        print('Unknown walker type {}'.format(wt))
        return None

    if estimator == 'back_propagated':
        bp_md = get_metadata(filename, path='Observables/BackPropagated/')
        naverages = bp_md['NumAverages'] if ix is None else 1
    else:
        naverages = 1

    pair_correlation_metadata = get_metadata(filename, path='Observables/BackPropagated/PairCorr/')
    num_correlators = pair_correlation_metadata['NCORRELATORS']
    correlator_names = [pair_correlation_metadata[f'CORRELATOR_NAME_{i}'].tobytes().decode('utf-8') for i in range(num_correlators)]

    P_ab_ij = numpy.zeros((naverages,num_correlators*num_correlators, nbasis*nbasis), dtype=numpy.complex128)
    dP_ab_ij = numpy.zeros((naverages,num_correlators*num_correlators, nbasis*nbasis), dtype=numpy.complex128)
    
    if ix is not None:
        iav_iter = [ix]
    else:
        iav_iter = range(naverages)
    
    for iav in iav_iter:
        for a in range(num_correlators):
            for b in range(num_correlators):
                mean, err = average_observable(
                    filename, 
                    'pair_correlation', 
                    eqlb=eqlb,
                    blocksize=blocksize,
                    estimator=estimator, 
                    dataset_name=f'P_{a}_{b}_ij',
                    ix=iav,
                    kappa=kappa
                )

                # adjust for autocorrelation
                P_ab_ij[iav,a*num_correlators+b] = mean.reshape((nbasis*nbasis))
                dP_ab_ij[iav,a*num_correlators+b] = err.reshape((nbasis*nbasis))

    return P_ab_ij,dP_ab_ij,correlator_names

def get_noons(filename, estimator='back_propagated', eqlb=1, blocksize=1, ix=None,
              nsamp=20, screen_factor=1, cutoff=1e-14, kappa=None):
    r"""Get NOONs from averaged AFQMC RDM.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    eqlb : int
        Number of blocks for equilibration. Default 1.
    blocksize : int
        the number of blocks to average over while reblocking.
        Default 1 (i.e. don't reblock).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).
    nsamp : int
        Number of perturbed RDMs to construct to estimate of error bar. Optional.
        Default: 20.
    screen_factor : int
        Zero RDM elements with abs(P[i,j]) < screen_factor*Perr[i,j]. Optional
        Default: 1.

    Returns
    -------
    noons : :class:`numpy.ndarray`
        NOONS.
    noons_err : :class:`numpy.ndarray`
        Estimate of error bar on NOONs.
    """
    P, Perr = average_one_rdm(filename, estimator='back_propagated', eqlb=1,
                              blocksize=1, ix=ix, kappa=kappa)
    if Perr.shape[0] == 2:
        # Collinear
        Perr = numpy.sqrt((Perr[0]**2 + Perr[1]**2))
    else:
        # Non-collinear / Closed
        Perr = Perr[0]
    # Sum over spin.
    P = numpy.sum(P, axis=0)
    P = 0.5 * (P + P.conj().T)
    Perr = 0.5 * (Perr + Perr.T)
    P[numpy.abs(P) < screen_factor*Perr] = 0.0
    noons = numpy.zeros((nsamp, P.shape[-1]))
    for s in range(nsamp):
        PT, X = regularised_ortho(P, cutoff=cutoff)
        Ppert = gen_sample_matrix(PT, Perr)
        e, ev = numpy.linalg.eigh(Ppert)
        noons[s] = e[::-1]

    PT, X = regularised_ortho(P, cutoff=cutoff)
    e, ev = numpy.linalg.eigh(PT)
    return e[::-1], numpy.std(noons, axis=0, ddof=1)

def solve_gen_eig(gamma, fock, cutoff=1e-14):
    r"""Solve generalised eigenvalue problem.
    
    Parameters
    ----------
    gamma : :class:`numpy.ndarray`
        Generalised density matrix.
    fock : :class:`numpy.ndarray`
        Generalised Fock matrix.
    cutoff : float
        Cutoff for singular values. Optional.

    Returns
    -------
    eig : :class:`numpy.ndarray`
        Eigenvalues.
    eigv : :class:`numpy.ndarray`
        Eigenvectors.
    """
    gamma, X = regularised_ortho(gamma, cutoff=cutoff)
    fock_trans = numpy.dot(X.conj().T, numpy.dot(fock, X))
    eig, C = numpy.linalg.eigh(fock_trans)
    # Rotate eigenvectors back to original basis
    return eig, numpy.dot(X, C)

def estimate_error_eig(gamma, gamma_err, fock=None, fock_err=None, nsamp=20, cutoff=1e-14):
    """Bootstrap estimate of error in eigenvalues."""
    eigs_tot = numpy.zeros((nsamp, gamma.shape[-1]))

    if fock is None:
        fock = numpy.eye(gamma.shape[-1])
        fock_err = numpy.zeros_like(fock)

    # TODO FIX THIS
    for s in range(nsamp):
        gamma_p = gen_sample_matrix(gamma, gamma_err)
        fock_p = gen_sample_matrix(fock, fock_err, herm=False)
        eigs, eigv = solve_gen_eig(gamma_p, fock_p)
        eigs_tot[s,:len(eigs)] = eigs
    return numpy.std(eigs_tot, axis=0, ddof=1)

def regularised_ortho(S, cutoff=1e-14):
    """Get orthogonalisation matrix."""
    sdiag, Us = numpy.linalg.eigh(S)
    sdiag[sdiag<cutoff] = 0.0
    X = Us[:,sdiag>cutoff] / numpy.sqrt(sdiag[sdiag>cutoff])
    Smod = numpy.dot(Us[:,sdiag>cutoff], numpy.diag(sdiag[sdiag>cutoff]))
    Smod = numpy.dot(Smod, Us[:,sdiag>cutoff].T.conj())
    return Smod, X

def gen_sample_matrix(mat, err, herm=True):
    """Perturb matrix by error bar."""
    a = numpy.zeros_like(mat)
    nbasis = mat.shape[0]
    assert a.shape[1] == nbasis
    assert a.shape == mat.shape
    # Dangerous. Discards imaginary part.
    for i in range(nbasis):
        for j in range(nbasis):
            a[i,j] = numpy.random.normal(loc=mat[i,j], scale=err[i,j], size=1)
    if herm:
        return 0.5*(a+a.conj().T)
    else:
        return a
