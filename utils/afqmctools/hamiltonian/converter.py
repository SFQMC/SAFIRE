# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import ast
from itertools import product
from warnings import warn

import h5py as h5
import numpy as np
import scipy.sparse
import scipy.linalg
from numba import jit

from afqmctools.utils.io import (
        to_complex,
        from_complex
        )
from afqmctools.hamiltonian.io import (
        write_sparse_basic,
        write_sparse_chol_chunk
        )


def fcidump_header(nel:int, norb:int, spin:int) -> str:
    """
    Writes a header for a FCIDUMP file.

    Parameters
    ----------
    nel : int
        Total number of electrons.
    norb : int
        Number of orbitals.
    spin : int
        Spin of the system.

    returns
    -------
    header : string
        Header for FCIDUMP file.
    """
    header = (
        "&FCI " +
        "NORB={:d}, ".format(norb) +
        "NELEC={:d}, ".format(nel) +
        "MS2={:d},\n".format(spin) +
        "ORBSYM=" +
        ",".join([str(1)]*norb) +
        ",\n" +
        "ISYM=1\n" +
        "&END\n"
        )
    return header

def read_fcidump_header(filename:str, mline=24) -> dict:
  """
  Read the header of a FCIDUMP file.

  Parameters
  ----------
  filename : string
      File containing integrals in FCIDUMP format.
  mline : int
      Maximum number of lines to read in header.

  Returns
  -------
  meta : dict
      Dictionary containing the metadata from the header.
  """
  meta = dict()
  found_end = False
  with open(filename) as f:
    for iline in range(mline):
      line = f.readline()
      if 'END' in line or '/' in line:
        found_end = True
        break
      for i in line.split(','):
        if 'NORB' in i:
          nbasis = int(i.split('=')[1])
          meta['nbasis'] = nbasis
        elif 'NELEC' in i:
          nelec = int(i.split('=')[1])
          meta['nelec'] = nelec
        elif 'MS2' in i:
          ms2 = int(i.split('=')[1])
          meta['ms2'] = ms2
        elif "ISYM" in i:
          isym = int(i.split('=')[1])
          meta['isym'] = isym
  if not found_end:
    msg = "FCIDUMP header longer than %d lines" % mline
    raise RuntimeError(msg)
  return meta

def read_fcidump(filename, symmetry=None, verbose=True):
    """Read in integrals from file.

    For FCIDUMP files with complex-valued integrals, 
    read_fcidump assumes that the real and imaginary parts
    are listed as `(real,imag)` as opposed to `real imag` 

    Parameters
    ----------
    filename : string
        File containing integrals in FCIDUMP format.
    symmetry : int, optional
        Permutational symmetry of two electron integrals. If not specified, 
        the symmetry is read from the header of the FCIDUMP file.
    verbose : bool
        Controls printing verbosity. Optional. Default: False.

    Returns
    -------
    h1e : :class:`np.ndarray`
        One-body part of the Hamiltonian.
    h2e : :class:`np.ndarray`
        Two-electron integrals.
    ecore : float
        Core contribution to the total energy.
    nelec : tuple
        Number of electrons.
    """
    meta = read_fcidump_header(filename)
    nbasis = meta['nbasis']
    nelec = meta['nelec']
    ms2 = meta['ms2']
    # Note: this is necessary since some codes do not correctly
    #         report the symmetry of the integrals.
    if symmetry is None:
        symmetry = meta['isym']
    assert(symmetry==1 or symmetry==4 or symmetry==8)
    if verbose:
        print ("# Reading integrals in plain text FCIDUMP format.")
    with open(filename) as f:
        while True:
            line = f.readline()
            if 'END' in line or '/' in line:
                break
        if verbose:
            print("# Number of orbitals: {}".format(nbasis))
            print("# Number of electrons: {}".format(nelec))
        h1e = np.zeros((nbasis, nbasis), dtype=np.complex128)
        h2e = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.complex128)
        lines = f.readlines()
        for l in lines:
            s = l.split()
            # ascii fcidump uses Chemist's notation for integrals.
            # each line contains v_{ijkl} i k j l
            # Note (ik|jl) = <ij|kl>.
            if l.strip().startswith('('):  # parenthesis complex value
                left, right = l.split(')')
                s = right.split()
                rt, it = left.split(',')
                int_r = float(rt.replace('(', ''))
                int_i = float(it)
                integral = int_r+1j*int_i
            elif len(s) == 6:
                # FCIDUMP from quantum package.
                integral = float(s[0]) + 1j*float(s[1])
                s = s[1:]
            else:
                try:
                    integral = float(s[0])
                except ValueError:
                    ig = ast.literal_eval(s[0].strip())
                    integral = ig[0] + 1j*ig[1]
            i, k, j, l = [int(x) for x in s[-4:]]
            if i == j == k == l == 0:
                ecore = integral
            elif j == 0 and l == 0:
                # <i|k> = <k|i>
                h1e[i-1,k-1] = integral
                h1e[k-1,i-1] = integral.conjugate()
            elif i > 0  and j > 0 and k > 0 and l > 0:
                # Assuming 8 fold symmetry in integrals.
                # <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
                # <kj|il> = <li|jk> = <il|kj> = <jk|li>
                # (ik|jl)
                h2e[i-1,k-1,j-1,l-1] = integral
                if symmetry == 1:
                    continue
                # (jl|ik)
                h2e[j-1,l-1,i-1,k-1] = integral
                # (ki|lj)
                h2e[k-1,i-1,l-1,j-1] = integral.conjugate()
                # (lj|ki)
                h2e[l-1,j-1,k-1,i-1] = integral.conjugate()
                if symmetry == 4:
                    continue
                # (ki|jl)
                h2e[k-1,i-1,j-1,l-1] = integral
                # (lj|ik)
                h2e[l-1,j-1,i-1,k-1] = integral
                # (ik|lj)
                h2e[i-1,k-1,l-1,j-1] = integral
                # (jl|ki)
                h2e[j-1,l-1,k-1,i-1] = integral
    if symmetry == 8:
        if np.any(np.abs(h1e.imag)) > 1e-18:
            print("# Found complex numbers in one-body Hamiltonian but 8-fold"
                  " symmetry specified.")
        if np.any(np.abs(h2e.imag)) > 1e-18:
            print("# Found complex numbers in two-body Hamiltonian but 8-fold"
                  " symmetry specified.")
    nalpha = (nelec + ms2) // 2
    nbeta = nalpha - ms2
    if ecore.imag > 1e-8:
      print(" Found complex core energy in FCIDUMP, ignoring imaginary part.")
    return h1e, h2e, ecore.real, (nalpha, nbeta)


def read_hamil_type(filename:str) -> str:
    r"""
    Read Hamiltonian type from internal format (HDF5, \*.h5).
    
    Parameters
    ----------
    filename : string
        Hamiltonian file (HDF5, \*.h5)

    Returns
    -------
    ham_type : string
        Type of Hamiltonian.
    """
    with h5.File(filename,'r') as fh5:
        if 'Hamiltonian/DenseFactorized/L' in fh5:
            return 'dense'
        elif 'Hamiltonian/Factorized/vals_0' in fh5:
            return 'sparse'
        elif 'Hamiltonian/KPFactorized/L0' in fh5:
            return 'kpoint'
        elif '/Interaction/Vq0' in fh5:
            return 'coqui_kpoint'
        elif 'Interaction/Vq0' in fh5:
            return 'kpoint_coqui'
        elif 'Hamiltonian/THC/Luv' in fh5:
            return 'thc'
        elif 'Hamiltonian/ModelHamiltonian/number_of_components' in fh5:
            return 'model'
        else:
            return None


def read_hamiltonian(filename, get_chol=True, walker_type=1):
    r"""Read Hamiltonian from internal format (\*.h5).

    Parameters
    ----------
    filename : string
        Hamiltonian file (\*.h5)

    Returns
    -------
    hamil : dict
        Data read from file.
    """
    ham_type = read_hamil_type(filename)
    if ham_type == 'kpoint':
        hc, chol, enuc, nmo, nelec, nmok, qkk2, nchol_pk, minus_k = (
                read_cholesky_kpoint(filename, get_chol=get_chol)
                )
        hamil = {
            'hcore': hc,
            'chol': chol,
            'enuc': enuc,
            'nelec': nelec,
            'nmo': nmo,
            'nmo_pk': nmok,
            'nchol_pk': nchol_pk,
            'minus_k': minus_k,
            'qk_k2': qkk2
            }
    elif ham_type == 'kpoint_coqui':
        hc, chol, enuc, nmo, nelec, nmok, qkk2, nchol_pk, minus_k = (
                read_cholesky_kpoint_coqui(filename, get_chol=get_chol)
                )
        hamil = {
            'hcore': hc,
            'chol': chol,
            'enuc': enuc,
            'nelec': nelec,
            'nmo': nmo,
            'nmo_pk': nmok,
            'nchol_pk': nchol_pk,
            'minus_k': minus_k,
            'qk_k2': qkk2
            }
    elif ham_type == 'sparse':
        hc, chol, enuc, nmo, nelec = read_sparse(filename,
                                                         get_chol=get_chol)
        hamil = {
            'hcore': hc,
            'chol': chol,
            'enuc': enuc,
            'nmo': nmo,
            'nelec': nelec
            }
        return hamil
    elif ham_type == 'dense':
        hcore, chol, enuc, nelec = read_dense(filename)
        hamil = {
            'hcore': hcore, 
            'chol': chol, 
            'enuc': enuc,
            'nmo' : hcore.shape[0],
            'nelec' : nelec
            }
        return hamil
    elif ham_type == 'thc':
        print(" Please implement THC hamiltonian reader.")
        hamil = {}
    elif ham_type == 'model':
        hamil = read_model(filename)
    else:
        hamil = None
        raise TypeError("Unknown Hamiltonian type.")
    return hamil


def read_model(filename):
    """
    Read model Hamiltonian from hdf5 file `fname`
    """
    raise NotImplementedError("[for devlopers] implement model H reader")


def read_sparse(filename, get_chol=True):
    r"""Read in sparse integrals from hdf5.

    Parameters
    ----------
    filename : string
        File containing integrals in internal format (\*.h5).

    Returns
    -------
    hcore : :class:`np.ndarray`
        One-body part of the Hamiltonian.
    chol_vecs : :class:`scipy.sparse.csr_array`
        Two-electron integrals. Shape: [nmo*nmo, nchol]
    ecore : float
        Core contribution to the total energy.
    nmo : int
        Number of orbitals.
    nelec : tuple
        Number of electrons.
    """
    warn("Sparse Hamiltonian format is deprecated. Use dense format and GPU acceleration instead.",DeprecationWarning)
    enuc, dims, hcore, real_ints = read_common_input(filename)
    chunks = dims[2]
    nmo = dims[3]
    nalpha = dims[4]
    nbeta = dims[5]
    nchol = dims[7]
    if get_chol:
        chol_vecs = read_cholesky(filename, real_ints=real_ints)
    else:
        chol_vecs = None
    return (hcore, chol_vecs, enuc, int(nmo), (int(nalpha), int(nbeta)))

def read_cholesky(filename:str, full=True, ichunk=None, real_ints=False):
    r"""
    Read in Cholesky integrals from internal HDF5 format.

    Parameters
    ----------
    filename : str
        File containing integrals in internal format (HDF5, \*.h5).
    full : bool
        If True, return full Cholesky matrix. Optional. Default: True.
    ichunk : int
        Index of Cholesky chunk to read. Optional. Default: None.
    real_ints : bool
        If True, read real integrals. Optional. Default: False.

    Returns
    -------
    chol_vecs : :class:`scipy.sparse.csr_array`
        Cholesky vectors.
    """
    with h5.File(filename, 'r') as fh5:
        if 'Hamiltonian/Factorized/block_sizes' not in fh5:
            raise RuntimeError(f"  no dataset called 'Hamiltonian/Factorized/block_sizes' in {filename}")
        block_sizes = fh5['Hamiltonian/Factorized/block_sizes'][...]
        nval = sum(block_sizes)
        s = 0
        dims = fh5['Hamiltonian/dims'][:]
        nmo = dims[3]
        nchol = dims[-1]
        if full:
            if real_ints:
                vals = np.zeros(nval, dtype=np.float64)
            else:
                vals = np.zeros(nval, dtype=np.complex128)
            row_ix = np.zeros(nval, dtype=np.int32)
            col_ix = np.zeros(nval, dtype=np.int32)
            for ic, bs in enumerate(block_sizes):
                ixs, vchunk = get_chunk(fh5, ic, real_ints)
                row_ix[s:s+bs] = ixs[::2]
                col_ix[s:s+bs] = ixs[1::2]
                vals[s:s+bs] = vchunk
                s += bs
            chol_vecs = scipy.sparse.csr_array((vals, (row_ix, col_ix)),
                                                shape=(nmo*nmo,nchol))
            return chol_vecs
        else:
            return get_chunk(fh5, ichunk, real_ints)


def get_chunk(fh5, ichunk, real_ints):
    """
    Get Cholesky chunk from hdf5 file.

    Parameters
    ----------
    fh5 : :class:`h5py.File`
        HDF5 file handle containing Cholesky vectors.
    ichunk : int
        Index of Cholesky chunk to read.
    real_ints : bool
        If True, read real integrals.

    Returns
    -------
    ixs : :class:`np.ndarray`
        Indices of Cholesky vectors.
    vals : :class:`np.ndarray`
        Cholesky vectors.
    
    Raises
    ------
    RuntimeError : if Cholesky chunk does not exist in file.
    """
    ixs = fh5['Hamiltonian/Factorized/index_%i'%ichunk][:]
    dset_name =  f'Hamiltonian/Factorized/vals_{ichunk}'
    if dset_name in fh5:
        vals = fh5[dset_name][...]
        if real_ints:
            vals = vals.ravel()
        else:
            vals = vals.view(np.complex128).ravel()
        return ixs, vals
    else:
        raise RuntimeError(f"  no dataset called '{dset_name}' in HDF5 file")
        

def check_sym(ikjl, nmo, sym):
    """Check permutational symmetry of integral

    Parameters
    ----------
    ikjl : tuple of ints
        Orbital indices of ERI.
    nmo : int
        Number of orbitals
    sym : int
        Desired permutational symmetry to check.

    Returns
    -------
    sym_allowed : bool
        True if integral is unique from set of equivalent.
    """
    if sym == 1:
        return True
    else:
        i, k, j, l = ikjl
        if sym == 4:
            kilj = (k,i,l,j)
            jlik = (j,l,i,k)
            ljki = (l,j,k,i)
            if (ikjl > jlik) or (ikjl > kilj) or (ikjl > ljki):
                return False
            else:
                return True
        else:
            ik = i + k*nmo
            jl = j + l*nmo
            return (i >= k and j >= l) and ik >= jl

@jit(nopython=True,cache=True)
def h1_spat2spin(h1e_new,h1e_old,Mspatial):
    r"""
    Convert 1-body Hamiltonian from spatial to spinor basis.

    Convert K_{ij} = 1-body Hamiltonian represented in a spatial orbital basis {phi_i(r)} 
        to a spinor basis { :math:`\phi_i(r) \times |sigma \rangle`  } for use in a FCIDUMP file.
        The spinor basis should be ordered with alternating up / down components.

    i.e. :math:`{ \phi_0(r) \times |\uparrow>,  \phi_0(r) \times |\downarrow \rangle, \phi_1(r) x |\uparrow \rangle,  \phi_1(r) x |\downarrow \rangle, ... }`

    Parameters
    ----------
    h1e_new : :class:`np.ndarray`
        New 1-body Hamiltonian in spinor basis.
    h1e_old : :class:`np.ndarray`
        Old 1-body Hamiltonian in spatial basis.
    Mspatial : int
        Number of spatial orbitals.

    Returns
    -------
    h1e_new : :class:`np.ndarray`
        New 1-body Hamiltonian.
    """
    for i in range(Mspatial):
        for j in range(Mspatial):
            h1e_new[i*2,j*2] = h1e_old[i,j]
            h1e_new[i*2 + 1,j*2 + 1] = h1e_old[i,j]
    return h1e_new

@jit(nopython=True,cache=True)
def h2_spat2spin(h2e_new,h2e_old,Mspatial):
    r"""
    Convert 2-body Hamiltonian from spatial to spinor basis.

    Convert :math:`V_{ijkl} = \int dr dr' \phi_i^*(r)\phi_k(r) \frac{1}{|r - r'|} phi_j^*(r') phi_l(r')`
        in a spatial orbital basis :math:`\{\phi_i(r)\}` to a spinor basis :math:`{ \phi_i(r) x |\sigma\rangle  }` for
        use in a FCIDUMP file. The spinor basis should be ordered with alternating up / down
        components. 

    i.e. :math:`{ \phi_0(r) \times |\uparrow>,  \phi_0(r) \times |\downarrow \rangle, \phi_1(r) x |\uparrow \rangle,  \phi_1(r) x |\downarrow \rangle, ... }`

    For :math:`(ik|jl)` Chemist's notation
      i.e. :math:`\langle ij|lk\rangle` in physicists notation.

    .. Warning::
        There should be no matrix elements between opposite spins in :math:`(ik|` or in :math:`|jl)` !

    Parameters
    ----------
    h2e_new : :class:`np.ndarray`
        New 2-body Hamiltonian in spinor basis.
    h2e_old : :class:`np.ndarray`
        Old 2-body Hamiltonian in spatial basis.
    Mspatial : int
        Number of spatial orbitals.
    """
    for i in range(Mspatial):
        for j in range(Mspatial):
            for k in range(Mspatial):
                for l in range(Mspatial):
                    h2e_new[i*2,j*2,k*2,l*2] = h2e_old[i,j,k,l]
                    h2e_new[i*2,j*2+1,k*2,l*2+1] = h2e_old[i,j,k,l]
                    h2e_new[i*2+1,j*2,k*2+1,l*2] = h2e_old[i,j,k,l]
                    h2e_new[i*2+1,j*2+1,k*2+1,l*2+1] = h2e_old[i,j,k,l]
    return h2e_new

def fmt_integral(intg, i, k, j, l, cplx, paren=False):
    """
    Format integral for FCIDUMP.

    FCIDUMP files contain integrals with one matrix element per line.
    Integral lines are formatted as:s

    * real 2-body integrals `  (ij|kl)  i  k  j  l`
    * complex 2-body integrals `  Re[(ij|kl)] Im[(ij|kl)] i k j l`  -OR- `  (Re[(ij|kl)], Im[(ij|kl)]) i k j l`
    * real 1-body integrals `  h_{ik} i k 0 0`
    * complex 1-body integrals `  Re[h_{ik}] Im[h_{ik}] i k 0 0` -OR- `  (Re[h_{ik}], Im[h_{ik}]) i k 0 0`
    * real constant integrals  `  C 0 0 0 0`
    * complex constant integrals `  Re[C] Im[C] 0 0 0 0` -OR- `  (Re[C], Im[C]) 0 0 0 0` 
    
    where the variation in complex-valued integral format depends on which code will be used to read
    the FCIDUMP file.

    Parameters
    ----------
    intg : float|complex
        Integral value.
    i,k,j,l : int
        Orbital indices.
    cplx : bool
        If True, then integrals are printed as complex-valued
    paren : bool
        If True, complex-valued integrals are printed in paranthesis
    """
    if cplx:
        if paren:
            fmt = '  ({: 13.8e}, {: 13.8e}) {:4d}  {:4d}  {:4d}  {:4d}\n'
        else:
            fmt = '  {: 13.8e}    {: 13.8e}  {:4d}  {:4d}  {:4d}  {:4d}\n'
        out = fmt.format(intg.real, intg.imag, i+1, k+1, j+1, l+1)
    else:
        fmt = '  {: 13.8e}    {:4d}  {:4d}  {:4d}  {:4d}\n'
        out = fmt.format(intg.real, i+1, k+1, j+1, l+1)
    return out


def write_fcidump(filename, hcore, chol, enuc, nmo, nelec, tol=1e-8, ctol=1e-12,
                  sym=1, cplx=True, paren=False, chol_is_eri=False, use_spinor=False):
    """Write FCIDUMP based from Cholesky factorised integrals.

    Parameters
    ----------
    filename : string
        Filename to write FCIDUMP to.
    hcore : :class:`np.ndarray`
        One-body hamiltonian.
    chol : :class:`np.ndarray`
        Cholesky matrix L[ik,n] or chemist's ERI (ik|jl) if chol_is_eri
    enuc : float
        Nuclear repulsion energy.
    nmo : int
        Total number of MOs.
    nelec : tuple
        Number of alpha and beta electrons.
    tol : float
        Only print eris above tol. Optional. Default 1e-8.
    sym : int
        Controls whether to only print symmetry inequivalent ERIS.
        Optional. Default 1, i.e. print everything.
    cplx : bool
        Write in complex format. Optional. Default : True.
    paren : bool
        Write complex numbers in parenthesis.
    chol_is_eri: bool
        Write chemist's ERI directly instead of using its Cholesky factorization
    use_spinor: bool
        Convret to a spinor basis before writting FCIDUMP
    """

    if use_spinor and cplx == False:
        print(
            "Requested real-valued integrals for spinor basis: "
            "using complex-valued integrals"
        )
        cplx = True
    
    if use_spinor and sym > 1:
        # Note: check_sym assumes a spatial orbital basis, not a spin orbital basis
        # TODO: add an option to check_sym to check symmetry for spinor basis as well.
        print(
            "write_fcidump not implemented for use_spinor with sym > 1: "
            "using sym = 1"
        )
        sym = 1
    
    if cplx and sym > 4:
        print("Warning: Requested 8-fold permutational "
              "symmetry with complex integrals.")
        cplx = False

    # Generate M_{(ik),(lj)} = (ik|jl)
    if chol_is_eri:  # chemist -> hermitian order
      # physicist <ij|kl> = chemist (ik|jl) = hermitian {(ik),(lj)}
      eris = chol.transpose((0,1,3,2))
    elif isinstance(chol,scipy.sparse.csr_array):
      eris = chol.dot(chol.conj().T).toarray().reshape((nmo,nmo,nmo,nmo))
    else:
      eris = chol.dot(chol.conj().T).reshape((nmo,nmo,nmo,nmo)) 
    
    if use_spinor:
        hcore = h1_spat2spin(
            h1e_new=np.zeros((2*nmo, 2*nmo), dtype=np.complex128),
            h1e_old=hcore,
            Mspatial=nmo
        )
        eris = h2_spat2spin(
            h2e_new=np.zeros((2*nmo, 2*nmo, 2*nmo, 2*nmo),dtype=np.complex128),
            h2e_old=eris,
            Mspatial=nmo
        )
        nmo = 2*nmo

    header = fcidump_header(sum(nelec), nmo, nelec[0]-nelec[1])
    #TODO: instead of generating all possible i,k,l,j coordinates, 
    #         just loop over the allowed values.
    with open(filename, 'w') as f:
        f.write(header)
        for i, k, l, j in product(range(nmo), repeat=4):
           sym_allowed = check_sym((i,k,j,l), nmo, sym)
           if abs(eris[i,k,l,j]) > tol:
             if sym_allowed:
               if not cplx:  # check imaginary part
                 if abs(eris[i,k,l,j].imag > ctol):
                   msg = "# Found complex integrals with cplx==False."
                   raise RuntimeError(msg)
               out = fmt_integral(eris[i,k,l,j], i, k, j, l,
                                  cplx, paren=paren)
               f.write(out)
             else:
               msg = str((i, j, k, l)) + ' not allowed by %d-fold symmetry' % sym
               #raise RuntimeError(msg)  # Cholesky can produce forbidden entries?

        for i in range(0,nmo):
            for j in range(0,i+1):
                if abs(hcore[i,j]) > tol:
                    out = fmt_integral(hcore[i,j], i, j, -1, -1,
                                       cplx, paren=paren)
                    f.write(out)

        f.write(fmt_integral(enuc+0j,-1,-1,-1,-1, cplx, paren=paren))


def read_cholesky_kpoint_coqui(filename, get_chol=True):
    r"""
    Read the CoQuí-format k-point factorized Hamiltonian from an HDF5 file.

    This reader follows the layout used by SAFIRE's CoQuí HDF5 files, using
    groups `/System` for one-body terms and metadata and `/Interaction` for
    two-body Cholesky factors. Complex data are stored as real-imag pairs on
    the last axis and are converted via `from_complex`.

    Data sources and meanings:
    - `/System` (attributes):
      - `number_of_bands` (int): number of bands per k-point (nmo per k).
      - `number_of_elec` (float|int): total electron count; split evenly into
        `(nalpha, nbeta)` assuming a closed shell.
      - `nuclear_energy` (float): nuclear-nuclear repulsion.
      - `frozen_core_energy` (float, optional): frozen-core energy; added to `enuc`.
      - `madelung_constant` (float, optional): electron self-interaction factor;
        subtracted as `madelung_constant * (nalpha + nbeta)`.
    - `/System/BZ` (group):
      - `number_of_kpoints` (attr, int): number of k-points `nkp`.
      - `qk_to_k2` (dataset, shape = `[nkp, nkp]`): map `(Q, K) -> K'`.
      - `qminus` (dataset, shape = `[nkp]`): map `Q -> -Q` for time-reversal symmetry.
      - `kp_to_ibz` (dataset, shape = `[nkp]`, optional): map full BZ k to IBZ index.
      - `kp_trev_pair` (dataset, shape = `[nkp]`, optional): time-reversal flags (>0 means conj).
    - `/System/H0` (dataset, shape = `[nspins, nkpts_or_ibz, nbnd, nbnd, 2]`):
      one-body Hamiltonian per spin and k-point (complex in last dim). If stored
      over IBZ, it is unfolded using `kp_to_ibz` and `kp_trev_pair`.
    - `/Interaction/Vq{Q}` (dataset per Q, shape = `[nchol, 1, nkp, nbnd, nbnd, 2]`):
      Cholesky factors of the two-body term. Normalized by `1/sqrt(nkp)`.

    Parameters
    ----------
    filename : str
        Path to the CoQuí HDF5 Hamiltonian file.
    get_chol : bool, optional
        If True, read and return the Cholesky vectors (default True).

    Returns
    -------
    hcore : list[np.ndarray]
        List of one-body Hamiltonians for each k-point. Each element has shape
        `(nbnd, nbnd)` and dtype `complex128` (first spin component used).
    chol_vecs : list[np.ndarray] or None
        If `get_chol` is True, a list over Q of arrays of shape
        `(nkp, nbnd*nbnd*nchol)` with dtype `complex128`, storing flattened
        `L[K][i,k,n]` per k-point. Otherwise `None`.
    enuc : float
        Effective nuclear energy (nuclear + frozen-core, minus Madelung term).
    nmo_tot : int
        Total number of orbitals over all k-points, `nbnd * nkp`.
    nelec : tuple[int, int]
        `(nalpha, nbeta)` electron counts, closed-shell split if only total given.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`, all entries equal to `nbnd`.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`, with time-reversal applied.
    minus_k : np.ndarray
        Time-reversal partner map `Q -> -Q`; shape `(nkp,)`.
    """
    with h5.File(filename, 'r') as fh5:
        sysgrp = fh5['System']
        enuc = sysgrp.attrs.get('nuclear_energy', 0.0)
        enuc += sysgrp.attrs.get('frozen_core_energy', 0.0)
        nbnd = int(sysgrp.attrs['number_of_bands'])
        try:
            nelec_total = sysgrp.attrs['number_of_elec']
            nalpha = nbeta = int(nelec_total // 2)
        except KeyError:
            nalpha = nbeta = 0
        enuc -= sysgrp.attrs.get('madelung_constant', 0.0) * (nalpha + nbeta)
        bz = fh5['System/BZ']
        nkp = int(bz.attrs['number_of_kpoints'])
        qk_k2 = bz['qk_to_k2'][:]
        minus_k = bz['qminus'][:]
        nmo_pk = np.full(nkp, nbnd, dtype=np.int32)
        nmo_tot = int(nbnd * nkp)
        h0 = fh5['System/H0'][:]
        nspins_h1, nkpts_h1 = h0.shape[0], h0.shape[1]
        unfold_ibz = nkpts_h1 != nkp
        hcore = []
        if unfold_ibz:
            kp_to_ibz = bz['kp_to_ibz'][:]
            kp_trev = bz['kp_trev_pair'][:]
            h0c = from_complex(h0, (nspins_h1, nkpts_h1, nbnd, nbnd))
            for k in range(nkp):
                hk = h0c[0, kp_to_ibz[k]].copy()
                if kp_trev[k] > 0:
                    hk = np.conj(hk)
                hcore.append(hk)
        else:
            h0c = from_complex(h0, (nspins_h1, nkp, nbnd, nbnd))
            for k in range(nkp):
                hcore.append(h0c[0, k].copy())
        nchol_pk = np.zeros(nkp, dtype=np.int32)
        for q in range(nkp):
            nchol_pk[q] = fh5[f'Interaction/Vq{q}'].shape[0]
        for q in range(nkp):
            j = minus_k[q]
            if j < q:
                nchol_pk[q] = nchol_pk[j]
        chol_vecs = None
        if get_chol:
            chol_vecs = [get_kpoint_chol_coqui(filename, nchol_pk, minus_k, q, qk_k2, nmo_pk) for q in range(nkp)]
    return (hcore, chol_vecs, enuc, nmo_tot, (int(nalpha), int(nbeta)), nmo_pk, qk_k2, nchol_pk, minus_k)

def get_kpoint_chol_coqui(filename, nchol_pk, minus_k, i, qk_k2, nmo_pk):
    r"""
    Read CoQuí-format Cholesky vectors for a given Q-vector.

    This function loads `/Interaction/Vq{Q}` for `Q = min(i, -i)` and returns
    the per-k-point Cholesky factors in a flattened `(nkp, nbnd*nbnd*nchol)`
    layout. For `Q` with `-Q < Q` (time-reversal pairing), the data are remapped
    using `qk_k2[i, k]` to select `K'` and conjugate-transposed to obtain the
    `Q` partner.

    Parameters
    ----------
    filename : str
        Path to the CoQuí HDF5 Hamiltonian file.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`.
    minus_k : np.ndarray
        Time-reversal partner map; shape `(nkp,)`.
    i : int
        Q-vector index to read.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`.

    Returns
    -------
    Lk : np.ndarray
        Cholesky vectors for Q-vector `i`, shape `(nkp, nbnd*nbnd*nchol)`, dtype `complex128`.
    """
    with h5.File(filename, 'r') as fh5:
        j = minus_k[i]
        qread = min(i, j)
        Vq_raw = fh5[f'Interaction/Vq{qread}'][:]
        nchol = Vq_raw.shape[0]
        nkp = Vq_raw.shape[2]
        nmo = int(nmo_pk[0])
        Vq = from_complex(Vq_raw, (nchol, 1, nkp, nmo, nmo)) / np.sqrt(nkp)
        Lk = np.zeros((nkp, nmo * nmo * nchol), dtype=np.complex128)
        if j < i:
            nmo_i = int(nmo_pk[i])
            for k1 in range(nkp):
                k2 = qk_k2[i, k1]
                v = Vq[:, 0, k2, :nmo_i, :nmo_i]
                Lk[k1] = np.conj(np.transpose(v, (2, 1, 0))).ravel()
        else:
            for K in range(nkp):
                v = Vq[:, 0, K, :, :]
                Lk[K] = np.transpose(v, (1, 2, 0)).ravel()
    return Lk

def read_cholesky_kpoint(filename, get_chol=True):
    r"""Read SAFIRE internal k-point factorized Hamiltonian (standard format).

    Reads datasets from the `Hamiltonian` group in the internal format and
    returns the one-body matrices and (optionally) per-Q Cholesky vectors.

    Reads
    -----
    - `Hamiltonian/dims` (array, len=8): overall dimensions. Used entries:
      - `dims[2] = nkp`: number of k-points.
      - `dims[3] = nmo_tot`: total orbitals across all k-points.
      - `dims[4] = nalpha`, `dims[5] = nbeta`: electron counts.
    - `Hamiltonian/NMOPerKP` (dataset, shape = `[nkp]`): orbitals per k-point.
    - `Hamiltonian/NCholPerKP` (dataset, shape = `[nkp]`): Cholesky count per Q.
    - `Hamiltonian/QKTok2` (dataset, shape = `[nkp, nkp]`): map `(Q, K) -> K'`.
    - `Hamiltonian/MinusK` (dataset, shape = `[nkp]`): time-reversal partner map.
    - `Hamiltonian/H1_kp{i}` (dataset per k, complex128 memory): one-body H for k.
    - `Hamiltonian/KPFactorized/L{Q}` (dataset per Q, complex128 memory): Cholesky
      factors for Q; remapped/conjugated for `-Q` via the time-reversal trick.

    Parameters
    ----------
    filename : str
        Path to the internal-format HDF5 file.
    get_chol : bool, optional
        If True, read and return per-Q Cholesky vectors (default True).

    Returns
    -------
    hcore : list[np.ndarray]
        List of one-body Hamiltonians per k-point; each `(nmo_k, nmo_k)` complex.
    chol_vecs : list[np.ndarray] or None
        If `get_chol` is True, list over Q of arrays with shape
        `(nkp, nmo_k*nmo_k*nchol_Q)` containing flattened `L[K][i,k,n]` per K.
        Otherwise `None`.
    enuc : float
        Core (nuclear) energy contribution read from `Hamiltonian/Energies`.
    nmo_tot : int
        Total number of orbitals across all k-points.
    nelec : tuple[int, int]
        `(nalpha, nbeta)` electron counts.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`, after time-reversal tie-in.
    minus_k : np.ndarray
        Time-reversal partner map `Q -> -Q`; shape `(nkp,)`.
    """
    enuc, dims, hcore, real_ints = read_common_input(filename, get_hcore=False)
    with h5.File(filename, 'r') as fh5:
        nmo_pk = fh5['Hamiltonian/NMOPerKP'][:]
        nchol_pk = fh5['Hamiltonian/NCholPerKP'][:]
        qk_k2 = fh5['Hamiltonian/QKTok2'][:]
        minus_k = fh5['Hamiltonian/MinusK'][:]
        hcore = []
        nkp = dims[2]
        nmo_tot = dims[3]
        nalpha = dims[4]
        nbeta = dims[5]
        if nmo_pk is None:
            raise KeyError("Could not read NMOPerKP dataset.")
        for i in range(0, nkp):
            hk = fh5['Hamiltonian/H1_kp{}'.format(i)][:]
            if hk is None:
                raise KeyError("Could not read one-body hamiltonian.")
            nmo = nmo_pk[i]
            hcore.append(hk.view(np.complex128).reshape(nmo,nmo))
        chol_vecs = []
        if nmo_pk is None:
            raise KeyError("Error nmo_pk dataset does not exist.")
        nmo_max = max(nmo_pk)
    if minus_k is None:
        raise KeyError("Error MinusK dataset does not exist.")
    if nchol_pk is None:
        raise KeyError("Error NCholPerKP dataset does not exist.")
    # unpack chol
    for i, nc in enumerate(nchol_pk):
      j = minus_k[i]
      if j<i:  # apply (Q, -Q) trick
        nchol_pk[i] = nchol_pk[j]
    if get_chol:
        for i in range(0, nkp):
            chol_vecs.append(get_kpoint_chol(filename, nchol_pk, minus_k, i, qk_k2, nmo_pk))
    else:
        chol_vecs = None

    return (hcore, chol_vecs, enuc, int(nmo_tot), (int(nalpha), int(nbeta)),
            nmo_pk, qk_k2, nchol_pk, minus_k)

def get_kpoint_chol(filename, nchol_pk, minus_k, i, qk_k2, nmo_pk):
    r"""
    Read standard-format Cholesky vectors for Q = i with time-reversal handling.

    Loads `Hamiltonian/KPFactorized/L{min(i, -i)}` and returns per-k-point
    Cholesky factors in a flattened `(nkp, nmo_i*nmo_i*nchol)` layout. If `-i < i`
    (time-reversal pairing), applies the `(Q, -Q)` trick: remap `K' = qk_k2[i, K]`
    and conjugate-transpose from the stored partner to obtain the `Q` data.

    Parameters
    ----------
    filename : str
        Path to the internal-format HDF5 file.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`.
    minus_k : np.ndarray
        Time-reversal partner map; shape `(nkp,)`.
    i : int
        Q-vector index to read.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`.

    Returns
    -------
    Lk : np.ndarray
        Cholesky vectors for Q `i`, shape `(nkp, nmo_i*nmo_i*nchol)`, complex128.
    """
    with h5.File(filename, 'r') as fh5:
        j = minus_k[i]
        Lk = fh5['Hamiltonian/KPFactorized/L{}'.format(min(i, j))][:]
        if Lk is None:
            msg = "Could not read Cholesky kpoint %d, with -Q %d." % (i, j)
            raise TypeError(msg)
        Lk = Lk.view(np.complex128)[:,:,0]
        if j<i:  # apply (Q, -Q) trick
            nchol = nchol_pk[j]
            # remap k-Q, then conjugate-transpose
            nmo = nmo_pk[i]
            assert nmo_pk[j] == nmo
            Lk1 = Lk.copy()
            for k1, k2 in enumerate(qk_k2[i]):
              cmat = Lk[k2].reshape(nmo, nmo, nchol)
              Lk1[k1] = cmat.transpose(1, 0, 2).conj().ravel()
            Lk = Lk1
    return Lk


def read_cholesky_kpoint_coqui(filename, get_chol=True):
    r"""Read in integrals from internal hdf5 format. kpoint dependent case.

    Parameters
    ----------
    filename : str
        Path to the internal-format HDF5 file.
    get_chol : bool, optional
        If True, read and return per-Q Cholesky vectors (default True).

    Returns
    -------
    hcore : list[np.ndarray]
        List of one-body Hamiltonians per k-point; each `(nmo_k, nmo_k)` complex.
    chol_vecs : list[np.ndarray] or None
        If `get_chol` is True, list over Q of arrays with shape
        `(nkp, nmo_k*nmo_k*nchol_Q)` containing flattened `L[K][i,k,n]` per K.
        Otherwise `None`.
    enuc : float
        Core (nuclear) energy contribution read from `Hamiltonian/Energies`.
    nmo_tot : int
        Total number of orbitals across all k-points.
    nelec : tuple[int, int]
        `(nalpha, nbeta)` electron counts.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`, after time-reversal tie-in.
    minus_k : np.ndarray
        Time-reversal partner map `Q -> -Q`; shape `(nkp,)`.
    """
    enuc, dims, hcore, real_ints = read_common_input(filename, get_hcore=False)
    with h5.File(filename, 'r') as fh5:
        nmo_pk = fh5['Hamiltonian/NMOPerKP'][:]
        nchol_pk = fh5['Hamiltonian/NCholPerKP'][:]
        qk_k2 = fh5['Hamiltonian/QKTok2'][:]
        minus_k = fh5['Hamiltonian/MinusK'][:]
        hcore = []
        nkp = dims[2]
        nmo_tot = dims[3]
        nalpha = dims[4]
        nbeta = dims[5]
        if nmo_pk is None:
            raise KeyError("Could not read NMOPerKP dataset.")
        for i in range(0, nkp):
            hk = fh5['Hamiltonian/H1_kp{}'.format(i)][:]
            if hk is None:
                raise KeyError("Could not read one-body hamiltonian.")
            nmo = nmo_pk[i]
            hcore.append(hk.view(np.complex128).reshape(nmo,nmo))
        chol_vecs = []
        if nmo_pk is None:
            raise KeyError("Error nmo_pk dataset does not exist.")
        nmo_max = max(nmo_pk)
    if minus_k is None:
        raise KeyError("Error MinusK dataset does not exist.")
    if nchol_pk is None:
        raise KeyError("Error NCholPerKP dataset does not exist.")
    # unpack chol
    for i, nc in enumerate(nchol_pk):
      j = minus_k[i]
      if j<i:  # apply (Q, -Q) trick
        nchol_pk[i] = nchol_pk[j]
    if get_chol:
        for i in range(0, nkp):
            chol_vecs.append(get_kpoint_chol(filename, nchol_pk, minus_k, i, qk_k2, nmo_pk))
    else:
        chol_vecs = None

    return (hcore, chol_vecs, enuc, int(nmo_tot), (int(nalpha), int(nbeta)),
            nmo_pk, qk_k2, nchol_pk, minus_k)


def get_kpoint_chol_coqui(filename, nchol_pk, minus_k, i, qk_k2, nmo_pk):
    r"""
    Read standard-format Cholesky vectors for Q = i with time-reversal handling.

    Loads `Hamiltonian/KPFactorized/L{min(i, -i)}` and returns per-k-point
    Cholesky factors in a flattened `(nkp, nmo_i*nmo_i*nchol)` layout. If `-i < i`
    (time-reversal pairing), applies the `(Q, -Q)` trick: remap `K' = qk_k2[i, K]`
    and conjugate-transpose from the stored partner to obtain the `Q` data.

    Parameters
    ----------
    filename : str
        Path to the internal-format HDF5 file.
    nchol_pk : np.ndarray
        Number of Cholesky vectors per Q; shape `(nkp,)`.
    minus_k : np.ndarray
        Time-reversal partner map; shape `(nkp,)`.
    i : int
        Q-vector index to read.
    qk_k2 : np.ndarray
        Momentum mapping `(Q, K) -> K'`; shape `(nkp, nkp)`.
    nmo_pk : np.ndarray
        Orbitals per k-point; shape `(nkp,)`.

    Returns
    -------
    Lk : np.ndarray
        Cholesky vectors for Q `i`, shape `(nkp, nmo_i*nmo_i*nchol)`, complex128.
    """
    with h5.File(filename, 'r') as fh5:
        j = minus_k[i]
        Lk = fh5['Hamiltonian/KPFactorized/L{}'.format(min(i, j))][:]
        if Lk is None:
            msg = "Could not read Cholesky kpoint %d, with -Q %d." % (i, j)
            raise TypeError(msg)
        Lk = Lk.view(np.complex128)[:,:,0]
        if j<i:  # apply (Q, -Q) trick
            nchol = nchol_pk[j]
            # remap k-Q, then conjugate-transpose
            nmo = nmo_pk[i]
            assert nmo_pk[j] == nmo
            Lk1 = Lk.copy()
            for k1, k2 in enumerate(qk_k2[i]):
              cmat = Lk[k2].reshape(nmo, nmo, nchol)
              Lk1[k1] = cmat.transpose(1, 0, 2).conj().ravel()
            Lk = Lk1
    return Lk



def read_dense(filename,walker_type=None):
    r"""Read in integrals from internal hdf5 format.

    Parameters
    ----------
    filename : string
        File containing integrals (\*.h5)
    walker_type : int
        integer describing the walker type:
          (1) : 'CLOSED' - i.e. closed-shell and RHF-like
          (2) : 'COLLINEAR' - i.e. open-shell and/or UHF-like
          (3) : 'NONCOLLINEAR' - i.e. GHF-like

    Returns
    -------
    hcore : :class:`np.ndarray`
        One-body part of the Hamiltonian.
    chol_vecs : :class:`np.ndarray`
        Two-electron integrals. Shape: [nmo*nmo, nchol]
    ecore : float
        Core contribution to the total energy.

    Raises
    ------
    RuntimeError : if Cholesky vectors are not present in
        the file called 'filename'
    """
    CHOLESKY_DATASET = 'Hamiltonian/DenseFactorized/L'
    real_ints = False
    enuc, dims, hcore, real_ints = read_common_input(filename)
    if walker_type == 3:
        npol = 2
    else:
        npol = 1
    with h5.File(filename, 'r') as fh5:
        nelec = (dims[4],dims[5])
        nmo = dims[3]
        nchol = dims[-1]
        if CHOLESKY_DATASET in fh5:
            chol = fh5[CHOLESKY_DATASET][...]
            if not real_ints:
                chol = from_complex(
                    data=chol,
                    shape=(nmo*nmo,nchol)
                    )
            assert chol.shape == (nmo*nmo, nchol)
            assert hcore.shape == (npol*nmo, npol*nmo)
            return hcore, chol, enuc, nelec # TODO: added 'nelec' to the return - this will (trivially) break some stuff!
        else:
            raise RuntimeError(f"  no dataset called '{CHOLESKY_DATASET}' in {filename}")


def read_common_input(filename, get_hcore=True):
    r"""Read common input from internal format (\*.h5).
    
    Parameters
    ----------
    filename : string
        File containing integrals in internal format (\*.h5).
    get_hcore : bool
        If True, read one-body Hamiltonian. Optional. Default: True.

    Returns
    -------
    enuc : float
        Nuclear repulsion energy.
    dims : :class:`np.ndarray`
        Dimensions of the Hamiltonian.
    hcore : :class:`np.ndarray`
        One-body part of the Hamiltonian.
    real_ints : bool
        True if integrals are real, False if complex
    """
    with h5.File(filename, 'r') as fh5:
        try:
            enuc = fh5['Hamiltonian/Energies'][:][0]
        except:
            print(" Error reading Hamiltonian/Energies dataset.")
            enuc = None
        try:
            dims = fh5['Hamiltonian/dims'][:]
        except:
            dims = None
        assert dims is not None, "Error reading Hamiltonian/dims data set."
        assert len(dims) == 8, "Hamiltonian dims data set has incorrect length."
        nmo = dims[3]
        real_ints = False
        if get_hcore:
            try:
                hcore = from_complex(fh5['Hamiltonian/hcore'][:], (nmo,nmo))
            except ValueError:
                hcore = fh5['Hamiltonian/hcore'][:]
                real_ints = True
            except KeyError:
                hcore = None
                pass
            if hcore is None:
                try:
                    # old sparse format only for complex.
                    hcore = fh5['Hamiltonian/H1'][:].view(np.complex128).ravel()
                    idx = fh5['Hamiltonian/H1_indx'][:]
                    row_ix = idx[::2]
                    col_ix = idx[1::2]
                    hcore = scipy.sparse.csr_array((hcore, (row_ix, col_ix))).toarray()
                    hcore = np.tril(hcore, -1) + np.tril(hcore, 0).conj().T
                except:
                    hcore = None
                    print("Error reading Hamiltonian/hcore data set.")
            try:
                complex_ints = bool(fh5['Hamiltonian/ComplexIntegrals'][0])
            except KeyError:
                complex_ints = None

            if complex_ints is not None:
                if hcore is not None:
                    hc_type = hcore.dtype
                else:
                    hc_type = type(hcore)
                msg = ("ComplexIntegrals flag conflicts with integral data type. "
                       "dtype = {:} ComplexIntegrals = {:}.".format(hc_type, complex_ints))
                assert real_ints ^ complex_ints, msg
        else:
            hcore = None
    return enuc, dims, hcore, real_ints


def write_fcidump_kpoint(filename, hcore, chol, enuc, nmo_tot, nelec,
                         nmo_pk, nchol_pk, qk_k2, tol=1e-8, sym=1,
                         paren=False, cplx=True, ctol=1e-12, use_spinor=False):
    """Write FCIDUMP based from Cholesky factorised integrals.

    Parameters
    ----------
    filename : string
        Filename to write FCIDUMP to.
    hcore : list
        One-body hamiltonian.
    chol : list
        Cholesky matrices L[Q][k_i][i,k]
    enuc : float
        Nuclear repulsion energy.
    nmo_tot : int
        Total number of MOs.
    nelec : tuple
        Number of alpha and beta electrons.
    nmo_pk : :class:`np.ndarray`
        Number of MOs per kpoint.
    nchol_pk : :class:`np.ndarray`
        Number of Cholesky vectors per kpoint.
    qk_k2 : :class:`np.ndarray`
        Array mapping (q,k) pair to kpoint: Q = k_i - k_k + G.
        qk_k2[iQ,ik_i] = i_kk.
    tol : float
        Only print eris above tol. Optional. Default 1e-8.
    sym : int
        Controls whether to only print symmetry inequivalent ERIS.
        Optional. Default 1, i.e. print everything.
    paren : bool
        Write complex numbers in parenthesis.
    use_spinor: bool
        Convret to a spinor basis before writting FCIDUMP
    """
    if use_spinor:
        raise NotImplementedError(
            "Conversion to spinor basis not yet implemented. Instead, run"
            "\n 1. `afqmc_to_fcidump` keeping the spatial basis" 
            "\n 2. `fcidump_spat2spin` to convert from spatial to spin"
            )

    header = fcidump_header(sum(nelec), nmo_tot, nelec[0]-nelec[1])
    nkp = len(nmo_pk)
    offsets = np.cumsum(nmo_pk)-nmo_pk[0]

    with open(filename, 'w') as f:
        f.write(header)
        for iq, lq_vec in enumerate(chol):
            nchol = nchol_pk[iq]
            lq = lq_vec.reshape(nkp, -1, nchol)
            for ki in range(nkp):
                for kl in range(nkp):
                    # decompress Cholesky vecs. to physicist's vijkl = <ij|kl>
                    #  i.e. c^\dag_i c^\dag_j c_l c_k
                    eri = np.dot(lq[ki], lq[kl].conj().T)
                    #  stored in hermitian order (i,k)|(l,j)
                    if not cplx:
                      if (abs(eri.imag).max() > ctol):
                        msg = "# Found complex integrals with cplx==False."
                        raise RuntimeError(msg)
                    ik = 0
                    for i in range(0, nmo_pk[ki]):
                        kk = qk_k2[iq,ki]
                        I = i + offsets[ki]
                        for k in range(0, nmo_pk[kk]):
                            kj = qk_k2[iq,kl]
                            K = k + offsets[kk]
                            lj = 0
                            for l in range(0, nmo_pk[kl]):
                                L = l + offsets[kl]
                                for j in range(0, nmo_pk[kj]):
                                    J = j + offsets[kj]
                                    sym_allowed = check_sym((I,K,J,L),
                                                            nmo_tot, sym)
                                    if abs(eri[ik,lj]) > tol:
                                      if sym_allowed:
                                        out = fmt_integral(eri[ik,lj],
                                                           I, K, J, L,
                                                           cplx, paren=paren)
                                        f.write(out)
                                      else:
                                        msg = str((I, J, K, L)) + ' not allowed by %d-fold symmetry' % sym
                                        #raise RuntimeError(msg)  # Cholesky can produce forbidden entries?
                                    lj += 1
                            ik += 1

        for ik, hk in enumerate(hcore):
            for i in range(nmo_pk[ik]):
                I = i + offsets[ik]
                for j in range(nmo_pk[ik]):
                    J = j + offsets[ik]
                    if I >= J and abs(hk[i,j]) > tol:
                        out = fmt_integral(hk[i,j], I, J, -1, -1,
                                           cplx, paren=paren)
                        f.write(out)

        out = fmt_integral(enuc+0j, -1, -1, -1, -1, cplx, paren=paren)
        f.write(out)

def sparse_to_dense(sparse_file, dense_file, real_chol=False):
    """
    Convert sparse Hamiltonian to dense format.

    Parameters
    ----------
    sparse_file : string
        File containing sparse Hamiltonian.
    dense_file : string
        File to write dense Hamiltonian to.
    real_chol : bool
        If True, write real Cholesky vectors. Optional. Default: False.
    """
    hcore, chol, enuc, nmo, nelec = read_sparse(sparse_file,
                                                        get_chol=False)
    with h5.File(dense_file, 'w') as fh5:
        fh5['Hamiltonian/Energies'] = np.array([enuc,0])
        with h5.File(sparse_file, 'r') as sph5:
            block_sizes = sph5['Hamiltonian/Factorized/block_sizes'][:]
            dims = sph5['Hamiltonian/dims'][:]
            nchol = dims[-1]
            if real_chol:
                fh5['Hamiltonian/hcore'] = hcore.real
                shape = (nmo*nmo,nchol)
            else:
                fh5['Hamiltonian/hcore'] = to_complex(hcore.astype(np.complex128))
                shape = (nmo*nmo,nchol,2)
            fh5['Hamiltonian/dims'] = dims
            chol_dset = fh5.create_dataset('Hamiltonian/DenseFactorized/L',
                                           shape,
                                           dtype=np.float64)
            s = 0
            for ic, bs in enumerate(block_sizes):
                ixs, vchunk = get_chunk(sph5, ic, real_chol)
                row_ix, col_ix = ixs[::2], ixs[1::2]
                sort = np.argsort(row_ix)
                row_ix = row_ix[sort]
                col_ix = col_ix[sort]
                vchunk = vchunk[sort]
                rows = np.unique(row_ix)
                for r in rows:
                    # H5PY array slicing is restrictive.
                    # index has to be in increasing order
                    rix = row_ix == r
                    cr = col_ix[rix]
                    vr = vchunk[rix]
                    sort = np.argsort(cr)
                    if real_chol:
                        chol_dset[r,cr[sort]] = np.real(vr[sort])
                    else:
                        chol_dset[r,cr[sort]] = to_complex(vr[sort])

def kpoint_to_sparse(kp_file, sp_file, real_chol=False,
                     thresh=1e-8):
    """
    Convert k-point Hamiltonian to sparse format.

    Parameters
    ----------
    kp_file : string
        File containing k-point Hamiltonian.
    sp_file : string
        File to write sparse Hamiltonian to.
    real_chol : bool
        If True, write real Cholesky vectors. Optional. Default: False.
    thresh : float
        Threshold for sparse integral values. Intagrals with absolute value less then 
        thresh are considered 0.0 Optional. Default: 1e-8.
    """
    warn("Sparse format is deprecated. Use k-point factorized Hamiltonian format instead.", DeprecationWarning)
    hamil = read_hamiltonian(kp_file, get_chol=False)
    hcore = hamil['hcore']
    enuc = hamil['enuc']
    nelec = hamil['nelec']
    nmo = hamil['nmo']
    nmo_pk = hamil['nmo_pk']
    minus_k = hamil['minus_k']
    qk_k2 = hamil['qk_k2']
    nkp = len(hcore)
    # 1. Extend hcore to block diagonal.
    hcore = scipy.linalg.block_diag(*hcore)
    nchol_pk = hamil['nchol_pk']
    nchol = sum(nchol_pk)
    orb_offset = np.zeros(nkp, dtype=np.int32)
    chol_offset = np.zeros(nkp, dtype=np.int32)
    write_sparse_basic(sp_file, hcore, enuc, nelec, real_chol=real_chol)
    for i in range(1, nkp):
        orb_offset[i] = orb_offset[i-1] + nmo_pk[i-1]
        chol_offset[i] = chol_offset[i-1] + nchol_pk[i-1]
    block_sizes = []
    for iq in range(nkp):
        s = chol_offset[iq]
        e = s + nchol_pk[iq]
        lq = get_kpoint_chol(kp_file, nchol_pk, minus_k, iq, qk_k2, nmo_pk)
        vals = []
        ixs = []
        for ki in range(nkp):
            for i in range(0, nmo_pk[ki]):
                kk = qk_k2[iq,ki]
                I = i + orb_offset[ki]
                for k in range(0, nmo_pk[kk]):
                    K = k + orb_offset[kk]
                    for nc in range(0,nchol_pk[iq]):
                        ll = lq[ki,(i*nmo_pk[ki]+k)*nchol_pk[iq]+nc]
                        if abs(ll) > thresh:
                            vals.append(ll)
                            ixs.append([I*nmo+K,s+nc])
        block_sizes.append(len(vals))
        write_sparse_chol_chunk(np.array(ixs).ravel(), vals, iq, filename=sp_file)

    with h5.File(sp_file, 'a') as fh5:
        fh5['Hamiltonian/Factorized/block_sizes'] = block_sizes
        fh5['Hamiltonian/dims'] = np.array([0, sum(block_sizes),
                                               nkp, nmo, nelec[0], nelec[1], 0,
                                               nchol])
