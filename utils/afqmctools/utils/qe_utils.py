# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Utilities for interacting with Quantum Espresso

Functionality:
1. Read orbitals and DFT density from PWSCF in HDF5 format
"""
from warnings import warn
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import h5py as h5

from numba import jit # TODO: numba the brute force loops

from afqmctools.utils.chemistry import atomic_symbol_to_num


@jit#(nopython=True)
def _make_g_grid(Nx,Ny,Nz,gvecs):
    rec_vecs = np.zeros((Nx,Ny,Nz,3),dtype=np.complex128)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                rec_vecs[i,j,k,:] = i*gvecs[0,:] + j*gvecs[1,:] + k*gvecs[2,:]

def _get_lattice_vecs(pwscf_xml_file):
    tree = ET.parse(pwscf_xml_file)

    root = tree.getroot()

    cell = root.find("./input/atomic_structure/cell")

    ndims = 3
    A = np.zeros((ndims,ndims))
    for d in range(ndims):
        A[d,:] = np.array(
            [ float(a) for a in cell.find(f"a{d+1}").text.split() ]
        )
    return A


def _get_common_xml(pwscf_xml_file):
    """
    Get common information from the Quantum Espresso (QE)
       checkpoint XML file AND compute/store some
       useful quantities.

    Returns
    -------
    - info : a dictionary containing:
        - A : 3x3 np.array containing the lattice vectors
        - volume : the volume of the cell used in the QE 
            calculation

    # TODO: consider making an info class instead of a dictionary    
    """
    tree = ET.parse(pwscf_xml_file)
    root = tree.getroot()
    cell = root.find("./input/atomic_structure/cell")

    NDIMS = 3 # TODO: get this from somewhere else
    A = np.zeros((NDIMS,NDIMS))
    for d in range(NDIMS):
        A[d,:] = np.array(
            [ float(a) for a in cell.find(f"a{d+1}").text.split() ]
        )

    volume = np.absolute( np.dot(np.cross(A[0,:],A[1,:]), A[2,:]))

    B = np.zeros((NDIMS,NDIMS))
    reciprocal = root.find("./output/basis_set/reciprocal_lattice")
    for d in range(NDIMS):
        B[d,:] = np.array( 
            [ float(b) for b in reciprocal.find(f"b{d+1}").text.split() ]
        )

    if (cell := root.find("./input/k_points_IBZ/monkhorst_pack")) is not None:
        nk = [
            int(cell.attrib['nk1']),
            int(cell.attrib['nk2']),
            int(cell.attrib['nk3'])
        ]
        nkpts = nk[0]*nk[1]*nk[2]
        kpts = [
            float(cell.attrib['k1']),
            float(cell.attrib['k2']),
            float(cell.attrib['k3'])
        ]

    elif (cell := root.find("./input/k_points_IBZ/nk")) is not None:
        nk = int(cell.text)
        kpts = [None]*nk # TODO: read form XML - low priority, this isn't used yet
        nkpts = nk

    nbands = int(root.find("./input/bands/nbnd").text)
    
    basis = root.find("./output/basis_set/fft_grid")
    fft_grid = [
        int(basis.attrib['nr1']),
        int(basis.attrib['nr2']),
        int(basis.attrib['nr3'])
    ]

    
    structure = root.find("./input/atomic_structure")
    a_lattice = float(structure.attrib['alat'])
    natom = int(structure.attrib['nat'])
    positions = np.zeros((natom,3))
    elememts = []
    for i,atom in enumerate(structure.findall("atomic_positions/atom")):
        positions[i,:] = np.array([ float(a) for a in atom.text.split()])
        elememts.append(atom.attrib['name'])

    info = {
        'A' : A,
        'a_lattice' : a_lattice,
        'B' : B * (2*np.pi/a_lattice), 
        'volume' : volume,
        'nkpts' : nkpts,
        'nbands' : nbands,
        'nk' : nk,
        'kpts' : kpts,
        'fft_grid' : fft_grid,
        'elememts' : elememts,
        'positions' : positions
    }

    return info

def _process_qeorbitals(evc,miller,gvecs,fft_grid=None):
    """
    Convert orbitals from the reciprocal space representation 
      to a spatial representation.

    Parameters
    ----------
    evc : np.ndarray with shape (Nbands,Ngvecs) where Ngvecs
        is the number of G-vectors in the reciprocal space
        representation of the orbitals. The G-vectors are
        assumed to be in the range (-1/2,1/2) in each dimension.
    miller : np.ndarray with shape (Ngvecs,3) containing the
        Miller indices of the G-vectors in the reciprocal
        space representation of the orbitals.
    gvecs : np.ndarray with shape (3,3) containing the
        reciprocal lattice vectors of the system.
    fft_grid : np.ndarray with shape (3,) containing the
        FFT grid of the system. If not provided, the FFT
        grid is computed from the Miller indices and the
        reciprocal lattice vectors.
    
    Returns
    -------
    orbitals : np.ndarray with shape (Nbands,Nx,Ny,Nz)
        The spatial representation of the orbitals, where
        Nx,Ny,Nz are the dimensions of the FFT grid.
    """
    bands = evc.shape[0]

    norm_qe = np.sqrt( np.dot(evc[0],evc[0].conj()).real )

    print("QE Normalization is ", norm_qe)

    Nx = np.max(miller[:,0])*2
    Ny = np.max(miller[:,1])*2
    Nz = np.max(miller[:,2])*2

    N = np.array([Nx,Ny,Nz])

    # we need to put evc into the appropriate shape
    a = np.zeros((Nx+1,Ny+1,Nz+1),dtype=np.complex128)
    orbitals = np.zeros((bands,*a.shape),dtype=np.complex128)

    for b in range(bands):
        for m,e in zip(miller,evc[b,:]):
            m_new = np.mod(m,N)
            a[ m_new[0], m_new[1], m_new[2] ] = e

        orbitals[b,:] = np.fft.ifftn(a)

    return orbitals


def shift_m(m,M):
    """

    """
    new_m = np.zeros_like(m)
    for i in range(new_m.shape[0]):
        if m[i] < 0:
            new_m[i] = m[i] + M[i]
        else:
            new_m[i] = m[i]

    return new_m


def get_qe_realspace_grid(fft_grid,A=None,cartesian=False):
    '''
    '''
    if cartesian and A is None:
        raise ValueError("Can't generate a realspace grid without lattice vectors, A")
    
    # TODO: read numpy docs, there must be a more direct way to do this!
    r_fracional_grid =np.array(
        np.meshgrid(
            np.linspace(0,1,fft_grid[0],endpoint=False),
            np.linspace(0,1,fft_grid[1],endpoint=False),
            np.linspace(0,1,fft_grid[2],endpoint=False),
            indexing='ij'
        )
    )

    if cartesian:
        print("Real-space grid is in Cartesian coordinates")
        cart_grid = np.tensordot(A.T,r_fracional_grid,axes=1)
        return cart_grid
    else:
        print("Real-space grid is in fractional coordinates")
        return r_fracional_grid


def read_qe_density(
        prefix,
        path=None,
        new_fft_grid=None,
        visualize=False,
        cartesian=True
    ):
    if path is not None:
        path = Path(path)

    # 1. Read Info
    # 1.A. Read XML -> kpoins, A, nbands
    common_info = _get_common_xml(Path(path / (prefix + ".xml")))

    volume = common_info['volume']
    A = common_info['A']
    rho_fft_grid = common_info['fft_grid'] # this is the density grid!

    # 1.B. Read HDF5
    with h5.File(Path(path / (prefix + ".save")) / "charge-density.hdf5", 'r' ) as f:
        miller = f["/MillerIndices"]
        miller_attrs = h5.AttributeManager(miller)
        B = np.array([
            miller_attrs.get("bg1"),
            miller_attrs.get("bg2"),
            miller_attrs.get("bg3")
        ])

        print("B = ", B)

        miller_inds = miller[...]

        rhotot_g = f["/rhotot_g"][...]
        # convert to complex!
        rhotot_g = rhotot_g[::2] + 1j*rhotot_g[1::2]

    # 2. determine N_i [already done above!]
    N = rho_fft_grid

    # 3. Choose new FFT grid M_i >= N_i
    if new_fft_grid is not None:
        M = np.array(new_fft_grid)
        if not np.all(np.greater_equal(M,N)):
            raise ValueError(f"New FFT grid {M} must be at least as big as old FFT grid {N}")
    else:
        M = N

    M1 = M[0]
    M2 = M[1]
    M3 = M[2]

    # 4. map evc (-1/2,1/2 ) -> (0,1)
    # TODO: numba this for performance!
    phi_reciprocal = np.zeros((M1,M2,M3),dtype=np.complex128)

    # 5. np.fft.ifft(rhotot_g) : no factor of Ngrid, but we will need the volume
    for m,e in zip(miller_inds,rhotot_g):
        new_m = shift_m(m,M)
        phi_reciprocal[ new_m[0], new_m[1], new_m[2] ] = e

    if visualize:
        isosurface_v2(
            phi_reciprocal.real.flatten(),
            min_coords=(0,0,0),
            max_coords=(M1,M2,M3),
            n=M
            )

    phi_real_space = np.fft.ifftn(phi_reciprocal)


    num_grid = np.prod(M)
    norm_factor = 1.0 * num_grid #* (np.sqrt(volume) / num_grid)
    return (
        get_qe_realspace_grid(fft_grid=M,A=A,cartesian=cartesian).reshape((3,num_grid)),
        norm_factor*phi_real_space.reshape((num_grid)),
        M
    )

@jit(nopython=True)
def _sparse_reciprocal_2dense(miller_inds,reciprocal,fft_grid):
    """
    Creates a dense 3d representation of the reciprocal space data

    This is achieved by mapping the sparse data onto a dense grid using
    numpy's convention for FFTs. The positive frequencies are stored in the
    first half of the array, while the negative frequencies are stored in the
    second half. The zero frequency mode is at index 0 in each dimenstino,
    and the Nyquist frequency mode is at the center of the array in each dimension.
    See the documentation for numpy.fft.ifftn for more details.

    Parameters
    ----------
    miller_inds : np.ndarray with shape (N_miller,3)
        The Miller indices for the reciprocal space data.
    reciprocal : np.ndarray with shape (N_miller,)
        The reciprocal space data.
    fft_grid : np.ndarray with shape (3,)
        The size of the FFT grid as (M1,M2,M3)

    Returns
    -------
    reciprocal_3d : np.ndarray with shape (M1,M2,M3)
        The dense 3d representation of the reciprocal space data.
    """
    reciprocal_3d = np.zeros(shape=(fft_grid[0],fft_grid[1],fft_grid[2]),dtype=np.complex128)
    new_m = np.zeros(3,dtype=np.int32)
    for m,e in zip(miller_inds,reciprocal[:]):
        for i in range(new_m.shape[0]):
            if m[i] < 0:
                new_m[i] = m[i] + fft_grid[i]
            else:
                new_m[i] = m[i]
        _inds = (new_m[0],new_m[1],new_m[2])
        reciprocal_3d[ new_m[0], new_m[1], new_m[2] ] = e
    return reciprocal_3d


def _rec2realspace(miller_inds,reciprocal,fft_grid,visualize=False):
    """
    Transfrom from reciprocal space data to real space

    Inputs:
    - reciprocal : np.ndarray with shape (N_band,N_fft) where N_fft is
                     the number of FFT grid points
    - fft_grid : np.ndarray specifying the size of the FFT grid
                     as (M1,M2,M3)
    """
    reciprocal_dense =  _sparse_reciprocal_2dense(
            np.array(miller_inds,dtype=np.int32),
            np.array(reciprocal),
            np.array(fft_grid,dtype=np.int32)
        )

    real_space = np.fft.ifftn(
        reciprocal_dense
    )
    return real_space


def _orbitals_to_dense_grid(miller_inds_k, evc_k, fft_grid, nkpts, nbands):
    """
    Convert sparse reciprocal space orbitals to a dense common grid.
    
    Parameters
    ----------
    miller_inds_k : list of np.ndarray
        List of Miller indices for each k-point, each with shape (n_gvecs_k, 3)
    evc_k : list of np.ndarray
        List of orbital coefficients for each k-point, each with shape (nbands, n_gvecs_k)
    fft_grid : np.ndarray
        FFT grid dimensions as (M1, M2, M3)
    nkpts : int
        Number of k-points
    nbands : int
        Number of bands
    
    Returns
    -------
    orbitals_dense : np.ndarray
        Dense orbital array with shape (nkpts, nbands, M1*M2*M3)
    """
    M = np.array(fft_grid, dtype=np.int32)
    num_grid = np.prod(M)
    
    # Initialize dense array for all k-points
    orbitals_dense = np.zeros((nkpts, nbands, num_grid), dtype=np.complex128)
    
    for k in range(nkpts):
        miller_k = np.array(miller_inds_k[k], dtype=np.int32)
        evc = evc_k[k]
        
        for b in range(nbands):
            # Map sparse coefficients to dense 3D grid
            dense_3d = _sparse_reciprocal_2dense(miller_k, evc[b], M)
            # Flatten to 1D
            orbitals_dense[k, b, :] = dense_3d.reshape(num_grid)
    
    return orbitals_dense


def read_orbitals(
        prefix,
        path=None, 
        new_fft_grid=None,
        cartesian=False,
        realspace=False,
        return_grid=False
    ):
    """
    Read orbitals from Quantum Espresso output.

    Parameters
    ----------
    prefix : str
        prefix of the output files
    path : str | Path, optional
        path to directory containing the output files. If not specified, 
        it is assumed that the files are in the current directory.
    new_fft_grid : list(int), optional
        new FFT grid to use for the orbitals.
    cartesian : bool, optional
        If True, return real space grid in Cartesian coordinates.
        If False, resturn real spcae grid in fractional coordinates.
        If 'realspace' is False, then this has no effect.
        If 'return_grid' is False, then this has no effect.
    realspace : bool, optional
        If True, return orbitals in real space. If False, return orbitals
        in reciprocal space.

    Returns
    -------
    tuple
        if realspace is True, then return a tuple containing:
        - dense represenation of the realspace grid : np.ndarray with shape (3,N_grid)
        - orbitals : np.ndarray with shape (N_kpts,N_bands,N_grid)
        - fft_grid : np.ndarray with shape (3,) containing the size of the FFT grid
        if realspace is False, then return a tuple containing:
        - metadata : dict containing metadata (includes 'common_grid_shape' and 'num_grid_points')
        - orbitals : np.ndarray with shape (N_kpts,N_bands,M1*M2*M3) on common reciprocal space grid
        - fft_grid : np.ndarray with shape (3,) containing the size of the FFT grid
    """
    if path is not None:
        path = Path(path)
    else:
        path = Path("./")

    # 1. Read Info
    # 1.A. Read XML -> kpoins, A, nbands
    common_info = _get_common_xml(Path(path / (prefix + ".xml")))

    volume = common_info['volume']
    nkpts = common_info['nkpts']
    nbands = common_info['nbands']
    rho_fft_grid = common_info['fft_grid']
    A = common_info['A']

    miller_inds_k = []
    evc_k = []

    # 1.B. Read HDF5
    for k in range(nkpts):
        with h5.File( Path(path / (prefix + ".save")) / f"wfc{k+1}.hdf5", 'r' ) as f:
            miller = f["/MillerIndices"]
            miller_inds = miller[...]

            evc = f["evc"][...]
            # convert to complex!
            evc = evc[:,::2] + 1j*evc[:,1::2]

            miller_inds_k.append(miller_inds)
            evc_k.append(evc)

    # 2. determine N_i [already done above!]
    N = rho_fft_grid

    # 3. Choose new FFT grid M_i >= N_i
    if new_fft_grid is not None:
        M = np.array(new_fft_grid)
        if not np.all(np.greater_equal(M,N)):
            raise ValueError(f"New FFT grid {M} must be at least as big as old FFT grid {N}")
    else:
        M = N


    # 4. map evc (-1/2,1/2 ) -> (0,1)
    if realspace:
        warn("Converting orbtials to real space orbitals. It is slow and memory intensive.")
        orbitals = np.zeros((nkpts,nbands,M[0],M[1],M[2]),dtype=np.complex128)
        for k in range(nkpts):
            for b in range(nbands):
                orbitals[k,b,:,:,:] = _rec2realspace(
                    miller_inds_k[k],
                    evc_k[k][b,:],
                    fft_grid=M,
                )

        num_grid = np.prod(M)
        # Note: the following factor leads to a density which integrates
        #   to the number of electrons in the cell
        norm_factor = np.sqrt(num_grid)
        if return_grid:
            dense_grid = get_qe_realspace_grid(fft_grid=M,A=A,cartesian=cartesian).reshape((3,num_grid))
        else:
            dense_grid = None
        return (
            dense_grid,
            norm_factor*orbitals.reshape((nkpts,nbands,num_grid)),
            M
        )
    else:
        print("Returning orbitals in reciprocal space on common grid")
        # Convert sparse orbitals to dense common grid
        orbitals_dense = _orbitals_to_dense_grid(miller_inds_k, evc_k, M, nkpts, nbands)
        meta = read_qe_metadata(prefix, path=path)
        meta["common_grid_shape"] = M
        meta["num_grid_points"] = np.prod(M)
        # Keep per-k-point Miller indices for reference/backward compatibility
        for k in range(nkpts):
            meta[f"gvecs_k{k}"] = miller_inds_k[k]
        return (
            meta,
            orbitals_dense,
            M
        )

def read_qe_metadata(prefix, path=None):
    """Read metadata from Quantum Espresso output.

    Parameters
    ----------
    forbs : str
        orbital HDF file from Coqui
    path : str
        path to directory containing the output files

    Returns
    -------
    meta : dict
        metadata
    orbs : numpy.ndarray
        orbitals
    """
    if path is not None:
        path = Path(path)

    common_info = _get_common_xml(Path(path / (prefix + ".xml")))

    meta = {
        "mesh": common_info['fft_grid'],  # FFT mesh
        "elem": [ atomic_symbol_to_num(elem) for elem in common_info['elememts'] ], 
        "pos": common_info['positions'],
        "cell": common_info['A'],
        "recvec": common_info['B'],
        "nkpts": common_info['nkpts'],
        "nbands": common_info['nbands'],
    }

    return meta


def get_rvecs(axes, mesh):
    """Regular grid in positive quadrant"""
    spaces = [np.arange(nx) for nx in mesh]
    gvecs = cubic_pos(spaces)
    fracs = axes/np.array(mesh)[:, np.newaxis]  # axes is row-major
    rvecs =  np.dot(gvecs, fracs)

    return rvecs

def cubic_pos(spaces):
    ndim = len(spaces)
    gvecs = np.stack(
        np.meshgrid(*spaces, indexing='ij'), axis=-1
    ).reshape(-1, ndim)
    return gvecs
