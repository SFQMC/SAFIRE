#!/usr/bin/env python3

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
from pathlib import Path

import tables # TODO: remove this dependency, replace by h5py
import numpy as np

from afqmctools.analysis.rdm import get_afqmc_rdm_samples, resample
from afqmctools.utils.qe_utils import read_orbitals, read_qe_metadata, _rec2realspace
from afqmctools.utils.aimbes_utils import read_coqui_orbitals

# TODO: remove - this isn't used
def read_dm(fstat, symmetrize=False):
    dm_dict = get_afqmc_rdm_samples(fstat)
    # !!!! HACK: assume one back propagated output
    dmm = dm_dict["a0"]["dm_mean"]
    dme = dm_dict["a0"]["dm_error"]
    # !!!! HACK: no spin resolution
    dm = dmm.sum(axis=0)
    de = (dme**2).sum(axis=0) ** 0.5
    if symmetrize:
        dm = 0.5 * (dm.T.conj() + dm)
    return dm, de

# TODO: move to generic rdm module
def dm_in_basis(dma, orbs, imag_tol=1e-6):
    """Put density matrix (dm) in orbital basis

    Parameters
    ----------
      dma:numpy.ndarray
          shape (number of samples, norb, norb)
      orbs (Array): shape (norb, nbas)
      imag_tol (float, optional): tolerance for imaginary component,
        which should be zero for physical observables, default 1e-6
    
    Returns
    -------
    ym : np.array
        mean of the rdm with shape (norb,norb)
    ye : np.array
        standard deviation of the rdm with shape (norb,norb)
    """
    # TODO: avoid np.einsum, still slow!
    yma = np.einsum("sij,jr->sir", dma, orbs)
    yma = np.einsum("ri,sir->sr", orbs.conj().T, yma)
    assert np.allclose(np.abs(yma.imag), 0, atol=imag_tol, rtol=0.001), "imaginary observable"
    ym = yma.real.mean(axis=0)
    ye = yma.real.std(axis=0, ddof=1)
    return ym, ye


def calc_nofk(dma, meta, orbs, kcut, **kwargs):
    """
    Calculate momentum distribution n(k).

    Parameters
    ----------
    dma : numpy.ndarray
        shape (number of samples, norb, norb)
    meta : dict
        dictionary containing system metadata
    orbs : numpy.ndarray
        orbitals on common reciprocal space grid, shape (nkpts, nbands, num_grid)
    kcut : float
        cutoff for k-points

    TODO: get k-point weights
    """
    nkpts = meta["nkpts"]
    nbands = meta["nbands"]
    
    # Generate k-vectors for the COMMON dense grid with centered indices
    mesh = meta.get("common_grid_shape", meta["mesh"])  # Use common grid if available
    gvecs_common = get_centered_gvecs(mesh)  # Centered Miller indices for full grid
    kvecs = gvecs_common @ meta["recvec"]  # Convert to k-space
    kmags = np.linalg.norm(kvecs, axis=-1)
    ksel = kmags < kcut  # Filter once, applies to all k-points
    
    warn("Assuming equally weighted k-points for now")
    nkm, nke = 0.0, 0.0 # just to initialize
    for k in range(nkpts):
        dma_k = dma[:, k*nbands:(k+1)*nbands, k*nbands:(k+1)*nbands]
        weight = 1.0 / nkpts # TODO: get this from metadata
        # Use SAME filtered orbitals for all k-points
        _nkm, _nke = dm_in_basis(dma_k, orbs[k][:, ksel], **kwargs)
        nkm += weight*_nkm
        nke += weight*_nke

    return kvecs[ksel], nkm, nke

# TODO: remove after testing new version
def calc_rhor_old(dma, meta, orbs, rvecs=None, **kwargs):
    """
    calculate charge density rho(r) from k-space orbitals

    Parameters
    ----------
    dma : numpy.ndarray
        shape (number of samples, norb, norb)
    meta : dict
        dictionary containing system metadata
    orbs : numpy.ndarray
        orbitals represented in terms of the FFT grid
    rvecs : numpy.ndarray, optional
        real-space grid points. If not specified, will be generated from the metadata.
    
    # TODO: get kpoint weights from metadata
    """
    if rvecs is None:
        rvecs = get_rvecs(meta["cell"], meta["mesh"])

    nkpts = meta["nkpts"]
    warn("Assuming equally weighted k-points for now")
    rhom, rhoe = 0.0, 0.0 # just to initialize
    for k in range(nkpts):
        weight = 1.0 / nkpts
        kvecs = meta[f"gvecs_k{k}"] @ meta["recvec"]
        # NOTE: the following line is highly memory intesnisve
        warn(
            "This is a memory intensive operation: use calc_rhor() instead for all but "
            "the smallest cells"
        )
        eikr = np.exp(1j * np.einsum('kd,rd->kr',kvecs, rvecs))
        volume = np.abs(np.linalg.det(meta["cell"]))
        rorbs = orbs[k] @ eikr / volume**0.5
        _rhom, _rhoe = dm_in_basis(dma, rorbs, **kwargs)
        rhom += weight*_rhom
        rhoe += weight*_rhoe

    return rhom, rhoe

#TODO: use jaxnumpy and jax.jit here
def calc_rhor(dma, meta, orbs, pwcut=None, **kwargs):
    r"""
    Compute charge density rho(r) from reciprocal-space orbitals
    
    Formally, the charge density is given by:

    .. math::
        \rho(\mathbf{r}) = \sum_{kk'} \sum_{ij} \rho^{kk'}_{ij} \\phi_{i}^{k}(\\mathbf{r}) \\phi_{j}^{k'*}(\\mathbf{r})

    Parameters
    ----------
    dma : numpy.ndarray
        shape (number of samples, norb, norb)
    meta : dict
        dictionary containing system metadata
    orbs : numpy.ndarray
        orbitals represented in terms of the FFT grid
    """
    nkpts = meta["nkpts"]
    nbands = meta["nbands"]
    numgrid = np.prod(meta["mesh"])
    # TODO: accumulate the mean (i.e. avoid storing nsamples of rho(r))
    warn("Assuming equally weighted k-points for now")
    rhom, rhoe = 0.0, 0.0 # just to initialize
    for k in range(nkpts):
        # looping to avoid memory usage
        for i in range(nbands): 
            orb_ik_dagger = orbs[k][i].conj()
            orb_ik_dagger = _rec2realspace(
                        meta[f"gvecs_k{k}"],
                        orb_ik_dagger,
                        fft_grid=meta["mesh"],
                    ).reshape(numgrid)
            for kprime in range(nkpts): # can probably sum over k' > k
                for j in range(nbands):
                    print(f"[For Sanity] k,i = {(k,i)} kprime,j = {(kprime,j)}",flush=True)
                    orb_jkprime = orbs[kprime][j]
                    orb_jkprime = _rec2realspace(
                            meta[f"gvecs_k{kprime}"],
                            orb_jkprime,
                            fft_grid=meta["mesh"],
                        ).reshape(numgrid)
                    _rhom = np.einsum("s,r,r->sr", dma[:,i,j], orb_ik_dagger, orb_jkprime)
                    _rhoe = _rhom.real.std(axis=0, ddof=1)
                    _rhom = _rhom.real.mean(axis=0)
                    rhom += _rhom
                    rhoe += _rhoe
    print(f"Integrated electron density per cell = {np.sum(rhom) / nkpts}")
    return rhom/nkpts, rhoe/nkpts

# TODO: use jaxnumpy and jax.jit here
def calc_rhor_from_r(dma, meta, rorbs, **kwargs):
    """
    Compute charge density rho(r) from real-space orbitals

    The density matrix may have multiple samples, and we use Welford's 
        algorithm to compute the mean and standard deviation of the charge density
        to avoid storing all the samples of rho(r) in memory at the same time.

    Parameters
    ----------
    dma : numpy.ndarray
        reduced one-body density matrix samples shape (number of samples, norb, norb)
    meta : dict
        dictionary containing system metadata
    rorbs : numpy.ndarray
        real-space orbitals with shape (number of k-points, norb, number of grid points)

    Returns
    -------
    rhom : numpy.ndarray
        mean charge density on real-space grid
    rhoe : numpy.ndarray
        stochastic uncertainty in charge density on real-space grid
    """
    rorbs = np.array(rorbs)
    nkpts = meta["nkpts"]
    nsamples = dma.shape[0]
    if dma.shape[1] % nkpts !=0 or dma.shape[2] % nkpts != 0:
        raise ValueError("Dimensions of 1-rdm are not divisible by number of k-points")
    nmo = dma.shape[1] // nkpts
    warn("Assuming equally weighted k-points for now")
    nbands_qe = rorbs.shape[1]
    if nmo < nbands_qe:
        print(f"Truncating QE orbitals ({nbands_qe} bands) to match number of bands in 1-rdm ({nmo} bands)")
        rorbs = rorbs[:,:nmo]
    robrs_k_dagger = rorbs.conj().transpose(0,2,1)
    # TODO: read k-point weights from metadata
    weight = 1.0 / nkpts
    rhom, rhoe = 0.0, 0.0 # rhom will be final mean, rhoe will be final error
    count = 0
    for sample in range(nsamples):
        count += 1
        dma_sample = dma[sample]
        rhom_k = 0.0
        for k in range(nkpts):
            for kprime in range(nkpts):
                print(f"[For Sanity] k = {k} kprime = {kprime}",flush=True)
                k_slice = slice(k*nmo, (k+1)*nmo)
                k_slice_prime = slice(kprime*nmo, (kprime+1)*nmo)

                _rhom_k = np.matmul(dma_sample[k_slice,k_slice_prime], rorbs[kprime])
                _rhom_k = np.einsum("ri,ir->r", robrs_k_dagger[k], _rhom_k)
                rhom_k += weight*_rhom_k
            
        # use Welford's algorithm to compute mean and standard deviation
        A = (rhom_k - rhom)
        rhom += A / count
        B = (rhom_k - rhom)
        rhoe += np.multiply(A,B)

    return rhom, rhoe / (nsamples - 1)

# ========================= qharv routines =========================

# from qharv.reel.config_h5
def open_read(fname, mode="r"):
    fp = tables.open_file(fname, mode=mode)
    return fp


def open_write(fname):
    filters = tables.Filters(complevel=5, complib="zlib")
    fp = tables.open_file(fname, mode="w", filters=filters)
    return fp


def read_group(grp):
    if type(grp) is tables.group.Group:
        data = dict()
        for g1 in grp:
            data[g1._v_name] = read_group(g1)
        return data
    else:
        return grp.read()


def load_dict(fname):
    data = dict()
    with open_read(fname) as h5file:
        for grp in h5file.root:
            data[grp._v_name] = read_group(grp)
        h5file.close()
    return data


def write_dict(fname, data):
    with open_write(fname) as h5file:
        save_dict(data, h5file)


def save_vec(vec, h5file, slab, name):
    try:
        len(vec)
    except TypeError:
        vec = np.array([vec])
    try:
        vec.dtype
    except AttributeError:
        vec = np.array([vec])
    atom = tables.Atom.from_dtype(vec.dtype)
    ca = h5file.create_carray(slab, name, atom, vec.shape)
    ca[:] = vec


def save_dict(arr_dict, h5file, slab=None):
    if slab is None:
        slab = h5file.root
    for key, arr in arr_dict.items():
        if isinstance(arr, dict):
            slab1 = h5file.create_group(slab, key)
            save_dict(arr, h5file, slab=slab1)
        else:
            save_vec(arr, h5file, slab, key)
    h5file.flush()


# from qharv.seed.hamwf_h5
def cubic_pos(spaces):
    ndim = len(spaces)
    gvecs = np.stack(np.meshgrid(*spaces, indexing="ij"), axis=-1).reshape(-1, ndim)
    return gvecs


def get_gvecs(mesh):
    spaces = [np.arange(nx) for nx in mesh]
    return cubic_pos(spaces)


def get_centered_gvecs(mesh):
    """
    Get centered Miller indices for FFT grid.
    
    Converts FFT convention indices [0, 1, ..., N-1] to centered indices
    [-N//2, ..., -1, 0, 1, ..., N//2-1] (for even N)
    or [-N//2, ..., -1, 0, 1, ..., N//2] (for odd N)
    
    Parameters
    ----------
    mesh : array-like
        FFT grid dimensions (Nx, Ny, Nz)
    
    Returns
    -------
    gvecs : np.ndarray
        Centered Miller indices with shape (Nx*Ny*Nz, 3)
    """
    spaces = []
    for nx in mesh:
        # Create centered indices using FFT convention
        indices = np.arange(nx)
        # Shift indices > nx//2 to negative values
        centered = np.where(indices > nx // 2, indices - nx, indices)
        spaces.append(centered)
    return cubic_pos(spaces)


def get_rvecs(axes, mesh, center=False):
    gvecs = get_gvecs(mesh)
    fracs = axes / np.array(mesh)[:, np.newaxis]  # axes is row-major
    rvecs = np.dot(gvecs, fracs)
    if center:
        c = 0.5 * np.ones(len(mesh)) @ (axes / np.array(mesh))
        rvecs += c
    return rvecs


#TODO: moves this to a generic "read orbitals" module
def _infer_orbital_format(orbital_source:str|Path):
    """Infer the format of the orbitals file.

    Parameters
    ----------
    orbital_source : str
        The path to the orbitals file.

    Returns
    -------
    str
        The format of the orbitals file.

    """
    orbital_source = Path(orbital_source)

    if orbital_source.name.endswith(".h5"):
        return "coqui", None
    elif orbital_source.is_dir():
        # Check for the presence of [prefix].save and [prefix].xml
        save_files = list(orbital_source.glob("*.save"))
        xml_files = list(orbital_source.glob("*.xml"))
        
        if not (len(save_files) == 1 and len(xml_files) == 1):
            raise ValueError(f"Could not find the Quantum Espresso output files in {orbital_source}")
        
        prefix = save_files[0].name.removesuffix(".save")
        
        if not prefix:
            raise ValueError(f"Could not infer Quantum Espresso prefix from {save_files[0]} and {xml_files[0]}")
        elif prefix != xml_files[0].name.removesuffix(".xml"):
            raise ValueError(f"Prefix mismatch between {save_files[0]} and {xml_files[0]} ")
        else:
            print(f"Reading Quantum Espresso output files with prefix {prefix}")

        return "qe", prefix

def charge_density(
        rdm,
        error_rdm,
        orbital_source:Path,
        rho_outfile:Path,
        nsample:int=32,
        verbose=False
    ):
    """Calculate and save the charge density from the 1-RDM.
    
    Will overwrite existing charge density file if it exists.

    Parameters
    ----------
    rdm : np.ndarray
        The 1-RDM. Shape (nspins, norbs, norbs).
    error_rdm : np.ndarray
        The error in the 1-RDM. Shape (nspins, norbs, norbs).
    orbital_source : Path
        The path to the orbitals file.
    rho_outfile : Path
        The path to the output charge density file.
    nsample : int
        The number of samples to use for resampling.
    verbose : bool
        Whether to print additional information.
    """
    orbital_source = Path(orbital_source)
    rho_outfile = Path(rho_outfile)

    #NOTE: This is spin-traced for now.
    if rdm.shape[0] == 2:
        warn("Spin-tracing the 1-RDM")
        rdm = rdm[0] + rdm[1]
        error_rdm = error_rdm[0] + error_rdm[1]

    rdm_samples = resample(rdm, error_rdm, nsample)

    if verbose:
        for i in range(nsample):
            print(f"sample {i} has trace {np.trace(rdm_samples[i])}")

    #TODO: have read_qe_orbitals return the meta data
    orbital_format,prefix = _infer_orbital_format(orbital_source)
    if "qe" in orbital_format:
        orbital_meta = read_qe_metadata(
            prefix=prefix,
            path=(orbital_source / f"{prefix}.xml").parent
        )
        _,orbitals_r,_ = read_orbitals(
            prefix=prefix,
            path=(orbital_source / f"{prefix}.xml").parent,
            realspace=True
        )
        rhom, rhoe = calc_rhor_from_r(rdm_samples, orbital_meta, orbitals_r, imag_tol=1e-2)
    elif "coqui" in orbital_format:
        orbital_meta, orbitals_k = read_coqui_orbitals(orbital_source)
        rhom, rhoe = calc_rhor(rdm_samples, orbital_meta, orbitals_k, imag_tol=1e-2)
    else:
        raise ValueError(f"[For Developers] Unsupported orbital format {orbital_format}")

    print(f" [+]  Ingrated charge density of rho(r) = {np.sum(rhom)}")

    print("==== Orbital metadata ====")
    for key, val in orbital_meta.items():
        print(f"{key}: {val}")

    print("saving rho(r) to %s" % rho_outfile)
    write_gaussian_cube(rho_outfile, {
            "axes": orbital_meta["cell"] / orbital_meta["mesh"],
            "elem": orbital_meta["elem"],
            "pos": orbital_meta["pos"],
            "data": rhom.reshape(orbital_meta["mesh"])
        },
        overwrite=True
    )
    write_gaussian_cube('error_' + rho_outfile.name, {
            "axes": orbital_meta["cell"] / orbital_meta["mesh"],
            "elem": orbital_meta["elem"],
            "pos": orbital_meta["pos"],
            "data": rhoe.reshape(orbital_meta["mesh"]),
        },
        overwrite=True
    )


def momentum_distribution(
        rdm,
        error_rdm,
        orbital_source:Path,
        kcut:float,
        nsample:int=32,
        verbose=False
    ):
    """Calculate the momentum distribution n(k) from the 1-RDM.

    Parameters
    ----------
    rdm : np.ndarray
        The 1-RDM. Shape (nspins, norbs, norbs).
    error_rdm : np.ndarray
        The error in the 1-RDM. Shape (nspins, norbs, norbs).
    orbital_source : Path
        The path to the orbitals file.
    kcut : float
        Cutoff for k-points in momentum space.
    nsample : int
        The number of samples to use for resampling.
    verbose : bool
        Whether to print additional information.

    Returns
    -------
    kvecs : np.ndarray
        K-vectors within the cutoff.
    nkm : np.ndarray
        Mean momentum distribution n(k).
    nke : np.ndarray
        Error in momentum distribution n(k).
    """
    orbital_source = Path(orbital_source)

    # NOTE: This is spin-traced for now.
    if rdm.shape[0] == 2:
        warn("Spin-tracing the 1-RDM", stacklevel=1)
        rdm = rdm[0] + rdm[1]
        error_rdm = error_rdm[0] + error_rdm[1]

    rdm_samples = resample(rdm, error_rdm, nsample)

    if verbose:
        for i in range(nsample):
            print(f"sample {i} has trace {np.trace(rdm_samples[i])}")

    # Infer orbital format and read reciprocal-space orbitals
    orbital_format, prefix = _infer_orbital_format(orbital_source)

    if "qe" in orbital_format:
        #orbital_meta = read_qe_metadata(
        #    prefix=prefix,
        #    path=(orbital_source / f"{prefix}.xml").parent
        #)
        orbital_meta, orbitals_k, _ = read_orbitals(
            prefix=prefix,
            path=(orbital_source / f"{prefix}.xml").parent,
            realspace=False  # We need reciprocal space orbitals for momentum distribution
        )
        kvecs, nkm, nke = calc_nofk(rdm_samples, orbital_meta, orbitals_k, kcut, imag_tol=1e-2)
    elif "coqui" in orbital_format:
        orbital_meta, orbitals_k = read_coqui_orbitals(orbital_source)
        kvecs, nkm, nke = calc_nofk(rdm_samples, orbital_meta, orbitals_k, kcut, imag_tol=1e-2)
    else:
        raise ValueError(f"[For Developers] Unsupported orbital format {orbital_format}")

    print(f" [+]  Integrated momentum distribution = {np.sum(nkm)}")

    if verbose:
        print("==== Orbital metadata ====")
        for key, val in orbital_meta.items():
            print(f"{key}: {val}")

    return kvecs, nkm, nke

# from qharv.reel.inspect.volumetric
def write_gaussian_cube(fcub, data, overwrite=False, **kwargs):
    """Write Gaussian cube file from volumetric data.

    Parameters
    ----------
    fcub : str
        The filename for the Gaussian cube file to be written.
    data :  dict
        Dictionary containing the following keys:
        - 'axes': (3, 3) matrix representing the grid axes.
        - 'data': Volumetric data as a numpy array.
        - 'elem': (optional) List of atomic numbers, default is (1,).
        - 'pos': (optional) List of atomic positions, default is ((0, 0, 0),).
        - 'origin': (optional) Coordinates of the origin, default is (0, 0, 0).
        - 'two_line_comment': (optional) Comments at the file head, default is "cube\nfile\n".
    overwrite : bool, optional
        If True, overwrite the existing file if it exists. Default is False.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the `write_gaussian_cube_text` function.
    
    Examples
    --------
    >>> data = {
    ...     "axes": np.diag((1.0, 1.0, 1.0)),
    ...     "data": np.random.rand(10, 10, 10), # Volumetric data!
    ...     "elem": [6, 6],
    ...     "pos": [(0, 0, 0), (1, 1, 1)],
    ...     "origin": (0, 0, 0),
    ...     "two_line_comment": "This is a test cube file\\nGenerated by write_gaussian_cube\\n"
    ... }
    >>> write_gaussian_cube("test.cube", data, overwrite=True)
        
    
    This will create a Gaussian cube file named "test.cube" with the provided data.
    """

    import os

    keys = data.keys()
    if os.path.isfile(fcub) and not overwrite:
        raise RuntimeError("%s exists" % fcub)
    # required inputs: grid axes (3, 3) matrix and volumetric data
    if "axes" not in keys:
        raise RuntimeError("grid axes is required")
    if "data" not in keys:
        raise RuntimeError("data grid is required")
    # optional inputs
    if "elem" in keys:
        elem = data["elem"]
    else:
        elem = (1,)
    if "pos" in keys:
        pos = data["pos"]
    else:
        pos = ((0, 0, 0),)
    if "origin" in data.keys():
        origin = data["origin"]
    else:
        origin = (0, 0, 0)
    text = write_gaussian_cube_text(
        data["data"], data["axes"], elem=elem, pos=pos, origin=origin, **kwargs
    )
    with open(fcub, "w") as f:
        f.write(text)


def write_gaussian_cube_text(
    vol,
    axes,
    elem=(1,),
    qs=None,
    pos=((0, 0, 0),),
    origin=(0, 0, 0),
    two_line_comment="cube\nfile\n",
):
    """Write Gaussian cube file using volumetric data

    Args:
      vol (np.array): volumetric data, shape (nx, ny, nz)
      axes (np.array): grid basis, e.g. np.diag((dx, dy, dz))
      elem (array-like, optional): list of atomic numbers, default (1,)
      qs (array-like, optional): list of atomic charges, default (0,)
      pos (array-like, optional): list of atomic positions
      origin (array-like, optional): coordinates of the origin
      two_line_comment (str, optional): comments at file head
    Return:
      str: Gaussian file content
    """
    text = two_line_comment
    # natom, origin
    natom = len(pos)
    if qs is None:
        qs = np.zeros(natom)
    x, y, z = origin
    line1 = "%4d %8.6f %8.6f %8.6f\n" % (natom, x, y, z)
    # grid, axes
    line2 = ""
    for n, vec in zip(vol.shape, axes, strict=True):
        x, y, z = vec
        line2 += "%4d %8.6f %8.6f %8.6f\n" % (n, x, y, z)
    # atoms
    line3 = ""
    for num, q, vec in zip(elem, qs, pos, strict=True):
        x, y, z = vec
        line3 += "%4d %4.1f %8.6f %8.6f %8.6f\n" % (num, q, x, y, z)
    # volumetric data (not human-readable format)
    dline = (len(vol.ravel()) * "%8.6e ") % tuple(vol.ravel())
    return text + line1 + line2 + line3 + dline

def read_gaussian_cube(filename):
    """
    Read a Gaussian cube file and return a dictionary with all volumetric data and metadata.
    
    Parameters
    ----------
    filename : str
        Path to the Gaussian cube file
    
    Returns
    -------
    dict
        Dictionary containing the cube file data with the following keys:
        
        - comments: List of two comment lines
        - n_atoms: Number of atoms
        - origin: Origin coordinates [x, y, z]
        - grid_dimensions: Grid dimensions [nx, ny, nz]
        - vectors: Cell vectors (3x3 array)
        - units: Units for each dimension ('Bohr' or 'Angstrom')
        - atoms: List of dictionaries for each atom with keys:
            - atomic_number
            - charge
            - position [x, y, z]
        - volumetric_data: 3D numpy array with the volumetric data
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Extract header comments (first two lines)
        comments = [lines[0].strip(), lines[1].strip()]
        
        # Parse atom count and origin (third line)
        parts = lines[2].split()
        n_atoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        
        # Parse grid dimensions and vectors (next three lines)
        grid_dims = []
        vectors = []
        for i in range(3, 6):
            parts = lines[i].split()
            grid_dims.append(int(parts[0]))
            vectors.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        # Get units (Bohr if positive, Angstroms if negative)
        units = ['Bohr' if dim > 0 else 'Angstrom' for dim in grid_dims]
        # Use absolute values for dimensions
        grid_dims = [abs(dim) for dim in grid_dims]
        nx, ny, nz = grid_dims
        vectors = np.array(vectors)
        
        # Parse atom information
        atoms = []
        for i in range(6, 6 + n_atoms):
            parts = lines[i].split()
            atom = {
                'atomic_number': int(parts[0]),
                'charge': float(parts[1]),
                'position': np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            }
            atoms.append(atom)
        
        # Parse volumetric data
        # Starting from line after atom data
        data_start_line = 6 + n_atoms
        
        # Flatten all remaining lines into a single list of values
        values = []
        for i in range(data_start_line, len(lines)):
            values.extend([float(val) for val in lines[i].split()])
        
        # Reshape the data according to the grid dimensions
        # The data is ordered with x as the outer loop, y as the middle loop, and z as the inner loop
        volumetric_data = np.array(values).reshape((nx, ny, nz))
        
        # Create and return the result dictionary
        result = {
            'comments': comments,
            'n_atoms': n_atoms,
            'origin': origin,
            'grid_dimensions': [nx, ny, nz],
            'vectors': vectors,
            'units': units,
            'atoms': atoms,
            'volumetric_data': volumetric_data
        }
        
        return result
    
    except Exception as e:
        raise ValueError(f"Error reading Gaussian cube file: {str(e)}")

        
# TODO Make this into an example, and remove from here
if __name__ == "__main__":
    fnk = "nofk.h5"
    frho = "rho.cube"

    forbs = "../ham/orbs.h5"
    fstat = "qmc.s000.stat.h5"

    nsample = 32  # bootstrap resample to propagate errorbar
    imag_tol = 1e-2  # !!!! large tolerance for testing
    # !!!! hard-code for Si
    elem_map = {"Si": 14}
    kcut = 2.11

    # program start
    dm, de = read_dm(fstat) # not this way, use function from rdm module
    dma = resample(dm, de, nsample)

    meta, orbs, _ = read_orbitals(forbs)

    print("calculating momentum distribution n(k)")
    kvecs, nkm, nke = calc_nofk(dma, meta, orbs, kcut=kcut, imag_tol=1e-2)
    print("saving n(k) to %s" % fnk)
    write_dict(fnk, {"kvecs": kvecs, "nk_mean": nkm, "nk_error": nke})

    print("calculating charge density rho(r)")
    rvecs = get_rvecs(meta["cell"], meta["mesh"])
    rhom, rhoe = calc_rhor(dma, meta, orbs, rvecs, imag_tol=1e-2)
    print("saving rho(r) to %s" % frho)
    elem_id = [elem_map[e] for e in meta["elem"]]
    data = {
        "axes": meta["cell"] / meta["mesh"],
        "elem": elem_id,
        "pos": meta["pos"],
        "data": rhom.reshape(meta["mesh"]),
    }
    write_gaussian_cube(frho, data)
