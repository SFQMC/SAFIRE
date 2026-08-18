# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import warnings
from pathlib import Path

import toml

import numpy as np
import h5py as h5
import scipy.sparse as sps

from afqmctools.hamiltonian.model.ham_class import SpinSymm,Hamiltonian,get_spin_symm_enum


spin_type = {
    SpinSymm.CLOSED : 'closed',
    SpinSymm.COLLINEAR : 'collinear',
    SpinSymm.NONCOLLINEAR : 'noncollinear'
}

inv_spin_type = {
    'close' : 1,
    'collinear' : 2,
    'noncollinear' : 3
}


def format_fixed_width_strings(strings):
    """Format a list of strings to be fixed width."""
    return ' '.join('{:>17}'.format(s) for s in strings)


def format_fixed_width_floats(floats):
    """Format a list of floats to be fixed width."""
    return ' '.join('{: .10e}'.format(f) for f in floats)


def to_complex(array:np.array):
    """Convert from numpy.complex128 to internal complex format
    """
    if array.dtype != np.complex128:
        array = array.astype(np.complex128,casting='same_kind')
    shape = array.shape
    return np.ascontiguousarray(array).view(np.float64).reshape(shape+(2,))

def from_complex(data, shape=None):
    """Convert from internal complex format to numpy.complex128
    """
    if shape is not None:
        return data.view(np.complex128).ravel().reshape(shape)
    else:
        return data.view(np.complex128).ravel()


def add_dataset(fh5:h5.File, name, value):
    """
    Adds value to dataset, and creates dataset if it does not exist.
    
    Parameters
    ----------
    fh5 : h5py.File
        File object to write to.
    name : str
        Name of dataset.
    value : np.array
        Value to write to dataset.
    """
    if name in fh5:
        fh5[name] = value
    else:
        fh5.create_dataset(name,data=value)


def add_group(fh5:h5.File, name):
    """
    Add a group called "name" to h5py.File instance "fh5",
       deleting fh5[name] *if it already exists*

    Parameters
    ----------
    fh5 : h5py.File
        File object to write to.
    name : str
        Name of group.

    Returns
    -------
    h5py.Group
        Group object.
    """
    if name in fh5:
        del fh5[name]
    return fh5.create_group(name)

def read_one_body(fname,format:str='csr'):
    """
    Read one-body Hamiltonian from file.

    Parameters
    ----------
    fname : str
        File name to read from.
    format : str
        Format of the file. Default is 'csr'.
    """
    # print(format)
    if format.lower() == 'csr':
        return _read_one_body_csr(fname=fname)
    else:
        raise NotImplementedError


def _from_complex_csr(data):

    if len(data.shape) == 1:
        return data
    elif data.shape[1] == 2:
        return data[:,0] + 1j* data[:,1]
    else:
        raise ValueError("Invalid shape for CSR matrix")

def _read_one_body_csr(fname:str|Path):
    """
    Read full one-body Hamiltonian in CSR format from an HDF5 file.

    We want to loop through /Hamiltonian/ModelHamiltonian/ModelComponent_[X];
        where the max X is found in /Hamiltonian/ModelHamiltonian/number_of_components

    Parameters
    ----------
    fname : str
        File name of HDF5 file containing the Hamiltonian.

    Returns
    -------
    scipy.sparse.csr_array
        One-body Hamiltonian in CSR format.
    """
    with h5.File(fname,'r') as f:
        num_comps = f['/Hamiltonian/ModelHamiltonian/number_of_components'][...]
        comps = []

        for n in range(num_comps):

            prefix = f'/Hamiltonian/ModelHamiltonian/ModelComponent_{n}/'


            if f[prefix + "model_type"][...] == b"one_body":
                print(f"inlcuding component {n}")
                data = _from_complex_csr(f[prefix + "tij/data_"][...])
                indices = f[prefix + "tij/jdata_"][...]
                indptr_begin = f[prefix + "tij/pointers_begin_"][...]
                indptr_last = f[prefix + "tij/pointers_end_"][...][-1]

                indptr = np.empty( (indptr_begin.shape[0] + 1) )
                indptr[:indptr_begin.shape[0]] = indptr_begin
                indptr[-1] = indptr_last

                comps.append(sps.csr_array((data,indices,indptr)))

        print("Read in:",comps)
        print("Unique vals:")
        for c in comps:
            print(np.unique(c.data))
    return sum(comps)


def get_hamiltonian_spin_symm(fname):
    """
    Get the spin symmetry of the Hamiltonian from a Hamiltonian HDF5 file.

    Parameters
    ----------
    fname : str
        File name of HDF5 file containing the Hamiltonian.

    Returns
    -------
    SpinSymm
        the spin symmetry of the Hamiltonian.
    """
    with h5.File(fname,'r') as f:
        spin_type = f['Hamiltonian/spin_type'].asstr()[...]

    return SpinSymm(inv_spin_type[str(spin_type)])


def read_nmo(fname):
    with h5.File(fname,'r') as f:
        nmo = f["/Hamiltonian/dims"][...][3]
    return nmo

def _format_dtype(array):
    if array.dtype == 'complex128':
        return to_complex(array)
    else:
      return array

def write_csr(f,csr_array:sps.csr_array,prefix):
    """
    Write CSR matrix to HDF5.

    Parameters
    ----------
    f : h5py.File or h5py.Group
        HDF5 object to write to.
    csr_array : sps.csr_array
        CSR matrix to write.
    prefix : str
        Prefix to write to.
    """

    dims = [
        csr_array.shape[0],
        csr_array.shape[1],
        csr_array.nnz
    ]
    
    f.create_dataset(
        name=prefix+'/dims',
        data=np.array(dims, dtype=np.int32)
    )
    f.create_dataset(
        name=prefix+'/data_',
        data=_format_dtype(csr_array.data)
    )
    f.create_dataset(
        name=prefix+'/jdata_',
        data=csr_array.indices.astype(np.int32,copy=False)
    )
    f.create_dataset(
        name=prefix+'/pointers_begin_',
        data=csr_array.indptr[:-1].astype(np.int32,copy=False)
    )
    f.create_dataset(
        name=prefix+'/pointers_end_',
        data=csr_array.indptr[1:].astype(np.int32,copy=False)
    )


def write_model_hamiltonian(
        hamiltonian:Hamiltonian,
        fname,
        prefix='Hamiltonian/ModelHamiltonian',
        nelec=None,
        spin_symm=None
    ):
    """
    Writes a lattice model Hamiltonian to an HDF5 file.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object to write.
    fname : str
        File name of HDF5 file to write to.
    prefix : str, optional
        Prefix to write to.
    nelec : tuple
        Number of electrons in the system.
    spin_symm : SpinSymm, optional
        Spin symmetry of the Walkers to use in AFQMC. If not provided, the spin symmetry is read from the Hamiltonian object.
    
    TODO: consider removing the prefix argument since it should always be the same.
    """
    if nelec is None:
        nelec = (0,0)

    # the AFQMC code expects all components to be either real or complex
    real_valued = hamiltonian.real_valued

    component_num = 0

    if spin_symm is not None:
        spin_symm = get_spin_symm_enum(spin_symm=spin_symm)
    else:
        spin_symm = hamiltonian.spin_symm

    if spin_symm == SpinSymm.CLOSED or spin_symm == SpinSymm.COLLINEAR:
        nup,ndn = nelec
    elif spin_symm == SpinSymm.NONCOLLINEAR:
        nup = sum(nelec)
        ndn = 0
    else:
        raise ValueError("Hamiltonian has invalid spin symmetry")

    # Track max_nnz per collection matrix in C++
    # collect_U has 4 categories (by hst_type): continuous_charge, continuous_spin, 
    #                                           discrete_charge, discrete_spin
    # collect_J has 3 categories (by hst_type): continuous_charge, continuous_spin,
    #                                           (discrete placeholder)
    # Multiple components can contribute to the same collection matrix, so
    #                                         we need to sum their max_nnz
    max_nnz_U = {'continuous_charge': 0, 'continuous_spin': 0, 'discrete_charge': 0, 'discrete_spin': 0}
    max_nnz_J = {'continuous_charge': 0, 'continuous_spin': 0}

    # First pass: accumulate max_nnz for components that go into the same collection matrix
    for key in hamiltonian.keys():
        for component in hamiltonian[key]:
            if key == 'Uij':  # Hubbard U components
                hst = component.metadata.get('hst_type', 'continuous_spin')
                max_nnz_U[hst] += component.max_nnz
            elif key == 'Jij':  # Hubbard J components
                hst = component.metadata.get('hst_type', 'continuous_spin')
                if hst in max_nnz_J:
                    max_nnz_J[hst] += component.max_nnz

    # Calculate overall maximum connectivity needed
    max_conn = max(
        max(max_nnz_U.values()),
        max(max_nnz_J.values()),
        12  # Default minimum
    )

    with h5.File(fname,'w') as f:

        f.create_dataset(
            'Hamiltonian/dims',
            data = np.array([0, 0, 0, hamiltonian.nsites*hamiltonian.nbands , nup, ndn, 0, 0], dtype=np.int64)
        )
        # write Energies!
        f.create_dataset(
            'Hamiltonian/Energies',
            data = np.array([0., 0.]),
            dtype=np.float64
        )

        f.create_dataset(
            'Hamiltonian/spin_type',
            data=spin_type[hamiltonian.spin_symm]
        )

        f.create_dataset(
            name=prefix+'/number_of_components',
            data=hamiltonian.num_components
        )

        for key in hamiltonian.keys():
            for component in hamiltonian[key]:
                component_prefix = prefix + f"/ModelComponent_{component_num}/"
                f.create_dataset(
                    name=component_prefix + 'model_type',
                    data=component.model_type
                )

                if hasattr(component,"spin_symm"):
                    f.create_dataset(
                        name=component_prefix + 'spin_type',
                        data=spin_type[component.spin_symm]
                    )

                # write component specific data
                for metakey,value in component.metadata.items():
                    if value is not None:
                        f.create_dataset(
                            name=component_prefix+metakey,
                            data=value
                        )
                if real_valued:
                    csr_array = component.csr_array
                else:
                    csr_array = component.csr_array.astype(np.complex128)

                write_csr(
                    f=f,
                    csr_array=csr_array,
                    prefix=component_prefix + key,
                    )
                component_num+=1

        f.create_dataset(
            name=prefix+'/maximum_connectivity',
            data=max_conn
        )
    

def write_pair_correlators(fname:str=None, pairs_dict=None):
    """
    Write pair correlators to an HDF5 file.

    .. warning:: This is an experimental feature and is not officially supported.
                 Use at your own risk.

    Parameters
    ----------
    fname : str
        File name of HDF5 file to write to.
    pairs_dict : dict
        Dictionary of pair correlators to write.
    """
    warnings.warn("Pair correlators are an experimental feature")

    max_num_pairs = max( [ len(pair_list) for pair_list in pairs_dict.values() ] ) # this is for c++ memory allocation
    num_correlators = len(pairs_dict.keys())

    with h5.File(fname,"a") as f:
        group_name = "PairCorrelator/orbital_map"
        if group_name in f:
            del f[group_name]
        g = f.create_group(group_name)
        g.create_dataset("num_pair",data=max_num_pairs)
        g.create_dataset("num_corr",data=num_correlators)
        for direction,pair in pairs_dict.items():
            g.create_dataset(direction,data=np.array([pair]).T)


def write_model_params(fname,params=None):
    """
    Write Model Hamiltonian parameters to a human-readable TOML file.

    The files produced by this function can be used as an input files via the 
        `read_input_params(fname)` function.

    Parameters
    ----------
    fname : str
        File name to write to.
    params : dict
        Dictionary of model parameters to write. This dictionary has the same conventions as the model
        Hamiltonian builder. See the User Guide for more information.
    """
    with open(fname,"w") as f:
        toml.dump(params, f)

def read_input_params(infile:str=None) -> dict:
    """Read parameters from a human-readable TOML input file and return them in a dictionary."""
    with open(infile,'r') as f:
        params = toml.loads(f.read())
    return params


def h5_as_dict(fname):
    """
    Load an HDF5 file as a dictionary.

    Parameters
    ----------
    fname : str
        File name of HDF5 file to load.

    Returns
    -------
    dict
        Dictionary containing the contents of the HDF5 file.
    """
    with h5.File(fname,'r') as f:
        return {key: f[key][...] for key in f.keys()}
    

def dump_dict(d:dict, offset=0, dict_title=None):
    if dict_title is not None:
        print(f"\n\n====== [{str(dict_title):^20s}] ======\n")
        
    for key,value in d.items():
        print(" "*offset + "[+] " f"{key:16s} = {str(value):16s}")

