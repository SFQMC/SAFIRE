# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Generic HDF5 read/write primitives shared by the Hamiltonian, Wavefunction,
and analysis layers. Nothing here knows about any particular on-disk schema.
"""

import numpy as np
import h5py as h5


def add_dataset(fh5: h5.File, name, value):
    """
    Add ``value`` as a dataset, overwriting it if it already exists.

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
        del fh5[name]
    fh5.create_dataset(name, data=value)


def add_group(fh5: h5.File, name):
    """
    Add a group called ``name`` to ``fh5``, deleting ``fh5[name]`` first if it
    already exists.

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


def to_complex(array: np.array):
    """Convert a numpy.complex128 array to SAFIRE's on-disk complex format:
    real and imaginary parts interleaved as a trailing length-2 axis.
    """
    if array.dtype != np.complex128:
        array = array.astype(np.complex128, casting='same_kind')
    shape = array.shape
    return np.ascontiguousarray(array).view(np.float64).reshape(shape + (2,))


def from_complex(data, shape=None):
    """Convert from SAFIRE's on-disk complex format back to numpy.complex128."""
    if shape is not None:
        return data.view(np.complex128).ravel().reshape(shape)
    else:
        return data.view(np.complex128).ravel()


def _read_group(group):
    data = {}
    for key, item in group.items():
        if isinstance(item, h5.Group):
            data[key] = _read_group(item)
        else:
            data[key] = item[...]
    return data


def h5_as_dict(fname):
    """
    Load an HDF5 file as a (possibly nested) dictionary.

    Parameters
    ----------
    fname : str
        File name of HDF5 file to load.

    Returns
    -------
    dict
        Dictionary containing the contents of the HDF5 file. Subgroups become
        nested dictionaries.
    """
    with h5.File(fname, 'r') as f:
        return _read_group(f)


def _write_group(group, data):
    for key, value in data.items():
        if isinstance(value, dict):
            _write_group(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value)


def dict_to_h5(fname, data):
    """
    Write a (possibly nested) dictionary of arrays to an HDF5 file, creating a
    subgroup for each nested dictionary.

    Parameters
    ----------
    fname : str
        File name of HDF5 file to write to.
    data : dict
        Dictionary to write.
    """
    with h5.File(fname, 'w') as f:
        _write_group(f, data)
