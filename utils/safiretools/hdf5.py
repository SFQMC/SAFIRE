# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""HDF5 read/write primitives shared by the Hamiltonian, Wavefunction and
analysis layers. Nothing here knows about any particular on-disk *schema*, but
`to_complex`/`from_complex` do implement SAFIRE's file-level convention for
storing complex arrays, which every schema builds on.
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


def to_complex(array):
    """
    Convert a complex array to SAFIRE's on-disk complex format: the real and
    imaginary parts interleaved as a trailing length-2 axis.

    Parameters
    ----------
    array : array_like
        Array to convert. Cast to ``complex128`` if it is not already.

    Returns
    -------
    numpy.ndarray
        Real ``float64`` array of shape ``array.shape + (2,)``.

    See Also
    --------
    from_complex : the inverse.
    """
    array = np.ascontiguousarray(np.asarray(array).astype(np.complex128, copy=False))
    return array.view(np.float64).reshape(array.shape + (2,))


def from_complex(data, real_ndim: int = 2):
    """
    Convert from SAFIRE's on-disk complex format back to ``complex128``,
    returning a dataset that was stored real unchanged.

    Parameters
    ----------
    data : array_like
        Dataset as stored.
    real_ndim : int, optional
        Rank this dataset has when it is stored *real*. Default 2, which suits
        matrices; pass 1 for a flat array such as a CSR matrix's ``data_``.

    Returns
    -------
    numpy.ndarray
        The dataset, complex if it was stored interleaved and untouched if it
        was stored real.

    Raises
    ------
    ValueError
        If `data` has neither the real nor the interleaved rank.

    Notes
    -----
    The two layouts are told apart by **rank**, not by the length of the
    trailing axis: `to_complex` appends an axis of length 2, so an interleaved
    dataset has rank ``real_ndim + 1``. Testing the trailing axis instead is
    ambiguous whenever the real dataset's own last axis happens to have length
    2 — a real ``(2, 2)`` one-body matrix, or a Cholesky matrix with two
    vectors — and silently reads it as complex.

    Both ranks occur in practice: the AFQMC executable accepts either for a
    model Hamiltonian's components, and a real-valued Hamiltonian is written
    real so that the file stays half the size.
    """
    data = np.asarray(data)

    if data.ndim == real_ndim + 1 and data.shape[-1] == 2:
        return np.ascontiguousarray(data).view(np.complex128).reshape(data.shape[:-1])

    if data.ndim != real_ndim:
        raise ValueError(
            f"dataset has rank {data.ndim}; expected {real_ndim} if real or "
            f"{real_ndim + 1} if interleaved complex"
        )

    return data


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
