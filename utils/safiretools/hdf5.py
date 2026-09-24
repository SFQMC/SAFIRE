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
`read_complex` and `write_csr`/`read_csr` implement the file-level conventions
for complex arrays and sparse matrices that the schemas build on.
"""

import numpy as np
import scipy.sparse as sps


def replace_dataset(parent, name, value):
    """
    Add ``value`` as a dataset to ``parent``, overwriting it if it already exists.
    """
    if name in parent:
        del parent[name]
    parent.create_dataset(name, data=value)


def replace_group(parent, name):
    """
    Add a group called ``name`` to ``parent``, deleting ``parent[name]`` first if it
    already exists. Returns the group.
    """
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def read_complex(dataset):
    """
    TRIQS/h5 will write complex dataset with an extra trailing length-2 axis and attribute
    __complex__. This wrapper allows reading arrays that may come in this format using h5py.

    Anything without the attribute is returned as h5py read it, which covers both real
    datasets and h5py's own complex ones.
    """
    data = dataset[...]

    if '__complex__' not in dataset.attrs:
        return data

    if data.ndim == 0 or data.shape[-1] != 2:
        raise ValueError(
            f"dataset '{dataset.name}' is marked __complex__ but has shape {data.shape}; "
            "expected a trailing axis of length 2"
        )

    complex_dtype = np.result_type(data.dtype, np.complex64)
    return np.ascontiguousarray(data).view(complex_dtype).reshape(data.shape[:-1])


def write_csr(parent, name, matrix):
    """
    Write the sparse `matrix` as group ``parent[name]`` in the CSR layout the AFQMC
    executable reads: ``dims``, ``data_``, ``jdata_``, ``pointers_begin_``,
    ``pointers_end_``.
    """
    matrix = sps.csr_array(matrix)
    group = parent.create_group(name)
    group.create_dataset(
        'dims', data=np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int32))
    group.create_dataset('data_', data=matrix.data)
    group.create_dataset('jdata_', data=matrix.indices.astype(np.int32, copy=False))
    group.create_dataset('pointers_begin_', data=matrix.indptr[:-1].astype(np.int32, copy=False))
    group.create_dataset('pointers_end_', data=matrix.indptr[1:].astype(np.int32, copy=False))


def read_csr(group):
    """Read back a ``scipy.sparse.csr_array`` written by `write_csr`."""
    nrows, ncols, nnz = (int(value) for value in group['dims'][...])
    data = read_complex(group['data_'])
    indices = group['jdata_'][...]

    indptr = np.zeros(nrows + 1, dtype=np.int64)
    indptr[:-1] = group['pointers_begin_'][...]
    if nrows:
        indptr[-1] = group['pointers_end_'][-1]

    return sps.csr_array((data[:nnz], indices[:nnz], indptr), shape=(nrows, ncols))
