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

FORMAT_VERSION = 1
"""
Version of the Hamiltonian, Wavefunction and results.h5 layouts, stored as the
``format_version`` attribute of the ``Hamiltonian`` and ``Wavefunction/NOMSD``/``PHMSD``
groups and of the results.h5 root. Bumped with every
incompatible change, in step with ``FORMAT_VERSION`` in
``src/AFQMC/Utilities/format_version.hpp``.
"""


def write_format_version(group) -> None:
    """Stamp `group` with the current `FORMAT_VERSION`."""
    # int32, as TRIQS/h5 writes an int: the executable reads attributes by exact type
    group.attrs['format_version'] = np.int32(FORMAT_VERSION)


def check_format_version(group) -> None:
    """
    Raise ``ValueError`` unless `group` carries the current `FORMAT_VERSION`.
    """
    version = group.attrs.get('format_version')
    if version is None:
        raise ValueError(
            f"'{group.name}' has no format_version attribute: it was written before "
            f"format version {FORMAT_VERSION}. Regenerate it with safiretools."
        )
    if int(version) != FORMAT_VERSION:
        raise ValueError(
            f"'{group.name}' has format_version {int(version)}, but safiretools reads "
            f"version {FORMAT_VERSION}. Regenerate it."
        )


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
    executable reads: ``shape``, ``row_pointers`` (``nrows + 1`` offsets),
    ``column_indices`` and ``values``.
    """
    matrix = sps.csr_array(matrix)
    group = parent.create_group(name)
    group.create_dataset('shape', data=np.array(matrix.shape, dtype=np.int32))
    group.create_dataset('row_pointers', data=matrix.indptr.astype(np.int32, copy=False))
    group.create_dataset('column_indices', data=matrix.indices.astype(np.int32, copy=False))
    group.create_dataset('values', data=matrix.data)


def read_csr(group):
    """
    Read back a ``scipy.sparse.csr_array`` written by `write_csr`, or one in the
    legacy layout CoQuí writes: ``dims`` (rows, columns, nonzeros), ``data_``,
    ``jdata_``, ``pointers_begin_`` and ``pointers_end_``.
    """
    if 'shape' in group:
        return sps.csr_array(
            (read_complex(group['values']), group['column_indices'][...],
             group['row_pointers'][...]),
            shape=tuple(int(n) for n in group['shape'][...]))

    nrows, ncols, nnz = (int(value) for value in group['dims'][...])
    data = read_complex(group['data_'])
    indices = group['jdata_'][...]

    indptr = np.zeros(nrows + 1, dtype=np.int64)
    indptr[:-1] = group['pointers_begin_'][...]
    if nrows:
        indptr[-1] = group['pointers_end_'][-1]

    return sps.csr_array((data[:nnz], indices[:nnz], indptr), shape=(nrows, ncols))
