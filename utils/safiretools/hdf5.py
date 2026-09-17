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
`read_complex` does implement the file-level convention for storing complex
arrays, which every schema builds on.
"""

import numpy as np
import h5py as h5


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
