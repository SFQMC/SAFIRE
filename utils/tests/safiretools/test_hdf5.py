# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import h5py as h5

from safiretools.hdf5 import (
    add_dataset,
    add_group,
    to_complex,
    from_complex,
    h5_as_dict,
    dict_to_h5,
)


def test_add_dataset_creates_and_overwrites(tmp_path):
    fname = tmp_path / 'test.h5'
    with h5.File(fname, 'w') as f:
        add_dataset(f, 'x', np.array([1, 2, 3]))
        add_dataset(f, 'x', np.array([4, 5, 6]))
        np.testing.assert_array_equal(f['x'][...], [4, 5, 6])


def test_add_group_replaces_existing(tmp_path):
    fname = tmp_path / 'test.h5'
    with h5.File(fname, 'w') as f:
        g = add_group(f, 'grp')
        g.create_dataset('a', data=1)
        g2 = add_group(f, 'grp')
        assert 'a' not in g2


def test_to_from_complex_round_trip():
    array = np.array([1 + 2j, 3 - 4j, 0.5j], dtype=np.complex128)
    on_disk = to_complex(array)
    assert on_disk.shape == array.shape + (2,)
    assert on_disk.dtype == np.float64
    recovered = from_complex(on_disk, shape=array.shape)
    np.testing.assert_allclose(recovered, array)


def test_dict_to_h5_round_trip_with_nested_groups(tmp_path):
    fname = tmp_path / 'nested.h5'
    data = {
        'a': np.array([1.0, 2.0, 3.0]),
        'group': {
            'b': np.array([[1, 2], [3, 4]]),
            'c': 42,
        },
    }
    dict_to_h5(fname, data)
    result = h5_as_dict(fname)

    np.testing.assert_array_equal(result['a'], data['a'])
    np.testing.assert_array_equal(result['group']['b'], data['group']['b'])
    assert result['group']['c'] == 42
