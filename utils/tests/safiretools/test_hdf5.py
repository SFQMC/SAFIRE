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
import pytest

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

    recovered = from_complex(on_disk, real_ndim=1)
    assert recovered.shape == array.shape
    np.testing.assert_allclose(recovered, array)


@pytest.mark.parametrize("shape", [(3,), (2, 2), (4, 2), (2, 3, 2)])
def test_to_from_complex_preserves_shape(shape):
    rng = np.random.default_rng(0)
    array = rng.random(shape) + 1j * rng.random(shape)

    recovered = from_complex(to_complex(array), real_ndim=len(shape))
    assert recovered.shape == shape
    np.testing.assert_allclose(recovered, array)


class TestFromComplexTellsTheLayoutsApartByRank:
    """
    `to_complex` appends a trailing length-2 axis, so the layouts differ in
    *rank*. Recognizing the complex one by "the last axis has length 2" instead
    is ambiguous whenever a real dataset's own last axis happens to be 2.
    """

    @pytest.mark.parametrize("array,real_ndim", [
        (np.zeros(2), 1),               # a flat real array of length 2
        (np.zeros((2, 2)), 2),          # a real 2x2 matrix
        (np.zeros((4, 2)), 2),          # a real matrix with two columns
    ])
    def test_real_data_is_returned_unchanged(self, array, real_ndim):
        recovered = from_complex(array, real_ndim=real_ndim)
        assert recovered.shape == array.shape
        assert not np.iscomplexobj(recovered)

    @pytest.mark.parametrize("real_ndim", [1, 2, 3])
    def test_interleaved_data_is_converted(self, real_ndim):
        rng = np.random.default_rng(1)
        shape = (2,) * real_ndim
        array = rng.random(shape) + 1j * rng.random(shape)

        recovered = from_complex(to_complex(array), real_ndim=real_ndim)
        assert np.iscomplexobj(recovered)
        np.testing.assert_allclose(recovered, array)

    def test_a_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="rank 3"):
            from_complex(np.zeros((2, 2, 3)), real_ndim=1)


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
