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

from safiretools.hdf5 import replace_dataset, replace_group, read_complex


def nda_complex(group, name, array):
    """
    Write `array` the way nda's h5 layer does: the real and imaginary parts as a
    trailing axis of length 2, plus the ``__complex__`` marker.
    """
    array = np.asarray(array)
    real_dtype = np.float32 if array.dtype == np.complex64 else np.float64
    interleaved = np.ascontiguousarray(array).view(real_dtype)
    dataset = group.create_dataset(name, data=interleaved.reshape(array.shape + (2,)))
    dataset.attrs['__complex__'] = '1'
    return dataset


def test_replace_dataset_creates_and_overwrites(tmp_path):
    with h5.File(tmp_path / 'test.h5', 'w') as f:
        replace_dataset(f, 'x', np.array([1, 2, 3]))
        replace_dataset(f, 'x', np.array([4, 5, 6]))
        np.testing.assert_array_equal(f['x'][...], [4, 5, 6])


def test_replace_group_replaces_existing(tmp_path):
    with h5.File(tmp_path / 'test.h5', 'w') as f:
        g = replace_group(f, 'grp')
        g.create_dataset('a', data=1)
        g2 = replace_group(f, 'grp')
        assert 'a' not in g2


class TestReadComplexOnTheMarkedLayout:
    """
    What nda and the C++ estimators write: a trailing axis of length 2 carrying the
    real and imaginary parts, marked with the ``__complex__`` attribute.
    """

    @pytest.mark.parametrize("shape", [(), (5,), (3, 2), (4, 3, 3)])
    def test_the_trailing_axis_is_dropped(self, tmp_path, shape):
        rng = np.random.default_rng(0)
        array = (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex128)

        with h5.File(tmp_path / 'marked.h5', 'w') as f:
            recovered = read_complex(nda_complex(f, 'x', array))

        assert recovered.shape == shape
        assert recovered.dtype == np.complex128
        np.testing.assert_array_equal(recovered, array)

    def test_single_precision_stays_single_precision(self, tmp_path):
        rng = np.random.default_rng(1)
        array = (rng.random(6) + 1j * rng.random(6)).astype(np.complex64)

        with h5.File(tmp_path / 'marked32.h5', 'w') as f:
            recovered = read_complex(nda_complex(f, 'x', array))

        assert recovered.dtype == np.complex64
        np.testing.assert_array_equal(recovered, array)

    def test_a_missing_trailing_axis_is_rejected(self, tmp_path):
        with h5.File(tmp_path / 'malformed.h5', 'w') as f:
            dataset = f.create_dataset('x', data=np.zeros((3, 3)))
            dataset.attrs['__complex__'] = '1'

            with pytest.raises(ValueError, match="trailing axis of length 2"):
                read_complex(dataset)


class TestReadComplexPassesEverythingElseThrough:
    """
    The marker attribute, not the shape, is what identifies the interleaved layout. A
    trailing axis of length 2 is ambiguous on its own -- a real ``(n, 2)`` matrix and a
    complex scalar observable of ``n`` bins are stored identically -- so an unmarked
    dataset is handed back exactly as h5py read it.

    That also covers h5py's own complex datasets, which are the ``{r, i}`` compound type
    on disk and carry no marker: h5py returns them as ``complex128`` already.
    """

    def test_native_h5py_complex_comes_back_complex(self, tmp_path):
        rng = np.random.default_rng(2)
        array = (rng.random((4, 3)) + 1j * rng.random((4, 3))).astype(np.complex128)

        with h5.File(tmp_path / 'native.h5', 'w') as f:
            recovered = read_complex(f.create_dataset('x', data=array))

        assert recovered.dtype == np.complex128
        assert recovered.shape == (4, 3)
        np.testing.assert_array_equal(recovered, array)

    @pytest.mark.parametrize("array", [
        np.zeros(2),                    # a flat real array of length 2
        np.zeros((2, 2)),               # a real 2x2 matrix
        np.zeros((4, 2)),               # a real matrix with two columns
        np.arange(4),                   # integers
    ])
    def test_real_data_is_returned_unchanged(self, tmp_path, array):
        with h5.File(tmp_path / 'real.h5', 'w') as f:
            recovered = read_complex(f.create_dataset('x', data=array))

        assert recovered.shape == array.shape
        assert recovered.dtype == array.dtype
        assert not np.iscomplexobj(recovered)

    def test_a_real_scalar_is_returned_unchanged(self, tmp_path):
        with h5.File(tmp_path / 'scalar.h5', 'w') as f:
            recovered = read_complex(f.create_dataset('x', data=3.5))

        assert recovered.shape == ()
        assert recovered == 3.5
