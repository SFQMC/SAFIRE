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

import numpy as np

from safiretools.stats import mean_and_error, reblock


def test_mean_and_error_real_known_mean():
    rng = np.random.default_rng(seed=0)
    true_mean = 5.0
    samples = rng.normal(loc=true_mean, scale=2.0, size=20000)

    mean, err = mean_and_error(samples)

    assert err > 0.0
    assert np.isfinite(err)
    assert abs(mean - true_mean) < 5 * err


def test_mean_and_error_complex_known_mean():
    rng = np.random.default_rng(seed=0)
    true_mean = -1.5 + 3.0j
    samples = (
        rng.normal(loc=true_mean.real, scale=1.0, size=20000)
        + 1j * rng.normal(loc=true_mean.imag, scale=4.0, size=20000)
    )

    mean, err = mean_and_error(samples)

    assert np.iscomplexobj(err)
    assert err.real > 0.0 and np.isfinite(err.real)
    assert err.imag > 0.0 and np.isfinite(err.imag)
    assert abs(mean.real - true_mean.real) < 5 * err.real
    assert abs(mean.imag - true_mean.imag) < 5 * err.imag


def test_mean_and_error_constant_series_is_detected_up_front():
    samples = np.full(50, 3.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, err = mean_and_error(samples)

    assert mean == 3.0
    assert err == 0.0


def test_reblock_uneven_sample_count_keeps_smaller_trailing_block():
    data = np.arange(10, dtype=np.float64)

    blocked = reblock(data, blocksize=3)

    np.testing.assert_allclose(blocked, [1.0, 4.0, 7.0, 9.0])


def test_reblock_evenly_divisible_unchanged_in_shape():
    data = np.arange(12, dtype=np.float64).reshape(6, 2)

    blocked = reblock(data, blocksize=2)

    assert blocked.shape == (3, 2)
    np.testing.assert_allclose(blocked[0], [1.0, 2.0])
