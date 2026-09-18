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
import pytest

from safiretools.results.stats import jackknife, rebinning_analysis, standard_error


def ar1(n, rho, rng):
    """
    A first-order autoregressive series, whose statistical inefficiency is known
    exactly: ``kappa = (1 + rho)/(1 - rho)``, so ``tau = rho/(1 - rho)``.
    """
    noise = rng.normal(size=n)
    values = np.empty(n)
    values[0] = noise[0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        values[i] = rho * values[i - 1] + noise[i]
    return values


# a ramp is the most autocorrelated series there is, so these four earn the
# under-binning warning; what they check is the bin arithmetic, not the error bar
ramp = pytest.mark.filterwarnings('ignore:rebin size')


@ramp
def test_bins_average_consecutive_samples():
    # the bin index has to be the slow one; averaging strided samples instead would
    # decorrelate the bins by construction and hide every correlation there is
    bins, _ = rebinning_analysis(np.arange(12.0), rebinsize=4)
    np.testing.assert_allclose(bins, [1.5, 5.5, 9.5])


@ramp
def test_remainder_samples_are_discarded():
    bins, _ = rebinning_analysis(np.arange(10.0), rebinsize=4)
    np.testing.assert_allclose(bins, [1.5, 5.5])


@ramp
def test_skip_drops_leading_samples():
    bins, _ = rebinning_analysis(np.arange(12.0), skip=4, rebinsize=4)
    np.testing.assert_allclose(bins, [5.5, 9.5])


@ramp
def test_component_shape_survives_rebinning():
    bins, _ = rebinning_analysis(np.arange(24.0).reshape(12, 2), rebinsize=4)

    assert bins.shape == (3, 2)
    np.testing.assert_allclose(bins[:, 0], [3.0, 11.0, 19.0])


def test_uncorrelated_series_has_no_autocorrelation_time():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=20000)

    bins, tau = rebinning_analysis(samples)

    assert tau == pytest.approx(0.0, abs=0.5)
    assert standard_error(bins) == pytest.approx(1 / np.sqrt(samples.size), rel=0.25)


def test_autocorrelated_series_recovers_its_correlation_time():
    rng = np.random.default_rng(1)
    rho = 0.9

    _, tau = rebinning_analysis(ar1(50000, rho, rng))

    assert tau == pytest.approx(rho / (1 - rho), rel=0.3)


def test_correlation_widens_the_error_bar():
    rng = np.random.default_rng(2)
    samples = ar1(50000, 0.9, rng)

    bins, _ = rebinning_analysis(samples)

    independent = samples.std(ddof=1) / np.sqrt(samples.size)
    assert standard_error(bins) > 3 * independent


def test_constant_series_is_not_taken_for_a_correlated_one():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bins, tau = rebinning_analysis(np.full((100, 4), 7.0))

    assert tau == 0.0
    np.testing.assert_array_equal(standard_error(bins), np.zeros(4))


def test_a_rebin_size_below_the_correlation_time_warns():
    rng = np.random.default_rng(3)

    with pytest.warns(UserWarning, match='autocorrelation time'):
        rebinning_analysis(ar1(400, 0.98, rng))


def test_too_few_bins_is_an_error():
    with pytest.raises(ValueError, match='too few'):
        rebinning_analysis(np.arange(5.0), rebinsize=4)


def test_standard_error_keeps_real_and_imaginary_apart():
    rng = np.random.default_rng(4)
    bins = rng.normal(0.0, 1.0, 500) + 1j * rng.normal(0.0, 4.0, 500)

    error = standard_error(bins)

    assert error.imag == pytest.approx(4 * error.real, rel=0.15)


def test_jackknife_reproduces_the_standard_error_of_a_linear_function():
    rng = np.random.default_rng(5)
    bins = rng.normal(size=(40, 3))

    value, error = jackknife(lambda x: x, bins)

    np.testing.assert_allclose(value, bins.mean(axis=0))
    np.testing.assert_allclose(error, standard_error(bins))


def test_jackknife_corrects_the_bias_of_a_nonlinear_function():
    # the mean squared has a closed-form bias-corrected estimator, mean**2 - var/nbins,
    # which the jackknife has to land on exactly
    rng = np.random.default_rng(8)
    bins = rng.normal(3.0, 1.5, 25)

    value, _ = jackknife(lambda x: x**2, bins)

    assert value == pytest.approx(bins.mean() ** 2 - bins.var(ddof=1) / bins.size)
    assert value < bins.mean() ** 2


def test_jackknife_error_belongs_to_the_corrected_value():
    # the pseudovalues are what the jackknife averages, so the error it quotes has to be
    # their standard error, not the spread of the uncorrected estimate
    rng = np.random.default_rng(10)
    numerator = rng.normal(4.0, 0.3, 30)
    denominator = rng.normal(2.0, 0.2, 30)

    value, error = jackknife(lambda a, b: a / b, numerator, denominator)

    nbins = numerator.size
    leave_one_out = np.array([
        np.delete(numerator, i).mean() / np.delete(denominator, i).mean()
        for i in range(nbins)
    ])
    pseudovalues = nbins * (numerator.mean() / denominator.mean()) - (nbins - 1) * leave_one_out

    assert value == pytest.approx(pseudovalues.mean())
    assert error == pytest.approx(standard_error(pseudovalues))


def test_jackknife_leaves_a_linear_function_alone():
    rng = np.random.default_rng(9)
    bins = rng.normal(size=(25, 2))

    value, _ = jackknife(lambda x: 2 * x + 1, bins)

    np.testing.assert_allclose(value, 2 * bins.mean(axis=0) + 1)


def test_jackknife_leaves_the_same_bin_out_of_every_series():
    # a - a is identically zero for every resample only if both series lose the same
    # bin, so any error at all here means the correlation between them was broken
    rng = np.random.default_rng(6)
    bins = rng.normal(size=(30, 2))

    _, error = jackknife(lambda a, b: a - b, bins, bins)

    np.testing.assert_allclose(error, 0.0, atol=1e-12)


def test_jackknife_of_a_ratio_agrees_with_error_propagation():
    rng = np.random.default_rng(7)
    numerator = rng.normal(4.0, 0.3, 200)
    denominator = rng.normal(2.0, 0.1, 200)

    value, error = jackknife(lambda a, b: a / b, numerator, denominator)

    propagated = abs(value) * np.hypot(
        standard_error(numerator) / numerator.mean(),
        standard_error(denominator) / denominator.mean(),
    )
    assert error == pytest.approx(propagated, rel=0.2)


def test_jackknife_rejects_series_of_different_length():
    bins = np.ones((10, 1))

    with pytest.raises(ValueError, match='bin count'):
        jackknife(lambda a, b: a + b, bins, bins[:5])


def test_jackknife_needs_more_than_one_bin():
    with pytest.raises(ValueError, match='at least 2 bins'):
        jackknife(lambda x: x, np.ones((1, 1)))
