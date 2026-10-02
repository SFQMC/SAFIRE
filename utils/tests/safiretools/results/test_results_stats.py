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
import pytest

from safiretools.results.stats import (
    _cramer_von_mises_sf,
    _fallback_rebinsize,
    _statistical_inefficiency,
    jackknife,
    optimal_rebinsize,
    rebinning_analysis,
    standard_error,
    stationarity_pvalue,
)


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


def drifting(rng, rho=0.9, num_samples=20000):
    """An `ar1` series with a decay from five standard deviations off added, slow compared
    to its correlation time."""
    steps = np.arange(num_samples)
    return ar1(num_samples, rho, rng) + 5 / np.sqrt(1 - rho**2) * np.exp(-steps / 500)


def test_bins_average_consecutive_samples():
    # the bin index has to be the slow one; averaging strided samples instead would
    # decorrelate the bins by construction and hide every correlation there is
    bins, _ = rebinning_analysis(np.arange(12.0), rebinsize=4)
    np.testing.assert_allclose(bins, [1.5, 5.5, 9.5])


def test_remainder_samples_are_discarded():
    bins, _ = rebinning_analysis(np.arange(10.0), rebinsize=4)
    np.testing.assert_allclose(bins, [1.5, 5.5])


def test_skip_drops_leading_samples():
    bins, _ = rebinning_analysis(np.arange(12.0), skip=4, rebinsize=4)
    np.testing.assert_allclose(bins, [5.5, 9.5])


def test_component_shape_survives_rebinning():
    bins, _ = rebinning_analysis(np.arange(24.0).reshape(12, 2), rebinsize=4)

    assert bins.shape == (3, 2)
    np.testing.assert_allclose(bins[:, 0], [3.0, 11.0, 19.0])


def test_uncorrelated_series_has_no_autocorrelation_time():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=20000)
    rebinsize, _ = optimal_rebinsize(samples)

    bins, tau = rebinning_analysis(samples, rebinsize=rebinsize)

    assert tau == pytest.approx(0.0, abs=0.5)
    assert standard_error(bins) == pytest.approx(1 / np.sqrt(samples.size), rel=0.25)


def test_autocorrelated_series_recovers_its_correlation_time():
    rng = np.random.default_rng(1)
    rho = 0.9

    samples = ar1(50000, rho, rng)
    rebinsize, _ = optimal_rebinsize(samples)

    _, tau = rebinning_analysis(samples, rebinsize=rebinsize)

    assert tau == pytest.approx(rho / (1 - rho), rel=0.3)


def test_correlation_widens_the_error_bar():
    rng = np.random.default_rng(2)
    samples = ar1(50000, 0.9, rng)
    rebinsize, _ = optimal_rebinsize(samples)

    bins, _ = rebinning_analysis(samples, rebinsize=rebinsize)

    independent = samples.std(ddof=1) / np.sqrt(samples.size)
    assert standard_error(bins) > 3 * independent


def test_constant_series_is_not_taken_for_a_correlated_one():
    samples = np.full((100, 4), 7.0)

    rebinsize, meets_criterion = optimal_rebinsize(samples)
    bins, tau = rebinning_analysis(samples, rebinsize=rebinsize)

    # an uncorrelated series needs no more than the noise floor (2N)**(1/3) sets
    assert (rebinsize, meets_criterion) == (8, True)
    assert tau == 0.0
    np.testing.assert_array_equal(standard_error(bins), np.zeros(4))


def test_uncorrelated_series_is_rebinned_at_the_noise_floor():
    # kappa = 1 leaves B**3 > 2N, and 64 is the first power of two past (2*20000)**(1/3)
    rng = np.random.default_rng(0)

    assert optimal_rebinsize(rng.normal(size=20000)) == (64, True)


def test_optimal_rebinsize_is_large_compared_to_the_correlation_time():
    rng = np.random.default_rng(1)
    rho = 0.9

    rebinsize, meets_criterion = optimal_rebinsize(ar1(50000, rho, rng))

    assert meets_criterion
    assert rebinsize > 10 * rho / (1 - rho)


def test_optimal_rebinsize_serves_the_slowest_component():
    rng = np.random.default_rng(12)
    fast = rng.normal(size=20000)
    slow = ar1(20000, 0.9, rng)

    assert optimal_rebinsize(slow)[0] > optimal_rebinsize(fast)[0]
    assert optimal_rebinsize(np.column_stack([fast, slow])) == optimal_rebinsize(slow)


def test_optimal_rebinsize_takes_real_and_imaginary_parts_apart():
    # a slow imaginary part must not be averaged away by a fast real one
    rng = np.random.default_rng(12)
    fast = rng.normal(size=20000)
    slow = ar1(20000, 0.9, rng)

    assert optimal_rebinsize(fast + 1j * slow) == optimal_rebinsize(slow)


def test_optimal_rebinsize_honours_skip():
    rng = np.random.default_rng(13)
    samples = np.concatenate([np.arange(1000.0), rng.normal(size=20000)])

    assert optimal_rebinsize(samples, skip=1000) == optimal_rebinsize(samples[1000:])


def test_a_series_too_short_for_the_criterion_is_reported():
    # a ramp has kappa_B = B, so B**3 > 2*N*B**2 never holds for B < N
    rebinsize, meets_criterion = optimal_rebinsize(np.arange(100.0))

    assert not meets_criterion
    assert 100 // rebinsize > 2


def test_a_short_correlated_series_keeps_enough_bins():
    # 180 samples at kappa = 19 fail the criterion; the largest size leaving two bins
    # would give an error bar of a single degree of freedom
    rebinsize, meets_criterion = optimal_rebinsize(ar1(180, 0.9, np.random.default_rng(6)))

    assert not meets_criterion
    assert 180 // rebinsize >= 5


def test_the_fallback_rebinsize_serves_the_slowest_component():
    fast = np.random.default_rng(7).normal(size=180)
    slow = ar1(180, 0.9, np.random.default_rng(5))

    assert _fallback_rebinsize(np.column_stack([fast, slow])) == _fallback_rebinsize(slow[:, None])
    assert _fallback_rebinsize(slow[:, None]) > _fallback_rebinsize(fast[:, None])


@pytest.mark.parametrize(
    'statistic, tail', [(0.347, 0.10), (0.461, 0.05), (0.743, 0.01), (1.168, 0.001)]
)
def test_cramer_von_mises_tail_matches_its_tabulated_critical_values(statistic, tail):
    assert _cramer_von_mises_sf(statistic) == pytest.approx(tail, rel=0.01)


def test_cramer_von_mises_tail_vanishes_for_a_large_statistic():
    for statistic in [10.0, 1e4, 1e6, np.inf]:
        assert _cramer_von_mises_sf(statistic) == 0.0
    # the series used to loop forever on a statistic this small, kve turning nan
    for statistic in [0.0, 1e-12]:
        assert _cramer_von_mises_sf(statistic) == 1.0
    assert np.isnan(_cramer_von_mises_sf(np.nan))


def test_a_stationary_series_is_not_taken_for_a_drifting_one():
    rng = np.random.default_rng(1)

    assert stationarity_pvalue(ar1(20000, 0.9, rng)) > 0.01


def test_stationary_series_are_flagged_at_the_nominal_rate():
    rng = np.random.default_rng(42)

    pvalues = np.array([stationarity_pvalue(rng.normal(size=2000)) for _ in range(300)])

    assert np.mean(pvalues < 0.05) == pytest.approx(0.05, abs=0.03)


def test_short_stationary_series_are_flagged_at_the_nominal_rate():
    # a short window often sums the autocorrelation noise to kappa < 1, which the floor
    # at 1 keeps from understating the variance
    rng = np.random.default_rng(42)

    pvalues = np.array([stationarity_pvalue(rng.normal(size=120)) for _ in range(1000)])

    assert np.mean(pvalues < 0.05) == pytest.approx(0.05, abs=0.02)


def test_statistical_inefficiency_recovers_that_of_an_ar1_series():
    rho = 0.9

    kappa = _statistical_inefficiency(ar1(20000, rho, np.random.default_rng(3)))

    assert kappa == pytest.approx((1 + rho) / (1 - rho), rel=0.2)
    assert _statistical_inefficiency(np.random.default_rng(4).normal(size=20000)) == pytest.approx(1.0, abs=0.1)


def test_statistical_inefficiency_has_nothing_to_measure_in_a_constant_series():
    # an exactly constant series has no variance to normalize the autocorrelation by
    assert _statistical_inefficiency(np.full(50, 3.0)) == 1.0
    assert _statistical_inefficiency(1.0 + 1e-15 * np.arange(50.0)) == 1.0
    assert _statistical_inefficiency(np.zeros(0)) == 1.0
    assert _statistical_inefficiency(np.ones(1)) == 1.0


def test_statistical_inefficiency_is_at_least_one():
    # the noise of a short window often sums to less than 1 for an uncorrelated series
    rng = np.random.default_rng(8)

    assert min(_statistical_inefficiency(rng.normal(size=60)) for _ in range(100)) == 1.0


def test_an_initial_transient_is_detected():
    samples = drifting(np.random.default_rng(2))

    assert stationarity_pvalue(samples) < 1e-6
    assert stationarity_pvalue(samples, skip=5000) > 0.01


def test_the_pvalue_moves_smoothly_with_skip():
    # a rebinned long-run variance jumps where the rebin size changes power of two, and
    # the p-value with it by up to an order of magnitude per few skipped samples
    samples = drifting(np.random.default_rng(2))

    log_pvalues = np.log10([stationarity_pvalue(samples, skip=skip) for skip in range(1000, 6001, 10)])

    assert np.max(np.abs(np.diff(log_pvalues))) < 0.2


def test_a_constant_series_is_stationary():
    assert stationarity_pvalue(np.full(100, 7.0)) == 1.0


def test_a_constant_series_is_stationary_up_to_roundoff():
    steps = np.arange(1000)
    # a step of one ulp halfway, which leaves the second half exactly constant
    step = np.concatenate([np.full(500, 0.1), np.full(500, np.nextafter(0.1, 1.0))])
    # a perfectly smooth drift, but one of roundoff size
    decay = 1.0 + 1e-15 * np.exp(-steps / 50)

    assert stationarity_pvalue(step) == 1.0
    assert stationarity_pvalue(decay) == 1.0


def test_a_series_that_moves_and_then_freezes_has_drifted():
    rng = np.random.default_rng(1)

    assert stationarity_pvalue(np.concatenate([rng.normal(size=500), np.full(500, 0.25)])) == 0.0


def test_a_strictly_alternating_series_is_stationary():
    # every bin averages to the same value, so there is no long-run variance to scale by
    assert stationarity_pvalue(np.tile([0.3, -0.1], 500)) == 1.0


def test_a_roundoff_imaginary_part_is_not_counted_as_tested():
    # a real observable carried as complex must not have its p-value doubled
    rng = np.random.default_rng(2)
    steps = np.arange(1000)
    real = rng.normal(size=steps.size)

    assert stationarity_pvalue(real + 1j * 1e-17 * np.exp(-steps / 50)) == stationarity_pvalue(real)
    assert stationarity_pvalue(real + 0j) == stationarity_pvalue(real)


def test_too_short_a_series_is_stationary():
    assert stationarity_pvalue(np.zeros(0)) == 1.0
    assert stationarity_pvalue(np.ones(1)) == 1.0
    assert stationarity_pvalue(np.arange(10.0), skip=10) == 1.0


def test_a_drift_in_the_imaginary_part_alone_is_detected():
    rng = np.random.default_rng(14)
    steps = np.arange(5000)
    real = rng.normal(size=steps.size)
    imag = rng.normal(size=steps.size)

    assert stationarity_pvalue(real + 1j * (imag + 3 * np.exp(-steps / 300))) < 1e-6
    assert stationarity_pvalue(real + 1j * imag) > 0.01


def test_the_stationarity_test_takes_only_a_scalar_series():
    with pytest.raises(ValueError, match='scalar'):
        stationarity_pvalue(np.zeros((100, 2)))


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
