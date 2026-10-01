# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Generic statistics primitives for equilibrated Monte Carlo time series."""

import numpy as np
from scipy.signal import correlate
from scipy.special import gammaln, kve

# spreads up to this are taken for roundoff
_ROUNDOFF = 1e-9


def optimal_rebinsize(samples, skip=0):
    """Choose the rebin size for `rebinning_analysis` from the series itself.

    The series is blocked in powers of two (Flyvbjerg and Petersen, J. Chem. Phys. 91, 461
    (1989)) and the smallest block size B with ``B**3 > 2*N*kappa_B**2`` is taken, N being
    the number of samples and kappa_B the statistical inefficiency seen at that block
    size, the squared ratio of the blocked to the unblocked standard error. This is the
    criterion of Lee et al., Phys. Rev. E 83, 066706 (2011), also used by pyblock: it
    balances the correlation left inside a bin against the noise of having few bins.

    As in `rebinning_analysis`, one size serves every component, the one the slowest of
    them needs; the real and imaginary parts of a complex series count as components of
    their own. Series that are to be combined, as in `jackknife`, have to share one bin
    count, so rebin all of them with the largest of their sizes.

    Returns the rebin size and whether it meets the criterion. If no size does, the series
    is too short for its autocorrelation time, and the size returned is the compromise of
    `_fallback_rebinsize`, whose error bar is underestimated.
    """
    samples = np.asarray(samples)[skip:, ...]
    num_samples = samples.shape[0]
    if num_samples < 2:
        # nothing to block; rebinning_analysis reports the lack of bins
        return 1, True

    components = samples.reshape(num_samples, -1)
    if np.iscomplexobj(components):
        components = np.concatenate([components.real, components.imag], axis=1)

    rebins = components
    rebinsize = 1
    base_variance = None
    while True:
        # squared standard error of the mean over this level's rebins
        variance = np.var(rebins, axis=0, ddof=1) / rebins.shape[0]
        if base_variance is None:
            base_variance = variance
        # a component that never fluctuates enters as the neutral kappa = 1
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.where(base_variance > 0.0, variance / base_variance, 1.0)
        if rebinsize**3 > 2 * num_samples * np.max(kappa) ** 2:
            return rebinsize, True

        # the next level has to leave at least two rebins for a variance
        if rebins.shape[0] < 4:
            return _fallback_rebinsize(components), False
        pairs = rebins.shape[0] // 2
        rebins = 0.5 * (rebins[0 : 2 * pairs : 2] + rebins[1 : 2 * pairs : 2])
        rebinsize *= 2


def _fallback_rebinsize(components):
    """Rebin size for a series too short for the criterion of `optimal_rebinsize`, given as
    its real components along the second axis.

    The criterion compares statistical inefficiencies estimated from the blocks
    themselves, which a series this short leaves too few of; the autocorrelation function
    still gives one, kappa, that of the slowest component. A bin of size B then misses a
    fraction ``(kappa**2 - 1)/(2*kappa*B)`` of the variance of the mean, as it does for
    an exponentially decaying autocorrelation, while m bins leave the variance itself
    uncertain by a relative ``2/(m - 1)``. The size with the smallest sum of the squared
    bias and that variance is taken; the largest size that leaves two bins, which has the
    least bias, would leave an error bar of a single degree of freedom.
    """
    num_samples = components.shape[0]
    kappa = max(_statistical_inefficiency(components[:, j]) for j in range(components.shape[1]))
    rebinsizes = np.arange(1, num_samples // 2 + 1)
    rebincounts = num_samples // rebinsizes
    squared_error = ((kappa**2 - 1) / (2 * kappa * rebinsizes)) ** 2 + 2 / (rebincounts - 1)
    return int(rebinsizes[np.argmin(squared_error)])


def rebinning_analysis(samples, *, rebinsize, skip=0):
    """Rebin a time series into decorrelated bins and estimate its autocorrelation time.

    The leading axis of `samples` runs over samples, every other index is a component.
    Returns the bin means, which the caller reduces to a mean and a standard error
    ``std(bins, ddof=1)/sqrt(len(bins))``, and the autocorrelation time tau, defined
    through the statistical inefficiency ``kappa = 1 + 2*tau`` so that uncorrelated
    samples give tau = 0. Components are correlated with each other, but that does not
    enter a per-component error bar; only correlation along the series does. One tau is
    reported for the whole vector, the largest over its components, since a bin size
    has to be long enough for the slowest of them. `optimal_rebinsize` chooses a
    `rebinsize` from the series.
    """
    samples = np.asarray(samples)[skip:, ...]
    num_samples = samples.shape[0]

    rebinsize = int(rebinsize)

    rebincount = num_samples // rebinsize
    if rebincount < 2:
        raise ValueError(
            f'{num_samples} samples in bins of {rebinsize} leave {rebincount} bins, '
            'too few to estimate an error'
        )
    samples = samples[: rebincount * rebinsize, ...]

    naive_std = np.std(samples, axis=0, ddof=1)
    # the bin index is the slow one: each bin has to average consecutive samples
    rebin_means = np.mean(samples.reshape(rebincount, rebinsize, *samples.shape[1:]), axis=1)
    rebin_std = np.std(rebin_means, axis=0, ddof=1)

    # kappa is the squared ratio of the error over the bins to the error the samples
    # would have if they were independent. A component that never fluctuates says
    # nothing about the correlation time, so it enters as the neutral kappa = 1.
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = rebinsize * (rebin_std / naive_std) ** 2
    kappa = np.where(naive_std > 0.0, kappa, 1.0)
    autocorrtime = max(0.5 * float(np.max(kappa) - 1.0), 0.0)

    return rebin_means, autocorrtime


def _cramer_von_mises_sf(statistic):
    """Upper tail ``P(W > statistic)`` of ``W = int_0^1 B(t)**2 dt``, B a Brownian bridge.

    Summed from the series of Anderson and Darling, Ann. Math. Stat. 23, 193 (1952), whose
    terms fall off monotonically.
    """
    if statistic <= 0.0:
        return 1.0
    # an infinite statistic would never let the terms fall off
    if np.isinf(statistic):
        return 0.0

    cdf = 0.0
    j = 0
    while True:
        y = 4 * j + 1
        q = y**2 / (16 * statistic)
        # kve(nu, q) = kv(nu, q) * exp(q), so exp(-2q) leaves the exp(-q) * kv(nu, q) wanted
        term = (
            np.exp(gammaln(j + 0.5) - gammaln(j + 1) - 2 * q)
            * np.sqrt(y)
            * kve(0.25, q)
            / (np.pi**1.5 * np.sqrt(statistic))
        )
        cdf += term
        # a large statistic has many terms of similar size, so stopping any earlier than
        # machine precision leaves a remainder that swamps its tiny upper tail
        if term < 1e-17:
            break
        j += 1
    return max(1.0 - cdf, 0.0)


def _statistical_inefficiency(samples, window_factor=5.0):
    """Statistical inefficiency ``kappa = 1 + 2*sum_t rho(t)`` of a real series, rho being
    its normalized autocorrelation function.

    The sum runs up to the smallest window M with ``M >= window_factor * kappa(M)``, the
    self-consistent window of Madras and Sokal, J. Stat. Phys. 50, 109 (1988), beyond which
    the noise of rho would outweigh its signal. A series too short for any window to
    qualify is summed over every lag it has.

    kappa is floored at 1: the noise of a short window often sums to less, which would
    understate a variance taken from it, and a Monte Carlo series is not anticorrelated.
    A series too short to correlate, or constant up to roundoff, has nothing to measure
    and gets the neutral 1.
    """
    num_samples = samples.shape[0]
    if num_samples < 2 or np.ptp(samples) <= _ROUNDOFF:
        return 1.0

    deviations = samples - samples.mean()
    # the full correlation runs over lags -(N-1) to N-1, the second half of it over t >= 0;
    # divided by N rather than N - t, which keeps the estimator positive semidefinite
    autocovariance = (
        correlate(deviations, deviations, mode='full', method='fft')[num_samples - 1 :]
        / num_samples
    )
    kappa = 1.0 + 2.0 * np.cumsum(autocovariance[1:] / autocovariance[0])

    windows = np.arange(1, num_samples)
    qualifies = windows >= window_factor * kappa
    return max(float(kappa[np.argmax(qualifies) if qualifies.any() else -1]), 1.0)


def _bridge_pvalue(samples):
    """`stationarity_pvalue` of a real series, or None if there is nothing to test: the
    series is too short, or constant up to roundoff."""
    num_samples = samples.shape[0]
    # a series that moves by no more than roundoff cannot drift, and one shorter than four
    # leaves a second half of a single sample
    if num_samples < 4 or np.ptp(samples) <= _ROUNDOFF:
        return None

    # the long-run variance comes from the second half, the part more likely to be
    # stationary; a drift in the first half would otherwise inflate it and hide itself
    second_half = samples[num_samples // 2 :]
    # it moved in the first half and then froze
    if np.ptp(second_half) <= _ROUNDOFF:
        return 0.0
    # The long-run variance is the variance times the statistical inefficiency, the
    # standard error of the mean being sigma/sqrt(N_eff) with N_eff = N/kappa. Chosen
    # because it is smooth in skip.
    longrun_variance = _statistical_inefficiency(second_half) * np.var(second_half)

    bridge = np.cumsum(samples - samples.mean()) / np.sqrt(num_samples * longrun_variance)
    return _cramer_von_mises_sf(np.mean(bridge**2))


def stationarity_pvalue(samples, skip=0):
    """Test a scalar time series for a drift, such as an initial transient not yet decayed.

    The partial sums of the deviations from the mean of a stationary series, scaled by
    ``sqrt(N)`` times its long-run standard deviation, approach a Brownian bridge B(t) as
    the number of samples N grows; a drift makes them run off to one side instead. The
    statistic is ``int_0^1 B(t)**2 dt``, as in the stationarity test of Heidelberger and
    Welch, Oper. Res. 31, 1109 (1983), who also take the long-run variance from the second
    half of the series only. Here it is the variance of the second half times its
    statistical inefficiency, summed from its autocorrelation function, so that the
    p-value changes smoothly with `skip`.

    Returns the p-value, the probability of a statistic at least this large if the series
    were stationary, so a small value is evidence of a drift. The real and imaginary parts
    of a complex series are tested apart, and the smaller p-value is returned with a
    Bonferroni correction for the number of parts tested.

    A series, or a real or imaginary part, whose spread is no more than 1e-9 counts as
    constant up to roundoff, and a constant one cannot drift, so it is left untested. A
    series that moves and then stays constant has drifted.
    """
    samples = np.asarray(samples)[skip:]
    if samples.ndim != 1:
        raise ValueError(
            f'the stationarity test takes a scalar series, not one of shape {samples.shape}'
        )

    parts = (samples.real, samples.imag) if np.iscomplexobj(samples) else (samples,)
    pvalues = [p for p in (_bridge_pvalue(part) for part in parts) if p is not None]
    if not pvalues:
        return 1.0
    return min(len(pvalues) * min(pvalues), 1.0)


def standard_error(bins):
    """Return the standard error of the mean over the leading axis of `bins`.

    The bins are taken to be independent of one another, which is what the rebinning in
    `rebinning_analysis` is for. The real and imaginary parts of complex bins get errors
    of their own: a complex `np.std` would fold the two into one magnitude, which is not
    what goes on the error bar of a quantity whose parts are read separately.
    """
    if np.iscomplexobj(bins):
        std = bins.real.std(axis=0, ddof=1) + 1j * bins.imag.std(axis=0, ddof=1)
    else:
        std = bins.std(axis=0, ddof=1)
    return std / np.sqrt(bins.shape[0])


def jackknife(func, *binned):
    """Jackknife value and error of `func` applied to the means of one or more binned series.
    """
    counts = {bins.shape[0] for bins in binned}
    if len(counts) != 1:
        raise ValueError(
            'jackknife needs one or more series agreeing in bin count, got '
            f'{[bins.shape[0] for bins in binned]}'
        )

    nbins = counts.pop()
    if nbins < 2:
        raise ValueError(f'a jackknife needs at least 2 bins, but there is only {nbins}')

    totals = [bins.sum(axis=0) for bins in binned]

    def without(index):
        return func(*(
            (total - bins[index]) / (nbins - 1)
            for total, bins in zip(totals, binned, strict=True)
        ))

    leave_one_out = np.array([without(index) for index in range(nbins)])
    plug_in = func(*(bins.mean(axis=0) for bins in binned))
    pseudovalues = nbins * plug_in - (nbins - 1) * leave_one_out

    return pseudovalues.mean(axis=0), standard_error(pseudovalues)
