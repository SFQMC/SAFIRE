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
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import gammaln, kve

# spreads up to this are taken for roundoff
_ROUNDOFF = 1e-9
# single-precision complex elements one batch of autocorrelation FFTs may hold, 128 MB
_FFT_CHUNK_ELEMENTS = 2**24


def optimal_rebinsize(samples, skip=0):
    """Choose the rebin size for `rebinning_analysis` from the series itself.

    A bin of size B misses a fraction ``(kappa**2 - 1)/(2*kappa*B)`` of the variance of the
    mean, as it does for an exponentially decaying autocorrelation with statistical
    inefficiency kappa, while m bins leave the variance itself uncertain by a relative
    ``2/(m - 1)``. The size with the smallest sum of the squared bias and that variance is
    taken, about ``B**3 = N*((kappa**2 - 1)/(2*kappa))**2``, so an uncorrelated series is
    not rebinned. kappa comes from the autocorrelation function of the slowest component;
    estimating it from the bins themselves cannot work, as it is exactly 1 at B = 1. The
    largest of many estimates overshoots the largest kappa there is, so the slowest
    component is chosen on one half of the series and its kappa estimated on the other,
    and the two ways round are averaged.
    """
    samples = np.asarray(samples)[skip:, ...]
    num_samples = samples.shape[0]
    if num_samples < 2:
        # nothing to rebin; rebinning_analysis reports the lack of bins
        return 1

    components = samples.reshape(num_samples, -1)
    if np.iscomplexobj(components):
        components = np.concatenate([components.real, components.imag], axis=1)

    first, second = (_statistical_inefficiency(half) for half in np.array_split(components, 2))
    kappa = 0.5 * float(second[np.argmax(first)] + first[np.argmax(second)])
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

    Summed from the series of Anderson and Darling, Ann. Math. Stat. 23, 193 (1952).
    """
    # the cdf is below 1e-17 here, and kve turns nan for a statistic below about 6e-11
    if statistic < 1e-3:
        return 1.0
    if statistic > 4.3:
        return 0.0

    cdf = 0.0
    for j in range(10):
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

        if term < 1e-13:
            break
    return max(1.0 - cdf, 0.0)


def _statistical_inefficiency(samples):
    """Statistical inefficiency ``kappa = 1 + 2*sum_t rho(t)`` of a real series, rho being
    its normalized autocorrelation function; of each column of a two-dimensional one, whose
    leading axis runs over samples.

    The sum is the initial positive sequence estimator of Geyer, Stat. Sci. 7, 473 (1992):
    rho is summed in pairs ``rho(2j) + rho(2j + 1)``, which are positive for a reversible
    Markov chain, up to the first pair that is not, beyond which the noise of rho outweighs
    its signal. Unlike a window proportional to kappa it does not stop early on a series
    whose estimate happens to come out small.

    kappa is floored at 1: the noise of a short series often sums to less, which would
    understate a variance taken from it, and a Monte Carlo series is not anticorrelated.
    A series too short to correlate, or constant up to roundoff, has nothing to measure
    and gets the neutral 1.
    """
    num_samples = samples.shape[0]
    columns = samples[:, None] if samples.ndim == 1 else samples
    kappa = np.ones(columns.shape[1])
    if num_samples >= 2:
        moving = np.flatnonzero(np.ptp(columns, axis=0) > _ROUNDOFF)
        # padded to at least 2N - 1, so the circular correlation of the FFT has no lag
        # wrap around onto another; in chunks of columns, to bound the transform's memory
        num_fft = next_fast_len(2 * num_samples - 1, real=True)
        chunk = max(1, _FFT_CHUNK_ELEMENTS // num_fft)
        for start in range(0, moving.size, chunk):
            selected = moving[start : start + chunk]
            # take is a contiguous copy, several times faster than fancy indexing here
            kappa[selected] = _geyer_kappa(np.take(columns, selected, axis=1), num_fft)
    return float(kappa[0]) if samples.ndim == 1 else kappa


def _geyer_kappa(columns, num_fft):
    """`_statistical_inefficiency` of every column of `columns`, none of them constant,
    with the autocorrelation taken through FFTs of length `num_fft`.

    The transforms run along contiguous rows and in single precision, which halves their
    cost and leaves rho accurate to about 1e-7, far below its statistical noise.
    """
    num_samples, num_columns = columns.shape
    # the mean comes off in double precision: in single precision an offset large against
    # the fluctuations would swallow them
    deviations = np.empty((num_columns, num_samples), dtype=np.float32)
    np.subtract(columns.T, columns.mean(axis=0)[:, None], out=deviations, casting='same_kind')
    # divided by N rather than N - t, which keeps the estimator positive semidefinite
    power = np.abs(rfft(deviations, n=num_fft, axis=1)) ** 2
    autocovariance = irfft(power, n=num_fft, axis=1)[:, :num_samples] / num_samples
    autocorrelation = autocovariance / autocovariance[:, :1]

    pairs = autocorrelation[:, : num_samples // 2 * 2].reshape(num_columns, -1, 2).sum(axis=2)
    nonpositive = pairs <= 0.0
    num_positive = np.where(nonpositive.any(axis=1), nonpositive.argmax(axis=1), pairs.shape[1])
    positive_sum = np.sum(pairs, axis=1, where=np.arange(pairs.shape[1]) < num_positive[:, None])
    # the pairs sum rho from t = 0, which counts the leading 1 of kappa twice
    return np.maximum(2.0 * positive_sum.astype(np.float64) - 1.0, 1.0)


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
