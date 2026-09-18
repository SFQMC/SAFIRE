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

from warnings import warn

import numpy as np

def rebinning_analysis(samples, skip=0, rebinsize=None):
    """Rebin a time series into decorrelated bins and estimate its autocorrelation time.

    The leading axis of `samples` runs over samples, every other index is a component.
    Returns the bin means, which the caller reduces to a mean and a standard error
    ``std(bins, ddof=1)/sqrt(len(bins))``, and the autocorrelation time tau, defined
    through the statistical inefficiency ``kappa = 1 + 2*tau`` so that uncorrelated
    samples give tau = 0. Components are correlated with each other, but that does not
    enter a per-component error bar; only correlation along the series does. One tau is
    reported for the whole vector, the largest over its components, since a bin size
    has to be long enough for the slowest of them.
    """
    samples = np.asarray(samples)[skip:, ...]
    num_samples = samples.shape[0]

    if rebinsize is None:
        min_rebincount = 10
        if num_samples < min_rebincount:
            rebinsize = 1
        else:
            rebinsize = int(np.sqrt(num_samples - min_rebincount)) + 1
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

    # residual correlation within a bin biases the error bar low by about tau/(2*rebinsize)
    if rebinsize < 10 * autocorrtime:
        warn(
            f'rebin size {rebinsize} is not large compared to the autocorrelation time '
            f'{autocorrtime:.3g}, so the error bar from these {rebincount} bins is too '
            f'small by roughly {50 * autocorrtime / rebinsize:.0f}%. Sample longer or '
            f'pass a larger rebinsize.',
            stacklevel=2,
        )

    return rebin_means, autocorrtime


def _std(values):
    """Take the standard deviation over the leading axis, real and imaginary parts apart.

    A complex `np.std` would fold the two into one magnitude, which is not what goes on
    the error bar of a quantity whose parts are read separately.
    """
    if np.iscomplexobj(values):
        return values.real.std(axis=0, ddof=1) + 1j * values.imag.std(axis=0, ddof=1)
    return values.std(axis=0, ddof=1)


def standard_error(bins):
    """Return the standard error of the mean over the leading axis of `bins`.

    The bins are taken to be independent of one another, which is what the rebinning in
    `rebinning_analysis` is for.
    """
    return _std(bins) / np.sqrt(bins.shape[0])


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
