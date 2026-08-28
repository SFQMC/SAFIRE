# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Generic statistics primitives for equilibrated Monte Carlo time series.
"""

import numpy as np
from numba import jit


@jit(nopython=True)  # pragma: no cover
def stddev(values):
    """
    Standard deviation, computed as
    ``1/(N-1) * sum_i |values[i] - mean(values)|**2``.

    Equivalent to ``np.std(values, ddof=1)``, which does not compile in
    Numba (``np.std(values)`` without ``ddof`` does).
    """
    n = len(values)
    mean = np.mean(values)
    s = 0
    for i in range(n):
        s += (values[i] - mean) ** 2
    return np.sqrt(s / (n - 1))


@jit(nopython=True)  # pragma: no cover
def corr(values: np.ndarray):
    """
    Compute the autocorrelation length of the input values.

    The autocorrelation length, l, of values a(i) is given by::

        l = 1 + sum_{k=1}^n ( sum_{i=1}^{n-k} t(a,i,k) )

    where::

        t(a,i,k) = T(a,i,k) if T(a,i,k) > 0.0 else 0.0
        T(a,i,k) = 2/sigma^2 * (a(i) - mean(a)) * (a(i+k) - mean(a))

    Returns ``np.inf`` for constant input (zero variance).
    """
    mean_value = np.mean(values)
    sigma = stddev(values)

    if np.isclose(sigma, 0.0):
        return np.inf
    n = values.shape[0]
    corr_len = 0.0
    for k in range(1, n + 1):
        ct = 0
        for i in range(1, n - k + 1):
            ct += (values[i - 1] - mean_value) * (values[(i - 1) + k] - mean_value)
        ct = (ct / sigma**2) / (n - k)
        if ct <= 0.0:
            break
        corr_len = corr_len + 2 * ct

    return corr_len + 1.0


def _real_mean_and_error(data, axis):
    """Mean, error, and autocorrelation length along ``axis`` for real data.

    Constant (zero-variance) slices along ``axis`` are detected up front from
    their standard deviation and assigned zero error directly, rather than
    inspecting the autocorrelation length for NaN/Inf after the fact.
    """
    ntrace = data.shape[axis]
    ymean = data.mean(axis=axis)
    ystd = data.std(axis=axis, ddof=1)
    degenerate = np.isclose(ystd, 0.0)

    kappa = np.apply_along_axis(corr, axis, data)
    kappa = np.where(degenerate, np.inf, kappa)

    with np.errstate(divide='ignore', invalid='ignore'):
        neffective = ntrace / kappa
        yerr = ystd / np.sqrt(neffective)
    yerr = np.where(degenerate, 0.0, yerr)

    return ymean, yerr, kappa


def mean_and_error(samples, axis=0):
    """
    Mean and standard error of a time series, corrected for autocorrelation.

    Works over arbitrary-shape data: a 1D scalar trace or a stack of RDM
    matrices are both averaged over ``axis``, independently for every other
    index.

    Parameters
    ----------
    samples : array_like
        Equilibrated time series data. May be real or complex.
    axis : int, optional
        Axis to average over (the sample axis), default 0.

    Returns
    -------
    mean : np.ndarray
        Mean of ``samples`` along ``axis``.
    error : np.ndarray
        Standard error of the mean, corrected for autocorrelation. For
        complex ``samples``, the real and imaginary parts of the error are
        computed independently (as real quantities) rather than combined into
        a single magnitude-style error; the autocorrelation length is
        estimated from the real part and reused for both.
    """
    samples = np.asarray(samples)

    if np.iscomplexobj(samples):
        ntrace = samples.shape[axis]
        real, imag = samples.real, samples.imag

        real_std = real.std(axis=axis, ddof=1)
        imag_std = imag.std(axis=axis, ddof=1)
        real_degenerate = np.isclose(real_std, 0.0)
        imag_degenerate = np.isclose(imag_std, 0.0)

        kappa = np.apply_along_axis(corr, axis, real)
        kappa = np.where(real_degenerate, np.inf, kappa)

        with np.errstate(divide='ignore', invalid='ignore'):
            neffective = ntrace / kappa
            real_err = real_std / np.sqrt(neffective)
            imag_err = imag_std / np.sqrt(neffective)
        real_err = np.where(real_degenerate, 0.0, real_err)
        imag_err = np.where(imag_degenerate, 0.0, imag_err)

        ymean = samples.mean(axis=axis)
        yerr = real_err + 1j * imag_err
        return ymean, yerr

    ymean, yerr, _ = _real_mean_and_error(samples, axis)
    return ymean, yerr


def _reblock_backend(data, blocksize):
    if blocksize == 1:
        return data
    return data.reshape(-1, blocksize, *data.shape[1:]).mean(axis=1)


def reblock(data, blocksize):
    """
    Reblock data along its first axis to reduce autocorrelation.

    Parameters
    ----------
    data : np.array
        The data to be reblocked. The leading dimension is the number of
        samples; remaining axes are unchanged.
    blocksize : int
        The size of each block. If the leading dimension is not evenly
        divisible by ``blocksize``, the trailing remainder samples form one
        smaller final block rather than being discarded.

    Returns
    -------
    np.array
        Reblocked data, with the first axis now the number of blocks.
    """
    data = np.asarray(data)
    if data.shape[0] % blocksize != 0:
        n_full_blocks = data.shape[0] // blocksize
        incomplete_block_size = data.shape[0] % blocksize
        data_first = _reblock_backend(data[: n_full_blocks * blocksize], blocksize)
        data_last = _reblock_backend(data[-incomplete_block_size:], incomplete_block_size)
        return np.concatenate((data_first, data_last), axis=0)
    return _reblock_backend(data, blocksize)
