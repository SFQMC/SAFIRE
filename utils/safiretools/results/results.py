# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""The observables of a finished AFQMC run, as written to its results.h5.

One `execute` block of the input writes its observables below a `Stage<N>` group of its
own, each of them as a `bins` dataset whose leading axis is the bin index. A bin is
already divided by the walker weight denominator when it is measured, so averaging over
bins is an unweighted mean.
"""

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from warnings import simplefilter, warn

import h5py as h5
import numpy as np

from safiretools.hdf5 import check_format_version, read_complex
from safiretools.results.stats import (
    jackknife,
    optimal_rebinsize,
    rebinning_analysis,
    standard_error,
    stationarity_pvalue,
)

# one `execute` block of the input writes its observables below one of these groups
STAGE_GROUP = re.compile(r"^Stage(\d+)$")


class RebinningWarning(UserWarning):
    """The bins of an observable do not support the average and error bar taken from them:
    either the series is too short for its autocorrelation time, so the error bar is
    underestimated, or it drifts, so the average is biased."""


# Python shows a warning once per line and text by default, which would hide the same
# observable failing again in another run. Appended, so that filters set from the command
# line or by the caller still decide.
simplefilter("always", RebinningWarning, append=True)


def _checked_rebinsize(name, series, skip, rebinsize):
    """Rebin size for the series of observable `name`, warning about what undermines it.

    A scalar series is put to `stationarity_pvalue` and taken to drift at 99.99%
    confidence. Without a `rebinsize`, the one `optimal_rebinsize` chooses is returned.
    """
    if series.ndim == 1:
        pvalue = stationarity_pvalue(series, skip)
        if pvalue < 0.0001:
            warn(
                f"'{name}' may not be equilibrated: if it were, a drift this large would have a "
                f"p-value of {pvalue:.2g}. Inspect `result.timeseries(\"{name}\")` visually and "
                f"pass a higher `skip` parameter to skip the unequilibrated part.",
                RebinningWarning,
                stacklevel=3,
            )

    if rebinsize is None:
        rebinsize, meets_criterion = optimal_rebinsize(series, skip)
        if not meets_criterion:
            num_bins = series[skip:].shape[0]
            warn(
                f"'{name}' is too short for its autocorrelation time: no rebin size meets "
                f"the rebinning criterion for its {num_bins} bins after skip. Defaulting to "
                f"{num_bins // rebinsize} rebins of size {rebinsize}.",
                RebinningWarning,
                stacklevel=3,
            )
    return rebinsize


class Results:
    """The observables of one stage of a results.h5.

    One `execute` block of the input writes its observables below a ``Stage<N>`` group of
    its own, and a `Results` reads one such stage. The file is opened for each read rather
    than held open, so a run still writing to it is picked up on the next call.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the results.h5 an AFQMC run wrote.
    stage : int or str, optional
        Stage to read, either by index or by group name, so ``0`` and ``"Stage0"`` both
        select the first one. Defaults to the last stage in the file. Pass ``""`` to read
        every stage at once, which puts the stage name on the front of each observable
        path — ``"Stage0/Energy"`` rather than ``"Energy"``.

    Raises
    ------
    ValueError
        If the file was written in another format version, if it has no ``Measurements``
        group, if there is no ``Stage<N>`` group below it, or if the file has no stage by
        the requested name.

    Examples
    --------
    >>> results = Results("afqmc.results.h5")
    >>> results.observable_names()
    ['BackPropEstimator/Steps=40/OneRDM', 'Energy']

    mean and error of one observable, dropping the first 20 bins as equilibration

    >>> energy, error = results.average("Energy", skip=20)

    a quantity built from several observables, whose correlations the error accounts for

    >>> ratio, error = results.evaluate(
    ...     lambda energy, rdm: energy.real / rdm[0].trace().real,
    ...     ["Energy", "BackPropEstimator/Steps=40/OneRDM"],
    ...     skip=20,
    ... )
    """

    def __init__(self, filename: str | Path, stage: int | str | None = None) -> None:
        self.filename = filename

        with h5.File(self.filename, "r") as f:
            check_format_version(f)
            measurements = f.get("Measurements")
            if measurements is None:
                raise ValueError(f"'{filename}' has no Measurements group")

            stages = sorted(
                (name for name in measurements if STAGE_GROUP.match(name)),
                key=lambda name: int(STAGE_GROUP.match(name).group(1)),
            )
            if not stages:
                raise ValueError(f"'{filename}' has no Stage<N> group below Measurements")

            if stage is None:
                stage = stages[-1]
            elif not isinstance(stage, str):
                stage = f"Stage{stage}"
            # "" is every stage, so there is nothing to look up
            if stage and stage not in stages:
                raise ValueError(f"'{filename}' has no {stage}, only {', '.join(stages)}")

        self.stage: str = stage

    def observable_names(self) -> list[str]:
        """Every observable this stage measured, or every stage's when `stage` is ``""``.

        Returns
        -------
        list of str
            The '/'-separated path of each observable below the stage group, sorted. A
            back-propagated observable keeps the estimator and the back-propagation length
            in its path, as in ``BackPropEstimator/Steps=40/OneRDM``; reading every stage
            puts the stage on the front as well.
        """
        names = []
        with h5.File(self.filename, "r") as f:
            meas_group = f[f"Measurements/{self.stage}"]

            def visit(name, obj):
                head, _, tail = name.rpartition("/")
                if tail == "bins" and isinstance(obj, h5.Dataset):
                    names.append(head)

            meas_group.visititems(visit)

        return sorted(names)

    def timeseries(self, observable_name: str) -> np.ndarray:
        """Read the bins of one observable.

        Parameters
        ----------
        observable_name : str
            Path of the observable below the stage group, as `Results.observable_names`
            lists it.

        Returns
        -------
        numpy.ndarray
            The bins, the leading axis running over them and the remaining axes carrying
            the shape of the observable. A complex observable comes back complex.

        Raises
        ------
        KeyError
            If this stage measured no observable by that name.
        """
        with h5.File(self.filename, "r") as f:
            dataset = f.get(f"Measurements/{self.stage}/{observable_name}/bins")
            if dataset is not None:
                return read_complex(dataset)

        raise KeyError(
            f"{self.stage or 'Measurements'} of '{self.filename}' has no observable "
            f"'{observable_name}'; it has {', '.join(self.observable_names())}"
        )

    def average(
        self, observable_name: str, skip: int = 0, rebinsize: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and stochastic error of one observable.

        The bins are rebinned before averaging, so that the error accounts for successive
        bins being correlated. Every component is averaged on its own: the components
        correlate with each other, but that does not enter their individual error bars,
        only correlation along the series does.

        Parameters
        ----------
        observable_name : str
            Path of the observable below the stage group, as `Results.observable_names`
            lists it.
        skip : int, optional
            Leading bins to drop as equilibration, default 0.
        rebinsize : int, optional
            Bins to average into one before taking the error. Defaults to the size
            `optimal_rebinsize` chooses for the bins left after `skip`.

        Returns
        -------
        mean : numpy.ndarray
            The observable averaged over its bins.
        error : numpy.ndarray
            Standard error of `mean`, component by component. For a complex observable the
            real and imaginary parts of `error` are the errors of the real and imaginary
            parts of `mean` rather than one combined magnitude.

        Warns
        -----
        RebinningWarning
            If the series is too short for its autocorrelation time, in which case `error`
            is too small, or if a scalar observable drifts after `skip`, in which case
            `mean` is biased.
        """
        series = self.timeseries(observable_name)
        rebinsize = _checked_rebinsize(observable_name, series, skip, rebinsize)
        bins, _ = rebinning_analysis(series, rebinsize=rebinsize, skip=skip)
        return bins.mean(axis=0), standard_error(bins)

    def evaluate(
        self,
        func: Callable[..., np.ndarray | complex],
        observable_names: Sequence[str],
        skip: int = 0,
        rebinsize: int | None = None,
    ) -> tuple[np.ndarray | complex, np.ndarray | complex]:
        """Mean and error of a quantity computed from the averages of several observables.

        If `func` is nonlinear, naive evaluation of `func(averages)` has a bias and getting
        its error would require knowledge of the correlation matrix. This function
        uses the jackknife method instead to correct the bias and calculate the errorbar.

        The jackknife requires that `func` is smooth in its arguments.

        Parameters
        ----------
        func : callable
            Called as ``func(*averages)``, one argument per name in `observable_names` and
            in that order, each of them that observable averaged over its bins. It may
            return a scalar or an array.
        observable_names : sequence of str
            Paths of the observables below the stage group, as `Results.observable_names`
            lists them. They all have to have the same number of bins so the jackknife lines
            up correctly.
        skip : int, optional
            Leading bins to drop as equilibration, default 0.
        rebinsize : int, optional
            Bins to average into one before the jackknife, the same for every observable.
            Defaults to the largest of the sizes `optimal_rebinsize` chooses for them, so
            that the slowest observable is rebinned far enough.

        Returns
        -------
        value : numpy.ndarray or scalar
            `func` evaluated on the averages of the observables, with the bias of applying
            a nonlinear `func` to averages taken out. The correction is identically zero
            for a linear `func`.
        error : numpy.ndarray or scalar
            Jackknife error of `value`. For a complex `value` the real and imaginary parts
            of `error` are the errors of the real and imaginary parts of `value`.

        Raises
        ------
        ValueError
            If the observables do not all have the same number of bins, or if fewer than
            two bins are left to resample.

        Warns
        -----
        RebinningWarning
            If one of the series is too short for its autocorrelation time, in which case
            `error` is too small, or if a scalar observable among them drifts after `skip`,
            in which case `value` is biased.
        """
        observable_names = list(observable_names)
        series = [self.timeseries(name) for name in observable_names]

        # every series is rebinned with one size, so they line up for the jackknife exactly
        # when they are equally long; checked first, since a size chosen for the longer ones
        # can leave a shorter one without enough bins to get that far
        if len({values.shape[0] for values in series}) != 1:
            raise ValueError(
                'observables can only be combined if they share a bin count, but '
                + ', '.join(
                    f"'{name}' has {values.shape[0]}"
                    for name, values in zip(observable_names, series, strict=True)
                )
            )

        # a loop rather than a comprehension, so the warnings point at the caller's frame
        rebinsizes = []
        for name, values in zip(observable_names, series, strict=True):
            rebinsizes.append(_checked_rebinsize(name, values, skip, rebinsize))
        rebinsize = max(rebinsizes)
        binned = [
            rebinning_analysis(values, rebinsize=rebinsize, skip=skip)[0] for values in series
        ]

        return jackknife(func, *binned)
