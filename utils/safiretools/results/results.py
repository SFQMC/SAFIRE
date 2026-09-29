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

import h5py as h5
import numpy as np

from safiretools.hdf5 import check_format_version, read_complex
from safiretools.results.stats import jackknife, rebinning_analysis, standard_error

# one `execute` block of the input writes its observables below one of these groups
STAGE_GROUP = re.compile(r"^Stage(\d+)$")


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
            Bins to average into one before taking the error. Defaults to roughly the
            square root of the number of bins left after `skip`.

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
        UserWarning
            If the rebin size does not come out large compared to the autocorrelation time
            of the series, in which case `error` is too small.
        """
        bins, _ = rebinning_analysis(self.timeseries(observable_name), skip, rebinsize)
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
            Bins to average into one before the jackknife. Defaults to roughly the square
            root of the number of bins left after `skip`.

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
        UserWarning
            If the rebin size does not come out large compared to the autocorrelation time
            of one of the series, in which case `error` is too small.
        """
        observable_names = list(observable_names)
        binned = [
            rebinning_analysis(self.timeseries(name), skip, rebinsize)[0]
            for name in observable_names
        ]

        if len({bins.shape[0] for bins in binned}) != 1:
            raise ValueError(
                'observables can only be combined if they share a bin count, but '
                + ', '.join(
                    f"'{name}' has {bins.shape[0]}"
                    for name, bins in zip(observable_names, binned, strict=True)
                )
            )

        return jackknife(func, *binned)
