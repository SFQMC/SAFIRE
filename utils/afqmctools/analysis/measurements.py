# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Reading and averaging the observables in a SAFIRE results.h5.

PLACEHOLDER. This does the least that the functional tests need: pull every bin series out
of the file and average it the way the old stats module averaged a block series. It is meant
to be replaced by a more carefully considered analysis interface.
"""

import re

import h5py as h5

from afqmctools.utils.io import read_dataset
from stats.stat_h5 import me2d

# one `execute` block of the input writes its observables below one of these groups
STAGE_GROUP = re.compile(r"^Stage(\d+)$")


def _read_bins(filename):
    """{"Stage<N>/<observable path>": bins} for every execute stage in a results.h5."""
    observables = {}
    found_stage = False

    with h5.File(filename, "r") as f:
        for stage, group in f["Measurements"].items():
            if STAGE_GROUP.match(stage) is None:
                continue
            found_stage = True

            def visit(name, obj, stage=stage):
                head, _, tail = name.rpartition("/")
                if tail == "bins" and isinstance(obj, h5.Dataset):
                    observables[f"{stage}/{head}"] = read_dataset(obj)

            group.visititems(visit)

    if not found_stage:
        raise ValueError(
            f"'{filename}' has no Stage<N> group below Measurements. A file written before "
            "the stages moved into the results file is not readable here."
        )
    return observables


def read_measurements(filename):
    """
    Read every observable in a results.h5.

    Parameters
    ----------
    filename : str | pathlib.Path
        path to the results.h5 written at the end of an AFQMC run

    Returns
    -------
    dict
        {name: bins}, where name is the observable's path below `Measurements` without the
        trailing `bins` component, e.g. `Stage0/Energy` or
        `Stage0/BackPropEstimator/Steps=40/OneRDM`. One `execute` block of the input is one
        stage, so an ordinary run has only `Stage0` names.

    Notes
    -----
    The leading axis of each array is the bin index. Every bin is already divided by the
    walker weight denominator when it is measured, and the denominator itself is not stored,
    so the average over bins is an unweighted mean.
    """
    return _read_bins(filename)


def average_measurements(filename, nequil=0):
    """
    Average every observable in a results.h5 over its bins.

    Parameters
    ----------
    filename : str | pathlib.Path
        path to the results.h5 written at the end of an AFQMC run
    nequil : int, optional
        number of leading bins to discard from every observable, default 0

    Returns
    -------
    dict
        {name: (mean, stochastic error)}, keyed as in `read_measurements`. The error is the
        one the old stats module computed, std / sqrt(N / kappa), with kappa the
        autocorrelation length of the real part.
    """
    return {name: me2d(bins[nequil:]) for name, bins in _read_bins(filename).items()}
