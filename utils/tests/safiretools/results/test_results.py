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
import h5py as h5
import pytest

from safiretools.hdf5 import write_format_version
from safiretools.results.results import RebinningWarning, Results
from safiretools.results.stats import (
    jackknife,
    optimal_rebinsize,
    rebinning_analysis,
    standard_error,
)

RDM = 'BackPropEstimator/Steps=40/OneRDM'


def test_the_last_stage_is_the_default(results_file):
    assert Results(results_file).stage == 'Stage1'


def test_a_stage_can_be_named_by_index_or_by_group(results_file):
    assert Results(results_file, stage=0).stage == 'Stage0'
    assert Results(results_file, stage='Stage0').stage == 'Stage0'


def test_a_stage_the_file_does_not_have_is_rejected(results_file):
    with pytest.raises(ValueError, match='no Stage7'):
        Results(results_file, stage=7)


def test_the_empty_stage_reads_every_stage(results_file, measured):
    """``stage=""`` spans the whole file, so the stage is part of each path."""
    results = Results(results_file, stage='')

    assert results.observable_names() == sorted(
        [f'Stage0/{name}' for name in measured] + ['Stage1/Energy'])


def test_reading_every_stage_finds_the_same_series(results_file, measured):
    across = Results(results_file, stage='').timeseries('Stage0/Energy')
    one = Results(results_file, stage=0).timeseries('Energy')

    assert np.array_equal(across, one)


def test_a_file_without_measurements_is_rejected(tmp_path):
    path = tmp_path / 'no_measurements.h5'
    with h5.File(path, 'w') as f:
        write_format_version(f)
        f.create_dataset('return_code', data=0)

    with pytest.raises(ValueError, match='no Measurements group'):
        Results(path)


def test_a_file_without_a_stage_is_rejected(tmp_path):
    path = tmp_path / 'no_stage.h5'
    with h5.File(path, 'w') as f:
        write_format_version(f)
        f.create_group('Measurements/NotAStage')

    with pytest.raises(ValueError, match='no Stage<N> group'):
        Results(path)


def test_an_unversioned_file_is_rejected(tmp_path):
    path = tmp_path / 'unversioned.h5'
    with h5.File(path, 'w') as f:
        f.create_group('Measurements/Stage0')

    with pytest.raises(ValueError, match='no format_version'):
        Results(path)


def test_observable_names_reach_through_nested_groups(results_file, measured):
    assert Results(results_file, stage=0).observable_names() == sorted(measured)


def test_complex_observables_come_back_complex(results_file, measured):
    series = Results(results_file, stage=0).timeseries('Energy')

    assert series.dtype == np.complex128
    np.testing.assert_allclose(series, measured['Energy'])


def test_real_observables_are_passed_through(results_file, measured):
    series = Results(results_file, stage=0).timeseries('Weight')

    assert series.dtype == np.float64
    np.testing.assert_allclose(series, measured['Weight'])


def test_the_shape_of_an_observable_survives_the_round_trip(results_file, measured):
    series = Results(results_file, stage=0).timeseries(RDM)

    assert series.shape == measured[RDM].shape
    np.testing.assert_allclose(series, measured[RDM])


def test_an_unknown_observable_says_what_there_is(results_file):
    with pytest.raises(KeyError, match='Energy'):
        Results(results_file, stage=0).timeseries('NotMeasured')


def test_average_agrees_with_rebinning_the_series_by_hand(results_file, measured):
    rebinsize, _ = optimal_rebinsize(measured['Energy'], skip=5)
    bins, _ = rebinning_analysis(measured['Energy'], rebinsize=rebinsize, skip=5)

    mean, error = Results(results_file, stage=0).average('Energy', skip=5)

    np.testing.assert_allclose(mean, bins.mean(axis=0))
    np.testing.assert_allclose(error, standard_error(bins))


def test_average_of_an_array_observable_keeps_its_shape(results_file, measured):
    mean, error = Results(results_file, stage=0).average(RDM)

    assert mean.shape == measured[RDM].shape[1:]
    assert error.shape == measured[RDM].shape[1:]


def test_evaluate_reproduces_average_for_a_linear_function(results_file):
    results = Results(results_file, stage=0)
    mean, error = results.average('Energy')

    value, jackknifed = results.evaluate(lambda energy: energy.real, ['Energy'])

    assert value == pytest.approx(mean.real)
    assert jackknifed == pytest.approx(error.real)


def test_evaluate_passes_the_observables_in_the_order_they_are_named(results_file):
    # a linear function, so that the bias correction cannot blur the comparison; one
    # rebin size for every call, since each drops a different remainder of bins otherwise
    results = Results(results_file, stage=0)
    energy, _ = results.average('Energy', rebinsize=8)
    norm, _ = results.average('Norm', rebinsize=8)

    value, _ = results.evaluate(lambda a, b: a - b, ['Energy', 'Norm'], rebinsize=8)

    assert value == pytest.approx(energy - norm)


def test_evaluate_debiases_a_nonlinear_function(results_file):
    # one rebin size, so that the difference is the bias correction and not a different
    # remainder of bins dropped
    results = Results(results_file, stage=0)
    energy, _ = results.average('Energy', rebinsize=8)
    norm, _ = results.average('Norm', rebinsize=8)

    value, _ = results.evaluate(lambda a, b: a / b, ['Energy', 'Norm'], rebinsize=8)

    # the correction is small but it is there, and it is not noise
    assert value != pytest.approx(energy / norm, rel=1e-9)
    assert value == pytest.approx(energy / norm, rel=1e-3)


def test_evaluate_rebins_every_observable_with_the_largest_size(results_file, measured):
    # the jackknife needs one bin count, so the observable needing the larger rebin
    # size sets it for all of them
    rebinsize = max(optimal_rebinsize(measured[name])[0] for name in ['Energy', 'Norm'])
    binned = [
        rebinning_analysis(measured[name], rebinsize=rebinsize)[0] for name in ['Energy', 'Norm']
    ]

    value, error = Results(results_file, stage=0).evaluate(lambda a, b: a / b, ['Energy', 'Norm'])

    expected, expected_error = jackknife(lambda a, b: a / b, *binned)
    assert value == pytest.approx(expected)
    assert error == pytest.approx(expected_error)


def test_evaluate_rejects_observables_of_different_length(results_file):
    with pytest.raises(ValueError, match="'Short' has"):
        Results(results_file, stage=0).evaluate(
            lambda energy, short: energy.real + short, ['Energy', 'Short']
        )


def add_real_observable(results_file, name, bins):
    """Write one more real observable into Stage0 of `results_file`."""
    with h5.File(results_file, 'a') as f:
        f.create_dataset(f'Measurements/Stage0/{name}/bins', data=bins)


def add_drift(results_file):
    """Add an observable ``Drift`` that decays from three standard deviations off."""
    rng = np.random.default_rng(15)
    steps = np.arange(120)
    add_real_observable(results_file, 'Drift', rng.normal(size=steps.size) + 3 * np.exp(-steps / 15))


def test_stationary_observables_average_without_a_warning(results_file, measured):
    # the 1-RDM is not a scalar, so it is not tested for drift at all
    results = Results(results_file, stage=0)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in measured:
            results.average(name)
        results.evaluate(lambda a, b: a / b, ['Energy', 'Norm'])
