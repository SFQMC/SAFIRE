# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import subprocess
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle

from safiretools import Lattice, vis


def _lattice(lattice_type, L1=3, L2=3):
    return Lattice.from_dict(dict(
        type=lattice_type, L1=L1, L2=L2, boundary1='pbc', boundary2='pbc'))


def _site_colors(ax):
    return np.array([p.get_facecolor() for p in ax.patches if isinstance(p, Circle)])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_import_does_not_load_matplotlib():
    code = "import sys, safiretools; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('lattice_type', ['square', 'triangular', 'honeycomb', 'kagome'])
def test_plot_lattice_draws_every_site(lattice_type):
    lattice = _lattice(lattice_type)
    ax = vis.plot_lattice(lattice)
    assert len(_site_colors(ax)) == lattice.N_sites


def test_plot_lattice_draws_into_given_axes():
    fig, ax = plt.subplots()
    assert vis.plot_lattice(_lattice('square'), ax=ax) is ax
    assert plt.get_fignums() == [fig.number]


def test_density_shapes_agree():
    lattice = _lattice('honeycomb')
    density = np.arange(lattice.N_sites, dtype=float)
    flat = _site_colors(vis.plot_lattice(lattice, density=density))
    cell = _site_colors(vis.plot_lattice(lattice, density=density.reshape(3, 3, 2)))
    np.testing.assert_array_equal(flat, cell)


def test_density_of_wrong_size_raises():
    lattice = _lattice('honeycomb')
    with pytest.raises(ValueError, match="one value per site"):
        vis.plot_lattice(lattice, density=np.zeros((3, 3)))


def test_unknown_norm_type_raises():
    lattice = _lattice('square')
    with pytest.raises(ValueError, match="norm_type"):
        vis.plot_lattice(lattice, density=np.ones(lattice.N_sites), norm_type='sqrt')
