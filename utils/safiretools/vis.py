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
Plotting helpers, used as ``from safiretools import vis``.

This is the only module that imports matplotlib, and ``safiretools/__init__.py``
does not import it, so ``import safiretools`` alone never loads matplotlib.

Every helper draws into the ``ax`` it is given, creating a figure only when
none is, and returns the axes. Showing or saving the figure is left to the
caller.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator

_HALO = [withStroke(linewidth=7, foreground='white')]


def plot_lattice(
        lattice,
        *,
        ax=None,
        density=None,
        density_label="Density",
        cmap='seismic',
        vmin=None,
        vmax=None,
        norm_type='linear',
        nth_neighbor=1,
        show_direct_nearest=True,
        show_image_nearest=True,
        show_labels=True,
        show_lattice_vecs=True,
        title=None,
        fig_scale=2,
):
    """
    Plot the sites of a lattice and the bonds to their nth-nearest neighbors.

    Bonds within the home cell are drawn solid, bonds to periodic images dotted.

    Parameters
    ----------
    lattice : Lattice
        A built lattice.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. By default a new figure is created, sized to the
        lattice.
    density : array_like, optional
        One value per site, used to color it. Either flat in site order or
        shaped ``(L1, L2, nb)``; ``(L1, L2)`` works when ``nb == 1``.
    density_label : str, optional
        Colorbar label. Default "Density".
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for `density`. Default 'seismic'.
    vmin, vmax : float, optional
        Color scale limits. Default the extremes of `density`.
    norm_type : {'linear', 'log'}, optional
        Color scale. Default 'linear'.
    nth_neighbor : int, optional
        Which neighbor shell to draw bonds for. Default 1.
    show_direct_nearest, show_image_nearest : bool, optional
        Draw the home-cell and image bonds. Default True.
    show_labels : bool, optional
        Label every site with its index and cell coordinate. Default True.
    show_lattice_vecs : bool, optional
        Draw ``a1`` and ``a2``, unless the lattice is one cell wide. Default
        True.
    title : str, optional
        Axes title. Default the lattice size and type.
    fig_scale : float, optional
        Figure inches per unit length, when a figure is created. Default 2.

    Returns
    -------
    matplotlib.axes.Axes
    """
    positions = np.array([site.position for site in lattice.get_sites()])

    if ax is None:
        width, height = fig_scale*(np.ptp(positions, axis=0) + 1)
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set(linewidth=4, edgecolor='0.5')

    ax.set_aspect(1.)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.tick_params(which='major', width=1.0, length=10, labelsize=14)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    if title is None:
        title = f"{lattice.L[0]}x{lattice.L[1]} {lattice._type} Lattice"
    ax.set_title(title, fontsize=20, verticalalignment='bottom', pad=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    if density is None:
        facecolors = ['lightgray']*lattice.N_sites
    else:
        density = np.asarray(density)
        if density.size != lattice.N_sites:
            raise ValueError(
                f"density has shape {density.shape}, but the lattice has "
                f"{lattice.N_sites} sites; expected one value per site, flat or shaped "
                f"{(*lattice.L, lattice.nb)}"
            )
        density = density.reshape(lattice.N_sites)

        vmin = density.min() if vmin is None else vmin
        vmax = density.max() if vmax is None else vmax
        # a constant density would otherwise give a degenerate norm
        if vmin == vmax:
            vmin -= 1e-10
            vmax += 1e-10

        if norm_type == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif norm_type == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f"unknown norm_type {norm_type!r}; expected 'linear' or 'log'")

        cmap = plt.get_cmap(cmap)
        facecolors = cmap(norm(density))
        colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        colorbar.set_label(density_label, fontsize=16)

    for site, facecolor in zip(lattice.sites, facecolors, strict=True):
        x, y = site.position
        ax.add_patch(Circle(
            (x, y),
            radius=0.1,
            clip_on=False,
            zorder=10,
            linewidth=1.0,
            edgecolor='black',
            facecolor=facecolor,
            path_effects=_HALO,
        ))

        if show_labels:
            ax.text(x, y - 0.2, str(site.index), zorder=100,
                    ha='center', va='top', weight='bold', style='italic',
                    color=(0, 20/256, 82/256), path_effects=_HALO)
            ax.text(x, y - 0.35, f"{site.coord[:2]}", zorder=100,
                    ha='center', va='top', fontfamily='monospace', fontsize='medium',
                    path_effects=_HALO)

    bonds = []
    if show_direct_nearest:
        bonds.append((lattice.get_nth_direct_neighbors(nth_neighbor), '-'))
    if show_image_nearest:
        bonds.append((lattice.get_nth_image_neighbors(nth_neighbor), ':'))
    for pairs, linestyle in bonds:
        for pair in pairs:
            # zorder above the sites, so they do not cover the arrowheads
            ax.arrow(*lattice[pair.i].position, *pair.r_relative,
                     color='red', linestyle=linestyle, zorder=20)

    if show_lattice_vecs and lattice.L[0] > 1 and lattice.L[1] > 1:
        origin = np.array([-0.825, -0.825])
        for vec, label, ha, va in ((lattice.a1, r'$\vec{a}_1$', 'center', 'top'),
                                   (lattice.a2, r'$\vec{a}_2$', 'right', 'center')):
            ax.arrow(*origin, *vec, color='black', width=0.04, head_width=0.1, shape='full')
            ax.text(*(origin + vec/2), label, color='black', fontsize=16, ha=ha, va=va)

    return ax
