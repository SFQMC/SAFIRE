# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from afqmctools.systems.lattice import Lattice,LatticeSite

def plot_site(ax,site:LatticeSite):

    ax.add_artist(
        Circle(
            xy = site.coord,
            radius = 0.1,
        )
    )

def plot_lattice(
        lattice:Lattice,
        show_direct_nearest=True,
        show_image_nearest=True,
        nth_neighbor=1,
        save=False,
        show_plot=True,
        show_labels = True,
        density = None,
        density_label = "Density",
        show_lattice_vecs = True,
        **kwargs
    ):
    """
    Plot a lattice with nearest neighbors.

    Parameters
    ----------
    lattice : afqmctools.systems.lattice.Lattice
        Lattice object to plot.
    show_direct_nearest : bool, optional
        Show direct nearest neighbors, by default True. Direct neighbors are connected by solid lines.
    show_image_nearest : bool, optional
        Show image nearest neighbors, by default True. Image neighbors are connected by dashed lines.
    save : bool, optional
        Save the plot to a file, by default False. If False, the plot is displayed.
    show_labels : bool, optional
        Show site labels, by default True.
    density : np.array, optional
        Array of site densities with shape (nsites,nsites), by default None. If provided, the site color is scaled by the density.
    norm_type : str, optional
        Type of normalization to use for the density, by default 'linear'. Options are 'linear' and 'log'.
    
    """
    num_sublattice = lattice.num_sublattice

    _cmap = kwargs.get("cmap",'seismic')

    Lx = lattice.L[0]
    Ly = lattice.L[1]

    # For plot limits
    # TODO: these are not correct for general {ai}, and/or for a basis!!
    Dx = Lx*lattice.a1[0] + Ly*lattice.a2[0]
    Dy = Lx*lattice.a1[1] + Ly*lattice.a2[1]

    if Dx == 0: Dx = max(1,Dy)
    if Dy == 0: Dy = max(1,Dx)

    if kwargs.get("fig_scale",None) is not None:
        fig_scale = kwargs.get("fig_scale")
    else:
        fig_scale = 2

    fig,ax = plt.subplots(1,1,figsize=(Dx*fig_scale,Dy*fig_scale))
    

    ax.xaxis.set_major_locator(MultipleLocator(1.000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(Dx))
    ax.yaxis.set_major_locator(MultipleLocator(1.000))
    ax.yaxis.set_minor_locator(AutoMinorLocator(Dy))
    ax.xaxis.set_minor_formatter("{x:.2f}")

    ax.minorticks_off()

    #ax.set_xlim(-0.5, Dx - 0.5) # doesn't work with basis, ignore for now
    #ax.set_ylim(-0.5, Dy - 0.5)
    ax.set_aspect(1.)

    ax.tick_params(which='major', width=1.0, length=10, labelsize=14)
    
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    lattice_type = getattr(lattice,"_type","")

    if kwargs.get("title",None) is not None:
        title_string = kwargs.get("title")
    else:
        title_string = f"{Lx}x{Ly} {lattice_type} Lattice"

    ax.set_title(title_string, fontsize=20, verticalalignment='bottom', pad = 20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    if density is not None:
        density = np.array(density)

        if density.shape == (Lx, Ly) and num_sublattice == 1:
            density = density[:, :, np.newaxis]

        vmin = kwargs.get('vmin', np.min(density))
        vmax = kwargs.get('vmax', np.max(density))
        # avoid machine precision issues when vmin == vmax
        if vmin == vmax:
            vmin -= 1e-10
            vmax += 1e-10

        norm_type = kwargs.pop('norm_type', 'linear')
        
        if norm_type == 'linear':
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colormap = cm.get_cmap(_cmap)(norm(density))  # Choose a colormap
        elif norm_type == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
            # for single sublattice case
            normed_density = norm(density[:, :, 0])[:,:,np.newaxis]
            colormap = cm.get_cmap(_cmap)( normed_density)  # Choose a colormap

    def annotate(x, y, text, coord):
    
        royal_blue = [0, 20/256, 82/256]
        black = [0.0,0.0,0.0]

        if density is not None:
            facecolor = colormap[tuple(coord)]  # Map density to a color
        else:
            facecolor = 'lightgray'  # Default color if density is not provided

        # Circle marker
        c = Circle(
            (x, y),
            radius=0.10,
            clip_on=False,
            zorder=10,
            linewidth=1.0,
            edgecolor=black,
            facecolor=facecolor,
            path_effects=[withStroke(linewidth=7, foreground='white')]
        )
        ax.add_artist(c)

        if not show_labels: return

        for path_effects in [[withStroke(linewidth=7, foreground='white')], []]:
            color = 'white' if path_effects else royal_blue
            ax.text(x, y-0.2, text, zorder=100,
                    ha='center', va='top', weight='bold', color=color,
                    style='italic', #fontfamily='Courier New',
                    path_effects=path_effects)

            color = 'white' if path_effects else 'black'
            ax.text(x, y-0.35, f"{coord[:2]}", zorder=100,
                    ha='center', va='top', weight='normal', color=color,
                    fontfamily='monospace', fontsize='medium',
                    path_effects=path_effects)


    for site in lattice.sites:
        annotate(
            x=site.position[0],
            y=site.position[1],
            text=site.index,
            coord=site.coord
        )

    def add_pair_arrows(pairs,offset=None,**kwargs):
        
        for pair in pairs:
            coord1 = lattice[pair.i].position
            if pair._abs_r is None:
                coord2 = lattice[pair.j].position
            else:
                coord2 = pair._abs_r

            delta_coord = [ c2 - c1 for c1,c2 in zip(coord1,coord2)]

            if offset is not None:
                coord1 = [ c + o for c,o in zip(coord1,offset)]

            # using zorder to avoid arrows being covered by sites
            ax.arrow(*coord1, *delta_coord, zorder=20, **kwargs)

    if show_direct_nearest:
        add_pair_arrows(
            pairs=lattice.get_nth_direct_neighbors(nth_neighbor),
            color='red'
        )
        
    if show_image_nearest:
        add_pair_arrows(
            pairs=lattice.get_nth_image_neighbors(nth_neighbor),
            color='red',
            linestyle=':'
        )

    if show_lattice_vecs:

        lattice_vec_settings = dict(
            color='black',
            width=0.04,
            head_width=0.1,
            shape='full'
        )
       
        if lattice.L[0] > 1 and lattice.L[1] > 1: # hide lattice vectors for 1-D lattices
            offset = 0.75
            offset = offset*1.1
            ax.arrow(-offset, -offset, *lattice.a1, **lattice_vec_settings)
            ax.text(-offset + lattice.a1[0] / 2, -offset + lattice.a1[1] / 2, r'$\vec{a}_1$', color='black', fontsize=16, ha='center', va='top')

            ax.arrow(-offset, -offset, *lattice.a2, **lattice_vec_settings)
            ax.text(-offset + lattice.a2[0] / 2, -offset + lattice.a2[1] / 2, r'$\vec{a}_2$', color='black', fontsize=16, ha='right', va='center')

    if density is not None:
        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap(_cmap)),ax=ax).set_label(density_label,fontsize=16)

    fig.patch.set(linewidth=4, edgecolor='0.5')

    if save:
        if isinstance(save, str):
            plt.savefig(save)
        else:
            plt.savefig("lattice.png")
    
    if show_plot:
        plt.show()

    return fig,ax


# ============================================================================
# Momentum Distribution Visualization Functions
# ============================================================================

def _extract_2d_slice(kvecs, nk, nk_err, slice_axis='z', slice_value=0.0, tolerance=0.1):
    """
    Extract k-points within tolerance of a plane for 2D slice visualization.
    
    Parameters
    ----------
    kvecs : np.ndarray
        K-vectors array with shape (nkpts, 3).
    nk : np.ndarray
        Momentum distribution values with shape (nkpts,).
    nk_err : np.ndarray or None
        Error estimates with shape (nkpts,).
    slice_axis : str
        Axis perpendicular to slice plane: 'x', 'y', or 'z'.
    slice_value : float
        Position of slice along slice_axis.
    tolerance : float
        Thickness of slice (k-points within ±tolerance are included).
    
    Returns
    -------
    dict
        Dictionary with keys: 'k1', 'k2', 'nk', 'nk_err', 'axis_names'
        where k1, k2 are the two in-plane k-directions.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[slice_axis.lower()]
    
    # Find k-points in slice
    in_slice = np.abs(kvecs[:, axis_idx] - slice_value) <= tolerance
    
    # Get in-plane axes
    other_axes = [i for i in range(3) if i != axis_idx]
    k1 = kvecs[in_slice, other_axes[0]]
    k2 = kvecs[in_slice, other_axes[1]]
    nk_slice = nk[in_slice]
    nk_err_slice = nk_err[in_slice] if nk_err is not None else None
    
    axis_names = ['kx', 'ky', 'kz']
    
    return {
        'k1': k1,
        'k2': k2,
        'nk': nk_slice,
        'nk_err': nk_err_slice,
        'axis_names': (axis_names[other_axes[0]], axis_names[other_axes[1]]),
        'n_points': len(k1)
    }


def _extract_linecut(kvecs, nk, nk_err, start_point, end_point, width=0.1, averaging='mean'):
    """
    Extract 1D line cut between two k-points with perpendicular averaging.
    
    Parameters
    ----------
    kvecs : np.ndarray
        K-vectors array with shape (nkpts, 3).
    nk : np.ndarray
        Momentum distribution values with shape (nkpts,).
    nk_err : np.ndarray or None
        Error estimates with shape (nkpts,).
    start_point : array-like
        Start k-point [kx, ky, kz].
    end_point : array-like
        End k-point [kx, ky, kz].
    width : float
        Perpendicular tolerance for averaging neighboring points.
    averaging : str
        Averaging method: 'mean', 'median', or 'max'.
    
    Returns
    -------
    dict
        Dictionary with keys: 'kpath', 'nk', 'nk_err', 'start', 'end'
    """
    start = np.array(start_point)
    end = np.array(end_point)
    direction = end - start
    path_length = np.linalg.norm(direction)
    direction_unit = direction / path_length if path_length > 0 else direction
    
    # Calculate position along path for each k-point
    relative_positions = kvecs - start
    along_path = np.dot(relative_positions, direction_unit)
    
    # Calculate perpendicular distance from path
    projection_on_path = along_path[:, np.newaxis] * direction_unit
    perpendicular = relative_positions - projection_on_path
    perp_distance = np.linalg.norm(perpendicular, axis=1)
    
    # Select points near the path
    on_path = (perp_distance <= width) & (along_path >= -width) & (along_path <= path_length + width)
    
    if np.sum(on_path) == 0:
        return {
            'kpath': np.array([]),
            'nk': np.array([]),
            'nk_err': np.array([]),
            'start': start,
            'end': end,
            'n_points': 0
        }
    
    kpath = along_path[on_path]
    nk_path = nk[on_path]
    nk_err_path = nk_err[on_path] if nk_err is not None else None
    
    # Sort by position along path
    sort_idx = np.argsort(kpath)
    kpath = kpath[sort_idx]
    nk_path = nk_path[sort_idx]
    if nk_err_path is not None:
        nk_err_path = nk_err_path[sort_idx]
    
    return {
        'kpath': kpath,
        'nk': nk_path,
        'nk_err': nk_err_path,
        'start': start,
        'end': end,
        'n_points': len(kpath)
    }


def compute_fermi_level(kvecs, nk, n_electrons=None, method='integrated', **kwargs):
    """
    Compute the Fermi wave vector kF from momentum distribution.
    
    Parameters
    ----------
    kvecs : np.ndarray
        K-vectors array with shape (nkpts, 3).
    nk : np.ndarray
        Momentum distribution values with shape (nkpts,).
    n_electrons : float or None
        Total number of electrons. If None, uses integral of n(k).
    method : str
        Method for computing kF:
        - 'integrated': Radially integrate n(k) to find kF where integral = N
        - 'threshold': Find |k| where n(k) crosses threshold
        - 'fit': Fit radial n(k) to Fermi function
    **kwargs
        nbins : int
            Number of radial bins (default: 50)
        threshold : float
            Threshold value for 'threshold' method (default: 0.5)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'kF': Fermi wave vector
        - 'kF_err': Uncertainty estimate
        - 'radial_profile': (k_radial, nk_radial, nk_err_radial)
        - 'method': Method used
        - 'n_integrated': Total electrons from integration
    """
    nbins = kwargs.get('nbins', 50)
    threshold = kwargs.get('threshold', 0.5)
    
    # Compute radial distance
    k_mag = np.linalg.norm(kvecs, axis=1)
    
    # Create radial bins
    k_max = np.max(k_mag)
    bin_edges = np.linspace(0, k_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Bin the data
    nk_radial = np.zeros(nbins)
    nk_radial_err = np.zeros(nbins)
    counts = np.zeros(nbins)
    
    for i in range(nbins):
        mask = (k_mag >= bin_edges[i]) & (k_mag < bin_edges[i + 1])
        if np.sum(mask) > 0:
            nk_radial[i] = np.mean(nk[mask])
            nk_radial_err[i] = np.std(nk[mask]) / np.sqrt(np.sum(mask))
            counts[i] = np.sum(mask)
    
    # Remove empty bins
    valid = counts > 0
    bin_centers = bin_centers[valid]
    nk_radial = nk_radial[valid]
    nk_radial_err = nk_radial_err[valid]
    
    if method == 'threshold':
        # Find where n(k) crosses threshold
        idx = np.where(nk_radial < threshold)[0]
        if len(idx) > 0:
            kF = bin_centers[idx[0]]
            kF_err = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
        else:
            kF = k_max
            kF_err = 0.1
        n_integrated = None
        
    elif method == 'fit':
        # Fit to Fermi-Dirac-like function
        def fermi_func(k, A, kF, sigma):
            return A / (1.0 + np.exp((k - kF) / sigma))
        
        # Initial guess
        p0 = [np.max(nk_radial), bin_centers[len(bin_centers)//2], 0.1]
        
        try:
            popt, pcov = curve_fit(fermi_func, bin_centers, nk_radial, p0=p0, 
                                   bounds=([0, 0, 0.001], [np.inf, k_max, 1.0]))
            kF = popt[1]
            kF_err = np.sqrt(pcov[1, 1]) if pcov is not None else 0.1
        except:
            # Fallback to threshold method
            idx = np.where(nk_radial < threshold)[0]
            kF = bin_centers[idx[0]] if len(idx) > 0 else k_max
            kF_err = 0.1
        n_integrated = None
        
    elif method == 'integrated':
        # Integrate radially: N(k) = ∫₀ᵏ 4πk'² n̄(k') dk'
        dk = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
        cumulative = np.cumsum(4 * np.pi * bin_centers**2 * nk_radial * dk)
        n_integrated = cumulative[-1]
        
        if n_electrons is None:
            n_electrons = n_integrated
        
        # Find where cumulative integral reaches n_electrons
        idx = np.where(cumulative >= n_electrons)[0]
        if len(idx) > 0:
            kF = bin_centers[idx[0]]
            kF_err = dk
        else:
            kF = k_max
            kF_err = dk
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'kF': kF,
        'kF_err': kF_err,
        'radial_profile': (bin_centers, nk_radial, nk_radial_err),
        'method': method,
        'n_integrated': n_integrated
    }


def plot_momentum_distribution_2d_slice(kvecs, nk, nk_err=None, **kwargs):
    """
    Plot 2D slice of momentum distribution n(k).
    
    Parameters
    ----------
    kvecs : np.ndarray
        K-vectors array with shape (nkpts, 3).
    nk : np.ndarray
        Momentum distribution values with shape (nkpts,).
    nk_err : np.ndarray or None
        Error estimates with shape (nkpts,).
    **kwargs
        slice_axis : str
            Axis perpendicular to slice: 'x', 'y', or 'z' (default: 'z')
        slice_value : float
            Position of slice along slice_axis (default: 0.0)
        tolerance : float
            Thickness of slice (default: 0.1)
        plot_type : str
            'scatter', 'contour', or 'both' (default: 'both')
            If 'both', creates two separate subplots
        interpolate : bool
            Whether to interpolate for contour plot (default: True)
        interpolation : str
            Interpolation method: 'linear', 'cubic', 'nearest' (default: 'cubic')
        grid_resolution : int
            Resolution for contour grid (default: 100)
        cmap : str
            Colormap (default: 'viridis')
        vmin, vmax : float
            Color scale limits
        fermi_level : None, 'auto', or float
            Fermi level to plot as circle. 'auto' computes it automatically.
        fermi_level_kwargs : dict
            Styling for Fermi circle (color, linestyle, linewidth, label)
        n_electrons : float
            For automatic Fermi level computation
        linecuts : list of dict or None
            Line cuts to overlay on the plot. Each dict should have 'start' and 'end'.
            The line cuts will be projected onto the slice plane.
        linecut_kwargs : dict
            Styling for line cut paths:
            - color: line color (default: 'red')
            - linestyle: line style (default: '-')
            - linewidth: line width (default: 2)
            - alpha: transparency (default: 0.7)
            - show_arrows: show direction arrows (default: True)
            - arrow_size: size of direction arrows (default: 0.15)
        figsize : tuple
            Figure size (default: (8, 6) for single plot, (14, 6) for both)
        dpi : int
            Figure DPI (default: 100)
        save : str or False
            Filename to save figure
        show : bool
            Display plot (default: True)
        title, xlabel, ylabel : str
            Plot labels
    
    Returns
    -------
    tuple
        (fig, ax, slice_data) where slice_data contains extracted data
        ax is a single axis or array of axes if plot_type='both'
    """
    # Extract parameters
    slice_axis = kwargs.get('slice_axis', 'z')
    slice_value = kwargs.get('slice_value', 0.0)
    tolerance = kwargs.get('tolerance', 0.1)
    plot_type = kwargs.get('plot_type', 'both')
    interpolate = kwargs.get('interpolate', True)
    interpolation = kwargs.get('interpolation', 'cubic')
    grid_resolution = kwargs.get('grid_resolution', 100)
    cmap = kwargs.get('cmap', 'viridis')
    fermi_level = kwargs.get('fermi_level', None)
    n_electrons = kwargs.get('n_electrons', None)
    linecuts = kwargs.get('linecuts', None)
    
    # Extract slice data
    slice_data = _extract_2d_slice(kvecs, nk, nk_err, slice_axis, slice_value, tolerance)
    
    if slice_data['n_points'] == 0:
        raise ValueError(f"No k-points found in slice at {slice_axis}={slice_value} ± {tolerance}")
    
    k1, k2 = slice_data['k1'], slice_data['k2']
    nk_slice = slice_data['nk']
    axis_names = slice_data['axis_names']
    
    # Determine which axes we're plotting (for line cut projection)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[slice_axis.lower()]
    other_axes = [i for i in range(3) if i != axis_idx]
    
    # Create figure - different layout for 'both'
    if plot_type == 'both':
        default_figsize = (14, 6)
        figsize = kwargs.get('figsize', default_figsize)
        dpi = kwargs.get('dpi', 100)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        axes = [ax1, ax2]
    else:
        default_figsize = (8, 6)
        figsize = kwargs.get('figsize', default_figsize)
        dpi = kwargs.get('dpi', 100)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        axes = [ax]
    
    # Determine color scale
    vmin = kwargs.get('vmin', np.min(nk_slice))
    vmax = kwargs.get('vmax', np.max(nk_slice))
    
    # Prepare interpolated grid if needed
    nk_grid = None
    if (plot_type in ['contour', 'both']) and interpolate:
        k1_min, k1_max = k1.min(), k1.max()
        k2_min, k2_max = k2.min(), k2.max()
        k1_grid = np.linspace(k1_min, k1_max, grid_resolution)
        k2_grid = np.linspace(k2_min, k2_max, grid_resolution)
        k1_mesh, k2_mesh = np.meshgrid(k1_grid, k2_grid)
        nk_grid = griddata((k1, k2), nk_slice, (k1_mesh, k2_mesh), method=interpolation)
        
        # Fill NaN values with 0 to avoid white corners
        nk_grid = np.nan_to_num(nk_grid, nan=0.0)
    
    # Compute Fermi level if needed
    kF = None
    if fermi_level is not None:
        if fermi_level == 'auto':
            fermi_data = compute_fermi_level(kvecs, nk, n_electrons=n_electrons, 
                                            method='integrated')
            kF = fermi_data['kF']
        else:
            kF = float(fermi_level)
    
    # Helper function to add Fermi circle, line cuts, and labels
    def decorate_axis(ax, title_suffix=''):
        # Plot Fermi level circle
        if kF is not None:
            fermi_kwargs = kwargs.get('fermi_level_kwargs', {})
            fermi_color = fermi_kwargs.get('color', 'white')
            fermi_ls = fermi_kwargs.get('linestyle', '--')
            fermi_lw = fermi_kwargs.get('linewidth', 2)
            fermi_label = fermi_kwargs.get('label', f'$k_F$ = {kF:.2f}')
            
            circle = Circle((0, 0), kF, fill=False, edgecolor=fermi_color, 
                           linestyle=fermi_ls, linewidth=fermi_lw, label=fermi_label,
                           zorder=100)
            ax.add_patch(circle)
        
        # Plot line cut paths (projected onto slice plane)
        if linecuts is not None:
            linecut_kwargs = kwargs.get('linecut_kwargs', {})
            lc_color = linecut_kwargs.get('color', 'red')
            lc_ls = linecut_kwargs.get('linestyle', '-')
            lc_lw = linecut_kwargs.get('linewidth', 2)
            lc_alpha = linecut_kwargs.get('alpha', 0.7)
            show_arrows = linecut_kwargs.get('show_arrows', True)
            arrow_size = linecut_kwargs.get('arrow_size', 0.15)
            
            for lc in linecuts:
                start = np.array(lc['start'])
                end = np.array(lc['end'])
                
                # Project start and end points onto the slice plane
                start_proj = np.array([start[other_axes[0]], start[other_axes[1]]])
                end_proj = np.array([end[other_axes[0]], end[other_axes[1]]])
                
                # Get color for this line cut (use individual color if specified)
                line_color = lc.get('color', lc_color)
                
                # Plot the line
                ax.plot([start_proj[0], end_proj[0]], 
                       [start_proj[1], end_proj[1]],
                       color=line_color, linestyle=lc_ls, linewidth=lc_lw, 
                       alpha=lc_alpha, zorder=90,
                       label=lc.get('label', None))
                
                # Add arrow to show direction
                if show_arrows:
                    # Calculate arrow position (at the end of the line)
                    dx = end_proj[0] - start_proj[0]
                    dy = end_proj[1] - start_proj[1]
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        # Position arrow at 90% of the line length
                        arrow_pos = start_proj + 0.9 * (end_proj - start_proj)
                        
                        # Arrow direction (normalized)
                        arrow_dx = dx / length * arrow_size
                        arrow_dy = dy / length * arrow_size
                        
                        ax.arrow(arrow_pos[0], arrow_pos[1], arrow_dx, arrow_dy,
                                head_width=arrow_size*0.4, head_length=arrow_size*0.6,
                                fc=line_color, ec=line_color, alpha=lc_alpha,
                                zorder=95, linewidth=0)
        
        # Set labels and styling
        ax.set_xlabel(kwargs.get('xlabel', axis_names[0]), fontsize=12)
        ax.set_ylabel(kwargs.get('ylabel', axis_names[1]), fontsize=12)
        
        base_title = kwargs.get('title', f'Momentum Distribution: {slice_axis}={slice_value:.2f}')
        ax.set_title(base_title + title_suffix, fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend if there are labeled items
        if kF is not None or (linecuts is not None and any('label' in lc for lc in linecuts)):
            ax.legend(loc='best', fontsize=10)
    
    # Plot based on type
    if plot_type == 'both':
        # Subplot 1: Scatter
        scatter = ax1.scatter(k1, k2, c=nk_slice, cmap=cmap, vmin=vmin, vmax=vmax, 
                             s=20, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax1, label='n(k)')
        decorate_axis(ax1, ' (Scatter)')
        
        # Subplot 2: Interpolated contour
        if interpolate and nk_grid is not None:
            contourf = ax2.contourf(k1_mesh, k2_mesh, nk_grid, levels=20, cmap=cmap, 
                                   vmin=vmin, vmax=vmax)
            ax2.contour(k1_mesh, k2_mesh, nk_grid, levels=10, colors='black', 
                       linewidths=0.5, alpha=0.3)
            plt.colorbar(contourf, ax=ax2, label='n(k)')
        else:
            # Fallback to scatter if interpolation is disabled
            scatter2 = ax2.scatter(k1, k2, c=nk_slice, cmap=cmap, vmin=vmin, vmax=vmax, 
                                  s=20, alpha=0.6, edgecolors='none')
            plt.colorbar(scatter2, ax=ax2, label='n(k)')
        decorate_axis(ax2, ' (Interpolated)')
        
        fig.tight_layout()
        ax_return = axes
        
    elif plot_type == 'scatter':
        scatter = ax.scatter(k1, k2, c=nk_slice, cmap=cmap, vmin=vmin, vmax=vmax, 
                           s=20, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='n(k)')
        decorate_axis(ax)
        ax_return = ax
        
    elif plot_type == 'contour':
        if interpolate and nk_grid is not None:
            contourf = ax.contourf(k1_mesh, k2_mesh, nk_grid, levels=20, cmap=cmap, 
                                  vmin=vmin, vmax=vmax)
            plt.colorbar(contourf, ax=ax, label='n(k)')
        else:
            # Fallback to scatter
            scatter = ax.scatter(k1, k2, c=nk_slice, cmap=cmap, vmin=vmin, vmax=vmax, 
                               s=20, alpha=0.6, edgecolors='none')
            plt.colorbar(scatter, ax=ax, label='n(k)')
        decorate_axis(ax)
        ax_return = ax
    
    # Save and show
    if kwargs.get('save', False):
        plt.savefig(kwargs['save'], dpi=dpi, bbox_inches='tight')
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig, ax_return, slice_data


def get_standard_linecuts(preset='axes', kvecs=None):
    """
    Generate standard line cut definitions.
    
    Parameters
    ----------
    preset : str
        Preset type:
        - 'axes': Along kx, ky, kz through origin
        - 'cube_edges': Γ→X, Γ→Y, Γ→Z
        - 'cube_face_diagonals': Γ→M (face centers)
        - 'cube_body_diagonal': Γ→R (body diagonal)
        - 'auto': Automatically detect extent from kvecs
    kvecs : np.ndarray or None
        K-vectors for automatic extent detection
    
    Returns
    -------
    list
        List of line cut dictionaries with 'start', 'end', 'label', 'color'
    """
    if preset == 'axes':
        if kvecs is not None:
            k_max = np.max(np.abs(kvecs))
        else:
            k_max = 1.0
        
        return [
            {'start': [0, 0, 0], 'end': [k_max, 0, 0], 'label': 'kx', 'color': 'C0'},
            {'start': [0, 0, 0], 'end': [0, k_max, 0], 'label': 'ky', 'color': 'C1'},
            {'start': [0, 0, 0], 'end': [0, 0, k_max], 'label': 'kz', 'color': 'C2'},
        ]
    
    elif preset == 'cube_edges':
        return [
            {'start': [0, 0, 0], 'end': [1, 0, 0], 'label': 'Γ→X', 'color': 'C0'},
            {'start': [0, 0, 0], 'end': [0, 1, 0], 'label': 'Γ→Y', 'color': 'C1'},
            {'start': [0, 0, 0], 'end': [0, 0, 1], 'label': 'Γ→Z', 'color': 'C2'},
        ]
    
    elif preset == 'cube_face_diagonals':
        return [
            {'start': [0, 0, 0], 'end': [1, 1, 0], 'label': 'Γ→M(xy)', 'color': 'C0'},
            {'start': [0, 0, 0], 'end': [1, 0, 1], 'label': 'Γ→M(xz)', 'color': 'C1'},
            {'start': [0, 0, 0], 'end': [0, 1, 1], 'label': 'Γ→M(yz)', 'color': 'C2'},
        ]
    
    elif preset == 'cube_body_diagonal':
        return [
            {'start': [0, 0, 0], 'end': [1, 1, 1], 'label': 'Γ→R', 'color': 'C0'},
        ]
    
    elif preset == 'auto':
        if kvecs is None:
            raise ValueError("kvecs required for 'auto' preset")
        k_max = np.max(np.abs(kvecs), axis=0)
        return [
            {'start': [0, 0, 0], 'end': [k_max[0], 0, 0], 'label': 'kx', 'color': 'C0'},
            {'start': [0, 0, 0], 'end': [0, k_max[1], 0], 'label': 'ky', 'color': 'C1'},
            {'start': [0, 0, 0], 'end': [0, 0, k_max[2]], 'label': 'kz', 'color': 'C2'},
        ]
    
    else:
        raise ValueError(f"Unknown preset: {preset}")


def plot_momentum_distribution_linecuts(kvecs, nk, nk_err=None, linecuts=None, **kwargs):
    """
    Plot 1D line cuts of momentum distribution n(k).
    
    Parameters
    ----------
    kvecs : np.ndarray
        K-vectors array with shape (nkpts, 3).
    nk : np.ndarray
        Momentum distribution values with shape (nkpts,).
    nk_err : np.ndarray or None
        Error estimates with shape (nkpts,).
    linecuts : list of dict or None
        List of line cut definitions. Each dict should have:
        - 'start': [kx, ky, kz] start point
        - 'end': [kx, ky, kz] end point
        - 'label': string label (optional)
        - 'color': plot color (optional)
        If None, uses preset='axes'.
    **kwargs
        preset : str
            Preset line cuts if linecuts is None: 'axes', 'cube_edges', etc.
        width : float
            Perpendicular averaging width (default: 0.1)
        averaging : str
            Averaging method: 'mean', 'median', 'max' (default: 'mean')
        show_errors : bool
            Show error bands (default: True if nk_err provided)
        error_style : str
            'band' or 'bars' (default: 'band')
        layout : str
            'single' (all on one plot) or 'grid' (subplots) (default: 'single')
        fermi_level : None, 'auto', or float
            Fermi level to mark on plot
        fermi_level_mode : str
            'vertical', 'horizontal', or 'both' (default: 'vertical')
        fermi_level_kwargs : dict
            Styling for Fermi level line
        kpoint_labels : dict or list of tuples
            User-defined k-point labels for path
        show_kpoint_markers : bool
            Show markers at labeled k-points (default: True if labels provided)
        n_electrons : float
            For automatic Fermi level computation
        figsize : tuple
        save : str or False
        show : bool
        title, xlabel, ylabel : str
    
    Returns
    -------
    tuple
        (fig, axes, linecut_data) where linecut_data is list of extracted data dicts
    """
    # Handle line cuts
    if linecuts is None:
        preset = kwargs.get('preset', 'axes')
        linecuts = get_standard_linecuts(preset, kvecs)
    
    width = kwargs.get('width', 0.1)
    averaging = kwargs.get('averaging', 'mean')
    layout = kwargs.get('layout', 'single')
    show_errors = kwargs.get('show_errors', nk_err is not None)
    error_style = kwargs.get('error_style', 'band')
    fermi_level = kwargs.get('fermi_level', None)
    fermi_level_mode = kwargs.get('fermi_level_mode', 'vertical')
    kpoint_labels = kwargs.get('kpoint_labels', None)
    show_kpoint_markers = kwargs.get('show_kpoint_markers', kpoint_labels is not None)
    
    # Extract all line cuts
    linecut_data = []
    for lc in linecuts:
        data = _extract_linecut(kvecs, nk, nk_err, lc['start'], lc['end'], 
                               width=width, averaging=averaging)
        data['label'] = lc.get('label', f"{lc['start']}→{lc['end']}")
        data['color'] = lc.get('color', None)
        linecut_data.append(data)
    
    # Compute Fermi level if needed
    kF = None
    if fermi_level is not None:
        if fermi_level == 'auto':
            fermi_data = compute_fermi_level(kvecs, nk, 
                                            n_electrons=kwargs.get('n_electrons', None),
                                            method='integrated')
            kF = fermi_data['kF']
        else:
            kF = float(fermi_level)

    # Create figure
    if layout == 'single':
        figsize = kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

        for data in linecut_data:
            if data['n_points'] == 0:
                continue

            color = data['color']
            label = data['label']

            # Plot line
            ax.plot(data['kpath'], data['nk'], label=label, color=color, linewidth=2)

            # Plot errors
            if show_errors and data['nk_err'] is not None:
                if error_style == 'band':
                    ax.fill_between(data['kpath'], 
                                   data['nk'] - data['nk_err'],
                                   data['nk'] + data['nk_err'],
                                   alpha=0.3, color=color)
                elif error_style == 'bars':
                    ax.errorbar(data['kpath'], data['nk'], yerr=data['nk_err'],
                              fmt='none', ecolor=color, alpha=0.3)

        # Plot Fermi level
        if kF is not None:
            fermi_kwargs = kwargs.get('fermi_level_kwargs', {})
            fermi_color = fermi_kwargs.get('color', 'black')
            fermi_ls = fermi_kwargs.get('linestyle', '--')
            fermi_lw = fermi_kwargs.get('linewidth', 1.5)

            if fermi_level_mode in ['vertical', 'both']:
                ax.axvline(kF, color=fermi_color, linestyle=fermi_ls, linewidth=fermi_lw,
                          label=f'$k_F$ = {kF:.2f}')

            if fermi_level_mode in ['horizontal', 'both']:
                # Would need to evaluate n(kF) - skip for now
                pass

        # K-point labels
        if kpoint_labels is not None:
            if isinstance(kpoint_labels, dict):
                kpoint_labels = list(kpoint_labels.items())

            for k_pos, k_label in kpoint_labels:
                ax.axvline(k_pos, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(k_pos, ax.get_ylim()[1], k_label, 
                       ha='center', va='bottom', fontsize=10)

        ax.set_xlabel(kwargs.get('xlabel', 'k-path'), fontsize=12)
        ax.set_ylabel(kwargs.get('ylabel', 'n(k)'), fontsize=12)
        ax.set_title(kwargs.get('title', 'Momentum Distribution Line Cuts'), fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif layout == 'grid':
        n_cuts = len(linecut_data)
        ncols = min(3, n_cuts)
        nrows = (n_cuts + ncols - 1) // ncols
        figsize = kwargs.get('figsize', (5 * ncols, 4 * nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, data in enumerate(linecut_data):
            ax = axes[idx]
            if data['n_points'] == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            color = data['color']
            label = data['label']

            ax.plot(data['kpath'], data['nk'], color=color, linewidth=2)

            if show_errors and data['nk_err'] is not None:
                if error_style == 'band':
                    ax.fill_between(data['kpath'], 
                                   data['nk'] - data['nk_err'],
                                   data['nk'] + data['nk_err'],
                                   alpha=0.3, color=color)

            ax.set_xlabel('k-path', fontsize=10)
            ax.set_ylabel('n(k)', fontsize=10)
            ax.set_title(label, fontsize=12)
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_cuts, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(kwargs.get('title', 'Momentum Distribution Line Cuts'), 
                    fontsize=14)
        fig.tight_layout()

    # Save and show
    if kwargs.get('save', False):
        plt.savefig(kwargs['save'], bbox_inches='tight')

    if kwargs.get('show', True):
        plt.show()

    return fig, axes, linecut_data
