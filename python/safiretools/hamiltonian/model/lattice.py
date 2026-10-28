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
Lattice geometry for the lattice-model Hamiltonian builder.

A ``Lattice`` is a 2-D lattice with an optional multi-site basis, a
boundary condition (open or periodic, optionally twisted) per axis, and lazily
computed neighbor lists. `Lattice.from_type` supplies the unit cell of the
built-in lattice types; the constructor takes any unit cell.

Lattices are never serialized on their own. They are persisted as embedded
state inside a ``LatticeHamiltonian``'s HDF5 file, so there is no
``to_hdf5``/``from_hdf5`` here. The exception is `DirectedPairs`, which is
derived from a lattice but written as an estimator's own input file.
"""

import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction
from warnings import warn

import h5py as h5
import numpy as np
import scipy.spatial as spatial

from safiretools.hdf5 import replace_group

logger = logging.getLogger(__name__)


@dataclass(order=True)
class LatticeSite:
    """
    Simple dataclass to hold basic metadata for each lattice site.
    """

    index: int
    """Ordered basis index."""
    coord: np.ndarray
    """Lattice coordinates."""
    position: np.ndarray
    """Spatial position. For square lattices, this is the same as coord."""

    def __str__(self) -> str:
        return f"{self.index} {self.coord}"


@dataclass
class NeighborPair:
    """
    Simple dataclass to relate a phase, given as `phase`, and used as
        Exp[-i*phase], with a pair of neighbor indices, (i,j).

    We have chosen to reject the case where i == j on the grounds
      that it usually does not make sense to regard a site as its
      own neighbor.
    """

    i: int
    j: int
    _shift: tuple
    r_relative: tuple
    phase: float = 0.0

    def __init__(self, i: int, j: int, _shift=None,
                 phase: float = 0.0, r_relative=None) -> None:
        if i == j:
            raise ValueError("Lattice sites are not allowed to be their own neighbor")

        self.i = i                  # index of first site
        self.j = j                  # index of second site

        self._shift = _shift        # supercell shift to obtain image
        self.phase = phase          # relative phase between sites

        self.r_relative = r_relative  # relative *spatial* position of site "j" with
                                      #    respect to "i"

    def __str__(self) -> str:
        return f"{self.i}->{self.j}  phase={self.phase} rad."


class Boundary:
    """
    The boundary condition along one lattice axis.

    Parameters
    ----------
    L : int
        Size of the lattice along the axis.
    direction : int
        The axis, 0 for a1 and 1 for a2.
    phase : float, optional
        Twist angle picked up when crossing the boundary. Default 0.0.
    """

    def __init__(self, L, direction, phase=0.0) -> None:
        self.L = L
        self.direction = direction
        self.phase = phase

    def is_valid_image(self, coordinate) -> bool:
        """True if `coordinate` is an image this boundary connects to."""
        return self.is_image(coordinate) and self.is_valid(coordinate)

    def is_image(self, coordinate) -> bool:
        r"""
        True if `coordinate`, in units of the lattice vectors :math:`\hat{a}_1,
        \hat{a}_2`, lies outside the lattice along this axis.
        """
        return coordinate[self.direction] % self.L != coordinate[self.direction]

    def is_valid(self, coordinate) -> bool:
        raise NotImplementedError


class PBCBoundary(Boundary):
    """A periodic, optionally twisted, boundary: every image is valid."""

    def is_valid(self, coordinate) -> bool:
        return True


class OpenBoundary(Boundary):
    """An open boundary: no image is valid, so a twist has no effect."""

    def __init__(self, L, direction, phase=0.0) -> None:
        super().__init__(L, direction)

    def is_valid(self, coordinate) -> bool:
        return False


def _angle_str_to_float(angle_string: str):
    """
    convert strings of type '3/2 pi' or '1.5' to a decimal number
    """

    angle = 1.0

    if angle_string.lower().endswith('pi'):
        angle_string = angle_string[:-2]
        angle *= np.pi

    angle *= float(Fraction(angle_string))

    return angle


def _parse_twists(twists):
    """
    parse the twist angle iterable based on what types are within the tuple

    accepted types:
    - str : a string representing a number. "pi" may be included
    - number : interpreted as the twist angle in radians
    """

    # Check if twists is iterable (but not a string, which is iterable but should be rejected)
    if isinstance(twists, (int, float)):
        raise TypeError(
            "Invalid 'twists' parameter: expected a 2-element iterable (tuple or list), "
            f"but got a single number: {twists}"
        )

    if isinstance(twists, str):
        raise TypeError(
            "Invalid 'twists' parameter: expected a 2-element iterable (tuple or list), "
            f"but got a string: '{twists}'"
        )

    # Check if twists has the __len__ attribute to avoid TypeError
    if not hasattr(twists, '__len__'):
        raise TypeError(
            "Invalid 'twists' parameter: expected a 2-element iterable (tuple or list), "
            f"but got type {type(twists).__name__}"
        )

    if len(twists) != 2:
        raise ValueError(
            f"Invalid 'twists' parameter given: must have length 2, got length {len(twists)}"
        )

    _twist = [0., 0.]

    for i, t in enumerate(twists):
        if isinstance(t, str):
            _twist[i] = _angle_str_to_float(t)
        elif isinstance(t, float):
            _twist[i] = t
        elif isinstance(t, int):
            _twist[i] = float(t)
        else:
            raise ValueError("Twist format not currently supported! "
                             "Please add a feature request")

    return _twist


class Lattice:
    """
    A two-dimensional lattice: a unit cell repeated ``Ls[0] x Ls[1]`` times.

    Parameters
    ----------
    lattice_vectors : array_like, shape (2, 2)
        The lattice vectors ``a1`` and ``a2``, one per row.
    basis : array_like, shape (nb, 2)
        Positions of the sites within the unit cell, in fractional coordinates
        of the lattice vectors: site ``b`` of cell ``(n1, n2)`` sits at
        ``A @ ((n1, n2) + basis[b])``.
    Ls : iterable(int)
        2-element lattice size, in units of the unit cell.
    boundaries : iterable(str)
        Boundary condition along each lattice vector: ``'open'`` or
        ``'periodic'`` (``'pbc'`` is accepted as an alias).
    twists : iterable, optional
        Twist angle in radians along each lattice vector, picked up when
        crossing a periodic boundary. Elements are numbers or strings such as
        ``'1/2 pi'``. Default is no twist, ``(0.0, 0.0)``.
    build : bool, optional
        Build the lattice sites immediately. Default is True.

    Notes
    -----
    ``A`` and ``basis`` are read-only properties, and the arrays they
    return are themselves immutable, so the geometry cannot be replaced or
    edited in place after construction.

    `build()` raises `ValueError` for a basis the neighbor search cannot handle:
    two basis vectors differing by a lattice translation, or an offset reaching
    a full supercell. See `_validate_basis`.

    Examples
    --------
    A Lieb lattice, a square lattice with a three-site basis:

    >>> lattice = Lattice([[1, 0], [0, 1]], [[0, 0], [0.5, 0], [0, 0.5]], (4, 4),
    ...                   boundaries=('periodic', 'periodic'))
    """

    def __init__(self, lattice_vectors, basis, Ls, *, boundaries, twists=(0.0, 0.0),
                 build=True) -> None:
        lattice_vectors = np.array(lattice_vectors, dtype=float)
        if lattice_vectors.shape != (2, 2):
            raise ValueError(
                f"lattice_vectors must have shape (2, 2), got {lattice_vectors.shape}")
        self._A = lattice_vectors.T.copy()
        self._basis = [np.array(b, dtype=float) for b in basis]
        if not self._basis or any(b.shape != (2,) for b in self._basis):
            raise ValueError("basis must be a non-empty list of 2-vectors")
        for array in (self._A, *self._basis):
            array.setflags(write=False)
        self._nb = len(self._basis)

        if not (hasattr(Ls, '__len__') and len(Ls) == 2):
            raise ValueError("Ls must be a 2-d Array-like")
        if Ls[0] < 1 or Ls[1] < 1:
            raise ValueError("Invalid lattice size; both entries of Ls must be >= 1")
        self.L = tuple(int(size) for size in Ls)

        if len(boundaries) != 2:
            raise ValueError(f"boundaries must have length 2, got {len(boundaries)}")
        twists = _parse_twists(twists)
        self.axis1_boundary = _boundary_from_str(boundaries[0])(
            L=self.L[0], direction=0, phase=twists[0])
        self.axis2_boundary = _boundary_from_str(boundaries[1])(
            L=self.L[1], direction=1, phase=twists[1])

        self._pairs_by_distance = dict()
        self._image_pairs_by_distance = dict()
        self.sites = list()
        self.N_sites = 0

        self._dist_map = None

        self._distances = None
        self._image_distances = None

        self._built = False
        if build:
            self.build()

    @classmethod
    def from_type(cls, type: str, Ls, *, cyl_mode=None, **kwargs) -> "Lattice":
        """
        A lattice with the unit cell of a built-in lattice type.

        Parameters
        ----------
        type : str
            'square', 'triangular', 'honeycomb' or 'kagome'.
        Ls : iterable(int)
            2-element lattice size, in units of the unit cell.
        cyl_mode : {None, 'XC', 'YC'}, optional
            Reshape the triangular unit cell into a rectangular one of twice the
            size, ``a2 -> -a1 + 2*a2``. 'XC' halves ``Ls[1]``. 'YC' halves
            ``Ls[0]`` and swaps the x and y components of the cell, so the lattice
            is ``(Ls[1], Ls[0] // 2)`` cells. Only for 'triangular'.
        **kwargs
            Forwarded to the `Lattice` constructor: ``boundaries``, ``twists``
            and ``build``.

        Examples
        --------
        A honeycomb lattice with a twist, open along the second axis:

        >>> lattice = Lattice.from_type('honeycomb', (6, 6),
        ...                             boundaries=('periodic', 'open'),
        ...                             twists=('1/2 pi', 0.0))
        """
        if type.lower() not in LATTICE_TYPES:
            raise ValueError(
                f"Unknown lattice type '{type}': supported types are {sorted(LATTICE_TYPES)}")
        lattice_vectors, basis = LATTICE_TYPES[type.lower()]

        if cyl_mode is not None:
            if cyl_mode not in ("XC", "YC"):
                raise ValueError(f"Unsupported cylinder mode {cyl_mode=}; try None, 'XC' or 'YC'")
            if type.lower() != 'triangular':
                # the reshaping is only the right rectangular cell for hexagonal geometry
                raise ValueError(
                    f"Unsupported cylinder mode {cyl_mode=} for lattice type '{type}': "
                    "XC/YC cell reshaping is only available for 'triangular'"
                )
            a1, a2 = np.array(lattice_vectors)
            lattice_vectors = [a1, -a1 + 2 * a2]
            # b and b + a2 in the new cell: a2 = (a1 + a2') / 2
            f1, f2 = np.transpose(basis)
            half = np.column_stack([f1 + f2 / 2, f2 / 2])
            basis = np.vstack([half, half + 0.5])
            if cyl_mode == "XC":
                Ls = (Ls[0], Ls[1] // 2)
            else:
                # swapping x and y of the cell leaves the fractional basis unchanged
                Ls = (Ls[1], Ls[0] // 2)
                lattice_vectors = [v[::-1] for v in lattice_vectors]

        return cls(lattice_vectors, basis, Ls, **kwargs)

    def __getitem__(self, index):
        return self.sites[index]

    @property
    def A(self):
        """
        The lattice vectors ``a1`` and ``a2`` as the *columns* of a 2x2 matrix,
        so site ``b`` of cell ``(n1, n2)`` sits at ``A @ ((n1, n2) + basis[b])``.
        Read-only; see `Lattice`.
        """
        return self._A

    @property
    def basis(self):
        """
        Basis vectors within the unit cell, in fractional coordinates of the
        lattice vectors. Read-only; see `Lattice`. The list and its arrays are
        both immutable.
        """
        return tuple(self._basis)

    @property
    def nb(self):
        """Number of sites in the unit cell (basis size)."""
        return self._nb

    @property
    def num_sublattice(self):
        """Number of sublattices; alias of `nb`."""
        return self._nb

    def _fail_if_not_built(self):
        if not self._built:
            raise RuntimeError(
                "Must build lattice instance via Lattice.build() "
                "before retrieving sites"
            )

    def add_site(self, coord):
        """
        Interface for external Builder
        """
        self.sites.append(
            LatticeSite(
                index=self.N_sites,
                coord=coord,
                position=self._position(coord)
            )
        )
        self.N_sites += 1

    def get_nth_neighbors(self, n=1):
        """
        High-Level interface to get all nth-nearest neighbors
        """
        return self.get_nth_direct_neighbors(n=n) + self.get_nth_image_neighbors(n=n)

    def get_nth_direct_neighbors(self, n=1):
        """
        get nth-order direct neighbors within the home cell only
        """
        self._fail_if_not_built()

        if n not in self._pairs_by_distance.keys():
            logger.debug("computing and storing %sth-nearest neighbors", n)
            self._build_nth_neighbors(n=n)

        return self._pairs_by_distance[n]

    def get_nth_image_neighbors(self, n=1):
        self._fail_if_not_built()

        if n not in self._image_pairs_by_distance.keys():
            logger.debug("computing and storing %sth-nearest image neighbors", n)
            self._build_nth_image_neighbors(n=n)

        return self._image_pairs_by_distance[n]

    def get_sites(self):
        self._fail_if_not_built()
        return self.sites

    def get_positions(self, sublattice_index=None):
        """
        Get the positions of all sites in the lattice.

        Parameters
        ----------
        sublattice_index : int, optional
            index of the sublattice to get positions for. If None, returns
            positions for all sites in the lattice. Default is None.

        Returns
        -------
        np.ndarray
            array of positions of all sites in the lattice
        """
        if sublattice_index is not None:
            return np.array([s.position for s in self.sites if s.coord[-1] == sublattice_index])
        else:
            return np.array([s.position for s in self.sites])

    def get_kvecs(self):
        """
        get b1,b2 the Bravais lattice vectors
        """
        self._fail_if_not_built()
        return 2*np.pi*np.linalg.inv(self.A)

    def get_planewaves(self, returnKVecs=False):
        """
        return unitary exp(ik*r) for each site, corresponding
        to the unit cell r. Currently fixed to periodic/periodic only
        """
        self._fail_if_not_built()
        periodicX = isinstance(self.axis1_boundary, PBCBoundary)
        periodicY = isinstance(self.axis2_boundary, PBCBoundary)
        if not (periodicX and periodicY):
            warn("get_planewaves() is only implemented for periodic/periodic boundaries")
        kvecs = self.get_kvecs()
        twist = np.array([self.axis1_boundary.phase,
                          self.axis2_boundary.phase])
        ks = np.array([s.coord[:2]@kvecs for s in self.sites])
        ks += twist
        rs = np.array([self.A@(s.coord[:2]/self.L) for s in self.sites])
        Uxtok = np.exp(1j*(ks@rs.T))/np.sqrt(self.L[0]*self.L[1])
        # we've mixed each basis, so we need to zero out those elements

        if self._nb > 1:
            mask = np.equal.outer([s.coord[-1] for s in self.sites],
                                  [s.coord[-1] for s in self.sites])
            Uxtok *= mask
        if returnKVecs:
            return ks, Uxtok
        return Uxtok

    def build(self):
        """
        Populate the lattice sites.

        Building twice is an error: the site list does not change once a
        lattice is built.
        """
        if self._built:
            raise RuntimeError(
                "Lattice instance is already built; the site list does not "
                "change once built"
            )

        self._validate_basis()

        self._build()
        self._built = True

    def _validate_basis(self):
        """
        Validates that a basis can be described by the neighbor machinery correctly.

        Checks for both of the failure modes below:

        1. **Two basis vectors differing by a lattice translation.** They
           describe the *same* site, so the lattice holds coincident sites and
           the neighbor lists are wrong.
        2. **A basis offset reaching a full supercell or more.**
           `_build_image_distances` only shifts by one supercell in each
           direction, so for larger offsets the true minimum image is never
           tested.

        Checked in fractional coordinates.
        Folding the basis into the unit cell satisfies both conditions for any
        lattice size.

        Both conditions depend on the boundaries as well as the size, since an
        open axis neither wraps nor contributes image shifts. A translation of
        one cell along an open axis of a 1-cell-wide lattice, for instance,
        lands outside the lattice and so collides with nothing.
        """
        periodic = (isinstance(self.axis1_boundary, PBCBoundary),
                    isinstance(self.axis2_boundary, PBCBoundary))
        for (i, first), (j, second) in itertools.combinations(enumerate(self._basis), 2):
            offset = second - first

            if np.allclose(offset, np.round(offset)):
                translation = np.round(offset).astype(int)
                # a periodic axis wraps, so any translation along it lands on a
                #   real site; an open one only does if it stays in the lattice
                reachable = all(
                    wraps or abs(component) <= length - 1
                    for component, length, wraps in zip(translation, self.L, periodic)
                )
                if reachable:
                    raise ValueError(
                        f"basis vectors {i} and {j} differ by the lattice translation "
                        f"{translation.tolist()}, so they describe the same site in a "
                        f"{tuple(self.L)} lattice: sites would coincide and neighbor "
                        "lists would be wrong. Every basis vector must be distinct "
                        "modulo the lattice vectors — fold the basis into the unit cell."
                    )

            # only periodic axes are searched for images, so only they can miss one
            too_far = [axis for axis, (component, length, wraps)
                       in enumerate(zip(offset, self.L, periodic))
                       if wraps and abs(component) >= length]
            if too_far:
                spans = np.abs(offset).round(3).tolist()
                raise ValueError(
                    f"basis vectors {i} and {j} are {spans} unit cells apart along "
                    f"(a1, a2), which reaches beyond the {tuple(self.L)} supercell on "
                    f"axis {too_far}. Neighbor searches only shift by one supercell, "
                    "fold the basis into the unit cell."
                )

    def _build(self):
        """
        We may want different cell shapes in the future, for now
            this builds multiples of the unit cell.
        """
        Lx, Ly = self.L
        for lx in range(Lx):
            for ly in range(Ly):
                for n in range(self._nb):
                    self.add_site(np.array((lx, ly, n)))

    def _index_map(self, coord):
        """
        returns basis index given the lattice coordinate, `coord`.
        """
        return coord[0]*self.L[1]*self._nb + coord[1]*self._nb + coord[2]

    def _position(self, coord):
        return self.A @ (coord[:2] + self._basis[coord[2]])

    def _build_distances(self):
        if self._distances is None:
            poses = np.asarray([s.position for s in self.sites])
            self._distances = spatial.distance.cdist(poses, poses)

    def _build_nth_neighbors(self, n):
        self._pairs_by_distance[n] = list()

        try:
            # if we need to build up the distances, initialize it now
            self._neighbor_distance_map(0)
            self._build_distances()
            dist_to_find = self._dist_map[n]

            distances = self._distances

            pairs = np.where(np.isclose(distances, dist_to_find))
            self._pairs_by_distance[n] = [
                NeighborPair(
                    i, j,
                    r_relative=self.sites[j].position-self.sites[i].position
                ) for i, j in zip(*pairs)
            ]

        except ValueError as e:
            # discard malformed neighbor lists before propagating exception
            self._pairs_by_distance.pop(n)
            raise ValueError(
                f"Could not build nth-neighbors with n={n} "
                f"due to an exception: {e}"
            ) from e

        if len(self.basis) > 1:
            return
        # the following only is true for fairly square lattices without a basis
        dist_pairs = [p.i for p in self._pairs_by_distance[n]]
        for i in range(self.N_sites):
            if i not in dist_pairs:
                warn(
                    f"attempted to build nth order neighbors for n={n} "
                    "where some lattice sites have no "
                    "neighbors within the home cell.",
                    RuntimeWarning, stacklevel=4)
                break

    def is_image(self, r) -> bool:
        """True if `r`, in lattice coordinates, lies outside the lattice."""
        return self.axis1_boundary.is_image(r) or self.axis2_boundary.is_image(r)

    def _build_image_distances(self):
        """
        Create the cache of shifts and corresponding distances
        Useful for fast computation, but memory intensive
        """
        if self._image_distances is None:
            self._image_distances = []
            poses = np.asarray([s.position for s in self.sites])
            periodicX = isinstance(self.axis1_boundary, PBCBoundary)
            periodicY = isinstance(self.axis2_boundary, PBCBoundary)
            xShifts, yShifts = [0], [0]
            vShift1, vShift2 = (self.A * self.L).T
            if periodicX:
                xShifts = [-1, 0, 1]
            if periodicY:
                yShifts = [-1, 0, 1]
            for shift1 in xShifts:
                for shift2 in yShifts:
                    if shift1 == shift2 == 0:
                        continue
                    shiftvec = shift1*vShift1+shift2*vShift2
                    distances = spatial.distance.cdist(poses, poses + shiftvec)
                    self._image_distances.append((shift1, shift2, shiftvec, distances))

    def _build_nth_image_neighbors(self, n):
        """
        building the image nth-order-neighbors from the cache of shifted
          distance matrices built by `_build_image_distances`.

          We reject the case where a site is it's own nth-order neighbor.
        """
        self._image_pairs_by_distance[n] = list()

        self._neighbor_distance_map(0)  # init _dist_map
        self._build_image_distances()
        dist_to_find = self._dist_map[n]
        Md = np.zeros((self.N_sites, self.N_sites))
        for shift1, shift2, shiftvec, distances in self._image_distances:
            pairs = np.where(np.isclose(distances, dist_to_find))
            # crossing a boundary forward (shift +1) picks up -phase, backward +phase;
            #   only periodic axes are ever shifted
            phase = -(shift1*self.axis1_boundary.phase + shift2*self.axis2_boundary.phase)
            # check if we found this yet, otherwise add
            for p in zip(*pairs):
                if Md[p] == 0:
                    pair = NeighborPair(*p)
                    r1 = self.sites[pair.i].position
                    r2 = self.sites[pair.j].position+shiftvec
                    image_pair = NeighborPair(pair.i, pair.j, phase=phase,
                                              _shift=(shift1, shift2),
                                              r_relative=r2-r1)
                    self._image_pairs_by_distance[n].append(image_pair)

            Md[pairs] = 1

    def _neighbor_distance_map(self, distance, min_distance=None):
        """
        compute neighbor distance map using scipy.spatial.distance_matrix()

        Parameters
        ----------
        distance : float
            distance to look up in the map.
        min_distance : float, optional
            smallest nonzero distance to keep in the map; shorter nonzero
            distances (e.g. small intra-cell basis separations) are dropped so
            that they do not occupy a neighbor shell. Only has an effect on the
            call that first builds the map.

        Returns
        -------
        int
            index of `distance` in the map, i.e. the `n` for which `distance`
            is the nth-neighbor separation.
        """

        def _positions_from_cell():
            logger.debug("Reading lattice site positions from Lattice")
            return np.array([site.position for site in self.sites])

        # 1. get all positions in an array
        if self._dist_map is None:

            logger.debug("Computing distance matrix of lattice")
            positions = _positions_from_cell()

            # 2. get distance matrix
            R = spatial.distance.cdist(
                positions,
                positions
            )

            # 3. find set of distances
            dist_set = set(np.round(R.flatten(), 8))

            # 4. make into a sorted list
            self._dist_map = list(dist_set)
            self._dist_map.sort()

            # 5. optionally drop separations that are too short to be a shell.
            #    The zero self-distance is kept so that _dist_map[n] remains the
            #    nth-neighbor separation.
            if min_distance is not None:
                self._dist_map = [
                    d for d in self._dist_map if d == 0.0 or d >= min_distance
                ]

        spot = np.where(np.isclose(self._dist_map, np.round(distance, 8)))[0]
        if len(spot) != 1:
            raise ValueError(f"{distance=} has no/multiple valid indices!")
        return spot[0]

    def get_directed_pairs(self, directions=None, nbands: int = 1):
        """
        The `DirectedPairs` the ``PairCorrelator`` estimator reads — see
        `get_directed_pairs`.
        """
        return get_directed_pairs(self, directions=directions, nbands=nbands)


LATTICE_TYPES = {
    'square': ([[1., 0.], [0., 1.]],
               [[0., 0.]]),
    'triangular': ([[1., 0.], [0.5, np.sqrt(3) / 2]],
                   [[0., 0.]]),
    # the honeycomb and kagome cells are the square-like ones netket uses, rather
    #   than the triangular cell with basis [(0, 0), (0, 1/sqrt(3))] and so on
    'honeycomb': ([[1., 0.], [0.5, np.sqrt(3) / 2]],
                  [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]),
    'kagome': ([[1., 0.], [0.5, np.sqrt(3) / 2]],
               [[0.5, 0.], [0., 0.5], [0.5, 0.5]]),
}
"""Lattice vectors and fractional basis of each `Lattice.from_type` type."""

_BOUNDARY_TYPES = {
    'open': OpenBoundary,
    'periodic': PBCBoundary,
    'pbc': PBCBoundary,
}


def _boundary_from_str(boundary_type):
    """
    Get the Boundary class named by the string `boundary_type`.

    Raises a ValueError for an unknown boundary type.
    """
    if boundary_type.lower() not in _BOUNDARY_TYPES:
        raise ValueError(
            f"Unknown boundary type '{boundary_type}': supported types are "
            f"{sorted(_BOUNDARY_TYPES)}"
        )
    return _BOUNDARY_TYPES[boundary_type.lower()]


DIRECTION_STEPS = {
    '0': (0, 0), 's': (0, 0), '': (0, 0),
    '+x': (1, 0), '-x': (-1, 0),
    '+y': (0, 1), '-y': (0, -1),
    '+x+y': (1, 1), '+y+x': (1, 1),
    '+x-y': (1, -1), '-y+x': (1, -1),
    '-x+y': (-1, 1), '+y-x': (-1, 1),
    '-x-y': (-1, -1), '-y-x': (-1, -1),
}
"""Unit step along each named direction, in lattice coordinates."""


@dataclass
class DirectedPairs:
    """
    For each direction, the orbital every orbital is paired with — what the
    ``PairCorrelator`` estimator reads.

    Attributes
    ----------
    offsets : dict
        Direction name to its ``(nbands*nsites,)`` array of orbital indices.
    """

    offsets: dict

    def to_hdf5(self, path) -> None:
        """
        Write the offsets to `path`, replacing any already in the file.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist.
        """
        with h5.File(path, 'a') as fh5:
            group = replace_group(fh5, 'PairCorrelator/orbital_map')
            for direction, offsets in self.offsets.items():
                group.create_dataset(
                    direction, data=np.asarray(offsets, dtype=np.int32)[:, np.newaxis])


def get_directed_pairs(lattice: Lattice, directions=None, nbands: int = 1):
    """
    Pair every orbital with the one `directions` away from it.

    Parameters
    ----------
    lattice : Lattice
        Lattice to walk. Every direction must lie along periodic axes — see
        Raises.
    directions : sequence of str
        Direction names, as `DIRECTION_STEPS` spells them: ``'s'`` for the site
        itself, ``'+x'``, ``'-y'``, ``'+x-y'`` and so on.
    nbands : int, optional
        Bands per site. Sites are expanded into orbitals as
        ``site*nbands + band``, matching the Hamiltonian's basis order.
        Default 1.

    Returns
    -------
    DirectedPairs

    Raises
    ------
    ValueError
        If a direction is not a recognized name, or if it steps along an axis
        with an open boundary. Stepping off an open edge would wrap around to
        the far side and pair two sites that are not neighbors at all.
    """
    periodic = (isinstance(lattice.axis1_boundary, PBCBoundary),
                isinstance(lattice.axis2_boundary, PBCBoundary))

    steps = {}
    for direction in directions:
        if direction not in DIRECTION_STEPS:
            raise ValueError(
                f"unknown direction {direction!r}; expected one of "
                f"{', '.join(sorted(DIRECTION_STEPS))}"
            )
        step = DIRECTION_STEPS[direction]
        for axis, (moves, wraps) in enumerate(zip(step, periodic)):
            if moves and not wraps:
                raise ValueError(
                    f"direction {direction!r} steps along axis {axis + 1}, which "
                    "has an open boundary; only periodic axes can be paired"
                )
        steps[direction] = step

    offsets = {direction: [] for direction in directions}
    for site in lattice.sites:
        for direction, step in steps.items():
            coord = site.coord.copy()
            coord[0] += step[0]
            coord[1] += step[1]

            # every axis stepped along is periodic, so the offset site is always
            #   reachable; none are encoded as invalid ("-1")
            coord[0] %= lattice.L[0]
            coord[1] %= lattice.L[1]

            index = lattice._index_map(coord)
            offsets[direction].extend(index * nbands + band
                                      for band in range(nbands))

    return DirectedPairs({direction: np.array(values, dtype=np.int32)
                          for direction, values in offsets.items()})
