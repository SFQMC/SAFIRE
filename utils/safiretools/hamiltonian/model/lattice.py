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
computed neighbor lists. Concrete subclasses differ only in their lattice
vectors and basis; everything else lives on the ABC.

The unit-cell geometry belongs to the lattice *type*. ``a1``, ``a2`` and
``basis`` always exist on an instance, but for the built-in types they are
pre-set, so they are not caller-settable. ``CustomLattice`` is
used to define your own unit cell is precisely. See `Lattice` for the full rule.

Lattices are never serialized on their own. They are persisted as embedded
state inside a ``LatticeHamiltonian``'s HDF5 file, so there is no
``to_hdf5``/``from_hdf5`` here.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from warnings import warn

import numpy as np
import scipy.spatial as spatial

logger = logging.getLogger(__name__)

GEOMETRY_KEYS = ('a1', 'a2', 'basis')
"""Unit-cell geometry keys, settable only for `CustomLattice`."""


def euclid_nd(coord1, coord2):
    """
    Compute the N-dimensional Euclidean distance between coordinate 1 and
    coordinate 2.
    """
    return np.sqrt(
        np.sum(
            np.square(coord1 - coord2)
        )
    )


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
    _abs_r: tuple
    _shift: tuple
    r_relative: tuple
    phase: float = 0.0

    def __init__(self, i: int, j: int, _abs_r=None, _shift=None,
                 phase: float = 0.0, r_relative=None) -> None:
        if i == j:
            raise ValueError("Lattice sites are not allowed to be their own neighbor")

        self.i = i                  # index of first site
        self.j = j                  # index of second site

        self._abs_r = _abs_r        # absolute lattice coordinate of "j". this differs from
                                    #    the coordinate of "j" when the second site is an image
        self._shift = _shift        # supercell shift to obtain image
        self.phase = phase          # relative phase between sites

        self.r_relative = r_relative  # relative *spatial* position of site "j" with
                                      #    respect to "i"

    def __str__(self) -> str:
        return f"{self.i}->{self.j}  phase={self.phase} rad."


class Boundary:
    """
    Base class for representing a boundary.
      Encapsulates the boundary condition.
    """

    def __init__(self, L, direction=None, *args, **kwargs) -> None:
        self.L = L
        self.direction = direction
        if not hasattr(self, 'phase'):
            self.phase = None

    def is_valid_image(self, coordinate):
        return self.is_image(coordinate) and self.is_valid(coordinate)

    def is_image(self, coordinate):
        r"""
        returns `True` if the site at a given coordinate is an image.

        Note: coordinate is expressed in units of the lattice vectors \hat{a}_1, \hat{a}_2
        """
        if coordinate[self.direction] % self.L != coordinate[self.direction]:
            return True
        else:
            return False

    def is_allowed(self, position) -> bool:
        """
        returns True if the site at a given position is allowed by the boundaries.
            all cites within the cell are "allowed", but some image sites may not
            be "allowed" if there is one or more open boundary.
        """
        raise NotImplementedError("Called 'is_allowed' for Boundary abstract base class")

    # TODO: remove when new "is_allowed" framework is implemented and tested
    def is_valid(self, position) -> bool:
        """
        returns True if the site at a given position is a valid image

        ( checks if site is an image first, a site that is not an image
         is not a valid image )
        """
        raise NotImplementedError("Called 'is_valid' for Boundary abstract base class")


class PBCBoundary(Boundary):
    """
    Concrete class for periodic boundary condition

    Parameters
    ----------
    L : int
        size of the lattice in the direction of the boundary
    direction : int
        direction of the boundary (i.e. along a1 or a2)
    phase : iterable(ints)
        phase angle for the boundary (i.e. the twist angle)
        if phase is not given, it is assumed to be (0,0);
        if phase is a 1-d iterable of length 2, it is interpreted as (phase1,phase2)
        where phase1 is applied when crossing the boundary along the a1 direction
        and phase2 is applied when crossing the boundary along the a2 direction.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if 'phase' in kwargs.keys():
            self.phase = kwargs['phase']
        else:
            self.phase = (0.0, 0.0)

    def is_valid(self, position):
        if self.is_image(position):
            return True
        else:
            return False

    def is_allowed(self, position) -> bool:
        return True


class OpenBoundary(Boundary):
    """
    concrete class for open boundary

    all images are invalid for open b.c.
    """

    def __init__(self, L, direction=None, **kwargs) -> None:
        super().__init__(
            L=L,
            direction=direction,
            **kwargs
        )

    def is_valid(self, position) -> bool:
        return False

    def is_allowed(self, position) -> bool:
        return not super().is_image(position)


def valid_L(L):
    """
    True if `L` is a 2-element specification of the lattice size.
    """
    if L is None:
        return False

    return hasattr(L, '__len__') and len(L) == 2


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


def _parse_twist(twist):
    """
    parse the twist angle iterable based on what types are within the tuple

    accepted types:
    - str : a string representing a number. "pi" may be included
    - number : interpreted as the twist angle in radians
    """

    # Check if twist is iterable (but not a string, which is iterable but should be rejected)
    if isinstance(twist, (int, float)):
        raise TypeError(
            "Invalid 'twist' parameter: expected a 2-element iterable (tuple or list), "
            f"but got a single number: {twist}"
        )

    if isinstance(twist, str):
        raise TypeError(
            "Invalid 'twist' parameter: expected a 2-element iterable (tuple or list), "
            f"but got a string: '{twist}'"
        )

    # Check if twist has the __len__ attribute to avoid TypeError
    if not hasattr(twist, '__len__'):
        raise TypeError(
            "Invalid 'twist' parameter: expected a 2-element iterable (tuple or list), "
            f"but got type {type(twist).__name__}"
        )

    if len(twist) != 2:
        raise ValueError(
            f"Invalid 'twist' parameter given: must have length 2, got length {len(twist)}"
        )

    _twist = [0., 0.]

    for i, t in enumerate(twist):
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


class Lattice(ABC):
    """
    Base class for representing lattices.

    **The unit-cell geometry belongs to the lattice type, not to the caller.**
    Every instance always has an ``a1``, an ``a2`` and a ``basis``, but for the
    built-in types those define the type. The geometry is supplied by the type's
    `_geometry()` implementation, and the constructor below — the only one —
    takes no geometry arguments at all.

    `CustomLattice` is used to define a custom lattice geometry: it is the
    one subclass whose constructor accepts ``a1``/``a2``/``basis``, and the only
    supported way to build a lattice whose geometry is not one of the built-in
    types. Passing ``a1``/``a2``/``basis`` to any other type raises `TypeError`,
    and `Lattice.from_dict` raises `ValueError` for the equivalent keys in a
    parameter dict; both used to be silently discarded.

    ``a1``, ``a2`` and ``basis`` are read-only properties, and the arrays they
    return are themselves immutable, so the geometry cannot be replaced or
    edited in place after construction. `build()` applying `cyl_mode` is
    the sole exception.
    """

    _type = None
    """Lattice type name, for reference only. Set by each concrete subclass."""

    @abstractmethod
    def _geometry(self):
        """
        Return this lattice type's ``(a1, a2, basis)``.

        ``a1`` and ``a2`` are required. ``basis`` may be ``None``, meaning a
        single site at the cell origin.
        """

    def __init__(
            self,
            L=None,
            *,
            metric=euclid_nd,
            axis1_boundary=None,
            axis2_boundary=None,
            build=True,
            twist=None,
            cyl_mode=None,
            _dist_map_min_distance=None,
    ) -> None:
        """
        Initialize with an "empty" lattice containing no sites; ``build()``
        populates them.

        There are deliberately no ``a1``/``a2``/``basis`` parameters here: the
        unit-cell geometry comes from this lattice type's `_geometry()`. Use
        `CustomLattice` to supply your own.

        Parameters
        ----------
        L : iterable(int)
            2-element lattice size, (L1, L2), in units of the unit cell.
        metric : callable, optional
            Distance function between two positions. Default is `euclid_nd`.
        axis1_boundary, axis2_boundary : type(Boundary), optional
            Boundary *classes* (not instances) applied along a1 and a2.
            Default is `OpenBoundary`.
        build : bool, optional
            Build the lattice sites immediately. Default is True.
        twist : iterable, optional
            2-element twist angle, one per axis. See `_parse_twist` for the
            accepted element types.
        cyl_mode : {None, 'XC', 'YC'}, optional
            Reshape the unit cell to be rectangular. The reshaping is hardcoded
            for hexagonal (triangular) geometry, so it is accepted **only** by
            `TriangularLattice` and raises `ValueError` for any other type.
            This is the one supported way the geometry changes after
            construction — `build()` applies it.
        _dist_map_min_distance : float, optional
            Smallest nonzero distance to keep in the neighbor distance map.
            Applied when the map is first built. See `_neighbor_distance_map`.
        """
        a1, a2, basis = self._geometry()

        if a1 is None or a2 is None:
            raise ValueError(
                f"{type(self).__name__}._geometry() returned no lattice vectors; "
                "every lattice type must define both 'a1' and 'a2'"
            )
        self._set_geometry(a1, a2, [np.zeros(2)] if basis is None else basis)

        self._pairs_by_distance = dict()
        self._image_pairs_by_distance = dict()
        self._metric = metric
        self._metric_v = np.vectorize(metric, signature="(n),(n)->()")
        self.sites = list()
        self.N_sites = 0

        # Change unit cell to be rectangular
        # default is nothing
        # with XC y dim is longer (1 in x dim)
        # and YC x is longer (1 in y dim)
        if cyl_mode not in ["XC", "YC", "none", "None", None, False]:
            raise ValueError(f"Unsupported cylinder mode {cyl_mode=}"
                             "\n Try None, 'XC', or 'YC'")
        if cyl_mode in ("XC", "YC") and self._type != "triangular":
            # the reshaping in build() rotates a2 onto -a1+2*a2 and doubles the
            #   basis along a2, which is only the right cell for hexagonal
            #   (triangular) geometry
            raise ValueError(
                f"Unsupported cylinder mode {cyl_mode=} for lattice type "
                f"'{self._type}': XC/YC cell reshaping is hardcoded for hexagonal "
                "(triangular) geometry and is only available on TriangularLattice"
            )
        self.cyl_mode = cyl_mode

        if valid_L(L):
            self.L = L
        else:
            raise ValueError("L must be a 2-d Array-like")

        if twist is None:
            twist = (0.0, 0.0)
        else:
            twist = _parse_twist(twist)

        if axis1_boundary is None:
            axis1_boundary = OpenBoundary

        if axis2_boundary is None:
            axis2_boundary = OpenBoundary

        self.axis1_boundary = axis1_boundary(L=L[0], direction=0, phase=twist[0])
        self.axis2_boundary = axis2_boundary(L=L[1], direction=1, phase=twist[1])

        self._dist_map = None
        self._dist_map_min_distance = _dist_map_min_distance

        self._distances = None
        self._image_distances = None

        self._built = False
        if build:
            self.build()

    @classmethod
    def from_dict(cls, params: dict, build=None):
        """
        Build a lattice from a parameter dictionary, dispatching on
        ``params['type']``.

        Parameters
        ----------
        params : dict
            Lattice parameters. ``L1`` and ``L2`` are required. Recognized
            optional keys: ``type`` ('square', 'triangular', 'honeycomb',
            'kagome', 'custom'; default 'square'), ``boundary1``/``boundary2``
            ('open', 'pbc'/'periodic'; default 'open'), ``build``, ``twist``,
            ``cyl_mode``, and — for ``type='custom'`` only — ``a1``, ``a2`` and
            ``basis``.
        build : bool, optional
            Build the lattice sites immediately, overriding ``params['build']``.

        Returns
        -------
        Lattice
            An instance of the concrete subclass named by ``params['type']``.

        Raises
        ------
        ValueError
            If ``a1``, ``a2`` or ``basis`` is given for a type other than
            'custom'. Those types define their own geometry (see `Lattice`), and
            a dict that sets them is asking for something it will not get — the
            old `get_lattice` discarded them silently. A key present but set to
            ``None`` is fine, so parameter templates that carry unused keys
            still work.
        """
        lattice_type = str(params.get('type', 'square')).lower()
        if lattice_type not in _LATTICE_TYPES:
            raise ValueError(
                f"Unknown lattice type '{lattice_type}': supported types are "
                f"{sorted(_LATTICE_TYPES)}"
            )
        lattice_cls = _LATTICE_TYPES[lattice_type]

        geometry = {
            key: params[key] for key in GEOMETRY_KEYS
            if params.get(key) is not None
        }
        if geometry and not issubclass(lattice_cls, CustomLattice):
            raise ValueError(
                f"Lattice type '{lattice_type}' defines its own unit-cell geometry; "
                f"{sorted(geometry)} can only be set for type='custom'"
            )

        L = (
            params['L1'],
            params['L2']
        )

        if L[0] < 1 or L[1] < 1:
            raise ValueError("Invalid lattice size; L1 and L2 must be >= 1")

        if build is None:
            build = params.get("build", True)

        return lattice_cls(
            L=L,
            axis1_boundary=_boundary_from_str(params.get("boundary1")),
            axis2_boundary=_boundary_from_str(params.get("boundary2")),
            build=build,
            twist=params.get("twist", None),
            cyl_mode=params.get("cyl_mode", None),
            **geometry,
        )

    def __getitem__(self, index):
        return self.sites[index]

    def _set_geometry(self, a1, a2, basis):
        """
        Install the unit-cell geometry, as immutable copies.

        The only two callers are `__init__` and `build()`'s `cyl_mode` reshaping;
        the geometry never changes after a lattice is built. Copies are taken so
        that freezing them cannot reach an array the caller still holds.
        """
        self._a1 = np.array(a1, dtype=float)
        self._a2 = np.array(a2, dtype=float)
        self._basis = [np.array(b, dtype=float) for b in basis]
        for array in (self._a1, self._a2, *self._basis):
            array.setflags(write=False)
        self._nb = len(self._basis)

    @property
    def a1(self):
        """
        First lattice vector. Read-only: the geometry belongs to the lattice
        type (see `Lattice`) and does not change once the lattice is built.
        """
        return self._a1

    @property
    def a2(self):
        """
        Second lattice vector. Read-only: the geometry belongs to the lattice
        type (see `Lattice`) and does not change once the lattice is built.
        """
        return self._a2

    @property
    def basis(self):
        """
        Basis vectors within the unit cell. Read-only: the geometry belongs to
        the lattice type (see `Lattice`) and does not change once the lattice is
        built. The list and its arrays are both immutable.
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

    def get_nth_neighbors(self, n=1, twist=None):
        """
        High-Level interface to get all nth-nearest neighbors
        """
        return self.get_nth_direct_neighbors(n=n) + self.get_nth_image_neighbors(n=n, twist=twist)

    def get_nth_direct_neighbors(self, n=1):
        """
        get nth-order direct neighbors within the home cell only
        """
        self._fail_if_not_built()

        if n not in self._pairs_by_distance.keys():
            logger.debug("computing and storing %sth-nearest neighbors", n)
            self._build_nth_neighbors(n=n)

        return self._pairs_by_distance[n]

    def get_nth_image_neighbors(self, n=1, twist=None):
        self._fail_if_not_built()

        if n not in self._image_pairs_by_distance.keys():
            logger.debug("computing and storing %sth-nearest image neighbors", n)
            self._build_nth_image_neighbors(n=n, twist=twist)

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
        return 2*np.pi*np.linalg.inv(np.array([self.a1, self.a2])).T

    def get_planewaves(self, returnKVecs=False):
        """
        return unitary exp(ik*r) for each site, corresponding
        to the unit cell r. Currently fixed to PBC/PBC only
        """
        self._fail_if_not_built()
        periodicX = isinstance(self.axis1_boundary, PBCBoundary)
        periodicY = isinstance(self.axis2_boundary, PBCBoundary)
        if not (periodicX and periodicY):
            warn("get_planewaves() is only implemented for PBC/PBC")
        kvecs = self.get_kvecs()
        twist = np.array([self.axis1_boundary.phase,
                          self.axis2_boundary.phase])
        ks = np.array([s.coord[:2]@kvecs for s in self.sites])
        ks += twist
        # we should order it, but ks isn't always in the 1st bz
        # order = np.argsort(np.linalg.norm(ks,axis=-1))
        # ks = ks[order]
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

        Applying `cyl_mode` here is the one place the unit-cell geometry changes
        after construction, and it is derived from the type's own `_geometry()`
        rather than supplied by the caller. Building twice is an error: neither
        the geometry nor the site list changes once a lattice is built.
        """
        if self._built:
            raise RuntimeError(
                "Lattice instance is already built; the unit-cell geometry and "
                "the site list do not change once built"
            )

        # Change unit cell to be 2x and make unit vectors match rectangular
        if self.cyl_mode == "XC" or self.cyl_mode == "YC":
            basis = self._basis + [b+self.a2 for b in self._basis]
            a1, a2 = self.a1, -self.a1+2*self.a2
            if not (np.allclose(a1[1], 0.) and np.allclose(a2[0], 0.)):
                warn("Warning, XC/YC mode is not perfectly rectangular. "
                     "\nCurrently hardcoded for hexagonal (triangular) lattices"
                     f"\n(new {a1=} new {a2=})")
            if self.cyl_mode == "XC":
                self.L = (self.L[0], self.L[1]//2)
            elif self.cyl_mode == "YC":
                self.L = (self.L[0]//2, self.L[1])
                self.L = self.L[::-1]
                a1, a2 = a1[::-1], a2[::-1]
                basis = [b[::-1] for b in basis]

            self._set_geometry(a1, a2, basis)

        # promised to have a1,a2 by now, so we can
        # do manipulations on them
        self.A = np.array([self.a1, self.a2]).T
        self.Ainv = np.linalg.inv(self.A)

        self._build()
        self._built = True

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
        return coord[0]*self.a1 + coord[1]*self.a2 + self.basis[coord[2]]

    def metric(self, coord1, coord2):
        return self._metric(coord1, coord2)

    def _build_distances(self):
        if self._distances is None:
            poses = np.asarray([s.position for s in self.sites])
            # fast numpy outer product
            self._distances = self._metric_v(poses[:, None], poses[None, :])

    def remove_distances(self):
        """
        Removes cached distance matrix.
        Useful if low on memory and the lattice is large
        """
        del self._distances
        del self._image_distances
        self._distances = None
        self._image_distances = None
        logger.debug("Removed distance matrix")

    def _build_nth_neighbors(self, n):
        if n not in self._pairs_by_distance.keys():
            self._pairs_by_distance[n] = list()
        else:
            raise RuntimeError(
                "Adding pair distances to existing distance list: probably a mistake!"
            )

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
                    _abs_r=self.sites[j].position,
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

    def _is_valid_image(self, r):
        # If you invert, be sure to use De Morgan's theorem!!
        if self.axis1_boundary.is_valid(r) and self.axis2_boundary.is_valid(r):
            return True
        else:
            return False

    def _is_allowed_site(self, r):
        if self.axis1_boundary.is_allowed(r) and self.axis2_boundary.is_allowed(r):
            return True
        else:
            return False

    def _get_image_axes(self, r):
        r"""
        Checks if a coordinate, r, is a valid image separately
            for both boundaries.

        Returns a list of boundary indices (either 0 - $\hat{a}_1$ or 1 - $\hat{a}_2$)
            for which r is a valid image. The list is empty if r is not a valid image
            for any boundary.

        NOTE: This function will NOT perform a cross check to see if the
                combination of boundaries would reject an image.
                use `self._is_valid_image(r)`
        """
        is_x_image = self.axis1_boundary.is_valid_image(r)
        is_y_image = self.axis2_boundary.is_valid_image(r)

        if is_x_image and is_y_image:
            return [0, 1]
        elif is_x_image:
            return [0]
        elif is_y_image:
            return [1]
        else:
            return []

    def is_image(self, r):
        # If you invert, be sure to use De Morgan's theorem!!
        if self.axis1_boundary.is_image(r) or self.axis2_boundary.is_image(r):
            return True
        else:
            return False

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
            vShift1, vShift2 = self.L[0]*self.a1, self.L[1]*self.a2
            if periodicX:
                xShifts = [-1, 0, 1]
            if periodicY:
                yShifts = [-1, 0, 1]
            for shift1 in xShifts:
                for shift2 in yShifts:
                    if shift1 == shift2 == 0:
                        continue
                    shiftvec = shift1*vShift1+shift2*vShift2
                    distances = self._metric_v(poses[:, None],
                                               poses[None, :]+shiftvec)
                    self._image_distances.append((shift1, shift2, shiftvec, distances))

    def _build_nth_image_neighbors(self, n, twist=None):
        """
        building the image nth-order-neighbors from the cache of shifted
          distance matrices built by `_build_image_distances`.

          We reject the case where a site is it's own nth-order neighbor.
        """

        if twist is not None:
            raise NotImplementedError("Need to recompute nth-order neighbors when given a twist")

        if n not in self._image_pairs_by_distance.keys():
            self._image_pairs_by_distance[n] = list()
        else:
            raise RuntimeError(
                "Adding image pair distances to existing distance list: probably a mistake!"
            )

        self._neighbor_distance_map(0)  # init _dist_map
        self._build_image_distances()
        dist_to_find = self._dist_map[n]
        Md = np.zeros((self.N_sites, self.N_sites))
        for shift1, shift2, shiftvec, distances in self._image_distances:
            pairs = np.where(np.isclose(distances, dist_to_find))
            # check if we found this yet, otherwise add
            for p in zip(*pairs):
                if Md[p] == 0:
                    pair = NeighborPair(*p)
                    r1 = self.sites[pair.i].position
                    r2 = self.sites[pair.j].position+shiftvec
                    r = self._to_lattice_basis(r2)
                    image_axes = self._get_image_axes(r)  # TODO: replace with shift1/2
                    phase = 0.0

                    for axis in image_axes:
                        # for twists, we need +/- depending on which direction we
                        #    cross the boundary in.
                        if r[axis] > 1:
                            sign = -1
                        else:
                            sign = 1
                        phase += sign*getattr(self, f"axis{axis+1}_boundary").phase

                    # TODO: add a unit test that confirms that invalid pairs are rejected
                    image_pair = NeighborPair(pair.i, pair.j, phase=phase,
                                              _abs_r=r2, _shift=(shift1, shift2),
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
            that they do not occupy a neighbor shell. Defaults to the
            `_dist_map_min_distance` given at construction. Only has an effect
            on the call that first builds the map.

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
            R = spatial.distance_matrix(
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
            if min_distance is None:
                min_distance = self._dist_map_min_distance
            if min_distance is not None:
                self._dist_map = [
                    d for d in self._dist_map if d == 0.0 or d >= min_distance
                ]

        spot = np.where(np.isclose(self._dist_map, np.round(distance, 8)))[0]
        if len(spot) != 1:
            raise ValueError(f"{distance=} has no/multiple valid indices!")
        return spot[0]

    def _to_lattice_basis(self, r):
        r"""
        convert from a position vector, r = r_x \hat{x} + r_y \hat{y}
            to a position vector expressed in the lattice basis,
            r = c_1 \hat{a}_1 + c_2 \hat{a}_2 + \hat{b}_{c_3}.

            For a set of non-orthogonal lattice vectors, this
                requires the solution to the linear equation:

            r_(position basis) = A r_(lattice basis) where A is
                matrix consisting of the lattice unit vectors.
        """
        candidates = [self.Ainv@(r-b) for b in self.basis]
        # find which basis vector brought us exactly to the correct spot
        basis_idx = np.argmin([np.linalg.norm(c-c.round().astype(int)) for c in candidates])
        return np.array((*candidates[basis_idx].round().astype(int), basis_idx))

    def _wrap_to_cell(self, r):
        """
        wrap input `r`, expressed in lattice basis,
        into the lattice cell
        """
        return np.array(
            [
                r[0] % self.L[0],
                r[1] % self.L[1],
                r[2]
            ]
        )


class SquareLattice(Lattice):
    """
    Specialization of Lattice to a square lattice.

    Here, we define a square lattice as a lattice in
      which sites are equidistant in the x-, and y-directions.

    The unit cell is fixed by the type: unit lattice vectors along x and y, one
    site per cell. Use `CustomLattice` for anything else.
    """

    _type = "square"

    def _geometry(self):
        return np.array([1., 0.]), np.array([0., 1.]), None

    def get_directed_pairs(self, directions=None):
        """
        Get directed pairs - i.e. for making pair correlators

        for now, we only have square lattice pair 'directions' implemented
           i.e. +/-x, +/-y. will need to work out the Triangular lattice
           possibilities.
        """
        return get_directed_pairs(self, directions=directions)


class TriangularLattice(Lattice):
    """
    Specialization of Lattice to a triangular lattice.

    The unit cell is fixed by the type and holds a single site. For a
    triangular Bravais lattice with a multi-site basis, use `HoneycombLattice`
    or `KagomeLattice`, which are implemented separately; `CustomLattice` can
    express other bases, with the caveat noted there.

    This is the only lattice type that accepts `cyl_mode`: the XC/YC cell
    reshaping `build()` performs is hardcoded for hexagonal geometry.
    """

    _type = "triangular"

    def _geometry(self):
        return np.array([1., 0.]), np.array([0.5, np.sqrt(3)/2]), None


class HoneycombLattice(Lattice):
    """
    Specialization of Lattice to a Honeycomb lattice.
    The underlying lattice is a triangular lattice with an "extra"
    site per cell

    Both the lattice vectors and the 2-site basis are fixed by the type.
    """

    _type = "honeycomb"

    def _geometry(self):
        # a1/a2 closer to the original triangular cell would be
        #   [1.,0.] and [0.5,sqrt(3)/2] with basis [(0,0),(0,1/sqrt(3))];
        #   the more square-like cell below follows netket's choices.
        return (
            np.array([1., 0.]),
            np.array([1/2, 0.75**0.5]),
            [np.array([0.5, 0.5 / 3**0.5]),
             np.array([1, 1 / 3**0.5])],
        )


class KagomeLattice(Lattice):
    """
    Specialization of Lattice to a Kagome lattice.
    The underlying lattice is a triangular lattice with 3
    site per cell

    Both the lattice vectors and the 3-site basis are fixed by the type.
    """

    _type = "kagome"

    def _geometry(self):
        # as for the honeycomb lattice, the more square-like cell here follows
        #   netket's choices rather than the original triangular cell.
        return (
            np.array([1., 0.]),
            np.array([1/2, np.sqrt(0.75)]),
            [np.array([0.5, 0]),
             np.array([0.25, np.sqrt(0.75) / 2]),
             np.array([0.75, np.sqrt(0.75) / 2])],
        )


class CustomLattice(Lattice):
    """
    The lattice type whose geometry the caller defines.

    This is the **only** subclass that accepts ``a1``/``a2``/``basis``, and the
    supported way to build a lattice whose unit cell is not one of the built-in
    types — defining the unit cell yourself is what makes a lattice "custom".
    The built-in types own their geometry (see `Lattice`) and reject these
    arguments.

    Parameters
    ----------
    L : iterable(int)
        2-element lattice size, (L1, L2), in units of the unit cell.
    a1, a2 : array_like
        The two lattice vectors. Required.
    basis : list(array_like), optional
        Basis vectors within the unit cell. Default is a single site at the
        cell origin.
    **kwargs
        Forwarded to `Lattice`; see that constructor for `metric`, `build`,
        `axis1_boundary`/`axis2_boundary`, `twist` and `cyl_mode`.

    Examples
    --------
    >>> lattice = CustomLattice(
    ...     L=(2, 2),
    ...     a1=np.array([1, 1]),
    ...     a2=np.array([1, -1]),
    ...     basis=[np.array([0, 0]), np.array([0, 0.5])],
    ... )

    Notes
    -----
    Currently only supports 2D lattices. Higher dimensions will be added in the
    future. Please contact the developers if you need this feature.

    BUG: if the magnitude of the basis vectors is too large, there are errors
    with computing direct neighbors and image neighbors. Try limiting basis
    vectors to the unit cell.
    """

    _type = "custom"

    def __init__(self, L=None, *, a1, a2, basis=None, **kwargs) -> None:
        self._custom_geometry = (a1, a2, basis)
        super().__init__(L=L, **kwargs)

    def _geometry(self):
        return self._custom_geometry


_LATTICE_TYPES = {
    'square': SquareLattice,
    'triangular': TriangularLattice,
    'honeycomb': HoneycombLattice,
    'kagome': KagomeLattice,
    'custom': CustomLattice,
}

_BOUNDARY_TYPES = {
    'open': OpenBoundary,
    'pbc': PBCBoundary,
    'periodic': PBCBoundary,
}


def _boundary_from_str(boundary_type):
    """
    Get the Boundary class named by the string `boundary_type`, or None if
    `boundary_type` is None.

    Raises a ValueError for an unknown boundary type.
    """
    if boundary_type is None:
        return None

    if boundary_type.lower() not in _BOUNDARY_TYPES:
        raise ValueError(
            f"Unknown boundary type '{boundary_type}': supported types are "
            f"{sorted(_BOUNDARY_TYPES)}"
        )
    return _BOUNDARY_TYPES[boundary_type.lower()]


def get_directed_pairs(lattice: Lattice, directions=None):

    _directed_pairs = {}
    for direction in directions:
        _directed_pairs[direction] = []

    for site in lattice.sites:
        for direction in directions:
            coord = site.coord.copy()
            if direction in ('0', 's', ''):
                pass  # since we catch unknown directions below!
            elif direction == "+x":
                coord[0] += 1
            elif direction == "-x":
                coord[0] -= 1
            elif direction == "+y":
                coord[1] += 1
            elif direction == "-y":
                coord[1] -= 1
            elif direction in ("+x+y", "+y+x"):
                coord[0] += 1
                coord[1] += 1
            elif direction in ("+x-y", "-y+x"):
                coord[0] += 1
                coord[1] -= 1
            elif direction in ("-x+y", "+y-x"):
                coord[0] -= 1
                coord[1] += 1
            elif direction in ("-x-y", "-y-x"):
                coord[0] -= 1
                coord[1] -= 1
            else:
                raise ValueError("Unknown 'direction' in 'directions'")

            # For now, we assume that both boundaries are periodic, so every
            #   offset site is reachable and none are encoded as invalid ("-1").
            if not lattice.is_image(coord):
                _directed_pairs[direction].append(lattice._index_map(coord))
            else:
                coord[0] = coord[0] % lattice.L[0]
                coord[1] = coord[1] % lattice.L[1]
                _directed_pairs[direction].append(lattice._index_map(coord))

    return _directed_pairs
