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
Lattice classes for Lattice Model Builder.
"""
import itertools
from fractions import Fraction
from dataclasses import dataclass

import numpy as np
import scipy.spatial as spatial

from warnings import warn

def euclid_nd(coord1,coord2):
    '''
    Compute the N-dimensional Euclidean
      distance between coordinate 1 and
      coordinate 2.
    '''
    return np.sqrt(
        np.sum(
            np.square(coord1 - coord2)
        )
    )


@dataclass(order=True)
class LatticeSite:
    """
    Simple dataclass to hold basic metadata for each lattice site.

    Attributes
    ----------
    index : int
        Ordered basis index
    coord : np.ndarray[int]
        Lattice coordinates
    position : np.ndarray[float]
        Spatial position. For square lattices, this is the same as coord.
    """

    index:int
    coord:np.ndarray
    position:np.ndarray

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

    i:int
    j:int
    _abs_r:tuple
    _shift:tuple
    r_relative:tuple
    phase:float=0.0

    def __init__(self,i:int,j:int,_abs_r=None,_shift=None,phase:float=0.0,r_relative=None) -> None:
        if i == j:
            raise ValueError("Lattice sites are not allowed to be their own neighbor")

        self.i = i                  # index of first site
        self.j = j                  # index of second site

        self._abs_r = _abs_r        # absolute lattice coordinate of "j". this differs from
                                    #    the coordinate of "j" when the second site is an image
        self._shift = _shift        # supercell shift to obtain image
        self.phase = phase          # relative phase between sites

        self.r_relative = r_relative # relative *spatial* position of site "j" with
                                     #    respect to "i"


    def __str__(self) -> str:
        return f"{self.i}->{self.j}  phase={self.phase} rad."


class Boundary:
    """
    Base class for respresenting a boundary.
      Encapsulates the boundary condition.
    """

    def __init__(self,L,direction=None,*args,**kwargs) -> None:
        self.L=L
        self.direction = direction
        if not hasattr(self,'phase'):
            self.phase = None

    def is_valid_image(self,coordinate):
        return self.is_image(coordinate) and self.is_valid(coordinate)

    def is_image(self,coordinate):
        r"""
        returns `True` if the site at a given coordinate is an image.

        Note: coordindate is expressed in units of the lattice vectors \hat{a}_1, \hat{a}_2
        """
        if coordinate[ self.direction ] % self.L != coordinate[ self.direction ]:
            return True
        else:
            return False

    def is_allowed(self,position) -> bool:
        """
        returns True if the site at a given position is allowed by the boundaries.
            all cites within the cell are "allowed", but some image sites may not
            be "allowed" if there is one or more open boundary.
        """
        raise NotImplementedError("Called 'is_allowed' for Boundary abstract base class")


    # TODO: remove when new "is_allowed" framework is implemetned and tested
    def is_valid(self,position) -> bool:
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
          where phase1 is applied when corssing the boundary along the a1 direction
          and phase2 is applied when crossing the boundary along the a2 direction.
    """

    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

        if 'phase' in kwargs.keys():
            self.phase = kwargs['phase']
        else:
            self.phase = (0.0,0.0)

    def is_valid(self,position):
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

    if L is None:
        return False

    if hasattr(L,"shape"):
        assert len(L.shape) == 2
        return True

    if len(L) == 2:
        return True

    return False

def _angle_str_to_float(angle_string:str):
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
    '''
    parse the twist angle iterable based on what types are within the tuple

    accepted types:
    - str : a string representing a number. "pi" may be included
    - number : interpreted as the twist anlge in radians
    '''

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

    _twist = [0.,0.]

    for i,t in enumerate(twist):
        if isinstance(t, str):
            _twist[i] = _angle_str_to_float(t)
        elif isinstance(t,float):
            _twist[i] = t
        elif isinstance(t,int):
            _twist[i] = float(t)
        else:
          raise ValueError("Twist format not currently supported! "
                           "Please add a feature request")

    return _twist

class Lattice:
    """
    Base class for representing lattices
    """

    ROTATION_GROUP = None

    def __init__(
            self,
            L=None,
            metric=euclid_nd,
            axis1_boundary=None,
            axis2_boundary=None,
            build=True,
            twist=None,
            cyl_mode=None,
            basis=None,
            _dist_map_min_distance=None,
            **kwargs
            ) -> None:
        '''
        Initialize with an "empty" lattice
          containing no sites. Concrete
          classes define how to build
        '''
        self._pairs_by_distance = dict()
        self._metric = metric
        self._metric_v = np.vectorize(metric,signature="(n),(n)->()")
        self.sites = list()
        self.N_sites = 0

        # Change unit cell to be rectangular
        # default is nothing
        # with XC y dim is longer (1 in x dim)
        # and YC x is longer (1 in y dim)
        if cyl_mode not in ["XC","YC","none","None",None,False]:
          raise ValueError(f"Unsupported cylinder mode {cyl_mode=}"
                            "\n Try None, 'XC', or 'YC'")
        self.cyl_mode = cyl_mode

        if valid_L(L):
            self.L = L
        else:
            raise ValueError("L must be a 2-d Array-like")

        if axis1_boundary is None:
            axis1_boundary = OpenBoundary

        if axis2_boundary is None:
            axis2_boundary = OpenBoundary

        self._image_pairs_by_distance = dict()

        if not hasattr(self,"a1"):
            self.a1 = None
        if not hasattr(self,"a2"):
            self.a2 = None
        if not hasattr(self,"basis"):
            if basis is None:
                self.basis = [np.zeros(2)]
            else:
                self.basis = basis

        # number of atoms in a cell with aliases
        self._nb = len(self.basis)
        self.nb = self._nb
        self.num_sublattice = self._nb

        if twist is None:
            twist = (0.0,0.0)
        else:
            twist = _parse_twist(twist)

        self.axis1_boundary=axis1_boundary(L=L[0],direction=0,phase=twist[0])
        self.axis2_boundary=axis2_boundary(L=L[1],direction=1,phase=twist[1])


        self._dist_map = None

        self._distances = None
        self._image_distances = None

        if _dist_map_min_distance is not None:
            self._neighbor_distance_map(
                distance=0.,
                min_distance=_dist_map_min_distance
                )

        self._built = False
        if build:
            self.build()

    def __getitem__(self,index):
        return self.sites[index]

    def _fail_if_not_built(self):
        if not self._built:
            raise RuntimeError(
                "Must build lattice instance via Lattice.build() "
                "before retrieving sites"
            )

    def add_site(self,coord):
        '''
        Interface for external Builder
        '''
        self.sites.append(
            LatticeSite(
                index=self.N_sites,
                coord=coord,
                position=self._position(coord)
            )
        )
        self.N_sites += 1


    def get_nth_neighbors(self,n=1,twist=None):
        '''
        High-Level interface to get all nth-nearest neighbors
        '''
        return self.get_nth_direct_neighbors(n=n) + self.get_nth_image_neighbors(n=n,twist=twist)


    def get_nth_direct_neighbors(self,n=1):
        '''
        get nth-order direct neighbors within the home cell only
        '''
        self._fail_if_not_built()

        if n not in self._pairs_by_distance.keys():
            print(f"computing and storing {n}th-nearest neighbors")
            self._build_nth_neighbors(n=n)

        return self._pairs_by_distance[n]

    def get_nth_image_neighbors(self,n=1,twist=None):
        self._fail_if_not_built()

        if n not in self._image_pairs_by_distance.keys():
            print(f"computing and storing {n}th-nearest image neighbors")
            self._build_nth_image_neighbors(n=n,twist=twist)

        return self._image_pairs_by_distance[n]

    def get_sites(self):
        self._fail_if_not_built()
        return self.sites


    def get_positions(self,sublattice_index=None):
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

    def get_kvecs(self,checkOverlap=True):
      '''
      get b1,b2 the Bravais lattice vectors

      checkOVerlap=True to check sign and values of a_i @ b_i
      '''
      self._fail_if_not_built()
      a1,a2 = np.array([*self.a1,0]),np.array([*self.a2,0])
      a3 = np.array([0,0,1]) # temp for formulas
      b1 = np.cross(a2, a3)
      b2 = np.cross(a3, a1)
      #b3 = np.cross(a1, a2)

      vol = np.cross(self.a1,self.a2)
      ratio = 2*np.pi/vol
      rvecs = ratio*np.vstack([b1[:-1],b2[:-1]])

      ovs = self.A@rvecs
      # check sign
      for i in range(2):
        if np.allclose(ovs[i,i],-2*np.pi):
            rvecs[i] *= -1
        elif checkOverlap and not np.isclose(ovs[i,i],2*np.pi):
              raise ValueError(f"Uh oh, b_{i} {rvecs[i]} not a \
                                reciprocal vector to a_{i} {self.A[i]}")

      return rvecs

    def get_planewaves(self, returnKVecs = False):
      '''
      return unitary exp(ik*r) for each site, corresponding
      to the unit cell r. Currently fixed to PBC/PBC only
      '''
      self._fail_if_not_built()
      periodicX = isinstance(self.axis1_boundary,PBCBoundary)
      periodicY = isinstance(self.axis2_boundary,PBCBoundary)
      if not (periodicX and periodicY):
            warn("get_planewaves() is only implemented for PBC/PBC")
      kvecs = self.get_kvecs()
      twist = np.array([self.axis1_boundary.phase,
                       self.axis2_boundary.phase])
      ks = np.array([s.coord[:2]@kvecs  for s in self.sites])
      ks += twist
      # we should order it, but ks isn't always in the 1st bz
      #order = np.argsort(np.linalg.norm(ks,axis=-1))
      #ks = ks[order]
      rs = np.array([self.A@(s.coord[:2]/self.L) for s in self.sites])
      Uxtok = np.exp(1j*(ks@rs.T))/np.sqrt(self.L[0]*self.L[1])
      # we've mixed each basis, so we need to zero out those elements

      if self._nb > 1:
        mask = np.equal.outer([s.coord[-1] for s in self.sites],
                              [s.coord[-1] for s in self.sites])
        Uxtok*=mask
      if returnKVecs: return ks,Uxtok
      return Uxtok

    def build(self):

        # Change unit cell to be 2x and make unit vectors match rectangular
        if self.cyl_mode == "XC" or self.cyl_mode == "YC":
          self.basis = self.basis+[b+self.a2 for b in self.basis]
          self.a1,self.a2 = self.a1,-self.a1+2*self.a2
          if not (np.allclose(self.a1[1],0.) and np.allclose(self.a2[0],0.)):
            warn("Warning, XC/YC mode is not perfectly rectangular. "
                 "\nCurrently hardcoded for hexagonal (triangular) lattices"
                 f"\n(new {self.a1=} new {self.a2=})")
          if self.cyl_mode == "XC":
            self.L = (self.L[0],self.L[1]//2)
          elif self.cyl_mode == "YC":
            self.L = (self.L[0]//2,self.L[1])
            self.L = self.L[::-1]
            self.a1,self.a2 = self.a1[::-1],self.a2[::-1]
            self.basis = [b[::-1] for b in self.basis]

          self._nb = len(self.basis)

        # promised to have a1,a2 by now, so we can
        # do manipulations on them
        self.A = np.array([self.a1,self.a2]).T
        self.Ainv = np.linalg.inv(self.A)

        self._build()
        self._built = True

    def _build(self):
        """
        We may want different cell shapes in the future, for now
            this builds multiples of the unit cell.
        """
        Lx,Ly = self.L
        for lx in range(Lx):
            for ly in range(Ly):
                for n in range(self._nb):
                    self.add_site(np.array((lx,ly,n)))

    def _index_map(self,coord):
        """
        returns basis index given the lattice coordinate, `coord`.
        """
        return coord[0]*self.L[1]*self._nb + coord[1]*self._nb + coord[2]

    def _position(self,coord):
        return coord[0]*self.a1 + coord[1]*self.a2 + self.basis[coord[2]]

    def metric(self,coord1,coord2):
        return self._metric(coord1,coord2)

    def _build_distances(self):
      if self._distances is None:
            poses = np.asarray([s.position for s in self.sites])
            self._distances = self._metric_v(poses[:,None],poses[None,:]) # fast numpy outer product

    def remove_distances(self):
      """
      Removes cached distance matrix.
      Useful if low on memory and the latice is large
      """
      del self._distances
      del self._image_distances
      self._distances = None
      self._image_distances = None
      print("Removed distance matrix")

    def _build_nth_neighbors(self,n):
        if n not in self._pairs_by_distance.keys():
            self._pairs_by_distance[n] = list()
        else:
            raise Warning("Adding pair distances to existing distance list: probably a mistake!")

        try:
            # if we need to build up the distances, initialize it now
            self._neighbor_distance_map(0)
            self._build_distances()
            dist_to_find = self._dist_map[n]

            distances = self._distances

            pairs = np.where(np.isclose(distances,dist_to_find))
            self._pairs_by_distance[n] = [
                NeighborPair(
                    i,j,
                    _abs_r=self.sites[j].position,
                    r_relative=self.sites[j].position-self.sites[i].position
                    ) for i,j in zip(*pairs)
                ]

        except ValueError as e:
            # discard malformed neighbor lists before propagating exception
            self._pairs_by_distance.pop(n)
            raise ValueError(
                f"Could not build nth-neighbors with n={n} "
                f"due to an exception: {e}"
            ) from e

        if len(self.basis)>1: return
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

    # TODO: check the triangular lattice against the next version below!
    def _is_valid_image_old(self,r,valid_axes=None):
        if valid_axes is None:
            valid_axes = self._get_image_axes(r=r)

        # TODO: think through this... this feels either incorrect or over-complicated
        #         I seem to recall that there was an issue with triangular lattices
        #         that this loop seemed to solve - start there!
        for axis in valid_axes:
            #NOTE: we are selecting the *other* axis
            other_boundary =  getattr(self,f"axis{(axis+1)%2 +1}_boundary")
            if isinstance(other_boundary,OpenBoundary) and other_boundary.is_image(r):
                return False

        return True


    def _is_valid_image(self,r):
        # If you invert, be sure to use De Morgan's theorem!!
        if self.axis1_boundary.is_valid(r) and self.axis2_boundary.is_valid(r):
            return True
        else:
            return False

    def _is_allowed_site(self,r):
        if self.axis1_boundary.is_allowed(r) and self.axis2_boundary.is_allowed(r):
            return True
        else:
            return False

    def _is_allowed_site(self,r):
        if self.axis1_boundary.is_allowed(r) and self.axis2_boundary.is_allowed(r):
            return True
        else:
            return False

    def _get_image_axes(self,r):
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
            return [0,1]
        elif is_x_image:
            return [0]
        elif is_y_image:
            return [1]
        else:
            return []


    def is_image(self,r):
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
          periodicX = isinstance(self.axis1_boundary,PBCBoundary)
          periodicY = isinstance(self.axis2_boundary,PBCBoundary)
          xShifts,yShifts = [0],[0]
          vShift1,vShift2 = self.L[0]*self.a1,self.L[1]*self.a2
          if periodicX: xShifts = [-1,0,1]
          if periodicY: yShifts = [-1,0,1,]
          for shift1 in xShifts:
            for shift2 in yShifts:
              #if shift1==shift2: continue
              if shift1==shift2==0: continue
              shiftvec = shift1*vShift1+shift2*vShift2
              distances = self._metric_v(poses[:,None],
                                            poses[None,:]+shiftvec)
              self._image_distances.append((shift1,shift2,shiftvec,distances))

    def _build_nth_image_neighbors(self,n,twist=None):
        """
        building the image nth-order-neighbors from
          the list of direct nth-order-neighbors.

        We use the fact that (nearly all) nth-order image neighbors are
            related to a direct nth-order neighbor by a rotation from the
            lattice's rotation group. All such image neighbors can be generated
            by rotating the vector `r_ij` between each pair of direct n-neighbors
            for all rotations in the lattice's rotation group.

            We reject the case where a site is it's own nth-order neighbor.
        """

        if twist is not None:
            raise NotImplementedError("Need to recompute nth-order neighbors when given a twist")

        if n not in self._image_pairs_by_distance.keys():
            self._image_pairs_by_distance[n] = list()
        else:
            raise Warning("Adding image pair distances to existing"
                          "distance list: probably a mistake!")

        def _check_add_image_neighbors(candidates,pair):
            for r_spatial in candidates:
                r = self._to_lattice_basis(r_spatial)
                if self.is_image(r) and self._is_allowed_site(r):
                    image_axes = self._get_image_axes(r)
                    latt_r = self._wrap_to_cell(r)

                    r_index = round(self._index_map(latt_r))

                    phase = 0.0

                    for axis in image_axes:
                        # for twists, we need +/- depending on which direction we
                        #    cross the boundary in.
                        if r[axis] > 1:
                            sign = -1
                        else:
                            sign = 1
                        phase += sign*getattr(self,f"axis{axis+1}_boundary").phase

                    #TODO: add a unit test that confirms that invalid pairs are rejected
                    image_pair = NeighborPair(pair.i,r_index,phase=phase,_abs_r=r_spatial)
                    self._image_pairs_by_distance[n].append( image_pair )

        self._neighbor_distance_map(0) # init _dist_map
        self._build_image_distances()
        dist_to_find = self._dist_map[n]
        Md = np.zeros((self.N_sites,self.N_sites))
        for shift1,shift2,shiftvec,distances in self._image_distances:
            pairs = np.where(np.isclose(distances,dist_to_find))
            # check if we found this yet, otherwise add
            for p in zip(*pairs):
              if Md[p] == 0:
                pair = NeighborPair(*p)
                r1 = self.sites[pair.i].position
                r2 = self.sites[pair.j].position+shiftvec
                r = self._to_lattice_basis(r2)
                image_axes = self._get_image_axes(r) # TODO: replace with shift1/2
                phase = 0.0

                for axis in image_axes:
                  # for twists, we need +/- depending on which direction we
                  #    cross the boundary in.
                  if r[axis] > 1:
                      sign = -1
                  else:
                      sign = 1
                  phase += sign*getattr(self,f"axis{axis+1}_boundary").phase

                #TODO: add a unit test that confirms that invalid pairs are rejected
                image_pair = NeighborPair(pair.i,pair.j,phase=phase,
                    _abs_r=r2,_shift=(shift1,shift2),r_relative=r2-r1)
                self._image_pairs_by_distance[n].append( image_pair )

                #_check_add_image_neighbors(
                #    candidates=[r2],
                #    pair=pair
                #)

            Md[pairs] = 1


    def _neighbor_distance_map(self,distance):
        """
        compute neighbor distance map using scipy.spatial.distance_matrix()
        """

        def _positions_from_cell():
            print("Reading lattice site positions from Lattice")
            return np.array([ site.position for site in self.sites ])

        # 1. get all positions in an array
        if self._dist_map is None:

            print("Computing distance matrix of lattice")
            positions = _positions_from_cell()

            # 2. get distance matrix
            R = spatial.distance_matrix(
                positions,
                positions
            )

            # 3. find set of distances
            dist_set = set( np.round(R.flatten(),8) )

            # 4. make into a sorted list
            self._dist_map = list(dist_set)
            self._dist_map.sort()

        spot = np.where(np.isclose(self._dist_map,np.round(distance,8)))[0]
        if len(spot)!=1:
          raise ValueError(f"{distance=} has no/multiple valid indices!")
        return spot[0]

    def _rotations(self,v,c=None):
        """
        Returns a list of position vectors,
            `v' = R(theta) @ v + c`
        based on the input vector `v` with
        rotations from the symmetry group of the
        lattice applied. A constant vector c
        can be optionally addded to the result.
        """
        if c is None:
            return [ R@v for R in self.ROTATION_GROUP ]
        else:
            return [ R@v + c for R in self.ROTATION_GROUP ]

    def _to_lattice_basis(self,r):
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
        return np.array((*candidates[basis_idx].round().astype(int),basis_idx))

    def _wrap_to_cell(self,r):
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

def _2d_rotation(theta):
    return np.array(
        [
            [np.cos(theta),-np.sin(theta)],
            [np.sin(theta),np.cos(theta)]
        ]
    )


class SquareLattice(Lattice):
    """
    Specialization of Lattice to a square lattice.

    Here, we define a square lattice as a lattice in
      which sites are equidistant in the x-, and y-directions.
    """

    ROTATION_ANGLES = ( np.pi/2, np.pi, 3*np.pi/2, 2*np.pi )
    ROTATION_GROUP = tuple(
        [ _2d_rotation(theta) for theta in ROTATION_ANGLES ]
    )

    def __init__(self, L=None, build=True, *args, **kwargs) -> None:

        self._type = "square" # for reference only

        self.a1 = np.array([1.,0.])
        self.a2 = np.array([0.,1.])

        super().__init__(L,*args,build=build,**kwargs)



    def get_directed_pairs(self,directions=None):
        """
        Get directed pairs - i.e. for making pair correlators

        for now, we only have square lattice pair 'directions' implemented
           i.e. +/-x, +/-y. will need to work out the Triangular lattice
           possibilities.
        """
        return get_directed_pairs(self,directions=directions)


class TriangularLattice(Lattice):
    """
    Specialization of Lattice to a triangular lattice.
    """

    pi = np.pi

    ROTATION_ANGLES = ((1/3)*pi,(2/3)*pi,pi,(4/3)*pi,(5/3)*pi,2*pi)
    ROTATION_GROUP = tuple(
        [ _2d_rotation(theta) for theta in ROTATION_ANGLES ]
    )

    def __init__(
        self,
        L=None,
        build=True,
        metric=euclid_nd,
        axis1_boundary=None,
        axis2_boundary=None,
        **kwargs
    ) -> None:

        self._type = "triangular" # for reference only

        self.a1 = np.array([1.,0.])
        self.a2 = np.array([0.5,np.sqrt(3)/2])


        if kwargs.get('basis'):
            # BUG: if the magnitude of the basis vectors is too large,
            #       there are errors with computing direct neighbors and 
            #       image neighbors. Try limiting basis vectors to the unit cell
            warn(
                "Triangular lattice does not fully support basis; proceed with caution! "
                " 'honeycomb' and 'kagome' lattices are implemented separately"
            )

        super().__init__(
            L=L,
            metric=metric,
            axis1_boundary=axis1_boundary,
            axis2_boundary=axis2_boundary,
            build=build,
            **kwargs
        )

class HoneycombLattice(Lattice):
    """
    Specialization of Lattice to a Honeycomb lattice.
    The underlying lattice is a triangular lattice with an "extra"
    site per cell
    """

    pi = np.pi

    #ROTATION_ANGLES = ((1/3)*pi,(2/3)*pi,pi,(4/3)*pi,(5/3)*pi,2*pi)
    ROTATION_ANGLES = ((2/3)*pi,(4/3)*pi,2*pi)
    ROTATION_GROUP = tuple(
        [ _2d_rotation(theta) for theta in ROTATION_ANGLES ]
    )

    def __init__(
        self,
        L=None,
        build=True,
        metric=euclid_nd,
        axis1_boundary=None,
        axis2_boundary=None,
        **kwargs
    ) -> None:

        self._type = "honeycomb" # for reference only
        # this is closer to the original triangular
        #self.a1 = np.array([1.,0.])
        #self.a2 = np.array([0.5,np.sqrt(3)/2])
        #self.basis = [np.zeros(2), np.array([0,1/np.sqrt(3)])]
        # more square like cell
        # based on netket choices
        self.a1 = np.array([1,0])
        self.a2 = np.array([1/2,0.75**0.5])
        self.basis = [np.array([0.5, 0.5 / 3**0.5]),
                      np.array([1, 1 / 3**0.5])]

        super().__init__(
            L=L,
            metric=metric,
            axis1_boundary=axis1_boundary,
            axis2_boundary=axis2_boundary,
            build=build,
            **kwargs
        )

class KagomeLattice(Lattice):
    """
    Specialization of Lattice to a Kagome lattice.
    The underlying lattice is a triangular lattice with 3
    site per cell
    """

    pi = np.pi

    ROTATION_ANGLES = (2*pi,) # no longer needed
    ROTATION_GROUP = tuple(
        [ _2d_rotation(theta) for theta in ROTATION_ANGLES ]
    )

    def __init__(
        self,
        L=None,
        build=True,
        metric=euclid_nd,
        axis1_boundary=None,
        axis2_boundary=None,
        **kwargs
    ) -> None:
        self._type = "kagome" # for reference only

        # this is closer to the original triangular
        #self.a1 = np.array([1.,0.])
        #self.a2 = np.array([0.5,np.sqrt(3)/2])
        # more square like cell
        # based on netket choices
        self.a1 = np.array([1,0])
        self.a2 = np.array([1/2, np.sqrt(0.75)])
        self.basis = [np.array([0.5, 0]),
                      np.array([0.25, np.sqrt(0.75) / 2]),
                      np.array([0.75, np.sqrt(0.75) / 2])]

        super().__init__(
            L=L,
            metric=metric,
            axis1_boundary=axis1_boundary,
            axis2_boundary=axis2_boundary,
            build=build,
            **kwargs
        )

def get_lattice(params:dict, build=None):
    """
    get a lattice instance from the input parameters
        in 'params'
    """
    def _get_boundary(boundary_type:str):
        """
        Get Boundary class from string 'boundary_type.'

        raises a ValueError for an unknown boundary type.
        """
        if boundary_type.lower() in {'open'}:
            return OpenBoundary
        elif boundary_type.lower() in {'pbc','periodic'}:
            return PBCBoundary
        else:
            raise ValueError("Uknown boundary type")

    def _get_lattice(lattice_type):
        """
        Get Lattice class from string 'lattice_type.'

        raises a ValueError for an unknown boundary type.
        """
        if lattice_type == 'square':
            return SquareLattice
        elif lattice_type == 'triangular':
            return TriangularLattice
        elif lattice_type == 'honeycomb':
            return HoneycombLattice
        elif lattice_type == 'kagome':
            return KagomeLattice
        elif lattice_type == 'custom':
            return CustomLattice
        else:
            raise ValueError("Unknown lattice type: supported types are 'square',"
                             "'traingular', 'honeycomb'")

    if 'type' in params.keys():
        lattice = _get_lattice(params['type'].lower())
    else:
        lattice = SquareLattice

    L = (
        params['L1'],
        params['L2']
    )

    if L[0] < 1 or L[1] < 1:
        raise ValueError("Invalid lattice size; L1 and L2 must be >= 1")

    if 'boundary1' in params.keys():
        Boundary1 = _get_boundary(params['boundary1'])
    else:
        Boundary1 = None

    if 'boundary2' in params.keys():
        Boundary2 = _get_boundary(params['boundary2'])
    else:
        Boundary2 = None

    if build is None:
        build = params.get("build",True)

    return lattice(
        L=L,
        axis1_boundary=Boundary1,
        axis2_boundary=Boundary2,
        build=build,
        twist=params.get("twist",None),
        basis=params.get("basis",None),
        a1 = params.get("a1",None),
        a2 = params.get("a2",None),
        cyl_mode = params.get("cyl_mode",None)
    )

class CustomLattice(Lattice):
    """
    custom lattice implementation that allows the 
    user to specify the lattice vectors and basis.
    """

    def __init__(self,
        L,
        a1,
        a2,
        basis,
        axis1_boundary=None,
        axis2_boundary=None,
        **kwargs):
            """
            custom lattice implementation that allows the 
            user to specify the lattice vectors and basis.

            Parameters
            ----------
            L : tuple
                length of the lattice in each direction
            a1 : np.ndarray
                first lattice vector
            a2 : np.ndarray
                second lattice vector
            basis : list
                list of basis vectors
            axis1_boundary : Boundary, optional
                boundary condition for the first axis. Options are "open", and "pbc". Default is "open".  
            axis2_boundary : Boundary, optional
                boundary condition for the second axis. Options are "open", and "pbc". Default is "open".
            **kwargs : dict
                additional keyword arguments to pass to the Lattice base class. See Lattice for more details.

            Examples
            --------
            >>> L = (2,2)
            >>> a1 = np.array([1,1])
            >>> a2 = np.array([1,-1])
            >>> basis = [np.array([0,0]),np.array([0,0.5])]
            >>> lattice = CustomLattice(L=L,a1=a1,a2=a2,basis=basis)

            Notes
            -----
            Currently only supports 2D lattices. Higher dimensions will be added in the future.
            Please contact the developers if you need this feature.
            """
            self.a1 = a1
            self.a2 = a2
            self.basis = basis
            super().__init__(L=L,
                axis1_boundary=axis1_boundary,
                axis2_boundary=axis2_boundary,
                **kwargs
        )

def get_directed_pairs(lattice:Lattice,directions=None):

    _directed_pairs = {}
    for direction in directions:
        _directed_pairs[direction] = []

    for site in lattice.sites:
        for direction in directions:
            coord = site.coord.copy()
            if direction in ('0','s',''):
                pass # since we catch unknown directions below!
            elif direction == "+x":
                coord[0] += 1
            elif direction == "-x":
                coord[0] -= 1
            elif direction == "+y":
                coord[1] += 1
            elif direction == "-y":
                coord[1] -= 1
            elif direction in ("+x+y","+y+x"):
                coord[0] += 1
                coord[1] += 1
            elif direction in ("+x-y","-y+x"):
                coord[0] += 1
                coord[1] -= 1
            elif direction in ("-x+y","+y-x"):
                coord[0] -= 1
                coord[1] += 1
            elif direction in ("-x-y","-y-x"):
                coord[0] -= 1
                coord[1] -= 1
            else:
                raise ValueError("Uknown 'direction' in 'directions'")

            # For now, we assume that both boundaries are periodic
            if not lattice.is_image(coord):
                _directed_pairs[direction].append(lattice._index_map(coord))
            elif lattice.is_image(coord):
                coord[0] = coord[0] % lattice.L[0]
                coord[1] = coord[1] % lattice.L[1]
                _directed_pairs[direction].append(lattice._index_map(coord))
            else:
                # We encode the case where the offset site is not valid with a "-1"
                _directed_pairs[direction].append(-1)

    return _directed_pairs
