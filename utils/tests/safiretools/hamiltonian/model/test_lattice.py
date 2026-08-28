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
import pytest

from safiretools.hamiltonian.model.lattice import (
    CustomLattice,
    HoneycombLattice,
    KagomeLattice,
    Lattice,
    OpenBoundary,
    PBCBoundary,
    SquareLattice,
    TriangularLattice,
)

LATTICE_CLASSES = {
    'square': SquareLattice,
    'triangular': TriangularLattice,
    'honeycomb': HoneycombLattice,
    'kagome': KagomeLattice,
}

# Known-good geometry, recorded from afqmctools.systems.lattice on a 3x3
# PBC/PBC cell before the dead rotation-group code was removed. `degree` is the
# number of nearest neighbors per site; `n_direct`/`n_image` split those pairs
# into home-cell and image pairs, which is what exercises the image-neighbor
# path specifically.
REFERENCE_3x3_PBC = {
    'square': dict(
        n_sites=9, nb=1, degree=4, n_direct=24, n_image=12,
        dist_map=[0.0, 1.0, 1.414214, 2.0],
    ),
    'triangular': dict(
        n_sites=9, nb=1, degree=6, n_direct=32, n_image=22,
        dist_map=[0.0, 1.0, 1.732051, 2.0],
    ),
    'honeycomb': dict(
        n_sites=18, nb=2, degree=3, n_direct=42, n_image=12,
        dist_map=[0.0, 0.57735, 1.0, 1.154701],
    ),
    'kagome': dict(
        n_sites=27, nb=3, degree=4, n_direct=86, n_image=22,
        dist_map=[0.0, 0.5, 0.866025, 1.0],
    ),
}


def _params(lattice_type, **overrides):
    params = {
        'type': lattice_type,
        'L1': 3,
        'L2': 3,
        'boundary1': 'PBC',
        'boundary2': 'PBC',
    }
    params.update(overrides)
    return params


class TestFromDict:
    """`Lattice.from_dict` replaces the free function `get_lattice`."""

    @pytest.mark.parametrize("lattice_type,expected", sorted(LATTICE_CLASSES.items()))
    def test_dispatches_on_type(self, lattice_type, expected):
        assert isinstance(Lattice.from_dict(_params(lattice_type)), expected)

    def test_type_is_case_insensitive(self):
        assert isinstance(Lattice.from_dict(_params('Kagome')), KagomeLattice)

    def test_defaults_to_square(self):
        params = _params('square')
        del params['type']
        assert isinstance(Lattice.from_dict(params), SquareLattice)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown lattice type"):
            Lattice.from_dict(_params('hexatic'))

    def test_unknown_boundary_raises(self):
        with pytest.raises(ValueError, match="Unknown boundary type"):
            Lattice.from_dict(_params('square', boundary1='reflecting'))

    @pytest.mark.parametrize("boundary,expected", [
        ('PBC', PBCBoundary),
        ('periodic', PBCBoundary),
        ('open', OpenBoundary),
    ])
    def test_boundary_strings(self, boundary, expected):
        lattice = Lattice.from_dict(_params('square', boundary1=boundary))
        assert isinstance(lattice.axis1_boundary, expected)

    def test_missing_boundary_defaults_to_open(self):
        params = _params('square')
        del params['boundary1']
        del params['boundary2']
        lattice = Lattice.from_dict(params)
        assert isinstance(lattice.axis1_boundary, OpenBoundary)
        assert isinstance(lattice.axis2_boundary, OpenBoundary)

    @pytest.mark.parametrize("L1,L2", [(0, 3), (3, 0), (-1, 1)])
    def test_invalid_size_raises(self, L1, L2):
        with pytest.raises(ValueError, match="Invalid lattice size"):
            Lattice.from_dict(_params('square', L1=L1, L2=L2))

    def test_build_argument_overrides_params(self):
        lattice = Lattice.from_dict(_params('square', build=True), build=False)
        assert lattice._built is False


class TestGeometryBelongsToTheType:
    """The unit-cell geometry is a property of the lattice type: `a1`/`a2`/
    `basis` always exist, but only `CustomLattice` lets a caller set them.

    These are regression tests for the old behavior, where `a1`/`a2` from the
    parameter dict were absorbed into `**kwargs` and silently discarded for
    every built-in lattice type.
    """

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    @pytest.mark.parametrize("key,value", [
        ('a1', [2.0, 0.0]),
        ('a2', [0.0, 3.0]),
        ('basis', [[0.0, 0.0], [0.25, 0.25]]),
    ])
    def test_from_dict_rejects_geometry_keys(self, lattice_type, key, value):
        params = _params(lattice_type, **{key: value})

        with pytest.raises(ValueError, match="defines its own unit-cell geometry"):
            Lattice.from_dict(params)

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    def test_from_dict_tolerates_geometry_keys_set_to_none(self, lattice_type):
        # parameter templates carrying unused keys must still work
        params = _params(lattice_type, a1=None, a2=None, basis=None)

        assert isinstance(Lattice.from_dict(params), LATTICE_CLASSES[lattice_type])

    @pytest.mark.parametrize("lattice_cls", sorted(LATTICE_CLASSES.values(), key=str))
    @pytest.mark.parametrize("key", ['a1', 'a2', 'basis'])
    def test_constructors_reject_geometry_arguments(self, lattice_cls, key):
        with pytest.raises(TypeError):
            lattice_cls(L=(3, 3), **{key: [[1.0, 1.0]]})

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    def test_geometry_comes_from_the_type(self, lattice_type):
        lattice = Lattice.from_dict(_params(lattice_type))
        a1, a2, basis = lattice._geometry()

        np.testing.assert_allclose(lattice.a1, a1)
        np.testing.assert_allclose(lattice.a2, a2)
        np.testing.assert_allclose(lattice.basis, basis if basis is not None else [[0.0, 0.0]])

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    def test_geometry_attributes_always_exist(self, lattice_type):
        lattice = Lattice.from_dict(_params(lattice_type))

        assert lattice.a1.shape == (2,)
        assert lattice.a2.shape == (2,)
        assert len(lattice.basis) == lattice.nb >= 1

    def test_unknown_keyword_is_not_swallowed(self):
        # the discarded-a1/a2 bug was caused by **kwargs absorbing them; no
        # constructor keyword may be silently ignored any more.
        with pytest.raises(TypeError):
            SquareLattice(L=(3, 3), a3=[1.0, 1.0])

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    @pytest.mark.parametrize("attr", ['a1', 'a2', 'basis'])
    def test_geometry_attributes_are_read_only(self, lattice_type, attr):
        lattice = Lattice.from_dict(_params(lattice_type))

        with pytest.raises(AttributeError):
            setattr(lattice, attr, [[9.0, 9.0]])

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_CLASSES))
    def test_geometry_arrays_cannot_be_edited_in_place(self, lattice_type):
        lattice = Lattice.from_dict(_params(lattice_type))

        with pytest.raises(ValueError):
            lattice.a1[0] = 9.0
        with pytest.raises(ValueError):
            lattice.a2[0] = 9.0
        with pytest.raises(ValueError):
            lattice.basis[0][0] = 9.0

    def test_basis_container_cannot_be_extended(self):
        lattice = Lattice.from_dict(_params('honeycomb'))

        assert not hasattr(lattice.basis, 'append')
        assert lattice.nb == 2

    def test_custom_geometry_is_copied_from_the_caller(self):
        a1 = np.array([2.0, 0.0])
        lattice = CustomLattice(L=(2, 2), a1=a1, a2=[0.0, 2.0])

        # freezing the lattice's copy must not reach the caller's array
        a1[0] = 5.0

        assert a1.flags.writeable
        np.testing.assert_allclose(lattice.a1, [2.0, 0.0])

    def test_building_twice_is_rejected(self):
        lattice = Lattice.from_dict(_params('square', build=False))
        lattice.build()

        with pytest.raises(RuntimeError, match="already built"):
            lattice.build()

    @pytest.mark.parametrize("lattice_type",
                             [t for t in sorted(LATTICE_CLASSES) if t != 'triangular'])
    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_cyl_mode_is_rejected_for_non_triangular_types(self, lattice_type, cyl_mode):
        # the XC/YC reshaping is hardcoded for hexagonal geometry
        with pytest.raises(ValueError, match="hardcoded for hexagonal"):
            Lattice.from_dict(_params(lattice_type, cyl_mode=cyl_mode))

    def test_cyl_mode_is_rejected_for_custom_lattices(self):
        with pytest.raises(ValueError, match="hardcoded for hexagonal"):
            CustomLattice(L=(4, 4), a1=[1.0, 0.0], a2=[0.5, 3 ** 0.5 / 2], cyl_mode='XC')

    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_cyl_mode_is_accepted_for_triangular(self, cyl_mode):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice = Lattice.from_dict(_params('triangular', L1=4, L2=4, cyl_mode=cyl_mode))

        assert lattice.cyl_mode == cyl_mode
        assert lattice.nb == 2

    def test_unsupported_cyl_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported cylinder mode"):
            Lattice.from_dict(_params('triangular', cyl_mode='ZC'))

    def test_cyl_mode_geometry_is_applied_during_build(self):
        params = _params('triangular', L1=4, L2=4, cyl_mode='XC', build=False)
        lattice = Lattice.from_dict(params)
        assert lattice.nb == 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice.build()

        # cyl_mode doubles the basis; nb/num_sublattice must follow it
        assert lattice.nb == 2
        assert lattice.num_sublattice == 2
        assert not lattice.a1.flags.writeable
        assert all(not b.flags.writeable for b in lattice.basis)

    def test_a_type_without_lattice_vectors_is_rejected(self):
        class GeometrylessLattice(Lattice):
            _type = "geometryless"

            def _geometry(self):
                return None, None, None

        with pytest.raises(ValueError, match="returned no lattice vectors"):
            GeometrylessLattice(L=(2, 2))

    def test_unknown_keyword_is_not_swallowed(self):
        # the bug fixed above was caused by **kwargs absorbing a1/a2; unknown
        # keywords must now be loud rather than silently ignored.
        with pytest.raises(TypeError):
            SquareLattice(L=(3, 3), a3=[1.0, 1.0])


class TestGeometryRegression:
    """Guards the dead rotation-group / `_is_valid_image_old` removal against
    any change in live geometry or neighbor maps.
    """

    @pytest.fixture(scope='class', params=sorted(REFERENCE_3x3_PBC))
    def case(self, request):
        lattice_type = request.param
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice = Lattice.from_dict(_params(lattice_type))
            direct = lattice.get_nth_direct_neighbors(1)
            image = lattice.get_nth_image_neighbors(1)
        return lattice, REFERENCE_3x3_PBC[lattice_type], direct, image

    def test_site_count(self, case):
        lattice, ref, _, _ = case
        assert lattice.N_sites == ref['n_sites']
        assert lattice.nb == ref['nb']
        assert lattice.num_sublattice == ref['nb']

    def test_distance_map(self, case):
        lattice, ref, _, _ = case
        np.testing.assert_allclose(lattice._dist_map[:4], ref['dist_map'], atol=1e-6)

    def test_nearest_neighbor_counts(self, case):
        lattice, ref, direct, image = case
        assert len(direct) == ref['n_direct']
        assert len(image) == ref['n_image']
        assert len(direct) + len(image) == ref['n_sites'] * ref['degree']

    def test_every_site_has_uniform_degree(self, case):
        lattice, ref, direct, image = case
        degrees = np.zeros(ref['n_sites'], dtype=int)
        for pair in direct + image:
            degrees[pair.i] += 1
        assert np.all(degrees == ref['degree'])

    def test_neighbor_set_is_symmetric(self, case):
        _, _, direct, image = case
        pairs = {(p.i, p.j) for p in direct + image}
        assert pairs == {(j, i) for i, j in pairs}

    def test_no_site_is_its_own_neighbor(self, case):
        _, _, direct, image = case
        assert all(p.i != p.j for p in direct + image)

    def test_untwisted_pairs_carry_no_phase(self, case):
        _, _, direct, image = case
        assert all(p.phase == 0.0 for p in direct + image)


def test_square_nearest_neighbors_match_independent_model():
    """Cross-check the 3x3 PBC square neighbor map against the offsets written
    out by hand, rather than against a golden list from the old code.
    """
    L = 3
    lattice = Lattice.from_dict(_params('square', L1=L, L2=L))
    pairs = lattice.get_nth_neighbors(1)

    expected = {
        (x * L + y, ((x + dx) % L) * L + (y + dy) % L)
        for x in range(L)
        for y in range(L)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    }
    assert {(p.i, p.j) for p in pairs} == expected


def test_twist_puts_a_phase_on_image_pairs_only():
    twisted = Lattice.from_dict(_params('square', twist=('1/2 pi', 0.0)))

    direct = twisted.get_nth_direct_neighbors(1)
    image = twisted.get_nth_image_neighbors(1)

    assert all(p.phase == 0.0 for p in direct)
    phases = {round(p.phase, 12) for p in image}
    assert phases == {0.0, round(np.pi / 2, 12), round(-np.pi / 2, 12)}


class TestMinDistance:
    """`_dist_map_min_distance` used to raise TypeError, since
    `_neighbor_distance_map` had no `min_distance` parameter.
    """

    def test_short_separations_are_dropped_from_the_map(self):
        # the honeycomb intra-cell separation, 1/sqrt(3), is the n=1 shell by default
        default = Lattice.from_dict(_params('honeycomb'))
        default._neighbor_distance_map(0.0)
        assert np.isclose(default._dist_map[1], 1 / np.sqrt(3))

        filtered = HoneycombLattice(L=(3, 3), axis1_boundary=PBCBoundary,
                                    axis2_boundary=PBCBoundary,
                                    _dist_map_min_distance=0.9)
        filtered._neighbor_distance_map(0.0)

        assert np.isclose(filtered._dist_map[1], 1.0)
        assert not any(0.0 < d < 0.9 for d in filtered._dist_map)

    def test_zero_self_distance_is_kept_at_index_zero(self):
        lattice = HoneycombLattice(L=(3, 3), _dist_map_min_distance=0.9)

        assert lattice._neighbor_distance_map(0.0) == 0
        assert lattice._dist_map[0] == 0.0

    def test_method_argument_takes_precedence(self):
        lattice = HoneycombLattice(L=(3, 3))

        lattice._neighbor_distance_map(0.0, min_distance=0.9)

        assert np.isclose(lattice._dist_map[1], 1.0)


class TestAbstractBaseClass:

    def test_lattice_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Lattice(L=(2, 2))

    @pytest.mark.parametrize("lattice_cls", sorted(LATTICE_CLASSES.values(), key=str))
    def test_concrete_subclasses_declare_a_type(self, lattice_cls):
        assert isinstance(lattice_cls._type, str)


class TestCustomLattice:
    """`CustomLattice` is the one type whose geometry the caller defines."""

    @pytest.mark.parametrize("kwargs", [
        {},
        {'a1': [1.0, 0.0]},
        {'a2': [0.0, 1.0]},
    ])
    def test_requires_both_lattice_vectors(self, kwargs):
        with pytest.raises(TypeError):
            CustomLattice(L=(2, 2), **kwargs)

    def test_basis_is_optional(self):
        lattice = CustomLattice(L=(2, 2), a1=[1.0, 0.0], a2=[0.0, 1.0])

        assert lattice.nb == 1
        np.testing.assert_allclose(lattice.basis, [[0.0, 0.0]])

    def test_from_dict_builds_a_custom_lattice(self):
        params = _params('custom', a1=[1.0, 1.0], a2=[1.0, -1.0],
                         basis=[[0.0, 0.0], [0.0, 0.5]])

        lattice = Lattice.from_dict(params)

        assert isinstance(lattice, CustomLattice)
        np.testing.assert_allclose(lattice.a1, [1.0, 1.0])
        np.testing.assert_allclose(lattice.a2, [1.0, -1.0])
        assert lattice.nb == 2
        assert lattice.N_sites == 9 * 2

    def test_geometry_reaches_the_site_positions(self):
        a1, a2 = [2.0, 0.0], [0.0, 3.0]
        basis = [[0.0, 0.0], [0.5, 0.5]]

        lattice = CustomLattice(L=(3, 3), a1=a1, a2=a2, basis=basis,
                                axis1_boundary=PBCBoundary,
                                axis2_boundary=PBCBoundary)

        np.testing.assert_allclose(lattice.A, np.array([a1, a2]).T)
        for basis_index in range(2):
            site = next(s for s in lattice.get_sites()
                        if tuple(s.coord) == (1, 2, basis_index))
            expected = np.array(a1) + 2 * np.array(a2) + np.array(basis[basis_index])
            np.testing.assert_allclose(site.position, expected)


class TestUnbuilt:

    @pytest.fixture
    def unbuilt(self):
        return Lattice.from_dict(_params('square', build=False))

    def test_not_built(self, unbuilt):
        assert unbuilt._built is False
        assert unbuilt.sites == []

    @pytest.mark.parametrize("method", [
        'get_sites',
        'get_nth_neighbors',
        'get_nth_direct_neighbors',
        'get_nth_image_neighbors',
        'get_kvecs',
    ])
    def test_accessors_raise_before_build(self, unbuilt, method):
        with pytest.raises(RuntimeError, match="Must build lattice instance"):
            getattr(unbuilt, method)()

    def test_build_populates_sites(self, unbuilt):
        unbuilt.build()

        assert unbuilt._built is True
        assert unbuilt.N_sites == 9


def test_triangular_rejects_a_basis():
    # the old implementation accepted a basis here and warned that it did not
    # really support one; honeycomb/kagome (or CustomLattice) are the answer.
    with pytest.raises(TypeError):
        TriangularLattice(L=(2, 2), basis=[[0.0, 0.0], [0.5, 0.0]])


def test_square_directed_pairs():
    lattice = Lattice.from_dict(_params('square', L1=3, L2=3))

    pairs = lattice.get_directed_pairs(directions=['+x', '-y', '0'])

    # site index = x*L2 + y for a single-site basis
    assert pairs['+x'] == [((i // 3 + 1) % 3) * 3 + i % 3 for i in range(9)]
    assert pairs['-y'] == [(i // 3) * 3 + (i % 3 - 1) % 3 for i in range(9)]
    assert pairs['0'] == list(range(9))

    with pytest.raises(ValueError, match="Unknown 'direction'"):
        lattice.get_directed_pairs(directions=['+z'])
