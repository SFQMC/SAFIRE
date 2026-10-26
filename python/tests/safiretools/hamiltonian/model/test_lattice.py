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
    LATTICE_TYPES,
    Lattice,
    OpenBoundary,
    PBCBoundary,
    get_directed_pairs,
)

# Known-good geometry, recorded from afqmctools.systems.lattice on a 3x3
# periodic/periodic cell before the dead rotation-group code was removed. `degree` is the
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


def _periodic(lattice_type, Ls=(3, 3), twists=(0.0, 0.0), **kwargs):
    return Lattice.from_type(lattice_type, Ls, boundaries=('periodic', 'periodic'),
                             twists=twists, **kwargs)


class TestFromType:
    """`Lattice.from_type` builds a lattice from the unit cell of a built-in type."""

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_TYPES))
    def test_matches_the_constructor(self, lattice_type):
        lattice = _periodic(lattice_type)
        lattice_vectors, basis = LATTICE_TYPES[lattice_type]
        expected = Lattice(lattice_vectors, basis, (3, 3),
                           boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))

        np.testing.assert_allclose(lattice.A, expected.A)
        np.testing.assert_allclose(lattice.basis, expected.basis)
        np.testing.assert_allclose(lattice.get_positions(), expected.get_positions())

    def test_type_is_case_insensitive(self):
        lattice = _periodic('Kagome')

        assert lattice.nb == 3
        np.testing.assert_allclose(lattice.basis, LATTICE_TYPES['kagome'][1])

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown lattice type"):
            _periodic('hexatic')

    def test_unknown_boundary_raises(self):
        with pytest.raises(ValueError, match="Unknown boundary type"):
            Lattice.from_type('square', (3, 3), boundaries=('reflecting', 'periodic'),
                              twists=(0.0, 0.0))

    @pytest.mark.parametrize("boundary,expected", [
        ('periodic', PBCBoundary),
        ('PERIODIC', PBCBoundary),
        ('pbc', PBCBoundary),
        ('open', OpenBoundary),
    ])
    def test_boundary_strings(self, boundary, expected):
        lattice = Lattice.from_type('square', (3, 3), boundaries=(boundary, 'periodic'),
                                    twists=(0.0, 0.0))
        assert isinstance(lattice.axis1_boundary, expected)

    def test_boundaries_are_required(self):
        with pytest.raises(TypeError, match='boundaries'):
            Lattice.from_type('square', (3, 3), twists=(0.0, 0.0))
        with pytest.raises(TypeError, match='boundaries'):
            Lattice(*LATTICE_TYPES['square'], (3, 3), twists=(0.0, 0.0))

    def test_twists_default_to_zero(self):
        lattice = Lattice.from_type('square', (3, 3), boundaries=('periodic', 'periodic'))
        assert lattice.axis1_boundary.phase == 0.0
        assert lattice.axis2_boundary.phase == 0.0

    @pytest.mark.parametrize("Ls", [(0, 3), (3, 0), (-1, 1)])
    def test_invalid_size_raises(self, Ls):
        with pytest.raises(ValueError, match="Invalid lattice size"):
            _periodic('square', Ls=Ls)

    def test_build_is_forwarded(self):
        lattice = _periodic('square', build=False)
        assert lattice._built is False

    @pytest.mark.parametrize("key,value", [
        ('a1', [2.0, 0.0]),
        ('a2', [0.0, 3.0]),
        ('basis', [[0.0, 0.0], [0.25, 0.25]]),
        ('lattice_vectors', [[2.0, 0.0], [0.0, 3.0]]),
        ('a3', [1.0, 1.0]),
    ])
    def test_rejects_unit_cell_arguments(self, key, value):
        # the unit cell comes from the type; a keyword trying to override it
        #   must fail rather than be silently ignored
        with pytest.raises(TypeError):
            _periodic('square', **{key: value})


class TestConstructor:
    """The `Lattice` constructor takes any unit cell."""

    @pytest.mark.parametrize("kwargs,match", [
        (dict(lattice_vectors=[[1.0, 0.0]]), "lattice_vectors must have shape"),
        (dict(lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
         "lattice_vectors must have shape"),
        (dict(basis=[]), "basis must be a non-empty list"),
        (dict(basis=[[0.0, 0.0, 0.0]]), "basis must be a non-empty list"),
        (dict(Ls=(3,)), "Ls must be a 2-d"),
        (dict(Ls=(3, 3, 3)), "Ls must be a 2-d"),
        (dict(boundaries=('periodic',)), "boundaries must have length 2"),
        (dict(twists=(0.0,)), "'twists'"),
    ])
    def test_invalid_arguments_raise(self, kwargs, match):
        params = dict(lattice_vectors=[[1.0, 0.0], [0.0, 1.0]], basis=[[0.0, 0.0]],
                      Ls=(3, 3), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))
        params.update(kwargs)

        with pytest.raises(ValueError, match=match):
            Lattice(**params)

    def test_builds_a_non_orthogonal_cell(self):
        lattice = Lattice(**dict(
            lattice_vectors=[[1.0, 1.0], [1.0, -1.0]], basis=[[0.0, 0.0], [0.0, 0.5]],
            Ls=(3, 3), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0),
        ))

        np.testing.assert_allclose(lattice.A, [[1.0, 1.0], [1.0, -1.0]])
        assert lattice.nb == 2
        assert lattice.N_sites == 9 * 2

    def test_size_is_a_tuple_of_ints(self):
        lattice = _periodic('square', Ls=np.array([3, 2]))

        assert lattice.L == (3, 2)
        assert all(type(size) is int for size in lattice.L)

    def test_unitcell_reaches_the_site_positions(self):
        a1, a2 = [2.0, 0.0], [0.0, 3.0]
        basis = [[0.0, 0.0], [0.5, 0.5]]

        lattice = Lattice([a1, a2], basis, (3, 3), boundaries=('periodic', 'periodic'),
                          twists=(0.0, 0.0))

        np.testing.assert_allclose(lattice.A, np.array([a1, a2]).T)
        for basis_index in range(2):
            site = next(s for s in lattice.get_sites()
                        if tuple(s.coord) == (1, 2, basis_index))
            # the basis is fractional: site b of cell (1, 2) sits at A @ ((1, 2) + basis[b])
            expected = ((1 + basis[basis_index][0]) * np.array(a1)
                        + (2 + basis[basis_index][1]) * np.array(a2))
            np.testing.assert_allclose(site.position, expected)

    def test_unitcell_is_copied_from_the_caller(self):
        a1 = np.array([2.0, 0.0])
        basis = np.array([[0.0, 0.0]])
        lattice = Lattice([a1, [0.0, 2.0]], basis, (2, 2),
                          boundaries=('open', 'open'), twists=(0.0, 0.0))

        # freezing the lattice's copy must not reach the caller's array
        a1[0] = 5.0
        basis[0, 0] = 5.0

        assert a1.flags.writeable
        assert basis.flags.writeable
        np.testing.assert_allclose(lattice.A[:, 0], [2.0, 0.0])
        np.testing.assert_allclose(lattice.basis, [[0.0, 0.0]])


class TestCylMode:
    """`cyl_mode` reshapes the triangular unit cell before construction."""

    @pytest.mark.parametrize("lattice_type",
                             [t for t in sorted(LATTICE_TYPES) if t != 'triangular'])
    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_rejected_for_non_triangular_types(self, lattice_type, cyl_mode):
        with pytest.raises(ValueError, match="only available for 'triangular'"):
            _periodic(lattice_type, cyl_mode=cyl_mode)

    def test_unsupported_cyl_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported cylinder mode"):
            _periodic('triangular', cyl_mode='ZC')

    @pytest.mark.parametrize("cyl_mode,L,a1,a2,basis", [
        ('XC', (4, 3), [1.0, 0.0], [0.0, 3 ** 0.5], [[0.0, 0.0], [0.5, 0.5]]),
        ('YC', (6, 2), [0.0, 1.0], [3 ** 0.5, 0.0], [[0.0, 0.0], [0.5, 0.5]]),
    ])
    def test_reshaped_cell(self, cyl_mode, L, a1, a2, basis):
        lattice = _periodic('triangular', Ls=(4, 6), cyl_mode=cyl_mode)

        assert lattice.L == L
        assert lattice.N_sites == 4 * 6
        np.testing.assert_allclose(lattice.A, np.array([a1, a2]).T)
        np.testing.assert_allclose(lattice.basis, basis)

    def test_reshaped_cell_is_fixed_at_construction(self):
        lattice = _periodic('triangular', Ls=(4, 4), cyl_mode='XC', build=False)

        # cyl_mode doubles the basis; nb/num_sublattice must follow it
        assert lattice.nb == 2
        assert lattice.num_sublattice == 2
        assert lattice.L == (4, 2)
        assert not lattice.A.flags.writeable
        assert all(not b.flags.writeable for b in lattice.basis)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice.build()

        assert lattice.nb == 2
        assert lattice.N_sites == 4 * 2 * 2

    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_boundaries_follow_the_reshaped_size(self, cyl_mode):
        lattice = _periodic('triangular', Ls=(4, 6), cyl_mode=cyl_mode)

        assert lattice.axis1_boundary.L == lattice.L[0]
        assert lattice.axis2_boundary.L == lattice.L[1]

    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_every_wrapping_bond_picks_up_the_twist(self, cyl_mode):
        twists = (0.3, 0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice = _periodic('triangular', Ls=(4, 6), twists=twists, cyl_mode=cyl_mode)
            image = lattice.get_nth_image_neighbors(1)

        assert image
        for pair in image:
            expected = -(pair._shift[0] * twists[0] + pair._shift[1] * twists[1])
            assert np.isclose(pair.phase, expected)


class TestUnitcellRegression:
    """Guards the dead rotation-group / `_is_valid_image_old` removal against
    any change in live geometry or neighbor maps.
    """

    @pytest.fixture(scope='class', params=sorted(REFERENCE_3x3_PBC))
    def case(self, request):
        lattice_type = request.param
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice = _periodic(lattice_type)
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
    """Cross-check the 3x3 periodic square neighbor map against the offsets written
    out by hand, rather than against a golden list from the old code.
    """
    L = 3
    lattice = _periodic('square', Ls=(L, L))
    pairs = lattice.get_nth_neighbors(1)

    expected = {
        (x * L + y, ((x + dx) % L) * L + (y + dy) % L)
        for x in range(L)
        for y in range(L)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    }
    assert {(p.i, p.j) for p in pairs} == expected


def test_twist_puts_a_phase_on_image_pairs_only():
    twisted = _periodic('square', twists=('1/2 pi', 0.0))

    direct = twisted.get_nth_direct_neighbors(1)
    image = twisted.get_nth_image_neighbors(1)

    assert all(p.phase == 0.0 for p in direct)
    phases = {round(p.phase, 12) for p in image}
    assert phases == {0.0, round(np.pi / 2, 12), round(-np.pi / 2, 12)}


@pytest.mark.parametrize("lattice_type", ['honeycomb', 'kagome'])
def test_twist_sign_on_a_one_cell_wide_periodic_axis(lattice_type):
    # with L == 1 both crossing directions land on coordinate +-1, so the sign
    #   must come from the shift, not from the image coordinate
    lattice = _periodic(lattice_type, Ls=(4, 1), twists=(0.0, 0.3))

    image = lattice.get_nth_image_neighbors(1)

    assert any(p._shift[1] != 0 for p in image)
    for p in image:
        assert np.isclose(p.phase, -0.3 * p._shift[1])


class TestMinDistance:
    """`_neighbor_distance_map(min_distance=...)` drops short separations from
    the neighbor shells.
    """

    def test_short_separations_are_dropped_from_the_map(self):
        # the honeycomb intra-cell separation, 1/sqrt(3), is the n=1 shell by default
        default = _periodic('honeycomb')
        default._neighbor_distance_map(0.0)
        assert np.isclose(default._dist_map[1], 1 / np.sqrt(3))

        filtered = _periodic('honeycomb')
        filtered._neighbor_distance_map(0.0, min_distance=0.9)

        assert np.isclose(filtered._dist_map[1], 1.0)
        assert not any(0.0 < d < 0.9 for d in filtered._dist_map)

    def test_zero_self_distance_is_kept_at_index_zero(self):
        lattice = _periodic('honeycomb')

        assert lattice._neighbor_distance_map(0.0, min_distance=0.9) == 0
        assert lattice._dist_map[0] == 0.0

    def test_only_the_call_that_builds_the_map_filters(self):
        lattice = _periodic('honeycomb')
        lattice._neighbor_distance_map(0.0)

        lattice._neighbor_distance_map(0.0, min_distance=0.9)

        assert np.isclose(lattice._dist_map[1], 1 / np.sqrt(3))


class TestBasisValidation:
    """`build()` rejects the two basis layouts that made the neighbor machinery
    silently disagree with the minimum-image answer.
    """

    @staticmethod
    def _custom(basis, L=(3, 3), a1=(1.0, 0.0), a2=(0.0, 1.0)):
        return Lattice([a1, a2], basis, L, boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))

    @pytest.mark.parametrize("basis", [
        [[0.0, 0.0], [1.0, 0.0]],          # differs by a1
        [[0.0, 0.0], [0.0, 2.0]],          # differs by 2*a2
        [[0.0, 0.0], [0.0, 0.0]],          # identical
        [[0.5, 0.5], [1.5, -0.5]],         # differs by a1 - a2
    ])
    def test_basis_differing_by_a_lattice_translation_is_rejected(self, basis):
        with pytest.raises(ValueError, match="describe the same site"):
            self._custom(basis)

    @pytest.mark.parametrize("basis", [
        [[0.0, 0.0], [3.0, 0.0]],          # exactly one supercell along a1
        [[0.0, 0.0], [3.5, 0.0]],
        [[0.0, 0.0], [0.5, 4.5]],          # beyond the supercell along a2
    ])
    def test_basis_reaching_beyond_the_supercell_is_rejected(self, basis):
        with pytest.raises(ValueError):
            self._custom(basis)

    @pytest.mark.parametrize("basis", [
        [[0.0, 0.0], [0.5, 0.5]],
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],       # Lieb
        [[0.0, 0.0], [2.5, 0.0]],                   # outside the cell but reachable
    ])
    def test_valid_bases_are_accepted(self, basis):
        lattice = self._custom(basis)

        assert lattice.N_sites == 9 * len(basis)

    def test_the_documented_lieb_example_still_builds(self):
        # the tilted-cell Lieb lattice from the docs
        lattice = Lattice(
            [[1.0, -1.0], [1.0, 1.0]],
            [[0, 0], [0.25, 0.25], [-0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [0.25, 0.75]],
            (4, 4), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0),
        )

        assert lattice.N_sites == 16 * 6

    @pytest.mark.parametrize("lattice_type", sorted(LATTICE_TYPES))
    def test_built_in_types_pass(self, lattice_type):
        assert _periodic(lattice_type).N_sites > 0

    @pytest.mark.parametrize("cyl_mode", ['XC', 'YC'])
    def test_cyl_mode_doubled_basis_passes(self, cyl_mode):
        # cyl_mode appends b + a2 for every b, so it must survive the check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lattice = _periodic('triangular', Ls=(4, 4), cyl_mode=cyl_mode)

        assert lattice.nb == 2

    def test_a_translation_that_lands_outside_an_open_lattice_is_allowed(self):
        """A molecule-in-a-box is not a broken basis.

        These two basis vectors differ by exactly `a1`, but the lattice is one
        cell wide with open boundaries, so nothing wraps and the sites are
        distinct. This is the water-molecule example from the lattice tutorial.
        """
        lattice = Lattice([[1.0, 0.0], [0.0, 1.0]],
                          [[0.0, 0.0], [-0.5, -0.5], [0.5, -0.5]], (1, 1),
                          boundaries=('open', 'open'), twists=(0.0, 0.0))

        assert lattice.N_sites == 3
        positions = lattice.get_positions()
        assert len({tuple(p) for p in positions}) == 3

    def test_same_translation_is_rejected_once_the_axis_wraps(self):
        # identical basis to the test above, but now the a1 axis is periodic,
        # so the two sites really are one and the same
        with pytest.raises(ValueError, match="describe the same site"):
            Lattice([[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [-0.5, -0.5], [0.5, -0.5]], (1, 1),
                    boundaries=('periodic', 'open'), twists=(0.0, 0.0))

    def test_validation_happens_at_build_not_construction(self):
        lattice = Lattice([(1.0, 0.0), (0.0, 1.0)], [[0.0, 0.0], [1.0, 0.0]], (3, 3),
                          boundaries=('open', 'open'), twists=(0.0, 0.0), build=False)

        with pytest.raises(ValueError, match="describe the same site"):
            lattice.build()

    @pytest.mark.parametrize("L,offset,should_build", [
        # the same offset is fine in a big cell and wrong in a small one, because
        # the image search only shifts by one supercell
        ((4, 4), 2.5, True),
        ((2, 2), 2.5, False),
        ((6, 6), 5.5, True),
        ((4, 4), 5.5, False),
    ])
    def test_rejection_tracks_the_minimum_image_answer(self, L, offset, should_build):
        """The guard must fire exactly when the lattice would be wrong.

        `builds` is compared against a brute-force minimum-image calculation
        over a wide range of supercell shifts, so this pins the guard to the
        physics rather than to the rule of thumb used to implement it.
        """
        basis = [[0.0, 0.0], [offset, 0.0]]
        try:
            lattice = self._custom(basis, L=L)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pairs = lattice.get_nth_neighbors(1)
            built = True
        except ValueError:
            built = False

        assert built is should_build

        if not built:
            return

        positions = lattice.get_positions()
        v1 = lattice.L[0] * lattice.A[:, 0]
        v2 = lattice.L[1] * lattice.A[:, 1]
        best = np.full((lattice.N_sites,) * 2, np.inf)
        for s1 in range(-6, 7):
            for s2 in range(-6, 7):
                separation = np.linalg.norm(
                    positions[:, None] - (positions[None, :] + s1 * v1 + s2 * v2), axis=-1)
                best = np.minimum(best, separation)
        np.fill_diagonal(best, np.inf)

        assert np.isclose(lattice._dist_map[1], best.min())
        assert len(pairs) == int(np.isclose(best, best.min()).sum())


def test_neighbor_pairs_are_directed_one_per_crossing():
    """Neighbor pairs are directed, one per boundary crossing — not deduplicated
    by minimum image.

    On a 2x2 periodic square each site has 4 nearest neighbors, so there are 16
    pairs even though only 8 distinct site-pairs are involved: site i reaches
    site j through both the +a1 and the -a1 image, and those two crossings carry
    opposite twist phases. Collapsing them would lose the sign the phase needs.
    """
    lattice = _periodic('square', Ls=(2, 2), twists=('1/2 pi', 0.0))

    direct = lattice.get_nth_direct_neighbors(1)
    image = lattice.get_nth_image_neighbors(1)

    # 4 sites x 4 nearest neighbors, split evenly between home cell and images
    assert len(direct) + len(image) == 2 * 2 * 4
    assert len(direct) == len(image) == 8

    degrees = np.zeros(lattice.N_sites, dtype=int)
    for pair in direct + image:
        degrees[pair.i] += 1
    assert np.all(degrees == 4)

    # the same ordered pair appears twice: inside the cell, and across the
    # boundary. Deduplicating by minimum image would merge these.
    assert (0, 2) in {(p.i, p.j) for p in direct}
    assert (0, 2) in {(p.i, p.j) for p in image}

    # home-cell neighbors never pick up a twist
    assert all(pair.phase == 0.0 for pair in direct)

    # crossing the twisted axis in opposite directions gives opposite phases
    crossings = {(pair.i, pair.j): (pair._shift, round(pair.phase, 12))
                 for pair in image}
    forward_shift, forward_phase = crossings[(0, 2)]
    reverse_shift, reverse_phase = crossings[(2, 0)]

    assert forward_shift == (-1, 0) and reverse_shift == (1, 0)
    assert forward_phase == round(np.pi / 2, 12)
    assert reverse_phase == -forward_phase

    # the untwisted axis contributes no phase
    assert all(round(phase, 12) == 0.0
               for shift, phase in crossings.values() if shift[0] == 0)


class TestUnbuilt:

    @pytest.fixture
    def unbuilt(self):
        return _periodic('square', build=False)

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


def test_square_directed_pairs():
    lattice = _periodic('square', Ls=(3, 3))

    offsets = lattice.get_directed_pairs(directions=['+x', '-y', '0']).offsets

    # site index = x*L2 + y for a single-site basis
    assert offsets['+x'].tolist() == [((i // 3 + 1) % 3) * 3 + i % 3 for i in range(9)]
    assert offsets['-y'].tolist() == [(i // 3) * 3 + (i % 3 - 1) % 3 for i in range(9)]
    assert offsets['0'].tolist() == list(range(9))

    with pytest.raises(ValueError, match="unknown direction"):
        lattice.get_directed_pairs(directions=['+z'])


@pytest.mark.parametrize("lattice_type", sorted(LATTICE_TYPES))
def test_directed_pairs_method_matches_the_function(lattice_type):
    lattice = _periodic(lattice_type)

    method = lattice.get_directed_pairs(directions=['+x', '+x-y'], nbands=2).offsets
    function = get_directed_pairs(lattice, directions=['+x', '+x-y'], nbands=2).offsets

    assert method.keys() == function.keys()
    for direction in method:
        np.testing.assert_array_equal(method[direction], function[direction])
