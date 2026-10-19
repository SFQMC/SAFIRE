# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`HamiltonianBuilder`: build steps, accessors, and the interaction-matrix
triangle convention."""

import numpy as np
import pytest

from safiretools import Lattice, SpinSymm
from safiretools.hamiltonian.model.builder import (
    HamiltonianBuilder,
    PinningField,
    intersite_band_matrix,
    onsite_band_matrix,
    skip_empty_params,
)
from safiretools.hamiltonian.model.lattice_hamiltonian import LatticeHamiltonian


@pytest.fixture
def square_2x2():
    return Lattice.from_type('square', (2, 2), boundaries=('periodic', 'periodic'),
                             twists=(0.0, 0.0))


@pytest.fixture
def square_4x4():
    return Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'),
                             twists=(0.0, 0.0))


@pytest.mark.parametrize("step", ["nth_neighbor_hopping", "onebody_onsite"])
def test_build_step_rejects_a_non_hermitian_band_matrix(square_2x2, step):
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=2)

    with pytest.raises(ValueError, match="not hermitian"):
        getattr(builder, step)(np.array([[5.0, 2.0], [9.0, 7.0]]))


# ----------------------------------------------------------------------
# the separate, deliberately diagonal-free U1/U2/J convention
# ----------------------------------------------------------------------

class TestBandMatrices:
    """
    The AFQMC executable reads a triangle of the whole interaction matrix,
    indexed by the combined ``site * nbands + band``; it has no notion of sites
    versus bands. U1, U2 and J must land strictly above that combined diagonal,
    which is left for the onsite Hubbard U.

    The onsite and inter-site band matrices differ because of how each is
    kroneckered into the combined index — see the two helpers' docstrings.
    """

    def test_onsite_form_is_strictly_upper_triangular(self):
        matrix = onsite_band_matrix(2.5, nbands=3).toarray()

        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(np.tril(matrix), 0.0)
        assert np.allclose(matrix[np.triu_indices(3, k=1)], 2.5)

    def test_single_band_onsite_form_is_empty(self):
        matrix = onsite_band_matrix(2.5, nbands=1)

        assert matrix.shape == (1, 1)
        assert matrix.nnz == 0

    def test_intersite_form_is_the_full_matrix(self):
        """
        Every ``(m, n)`` is band ``m`` on one site interacting with band ``n`` on
        the other, and all of them land in the combined upper triangle — so the
        lower triangle must be filled too.
        """
        matrix = intersite_band_matrix(2.5, nbands=3).toarray()

        assert np.allclose(matrix, 2.5)
        assert matrix.shape == (3, 3)

    def test_single_band_intersite_form_is_the_lone_element(self):
        matrix = intersite_band_matrix(2.5, nbands=1).toarray()

        assert np.allclose(matrix, [[2.5]])


INTERACTION_STEPS = [
    ("hubbard_U1_density_density", 'U1ij'),
    ("hubbard_U2_spin_spin", 'U2ij'),
    ("hubbard_Jij", 'Jij'),
]


@pytest.mark.parametrize("step,key", INTERACTION_STEPS)
@pytest.mark.parametrize("nbands,nth_neighbor", [
    (3, 0),   # onsite: strictly upper comes from the band matrix
    (3, 1),   # inter-site, multiband: the site matrix already forces it
    (1, 1),   # inter-site, single band: the band matrix is diagonal-only
])
def test_interaction_terms_stay_strictly_above_the_combined_diagonal(
        square_2x2, step, key, nbands, nth_neighbor):
    """
    The executable indexes the interaction matrix by the combined
    ``site * nbands + band`` and keeps ``i < j`` for the spin-spin and J blocks,
    leaving the diagonal for the onsite Hubbard U. So every U1/U2/J contribution
    has to land strictly above the combined diagonal.
    """
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=nbands)
    getattr(builder, step)(1.5, nth_neighbor=nth_neighbor)

    matrix = builder.get_hamiltonian()[key][0].toarray()
    assert matrix.any(), "expected a non-empty interaction term"
    assert np.allclose(np.tril(matrix), 0.0)


@pytest.mark.parametrize("step,key", INTERACTION_STEPS)
def test_intersite_terms_cover_every_band_pair_across_the_bond(step, key):
    """
    Regression: afqmctools built the inter-site band matrix with
    ``combinations_with_replacement``, i.e. ``m <= n`` only, which dropped every
    ``m > n`` pair. On two sites with two bands that silently omitted the
    ``s0b1``-``s1b0`` interaction, even though its combined index (1, 2) is in
    the triangle the executable reads.
    """
    nbands = 2
    lattice = Lattice.from_type('square', (2, 1), boundaries=('open', 'open'), twists=(0.0, 0.0))

    builder = HamiltonianBuilder(lattice=lattice, nbands=nbands)
    getattr(builder, step)(1.5, nth_neighbor=1)

    matrix = builder.get_hamiltonian()[key][0].toarray()
    labels = [f's{i}b{m}' for i in range(2) for m in range(nbands)]
    present = {(labels[r], labels[c]) for r, c in zip(*np.nonzero(matrix))}

    assert present == {
        ('s0b0', 's1b0'), ('s0b0', 's1b1'),
        ('s0b1', 's1b0'), ('s0b1', 's1b1'),
    }
    # and every one of them is still in the range the executable reads
    assert np.allclose(np.tril(matrix), 0.0)


@pytest.mark.parametrize("step,key", INTERACTION_STEPS)
def test_interaction_terms_never_touch_the_onsite_diagonal(square_2x2, step, key):
    """
    The combined diagonal carries the onsite Hubbard U, so U1/U2/J must leave it
    empty — otherwise `LatticeHamiltonian._split_hubbard_u` could not recover U
    from a written file.
    """
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=3)
    getattr(builder, step)(1.5)

    assert np.allclose(np.diag(builder.get_hamiltonian()[key][0].toarray()), 0.0)


@pytest.mark.parametrize("step", [
    "hubbard_U1_density_density",
    "hubbard_U2_spin_spin",
    "hubbard_Jij",
])
def test_onsite_interaction_needs_more_than_one_band(square_2x2, step):
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=1)

    with pytest.raises(ValueError, match="not supported for nbands=1"):
        getattr(builder, step)(1.5)


# ----------------------------------------------------------------------
# accessors and construction
# ----------------------------------------------------------------------

def test_accessors_replace_bare_attribute_access(square_4x4):
    builder = HamiltonianBuilder(lattice=square_4x4, nbands=2,
                                 spin_symm=SpinSymm.COLLINEAR)

    assert builder.get_lattice() is square_4x4
    assert isinstance(builder.get_hamiltonian(), LatticeHamiltonian)
    assert builder.get_hamiltonian().nbands == 2
    assert builder.get_hamiltonian().spin_symm is SpinSymm.COLLINEAR


def test_builder_requires_a_lattice():
    with pytest.raises(ValueError, match="must be defined on a 'Lattice'"):
        HamiltonianBuilder(lattice=None)


def test_unknown_builder_parameter_raises(square_4x4):
    with pytest.raises(TypeError, match="nbandz"):
        HamiltonianBuilder(lattice=square_4x4, nbandz=2)


def test_from_input_builds_and_finalizes(square_4x4):
    builder = HamiltonianBuilder.from_input(square_4x4, t=1.0, U=4.0)
    hamiltonian = builder.get_hamiltonian()

    assert sorted(hamiltonian.keys()) == ['Uij', 'tij']
    assert hamiltonian.num_components == 2
    assert hamiltonian.spin_symm is SpinSymm.COLLINEAR
    assert hamiltonian['Uij'][0].hubbard_strat_type == 'discrete_spin'


def test_from_input_rejects_unknown_keys(square_4x4):
    with pytest.raises(TypeError, match="twist"):
        HamiltonianBuilder.from_input(square_4x4, U=4.0, twist=(0.1, 0.0))


def test_from_input_passes_hst_types(square_4x4):
    hamiltonian = HamiltonianBuilder.from_input(
        square_4x4, U=4.0, hst_types={'U': 'continuous_charge'}).get_hamiltonian()

    assert hamiltonian['Uij'][0].hubbard_strat_type == 'continuous_charge'


def test_from_input_rejects_hst_types_for_one_body_terms(square_4x4):
    with pytest.raises(ValueError, match="non-interaction"):
        HamiltonianBuilder.from_input(square_4x4, hst_types={'t': 'discrete_spin'})


def test_negative_and_positive_u_become_separate_components(square_2x2):
    """
    Opposite signs call for different Hubbard-Stratonovich transformations, so
    they must not be combined.
    """
    builder = HamiltonianBuilder(lattice=square_2x2)
    builder.onsite_hubbard(np.array([4.0, -2.0, 4.0, -2.0]))
    builder.finalize()

    hst_types = {component.hubbard_strat_type
                 for component in builder.get_hamiltonian()['Uij']}
    assert hst_types == {'discrete_spin', 'discrete_charge'}


def test_mixed_sign_scalar_u_cannot_choose_an_hst_type(square_2x2):
    builder = HamiltonianBuilder(lattice=square_2x2)

    with pytest.raises(ValueError, match="unambiguously choose"):
        builder._get_hst_type(np.array([1.0, -1.0]))


def test_invalid_hst_type_override_raises(square_2x2):
    builder = HamiltonianBuilder(lattice=square_2x2)

    with pytest.raises(ValueError, match="Invalid hst_type"):
        builder.onsite_hubbard(4.0, hst_type='continuous_sideways')


@pytest.mark.parametrize("hst_type,cleaned", [
    ('discrete charge', 'discrete_charge'),
    ('DISCRETE_SPIN', 'discrete_spin'),
    ('Continuous Spin', 'continuous_spin'),
])
def test_hst_type_spellings_are_normalized(square_2x2, hst_type, cleaned):
    builder = HamiltonianBuilder(lattice=square_2x2)
    assert builder._clean_hst_type(hst_type) == cleaned


def test_rashba_soc_requires_noncollinear(square_2x2):
    builder = HamiltonianBuilder(lattice=square_2x2, spin_symm=SpinSymm.COLLINEAR)

    # rashba_lambda has to be positional: skip_empty_params wraps it as `params`
    with pytest.raises(ValueError, match="non-collinear"):
        builder.rashba_soc(0.1)


def pinning_potentials(builder):
    """The (up, down) on-site potentials of the builder's single one-body term."""
    nbasis = builder.get_hamiltonian().nbasis
    matrix = builder.get_hamiltonian()['tij'][0].csr_array.toarray()
    return np.diag(matrix[:nbasis]), np.diag(matrix[nbasis:])


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("field,expected", [
    (PinningField.afm(0.5, staggered=True), lambda n, edge: (0.25 * (-1)**(n + edge), -0.25 * (-1)**(n + edge))),
    (PinningField.afm(0.5, staggered=False), lambda n, edge: (0.25 * (-1)**n, -0.25 * (-1)**n)),
    (PinningField.fm(0.5, staggered=True), lambda n, edge: (0.25, -0.25) if edge else (-0.25, 0.25)),
    (PinningField.fm(0.5, staggered=False), lambda n, edge: (0.25, -0.25)),
    (PinningField.charge(0.5, staggered=True), lambda n, edge: (0.25, 0.25) if edge else (-0.25, -0.25)),
    (PinningField.charge(0.5, staggered=False), lambda n, edge: (0.25, 0.25)),
    (PinningField(lambda n, edge: (n, -edge)), lambda n, edge: (n, -edge)),
])
def test_pinning_acts_on_the_edges_of_one_axis(square_4x4, axis, field, expected):
    builder = HamiltonianBuilder(lattice=square_4x4)
    builder.pinning((field, None) if axis == 0 else (None, field))
    up, down = pinning_potentials(builder)

    for site in square_4x4.get_sites():
        r = site.coord[axis]
        if r in (0, 3):
            want = expected(site.coord[(axis + 1) % 2], int(r == 3))
        else:
            want = (0.0, 0.0)
        assert (up[site.index], down[site.index]) == pytest.approx(want)


def test_pinning_on_both_axes_adds_at_the_corners(square_4x4):
    builder = HamiltonianBuilder(lattice=square_4x4)
    builder.pinning((PinningField.charge(0.5, staggered=False),
                     PinningField.charge(0.5, staggered=False)))
    up, _ = pinning_potentials(builder)

    corner = next(site.index for site in square_4x4.get_sites()
                  if site.coord[0] == 0 and site.coord[1] == 0)
    assert up[corner] == pytest.approx(0.5)


def test_pinning_needs_one_field_per_axis(square_4x4):
    builder = HamiltonianBuilder(lattice=square_4x4)

    with pytest.raises(ValueError, match="one entry per lattice axis"):
        builder.pinning((PinningField.charge(0.5, staggered=False),))


@pytest.mark.parametrize("params,expected_calls", [
    (0.0, 0),
    (None, 0),
    ([0.0, 0.0], 0),
    ([None, 0.0], 0),
    (1.0, 1),
    ([0.0, 1.0], 1),
])
def test_skip_empty_params(params, expected_calls):
    class Recorder:
        def __init__(self):
            self.calls = []

        @skip_empty_params
        def step(self, params, *args, **kwargs):
            self.calls.append(params)

    recorder = Recorder()
    recorder.step(params)
    assert len(recorder.calls) == expected_calls


class TestBuildStepsArePassableByKeyword:
    """
    The build-step decorators must preserve the wrapped signature, so the
    amplitude parameter each step documents (`t`, `U`, `U1`, `J`, ...) can be
    given by keyword. A wrapper declared as ``(self, params, *args, **kwargs)``
    renames every one of them to ``params`` and raises
    ``TypeError: missing 1 required positional argument: 'params'`` instead.
    """

    @pytest.mark.parametrize("step,kwargs", [
        ('nth_neighbor_hopping', dict(t=[1.0, 0.5])),
        ('onebody_onsite', dict(epsilon=0.1)),
        ('onsite_hubbard', dict(U=4.0)),
        ('hubbard_U1_density_density', dict(U1=1.5)),
        ('hubbard_U2_spin_spin', dict(U2=1.0)),
        ('hubbard_Jij', dict(J=0.5)),
        ('heisenberg_J', dict(J=0.3)),
        ('nth_order_hubbard_Vij', dict(V=2.0)),
    ])
    def test_amplitude_by_keyword(self, square_2x2, step, kwargs):
        builder = HamiltonianBuilder(lattice=square_2x2, nbands=2,
                                     spin_symm=SpinSymm.COLLINEAR)
        getattr(builder, step)(**kwargs)
        assert builder.get_hamiltonian().num_components > 0

    @pytest.mark.parametrize("step,kwargs", [
        ('nth_neighbor_hopping', dict(t=1.0)),
        ('onsite_hubbard', dict(U=4.0)),
    ])
    def test_keyword_and_positional_agree(self, square_2x2, step, kwargs):
        (value,) = kwargs.values()

        positional = HamiltonianBuilder(lattice=square_2x2)
        getattr(positional, step)(value)

        keyword = HamiltonianBuilder(lattice=square_2x2)
        getattr(keyword, step)(**kwargs)

        for key in positional.get_hamiltonian().keys():
            a = sum(positional.get_hamiltonian()[key]).csr_array
            b = sum(keyword.get_hamiltonian()[key]).csr_array
            assert (a != b).nnz == 0

    @pytest.mark.parametrize("step,amplitude", [
        ('nth_neighbor_hopping', 't'),
        ('onsite_hubbard', 'U'),
        ('nth_order_hubbard_Vij', 'V'),
        ('rashba_soc', 'rashba_lambda'),
    ])
    def test_signature_survives_decoration(self, step, amplitude):
        """Introspection (and therefore autodoc) sees the real parameter names."""
        import inspect

        parameters = inspect.signature(getattr(HamiltonianBuilder, step)).parameters
        assert list(parameters)[:2] == ['self', amplitude]
        assert 'params' not in parameters

    def test_rashba_soc_iterates_a_list_of_hoppings(self, square_4x4):
        """
        `rashba_soc` recurses for a list-valued `t`, passing `rashba_lambda` by
        keyword — impossible while the decorator renamed the first parameter.
        """
        builder = HamiltonianBuilder(lattice=square_4x4,
                                     spin_symm=SpinSymm.NONCOLLINEAR)
        builder.nth_neighbor_hopping([1.0, 0.5])
        builder.rashba_soc(rashba_lambda=0.3, t=[1.0, 0.5])
        builder.finalize()

        assert builder.get_hamiltonian()['tij']

    def test_a_zero_amplitude_given_by_keyword_still_skips(self, square_2x2):
        builder = HamiltonianBuilder(lattice=square_2x2)
        builder.onsite_hubbard(U=0.0)
        builder.hubbard_Jij(J=None)
        assert builder.get_hamiltonian().num_components == 0
