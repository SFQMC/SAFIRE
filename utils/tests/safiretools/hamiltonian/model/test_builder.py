# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`HamiltonianBuilder`: build steps, accessors, and the two Hermiticity
conventions that must not be confused with each other."""

import numpy as np
import pytest
import scipy.sparse as sps

from safiretools import Lattice, SpinSymm
from safiretools.hamiltonian.model.builder import (
    HamiltonianBuilder,
    force_hermitian,
    intersite_band_matrix,
    is_hermitian,
    onsite_band_matrix,
    skip_empty_params,
)
from safiretools.hamiltonian.model.lattice_hamiltonian import LatticeHamiltonian


@pytest.fixture
def square_2x2():
    return Lattice.from_dict(dict(L1=2, L2=2, boundary1='pbc', boundary2='pbc'))


@pytest.fixture
def square_4x4():
    return Lattice.from_dict(dict(L1=4, L2=4, boundary1='pbc', boundary2='pbc'))


# ----------------------------------------------------------------------
# force_hermitian: the diagonal must survive
# ----------------------------------------------------------------------

class TestForceHermitian:
    """
    afqmctools' ``force_herm`` started from ``triu(M, 1)``, which zeroed the
    diagonal, so every onsite/diagonal term of a non-Hermitian input was
    silently discarded.
    """

    NON_HERMITIAN = np.array([
        [1.0, 2.0, 3.0],
        [9.0, 4.0, 5.0],
        [8.0, 7.0, 6.0],
    ])

    def test_diagonal_is_preserved(self):
        result = force_hermitian(self.NON_HERMITIAN)

        assert np.allclose(np.diag(result), np.diag(self.NON_HERMITIAN))
        assert is_hermitian(result)

    def test_upper_triangle_is_kept_and_mirrored(self):
        result = force_hermitian(self.NON_HERMITIAN)

        expected = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ])
        assert np.allclose(result, expected)

    def test_complex_diagonal_is_preserved_and_result_is_hermitian(self):
        matrix = self.NON_HERMITIAN + 1j * np.tril(np.ones((3, 3)), -1)
        result = force_hermitian(matrix)

        assert np.allclose(np.diag(result), np.diag(matrix))
        assert is_hermitian(result)

    def test_sparse_diagonal_is_preserved(self):
        result = force_hermitian(sps.csr_array(self.NON_HERMITIAN))

        assert np.allclose(result.toarray().diagonal(), np.diag(self.NON_HERMITIAN))
        assert is_hermitian(result)

    def test_average_method(self):
        result = force_hermitian(self.NON_HERMITIAN, method='average')

        assert is_hermitian(result)
        assert np.allclose(np.diag(result), np.diag(self.NON_HERMITIAN))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown force_hermitian method"):
            force_hermitian(self.NON_HERMITIAN, method='sideways')


NON_HERMITIAN_BAND = np.array([[5.0, 2.0], [9.0, 7.0]])
"""A non-Hermitian 2-band amplitude whose diagonal the old code discarded."""


def test_onebody_onsite_keeps_the_band_diagonal(square_2x2):
    """
    Regression for the diagonal-zeroing bug at `epsilon_band` in
    `onebody_onsite`, where the band diagonal lands on the matrix diagonal.
    """
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=2,
                                 spin_symm=SpinSymm.CLOSED)
    builder.onebody_onsite(NON_HERMITIAN_BAND, force_herm=True)

    matrix = builder.get_hamiltonian()['tij'][0].toarray()
    assert np.allclose(np.diag(matrix), [5.0, 7.0] * square_2x2.N_sites)
    assert np.allclose(matrix, matrix.conj().T)


def test_nth_neighbor_hopping_keeps_the_band_diagonal(square_2x2):
    """
    Regression for the same bug at `tband` in `nth_neighbor_hopping`. Hopping
    kroneckers the band matrix with a neighbor graph that has no self-neighbors,
    so the band diagonal shows up inside the *off-site* blocks, not on the
    matrix diagonal.
    """
    nbands = 2
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=nbands,
                                 spin_symm=SpinSymm.CLOSED)
    builder.nth_neighbor_hopping(NON_HERMITIAN_BAND, force_herm=True)

    matrix = builder.get_hamiltonian()['tij'][0].toarray()
    # sites 0 and 1 are nearest neighbors on a 2x2 periodic lattice
    block = matrix[0:nbands, nbands:2 * nbands]
    assert not np.allclose(block, 0.0)
    # the graph weight is an integer multiple of -1, so the band diagonal is
    #   proportional to (-5, -7) rather than zero
    assert np.allclose(np.diag(block) / np.diag(block)[0], [1.0, 7.0 / 5.0])
    assert np.allclose(matrix, matrix.conj().T)


@pytest.mark.parametrize("step", ["nth_neighbor_hopping", "onebody_onsite"])
def test_build_step_rejects_a_non_hermitian_band_matrix_by_default(square_2x2, step):
    builder = HamiltonianBuilder(lattice=square_2x2, nbands=2)

    with pytest.raises(ValueError, match="force_herm"):
        getattr(builder, step)(np.array([[5.0, 2.0], [9.0, 7.0]]))


# ----------------------------------------------------------------------
# the separate, deliberately diagonal-free U1/U2/J convention
# ----------------------------------------------------------------------

class TestBandMatrices:
    """
    The AFQMC executable reads a triangle of the whole interaction matrix,
    indexed by the combined ``site * nbands + band``; it has no notion of sites
    versus bands. U1, U2 and J must land strictly above that combined diagonal,
    which is left for the onsite Hubbard U. So this convention must *not* pick
    up the `force_hermitian` fix above.

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

    This also guards against accidentally applying the `force_hermitian`
    diagonal fix to this path.
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
    lattice = Lattice.from_dict(dict(L1=2, L2=1, boundary1='open', boundary2='open'))

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
                                 spin_symm=SpinSymm.COLLINEAR, nelec=(8, 8))

    assert builder.get_lattice() is square_4x4
    assert isinstance(builder.get_hamiltonian(), LatticeHamiltonian)
    assert builder.get_hamiltonian().nbands == 2
    assert builder.get_hamiltonian().nelec == (8, 8)


def test_builder_requires_a_lattice():
    with pytest.raises(ValueError, match="must be defined on a 'Lattice'"):
        HamiltonianBuilder(lattice=None)


def test_unknown_builder_parameter_raises(square_4x4):
    with pytest.raises(ValueError, match="Unknown HamiltonianBuilder parameters"):
        HamiltonianBuilder(lattice=square_4x4, nbandz=2)


def test_from_input_builds_and_finalizes(square_4x4):
    builder = HamiltonianBuilder.from_input(
        {'hamiltonian': dict(t=1.0, U=4.0, nelec=(8, 8))},
        lattice=square_4x4,
    )
    hamiltonian = builder.get_hamiltonian()

    assert sorted(hamiltonian.keys()) == ['Uij', 'tij']
    assert hamiltonian.num_components == 2
    assert hamiltonian.spin_symm is SpinSymm.COLLINEAR
    assert hamiltonian['Uij'][0].hubbard_strat_type == 'discrete_spin'


def test_from_input_builds_its_own_lattice():
    builder = HamiltonianBuilder.from_input({
        'lattice': dict(L1=3, L2=3, boundary1='pbc', boundary2='pbc'),
        'hamiltonian': dict(t=1.0, U=4.0),
    })

    assert builder.get_lattice().N_sites == 9
    assert builder.get_hamiltonian().nsites == 9


def test_from_input_reads_a_toml_file(tmp_path):
    import toml

    path = tmp_path / 'input.toml'
    with open(path, 'w') as f:
        toml.dump({
            'lattice': dict(L1=2, L2=2, boundary1='pbc', boundary2='pbc'),
            'hamiltonian': dict(t=1.0, U=4.0),
        }, f)

    builder = HamiltonianBuilder.from_input(path)
    assert builder.get_hamiltonian().nsites == 4


def test_from_input_rejects_other_sources():
    with pytest.raises(ValueError, match="Invalid parameter source"):
        HamiltonianBuilder.from_input(42)


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


def test_fm_pinning_reads_fm_pin_type(square_4x4):
    """
    afqmctools' `fm_pinning` read ``afm_pin_type``, which made ``fm_pin_type``
    dead input.
    """
    builder = HamiltonianBuilder(lattice=square_4x4)
    hamiltonian = builder.get_hamiltonian()
    hamiltonian.afm_pin_type = 'sideways'   # would raise if it were consulted
    hamiltonian.fm_pin_type = 'fm'

    builder.fm_pinning(0.25)
    assert hamiltonian['tij']


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
