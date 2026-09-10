# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`LatticeHamiltonian`: the term container, its HDF5 format, and the round trip
through `to_hdf5`/`from_hdf5` — including splitting the combined ``Uij`` matrix
back into U, U1 and U2."""

import h5py as h5
import numpy as np
import pytest
import scipy.sparse as sps

from safiretools import Lattice, SpinSymm
from safiretools.hamiltonian.base import Hamiltonian
from safiretools.hamiltonian.model.builder import HamiltonianBuilder
from safiretools.hamiltonian.model.lattice_hamiltonian import (
    HamiltonianComponent,
    LatticeHamiltonian,
    lattice_metadata_from,
)

CASES = {
    'hubbard': dict(
        lattice=dict(L1=4, L2=4, boundary1='pbc', boundary2='pbc'),
        hamiltonian=dict(t=1.0, U=4.0, nelec=(8, 8)),
    ),
    'attractive_hubbard': dict(
        lattice=dict(L1=3, L2=3, boundary1='pbc', boundary2='open'),
        hamiltonian=dict(t=1.0, U=-4.0),
    ),
    'kanamori': dict(
        lattice=dict(L1=6, L2=1, boundary1='pbc', boundary2='open'),
        hamiltonian=dict(nbands=2, t=1.0, U=4.0, U1=2.0, U2=1.0, J=0.5, nelec=(6, 6)),
    ),
    'heisenberg': dict(
        lattice=dict(L1=4, L2=2, boundary1='pbc', boundary2='pbc'),
        hamiltonian=dict(t=1.0, J_heisenberg=1.0),
    ),
    'twisted_honeycomb': dict(
        lattice=dict(type='honeycomb', L1=3, L2=3, boundary1='pbc', boundary2='pbc',
                     twist=('1/2 pi', 0.0)),
        hamiltonian=dict(t=1.0, U=4.0),
    ),
    'custom': dict(
        lattice=dict(type='custom', L1=3, L2=3, boundary1='pbc', boundary2='open',
                     a1=[1.0, 0.0], a2=[0.0, 1.0], basis=[[0.0, 0.0], [0.5, 0.0]]),
        hamiltonian=dict(t=1.0, U=2.0),
    ),
}


def fingerprint(hamiltonian):
    """Every term's shape, metadata and nonzero entries, order-independent."""
    out = {}
    for key in sorted(hamiltonian.keys()):
        for i, component in enumerate(hamiltonian[key]):
            csr = sps.csr_array(component.csr_array)
            csr.sum_duplicates()
            coo = csr.tocoo()
            order = np.lexsort((coo.col, coo.row))
            out[f'{key}[{i}]'] = (
                tuple(csr.shape),
                str(component.model_type),
                str(component.hubbard_strat_type),
                int(component.spin_symm),
                tuple(coo.row[order].tolist()),
                tuple(coo.col[order].tolist()),
                tuple(np.round(coo.data[order], 12).tolist()),
            )
    return out


def h5_fingerprint(path):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5.Dataset):
            value = obj[...]
            out[name] = (str(value) if value.dtype.kind in 'OSU'
                         else np.round(np.asarray(value, dtype=float), 12).tolist())

    with h5.File(path, 'r') as f:
        f.visititems(visit)
    return out


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param, CASES[request.param]


def test_from_dict_builds_a_finalized_hamiltonian(case):
    _, source = case
    hamiltonian = LatticeHamiltonian.from_dict(source)

    assert isinstance(hamiltonian, LatticeHamiltonian)
    assert hamiltonian.num_components == sum(
        len(components) for components in hamiltonian.terms.values())
    assert hamiltonian.spin_symm is not None


def test_hdf5_round_trip_preserves_the_terms(case, tmp_path):
    """
    A file holds the *combined* Hubbard matrix, so reading it back has to invert
    the merge. Re-finalizing the split terms must reproduce the original.
    """
    name, source = case
    hamiltonian = LatticeHamiltonian.from_dict(source)

    path = tmp_path / f'{name}.h5'
    hamiltonian.to_hdf5(path)

    restored = Hamiltonian.from_hdf5(path)
    assert isinstance(restored, LatticeHamiltonian)
    assert restored.nbands == hamiltonian.nbands
    assert restored.nsites == hamiltonian.nsites
    assert restored.nelec == hamiltonian.nelec
    assert restored.spin_symm is hamiltonian.spin_symm

    _refinalize(restored)
    assert fingerprint(restored) == fingerprint(hamiltonian)


def test_hdf5_round_trip_reproduces_the_file(case, tmp_path):
    name, source = case
    hamiltonian = LatticeHamiltonian.from_dict(source)

    first = tmp_path / f'{name}.h5'
    second = tmp_path / f'{name}_again.h5'
    hamiltonian.to_hdf5(first)

    restored = Hamiltonian.from_hdf5(first)
    _refinalize(restored)
    restored.to_hdf5(second)

    assert h5_fingerprint(first) == h5_fingerprint(second)


def _refinalize(hamiltonian):
    """
    Recombine the U/U1/U2 terms `from_hdf5` split apart, without a lattice —
    which `finalize` only needs for building, not for combining.
    """
    builder = HamiltonianBuilder.__new__(HamiltonianBuilder)
    builder._hamiltonian = hamiltonian
    builder._lattice = None
    builder._combine_hubbard_u()
    builder._find_max_spin_symm()


class TestHubbardSplit:
    """The inverse of `HamiltonianBuilder._combine_hubbard_u`."""

    @pytest.fixture
    def kanamori(self, tmp_path):
        hamiltonian = LatticeHamiltonian.from_dict(CASES['kanamori'])
        path = tmp_path / 'kanamori.h5'
        hamiltonian.to_hdf5(path)
        return hamiltonian, Hamiltonian.from_hdf5(path)

    def test_u_u1_and_u2_come_back_separately(self, kanamori):
        _, restored = kanamori
        assert sorted(restored.keys()) == ['Jij', 'U1ij', 'U2ij', 'Uij', 'tij']

    def test_onsite_u_is_the_diagonal(self, kanamori):
        original, restored = kanamori
        nbasis = original.nbasis

        combined = original['Uij'][0].toarray()
        onsite = restored['Uij'][0].toarray()

        assert np.allclose(onsite, np.diag(np.diag(combined[:nbasis, :nbasis])))

    def test_u1_is_the_off_diagonal_of_the_upper_block(self, kanamori):
        original, restored = kanamori
        nbasis = original.nbasis

        upper = original['Uij'][0].toarray()[:nbasis, :nbasis]
        u1 = restored['U1ij'][0].toarray()

        assert np.allclose(np.diag(u1), 0.0)
        assert np.allclose(u1, upper - np.diag(np.diag(upper)))

    def test_u2_is_the_lower_block(self, kanamori):
        original, restored = kanamori
        nbasis = original.nbasis

        assert np.allclose(restored['U2ij'][0].toarray(),
                           original['Uij'][0].toarray()[nbasis:, :nbasis])

    def test_an_unexpected_shape_is_rejected(self):
        from safiretools.hamiltonian.model.lattice_hamiltonian import _split_hubbard_u

        component = HamiltonianComponent(sps.csr_array(np.eye(5)), 'hubbard_u')
        with pytest.raises(ValueError, match="neither"):
            list(_split_hubbard_u(component, nbasis=4))


class TestLatticeMetadata:
    """
    Lattice shape is recorded dimension-agnostically (``L``/``boundaries`` as
    per-axis sequences, the unit cell as a ``lattice_vectors`` matrix), so files
    written now stay readable when `Lattice` becomes N-dimensional.
    """

    def test_metadata_is_extracted_from_the_lattice(self):
        lattice = Lattice.from_dict(dict(type='honeycomb', L1=3, L2=4,
                                         boundary1='pbc', boundary2='open',
                                         twist=('1/2 pi', 0.0)))
        metadata = lattice_metadata_from(lattice)

        assert metadata['type'] == 'honeycomb'
        assert metadata['L'] == [3, 4]
        assert metadata['boundaries'] == ['pbc', 'open']
        assert np.isclose(metadata['twist'][0], np.pi / 2)
        assert np.isclose(metadata['twist'][1], 0.0)
        assert np.allclose(metadata['lattice_vectors'], [lattice.a1, lattice.a2])
        assert np.allclose(metadata['basis'], lattice.basis)

    def test_metadata_survives_the_round_trip(self, case, tmp_path):
        name, source = case
        hamiltonian = LatticeHamiltonian.from_dict(source)

        path = tmp_path / f'{name}.h5'
        hamiltonian.to_hdf5(path)
        restored = Hamiltonian.from_hdf5(path)

        assert restored.lattice_metadata == hamiltonian.lattice_metadata

    def test_lattice_params_rebuild_the_same_lattice(self, case):
        _, source = case
        hamiltonian = LatticeHamiltonian.from_dict(source)

        original = Lattice.from_dict(source['lattice'])
        rebuilt = Lattice.from_dict(hamiltonian.lattice_params)

        assert rebuilt.N_sites == original.N_sites
        assert np.allclose(rebuilt.a1, original.a1)
        assert np.allclose(rebuilt.a2, original.a2)
        assert np.allclose(rebuilt.basis, original.basis)
        assert np.allclose([site.position for site in rebuilt.get_sites()],
                           [site.position for site in original.get_sites()])

    def test_a_hamiltonian_with_no_metadata_writes_no_lattice_group(self, tmp_path):
        hamiltonian = LatticeHamiltonian(nsites=4, spin_symm=SpinSymm.CLOSED)
        hamiltonian.add_term('tij', HamiltonianComponent(
            sps.csr_array(np.eye(4)), 'one_body'))

        path = tmp_path / 'bare.h5'
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert 'Hamiltonian/ModelHamiltonian/Lattice' not in fh5

        assert Hamiltonian.from_hdf5(path).lattice_params == {}


class TestRealValued:
    """
    afqmctools stored the *inverse* of this under the name ``_real_valued``, so
    every real model Hamiltonian was upcast to complex on write.
    """

    def test_a_real_hamiltonian_is_real_valued(self):
        hamiltonian = LatticeHamiltonian.from_dict(CASES['hubbard'])
        assert hamiltonian.real_valued

    def test_a_twisted_hamiltonian_is_not(self):
        hamiltonian = LatticeHamiltonian.from_dict(CASES['twisted_honeycomb'])
        assert not hamiltonian.real_valued

    def test_a_real_hamiltonian_writes_rank_1_data(self, tmp_path):
        path = tmp_path / 'real.h5'
        LatticeHamiltonian.from_dict(CASES['hubbard']).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            data = fh5['Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/data_']
            assert data.ndim == 1

    def test_a_complex_hamiltonian_writes_interleaved_data(self, tmp_path):
        path = tmp_path / 'complex.h5'
        LatticeHamiltonian.from_dict(CASES['twisted_honeycomb']).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            data = fh5['Hamiltonian/ModelHamiltonian/ModelComponent_0/tij/data_']
            assert data.shape[-1] == 2


class TestHamiltonianComponent:

    def test_components_of_the_same_type_add(self):
        a = HamiltonianComponent(sps.csr_array(np.eye(2)), 'one_body',
                                 spin_symm=SpinSymm.CLOSED)
        b = HamiltonianComponent(sps.csr_array(2 * np.eye(2)), 'one_body',
                                 spin_symm=SpinSymm.COLLINEAR)

        total = a + b
        assert np.allclose(total.toarray(), 3 * np.eye(2))
        # the sum is only as symmetric as its least symmetric part
        assert total.spin_symm is SpinSymm.COLLINEAR

    def test_sum_starts_from_zero(self):
        components = [HamiltonianComponent(sps.csr_array(np.eye(2)), 'one_body')
                      for _ in range(3)]
        assert np.allclose(sum(components).toarray(), 3 * np.eye(2))

    def test_different_hst_types_cannot_be_added(self):
        a = HamiltonianComponent(sps.csr_array(np.eye(2)), 'hubbard_u',
                                 hst_type='discrete_spin')
        b = HamiltonianComponent(sps.csr_array(np.eye(2)), 'hubbard_u',
                                 hst_type='discrete_charge')

        with pytest.raises(ValueError, match="Hubbard-Stratonovich"):
            a + b

    def test_different_model_types_cannot_be_added(self):
        a = HamiltonianComponent(sps.csr_array(np.eye(2)), 'one_body')
        b = HamiltonianComponent(sps.csr_array(np.eye(2)), 'hubbard_u')

        with pytest.raises(ValueError, match="Can't add"):
            a + b


def test_writing_without_a_spin_symmetry_raises(tmp_path):
    hamiltonian = LatticeHamiltonian(nsites=4)
    hamiltonian.spin_symm = None

    with pytest.raises(ValueError, match="no spin symmetry"):
        hamiltonian.to_hdf5(tmp_path / 'ham.h5')


def test_noncollinear_merges_the_spin_sectors_in_dims(tmp_path):
    hamiltonian = LatticeHamiltonian(nsites=4, spin_symm=SpinSymm.NONCOLLINEAR,
                                     nelec=(3, 2))
    hamiltonian.add_term('tij', HamiltonianComponent(
        sps.csr_array(np.eye(8)), 'one_body', spin_symm=SpinSymm.NONCOLLINEAR))

    path = tmp_path / 'ghf.h5'
    hamiltonian.to_hdf5(path)

    with h5.File(path, 'r') as fh5:
        dims = fh5['Hamiltonian/dims'][...]
    assert (dims[4], dims[5]) == (5, 0)
