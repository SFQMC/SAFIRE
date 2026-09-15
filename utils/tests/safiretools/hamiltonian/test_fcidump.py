# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""FCIDUMP read/write."""

import numpy as np
import pytest

from safiretools.hamiltonian.fcidump import (
    check_sym,
    fcidump_header,
    h1_spat2spin,
    h2_spat2spin,
    read_fcidump,
    read_fcidump_header,
    write_fcidump,
)


@pytest.fixture
def integrals():
    """
    A small real Hamiltonian with a Cholesky factorization of its ERIs.

    Each Cholesky vector is symmetrized in its orbital pair, so the resulting
    ERI tensor has the full 8-fold permutational symmetry that `write_fcidump`
    assumes when it writes only symmetry-unique entries.
    """
    rng = np.random.default_rng(7)
    nmo = 4

    hcore = rng.random((nmo, nmo))
    hcore = 0.5 * (hcore + hcore.T)

    vectors = rng.random((6, nmo, nmo))
    vectors = 0.5 * (vectors + vectors.transpose((0, 2, 1)))
    chol = vectors.reshape(6, nmo * nmo).T

    return nmo, hcore, chol, 1.234


def test_header_round_trips(tmp_path):
    path = tmp_path / 'FCIDUMP'
    path.write_text(fcidump_header(nel=6, norb=4, spin=0))

    assert read_fcidump_header(path) == {'nbasis': 4, 'nelec': 6, 'ms2': 0, 'isym': 1}


def test_a_header_with_no_terminator_is_rejected(tmp_path):
    path = tmp_path / 'FCIDUMP'
    path.write_text("&FCI NORB=4,\n" * 30)

    with pytest.raises(ValueError, match="longer than"):
        read_fcidump_header(path, mline=5)


@pytest.mark.parametrize("sym", [1, 4, 8])
def test_write_then_read_reproduces_the_integrals(integrals, tmp_path, sym):
    nmo, hcore, chol, enuc = integrals
    nelec = (2, 2)

    path = tmp_path / 'FCIDUMP'
    write_fcidump(path, hcore, chol, enuc, nmo, nelec, sym=sym, cplx=False, tol=1e-12)

    h1e, h2e, ecore, nelec_read = read_fcidump(path, symmetry=sym, verbose=False)

    assert nelec_read == nelec
    assert np.isclose(ecore, enuc)
    assert np.allclose(h1e, hcore)

    # FCIDUMP holds (ik|jl); the Cholesky product is stored in hermitian
    #   (ik),(lj) order, so undo that transposition before comparing
    eris = (chol @ chol.conj().T).reshape((nmo,) * 4)
    assert np.allclose(h2e.transpose((0, 1, 3, 2)), eris)


@pytest.mark.parametrize("paren", [False, True])
def test_complex_integrals_round_trip(integrals, tmp_path, paren):
    nmo, hcore, chol, enuc = integrals
    complex_chol = chol + 1j * np.roll(chol, 1, axis=0)

    path = tmp_path / 'FCIDUMP'
    write_fcidump(path, hcore.astype(complex), complex_chol, enuc, nmo, (2, 2),
                  sym=1, cplx=True, paren=paren, tol=1e-12)

    h1e, h2e, ecore, _ = read_fcidump(path, symmetry=1, verbose=False)

    assert np.isclose(ecore, enuc)
    assert np.allclose(h1e, hcore)
    eris = (complex_chol @ complex_chol.conj().T).reshape((nmo,) * 4)
    assert np.allclose(h2e.transpose((0, 1, 3, 2)), eris)


def test_writing_complex_integrals_as_real_is_rejected(integrals, tmp_path):
    nmo, hcore, chol, enuc = integrals
    complex_chol = chol + 1j * np.roll(chol, 1, axis=0)

    with pytest.raises(ValueError, match="complex integrals with cplx=False"):
        write_fcidump(tmp_path / 'FCIDUMP', hcore, complex_chol, enuc, nmo, (2, 2),
                      sym=1, cplx=False)


def test_an_unsupported_symmetry_is_rejected(integrals, tmp_path):
    nmo, hcore, chol, enuc = integrals
    path = tmp_path / 'FCIDUMP'
    write_fcidump(path, hcore, chol, enuc, nmo, (2, 2), sym=1, cplx=False)

    with pytest.raises(ValueError, match="Unsupported permutational symmetry"):
        read_fcidump(path, symmetry=2, verbose=False)


def test_eight_fold_symmetry_writes_each_integral_once(integrals, tmp_path):
    nmo, hcore, chol, enuc = integrals
    path = tmp_path / 'FCIDUMP'
    write_fcidump(path, hcore, chol, enuc, nmo, (2, 2), sym=8, cplx=False, tol=1e-12)

    # read with symmetry=1, so only the entries actually written are populated
    _, h2e, _, _ = read_fcidump(path, symmetry=1, verbose=False)
    i, k, j, l = 0, 1, 2, 3

    written = [c for c in [(i, k, j, l), (j, l, i, k), (k, i, l, j), (l, j, k, i),
                           (k, i, j, l), (l, j, i, k), (i, k, l, j), (j, l, k, i)]
               if abs(h2e[c[0], c[1], c[3], c[2]]) > 0]
    assert len(written) == 1


def test_spinor_basis_doubles_the_orbital_count(integrals, tmp_path):
    nmo, hcore, chol, enuc = integrals
    path = tmp_path / 'FCIDUMP'
    write_fcidump(path, hcore, chol, enuc, nmo, (2, 2), sym=1, cplx=True,
                  use_spinor=True, tol=1e-12)

    assert read_fcidump_header(path)['nbasis'] == 2 * nmo


class TestCheckSym:

    def test_symmetry_one_keeps_everything(self):
        assert all(check_sym((i, k, j, l), 4, 1)
                   for i in range(4) for k in range(4) for j in range(4) for l in range(4))

    @pytest.mark.parametrize("sym", [4, 8])
    def test_exactly_one_of_each_equivalent_set_is_kept(self, sym):
        nmo = 4
        kept = {}
        for i in range(nmo):
            for k in range(nmo):
                for j in range(nmo):
                    for l in range(nmo):
                        if check_sym((i, k, j, l), nmo, sym):
                            kept[(i, k, j, l)] = True

        # every kept index tuple must be the unique representative: applying the
        #   4-fold permutations must land outside the kept set unless it is a fixed point
        for i, k, j, l in kept:
            for other in [(j, l, i, k), (k, i, l, j), (l, j, k, i)]:
                assert other == (i, k, j, l) or other not in kept


class TestSpatToSpin:

    def test_h1_alternates_up_and_down(self):
        h1 = np.arange(4, dtype=np.complex128).reshape(2, 2)
        out = h1_spat2spin(np.zeros((4, 4), dtype=np.complex128), h1)

        assert np.allclose(out[0::2, 0::2], h1)   # up-up block
        assert np.allclose(out[1::2, 1::2], h1)   # down-down block
        assert np.allclose(out[0::2, 1::2], 0.0)  # no up-down coupling

    def test_h2_has_no_opposite_spin_coupling_within_a_pair(self):
        h2 = np.arange(16, dtype=np.complex128).reshape((2,) * 4)
        out = h2_spat2spin(np.zeros((4,) * 4, dtype=np.complex128), h2)

        assert np.allclose(out[0::2, 0::2, 0::2, 0::2], h2)
        assert np.allclose(out[0::2, 1::2, 0::2, 1::2], h2)
        # (up,down| within one pair index is forbidden
        assert np.allclose(out[0, 0, 1, 0], 0.0)
