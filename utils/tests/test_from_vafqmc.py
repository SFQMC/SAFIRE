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
Tests for utils/afqmctools/inputs/from_vafqmc.py.

`hafqmc` is an optional dependency (the VAFQMC extra), so the whole module is
skipped without it.
"""

import h5py as h5
import numpy as np
import pytest

pytest.importorskip("hafqmc")

from afqmctools.inputs.from_vafqmc import (
    _chol_to_safire_layout,
    infer_walker_type,
    is_spin_doubled,
    spin_mixing_magnitude,
    to_polarized_operator,
    to_polarized_orbitals,
    write_hamiltonian_dense,
    write_nomsd_wfn,
)
from afqmctools.utils.io import from_complex


NMO, NCHOL, NA, NB = 6, 5, 3, 2


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


class TestCholeskyLayout:

    def test_rank3_is_transposed_to_flat(self, rng):
        chol = rng.standard_normal((NCHOL, NMO, NMO))
        out = _chol_to_safire_layout(chol)
        assert out.shape == (NMO * NMO, NCHOL)
        for k in range(NCHOL):
            assert np.array_equal(out[:, k], chol[k].ravel())

    def test_rank2_passes_through(self, rng):
        chol = rng.standard_normal((NMO * NMO, NCHOL))
        assert np.array_equal(_chol_to_safire_layout(chol), chol)

    def test_bad_rank_raises(self, rng):
        with pytest.raises(ValueError):
            _chol_to_safire_layout(rng.standard_normal(NMO))


class TestPolarizedEmbedding:

    def test_operator_is_block_diagonal(self, rng):
        a = rng.standard_normal((NMO, NMO))
        out = to_polarized_operator(a, NMO)
        assert out.shape == (2 * NMO, 2 * NMO)
        assert np.array_equal(out[:NMO, :NMO], a)
        assert np.array_equal(out[NMO:, NMO:], a)
        assert not out[:NMO, NMO:].any()
        assert not out[NMO:, :NMO].any()

    def test_stacked_operator_keeps_leading_axis(self, rng):
        a = rng.standard_normal((NCHOL, NMO, NMO))
        out = to_polarized_operator(a, NMO)
        assert out.shape == (NCHOL, 2 * NMO, 2 * NMO)

    def test_already_doubled_is_a_noop(self, rng):
        a = rng.standard_normal((2 * NMO, 2 * NMO))
        assert to_polarized_operator(a, NMO) is a

    def test_v0_is_reproduced_on_the_doubled_space(self, rng):
        # blockdiag(L, L)**2 == blockdiag(L**2, L**2), so SAFIRE's internally re-added
        # v0 = -1/2 sum_k L_k L_k is the same operator before and after embedding.
        chol = rng.standard_normal((NCHOL, NMO, NMO))
        v0 = -0.5 * np.einsum("kpr,krs->ps", chol, chol)
        pol = to_polarized_operator(chol, NMO)
        v0_pol = -0.5 * np.einsum("kpr,krs->ps", pol, pol)
        assert np.allclose(v0_pol, to_polarized_operator(v0, NMO))

    def test_bad_dimension_raises(self, rng):
        with pytest.raises(ValueError):
            to_polarized_operator(rng.standard_normal((NMO + 1, NMO + 1)), NMO)

    def test_orbitals_are_block_diagonal(self, rng):
        wa = rng.standard_normal((NMO, NA))
        wb = rng.standard_normal((NMO, NB))
        out = to_polarized_orbitals(wa, wb, NMO)
        assert out.shape == (2 * NMO, NA + NB)
        assert np.array_equal(out[:NMO, :NA], wa)
        assert np.array_equal(out[NMO:, NA:], wb)
        assert not out[NMO:, :NA].any()
        assert not out[:NMO, NA:].any()


class TestSpinDetection:

    def test_is_spin_doubled(self, rng):
        assert not is_spin_doubled(rng.standard_normal((NMO, NMO)),
                                   rng.standard_normal((NCHOL, NMO, NMO)), NMO)
        assert is_spin_doubled(rng.standard_normal((2 * NMO, 2 * NMO)),
                               rng.standard_normal((NCHOL, 2 * NMO, 2 * NMO)), NMO)

    def test_offdiag_ratio_is_zero_for_a_block_diagonal_operator(self, rng):
        hmf = to_polarized_operator(rng.standard_normal((NMO, NMO)), NMO)
        vhs = to_polarized_operator(rng.standard_normal((NCHOL, NMO, NMO)), NMO)
        mags = spin_mixing_magnitude(hmf, vhs, NMO)
        assert mags["offdiag_ratio"] == 0.0
        assert mags["vhs_aa"] > 0.0


class TestInferWalkerType:
    """Electron counts alone cannot see a broken-symmetry singlet."""

    def test_explicit_request_is_honoured(self, rng):
        w = rng.standard_normal((NMO, NA))
        assert infer_walker_type((NA, NA), "closed", w, w + 1.0) == "closed"

    def test_unequal_counts_are_collinear(self, rng):
        w = rng.standard_normal((NMO, NA))
        assert infer_walker_type((NA, NB), "auto", w, w) == "collinear"

    def test_identical_orbitals_are_closed(self, rng):
        w = rng.standard_normal((NMO, NA))
        assert infer_walker_type((NA, NA), "auto", w, w.copy()) == "closed"

    def test_broken_symmetry_singlet_is_collinear(self, rng):
        # na == nb, so the count-based answer would be "closed" and would silently
        # replace the beta orbitals with the alpha ones.
        w = rng.standard_normal((NMO, NA))
        assert infer_walker_type((NA, NA), "auto", w, w + 1e-3) == "collinear"

    def test_falls_back_to_counts_without_orbitals(self):
        assert infer_walker_type((NA, NA), "auto") == "closed"


class TestWriteHamiltonianDense:

    def test_dims_and_layout(self, tmp_path, rng):
        hcore = rng.standard_normal((NMO, NMO))
        chol = rng.standard_normal((NCHOL, NMO, NMO))
        fname = tmp_path / 'ham.h5'
        write_hamiltonian_dense(hcore, chol, (NA, NB), NMO, 1.5, fname)

        with h5.File(fname, 'r') as fh5:
            dims = fh5['Hamiltonian/dims'][()]
            assert list(dims) == [0, 0, 0, NMO, NA, NB, 0, NCHOL]
            assert np.array_equal(fh5['Hamiltonian/hcore'][()], hcore)
            assert fh5['Hamiltonian/DenseFactorized/L'].shape == (NMO * NMO, NCHOL)
            assert np.array_equal(fh5['Hamiltonian/Energies'][()], np.array([1.5, 0.0]))
            assert fh5['Hamiltonian/ComplexIntegrals'][()][0] == 0
            assert 'Hamiltonian/X' not in fh5

    def test_hermitian_complex_hcore_survives(self, tmp_path, rng):
        a = rng.standard_normal((NMO, NMO)) + 1j * rng.standard_normal((NMO, NMO))
        hcore = a + a.conj().T
        chol = rng.standard_normal((NCHOL, NMO, NMO))
        fname = tmp_path / 'ham.h5'
        write_hamiltonian_dense(hcore, chol, (NA, NB), NMO, 0.0, fname)

        with h5.File(fname, 'r') as fh5:
            stored = fh5['Hamiltonian/hcore'][()]
        assert np.array_equal(from_complex(stored, shape=(NMO, NMO)), hcore)

    def test_complex_cholesky_sets_the_flag(self, tmp_path, rng):
        chol = (rng.standard_normal((NCHOL, NMO, NMO))
                + 1j * rng.standard_normal((NCHOL, NMO, NMO)))
        fname = tmp_path / 'ham.h5'
        write_hamiltonian_dense(rng.standard_normal((NMO, NMO)), chol,
                                (NA, NB), NMO, 0.0, fname)

        with h5.File(fname, 'r') as fh5:
            assert fh5['Hamiltonian/ComplexIntegrals'][()][0] == 1
            assert fh5['Hamiltonian/DenseFactorized/L'].shape == (NMO * NMO, NCHOL, 2)

    def test_ortho_is_written(self, tmp_path, rng):
        ortho = rng.standard_normal((NMO, NMO))
        fname = tmp_path / 'ham.h5'
        write_hamiltonian_dense(rng.standard_normal((NMO, NMO)),
                                rng.standard_normal((NCHOL, NMO, NMO)),
                                (NA, NB), NMO, 0.0, fname, ortho=ortho)
        with h5.File(fname, 'r') as fh5:
            assert np.array_equal(fh5['Hamiltonian/X'][()], ortho)

    def test_rewrite_truncates_rather_than_merging(self, tmp_path, rng):
        # write_dense opens in append mode; these conversions are re-run in place, so a
        # stale wider Cholesky block must not survive underneath a narrower one.
        fname = tmp_path / 'ham.h5'
        write_hamiltonian_dense(rng.standard_normal((NMO, NMO)),
                                rng.standard_normal((NCHOL + 3, NMO, NMO)),
                                (NA, NB), NMO, 0.0, fname)
        write_hamiltonian_dense(rng.standard_normal((NMO, NMO)),
                                rng.standard_normal((NCHOL, NMO, NMO)),
                                (NA, NB), NMO, 0.0, fname)
        with h5.File(fname, 'r') as fh5:
            assert fh5['Hamiltonian/DenseFactorized/L'].shape == (NMO * NMO, NCHOL)
            assert fh5['Hamiltonian/dims'][()][-1] == NCHOL


class TestWriteNomsdWfn:

    def _read(self, fname):
        with h5.File(fname, 'r') as fh5:
            g = fh5['Wavefunction/NOMSD']
            return {k: g[k][()] for k in ('dims',)}, sorted(g.keys())

    def test_closed_writes_one_spin_sector(self, tmp_path, rng):
        w = rng.standard_normal((NMO, NA))
        fname = tmp_path / 'wfn.h5'
        write_nomsd_wfn(fname, w, w, (NA, NA), walker_type='closed')
        info, keys = self._read(fname)
        assert list(info['dims']) == [NMO, NA, NA, 1, 1]
        assert 'PsiT_0' in keys and 'PsiT_1' not in keys
        assert 'Psi0_beta' not in keys

    def test_collinear_writes_both_spin_sectors(self, tmp_path, rng):
        fname = tmp_path / 'wfn.h5'
        write_nomsd_wfn(fname, rng.standard_normal((NMO, NA)),
                        rng.standard_normal((NMO, NB)), (NA, NB),
                        walker_type='collinear')
        info, keys = self._read(fname)
        assert list(info['dims']) == [NMO, NA, NB, 2, 1]
        assert 'PsiT_0' in keys and 'PsiT_1' in keys
        assert 'Psi0_beta' in keys

    def test_polarized_still_writes_the_empty_beta_sector(self, tmp_path, rng):
        # SAFIRE's reader indexes PsiT_{2*idet+1} for every determinant whenever the
        # walker type is COLLINEAR, so a zero-column beta block must be on file.
        nel = NA + NB
        w = rng.standard_normal((2 * NMO, nel))
        fname = tmp_path / 'wfn.h5'
        write_nomsd_wfn(fname, w, w[:, :0], (nel, 0), walker_type='collinear')
        info, keys = self._read(fname)
        assert list(info['dims']) == [2 * NMO, nel, 0, 2, 1]
        assert 'PsiT_1' in keys
        with h5.File(fname, 'r') as fh5:
            assert fh5['Wavefunction/NOMSD/Psi0_beta'].shape[1] == 0

    def test_uhf_is_an_alias_for_collinear(self, tmp_path, rng):
        wa, wb = rng.standard_normal((NMO, NA)), rng.standard_normal((NMO, NB))
        a, b = tmp_path / 'a.h5', tmp_path / 'b.h5'
        write_nomsd_wfn(a, wa, wb, (NA, NB), walker_type='uhf')
        write_nomsd_wfn(b, wa, wb, (NA, NB), walker_type='collinear')
        with h5.File(a) as fa, h5.File(b) as fb:
            assert np.array_equal(fa['Wavefunction/NOMSD/dims'][()],
                                  fb['Wavefunction/NOMSD/dims'][()])

    def test_orthonormalizes_by_default(self, tmp_path, rng):
        w = rng.standard_normal((NMO, NA))
        fname = tmp_path / 'wfn.h5'
        write_nomsd_wfn(fname, w, w, (NA, NA), walker_type='closed')
        with h5.File(fname) as fh5:
            psi = from_complex(fh5['Wavefunction/NOMSD/Psi0_alpha'][()], shape=(NMO, NA))
        assert np.allclose(psi.conj().T @ psi, np.eye(NA))

    def test_orthonormalize_can_be_disabled(self, tmp_path, rng):
        w = rng.standard_normal((NMO, NA))
        fname = tmp_path / 'wfn.h5'
        write_nomsd_wfn(fname, w, w, (NA, NA), walker_type='closed', orthonormalize=False)
        with h5.File(fname) as fh5:
            psi = from_complex(fh5['Wavefunction/NOMSD/Psi0_alpha'][()], shape=(NMO, NA))
        assert np.allclose(psi, w)

    def test_noncollinear_is_rejected(self, tmp_path, rng):
        w = rng.standard_normal((NMO, NA))
        with pytest.raises(ValueError, match="spin-orbital"):
            write_nomsd_wfn(tmp_path / 'wfn.h5', w, w, (NA, NA), walker_type='ghf')
