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
The `Wavefunction` ABC: derived shape, the initial Slater determinant,
orthonormality, format detection, and the file semantics of ``to_hdf5``.
"""

import h5py as h5
import numpy as np
import pytest

from safiretools import (
    LatticeHamiltonian,
    NOMSDWavefunction,
    PHMSDWavefunction,
    SpinSymm,
    Wavefunction,
)
from safiretools.wavefunction.base import (
    is_orthonormal,
    modified_gram_schmidt,
    wavefunction_format,
)


class TestDerivedShape:

    @pytest.mark.parametrize('spin_symm, nspin, npol, per_spin, on_disk', [
        ('closed', 1, 1, (3,), (3, 3)),
        ('collinear', 2, 1, (3, 2), (3, 2)),
        ('noncollinear', 1, 2, (5,), (5, 0)),
    ])
    def test_spin_structure_follows_the_symmetry(self, make_nomsd, spin_symm,
                                                 nspin, npol, per_spin, on_disk):
        nelec = (3, 3) if spin_symm == 'closed' else (3, 2)
        wavefunction = make_nomsd(spin_symm, nelec=nelec)

        assert wavefunction.nspin == nspin
        assert wavefunction.npol == npol
        assert wavefunction.nelec_per_spin == per_spin
        assert wavefunction.nrows == npol * wavefunction.nmo

    @pytest.mark.parametrize('spin_symm, nelec, expected', [
        ('closed', (3, 3), (3, 3)),
        ('collinear', (3, 2), (3, 2)),
        ('collinear', (3, 0), (3, 0)),
        ('noncollinear', (3, 2), (5, 0)),
    ])
    def test_noncollinear_merges_the_spin_channels_on_disk(self, make_nomsd,
                                                           spin_symm, nelec,
                                                           expected):
        assert make_nomsd(spin_symm, nelec=nelec).nelec_on_disk == expected

    def test_a_closed_shell_wavefunction_needs_equal_populations(self,
                                                                 orthonormal):
        with pytest.raises(ValueError, match="equal spin populations"):
            NOMSDWavefunction(coeffs=[1.0], dets=orthonormal(6, 3)[np.newaxis],
                              nelec=(3, 2), spin_symm='closed')

    def test_coeffs_must_be_one_dimensional(self, orthonormal):
        with pytest.raises(ValueError, match="one-dimensional"):
            NOMSDWavefunction(coeffs=[[1.0]],
                              dets=orthonormal(6, 3)[np.newaxis],
                              nelec=(3, 3), spin_symm='closed')

    def test_nelec_must_be_a_pair(self, orthonormal):
        with pytest.raises(ValueError, match=r"\(nup, ndown\) pair"):
            NOMSDWavefunction(coeffs=[1.0], dets=orthonormal(6, 3)[np.newaxis],
                              nelec=(3,), spin_symm='closed')


class TestPsi0:

    def test_it_defaults_to_the_leading_determinant(self, make_nomsd):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), ndets=2)
        alpha, beta = wavefunction.psi0

        assert np.allclose(alpha, wavefunction.dets[0][:, :3])
        assert np.allclose(beta, wavefunction.dets[0][:, 3:])

    def test_the_default_is_a_copy(self, make_nomsd):
        wavefunction = make_nomsd('collinear')
        wavefunction.psi0[0][0, 0] = 1234.0

        assert wavefunction.dets[0][0, 0] != 1234.0

    def test_an_explicit_psi0_is_kept(self, make_nomsd, orthonormal):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), nmo=6)
        psi0 = (orthonormal(6, 3), orthonormal(6, 2))
        wavefunction.psi0 = psi0

        assert np.allclose(wavefunction.psi0[0], psi0[0])
        assert np.allclose(wavefunction.psi0[1], psi0[1])

    def test_it_must_have_one_block_per_spin_channel(self, make_nomsd,
                                                     orthonormal):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), nmo=6)

        with pytest.raises(ValueError, match="one block per spin channel"):
            wavefunction.psi0 = (orthonormal(6, 3),)

    def test_each_block_must_have_the_right_shape(self, make_nomsd,
                                                  orthonormal):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), nmo=6)

        with pytest.raises(ValueError, match="psi0 block 1 has shape"):
            wavefunction.psi0 = (orthonormal(6, 3), orthonormal(6, 3))


class TestOrthonormality:

    def test_gram_schmidt_orthonormalizes(self, rng):
        matrix = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        orthonormalized = modified_gram_schmidt(matrix)

        assert is_orthonormal(orthonormalized)
        # the column space is preserved
        assert np.linalg.matrix_rank(np.hstack([matrix, orthonormalized])) == 3

    def test_gram_schmidt_rejects_linearly_dependent_columns(self):
        matrix = np.ones((4, 2))

        with pytest.raises(ValueError, match="linearly dependent"):
            modified_gram_schmidt(matrix)

    def test_gram_schmidt_rejects_a_non_matrix(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            modified_gram_schmidt(np.zeros(4))

    def test_an_empty_block_counts_as_orthonormal(self):
        assert is_orthonormal(np.zeros((6, 0), dtype=complex))

    def test_orthonormalize_does_not_touch_the_original(self, rng):
        dets = (rng.normal(size=(1, 6, 5))
                + 1j * rng.normal(size=(1, 6, 5)))
        wavefunction = NOMSDWavefunction(coeffs=[1.0], dets=dets, nelec=(3, 2),
                                         spin_symm='collinear')
        before = wavefunction.dets.copy()

        fixed = wavefunction.orthonormalize()

        assert np.array_equal(wavefunction.dets, before)
        assert is_orthonormal(fixed.dets[0][:, :3])
        assert is_orthonormal(fixed.dets[0][:, 3:])

    def test_orthonormalize_leaves_orthonormal_blocks_exactly_alone(self,
                                                                    make_nomsd):
        wavefunction = make_nomsd('collinear', ndets=2)

        assert np.array_equal(wavefunction.orthonormalize().dets,
                              wavefunction.dets)

    def test_writing_warns_about_non_orthonormal_matrices(self, rng, tmp_path):
        dets = rng.normal(size=(1, 6, 5)) + 0j
        wavefunction = NOMSDWavefunction(coeffs=[1.0], dets=dets, nelec=(3, 2),
                                         spin_symm='collinear')

        with pytest.warns(UserWarning, match="not orthonormal"):
            wavefunction.to_hdf5(tmp_path / 'wfn.h5')

    def test_writing_does_not_orthonormalize(self, rng, tmp_path):
        dets = rng.normal(size=(1, 6, 3)) + 0j
        wavefunction = NOMSDWavefunction(coeffs=[1.0], dets=dets, nelec=(3, 3),
                                         spin_symm='closed')
        path = tmp_path / 'wfn.h5'

        with pytest.warns(UserWarning):
            wavefunction.to_hdf5(path)

        assert np.allclose(Wavefunction.from_hdf5(path).dets, dets)


class TestFormatDetection:

    def test_it_names_the_representation_in_the_file(self, make_nomsd,
                                                    make_phmsd, tmp_path):
        nomsd = tmp_path / 'nomsd.h5'
        phmsd = tmp_path / 'phmsd.h5'

        with pytest.warns(UserWarning):
            make_nomsd('collinear').to_hdf5(nomsd)
        make_phmsd().to_hdf5(phmsd)

        assert wavefunction_format(nomsd) == 'nomsd'
        assert wavefunction_format(phmsd) == 'phmsd'

    def test_a_file_with_no_wavefunction_is_rejected(self, tmp_path):
        path = tmp_path / 'empty.h5'
        with h5.File(path, 'w') as fh5:
            fh5['something/else'] = 1

        with pytest.raises(ValueError, match="no wavefunction"):
            wavefunction_format(path)

    def test_from_hdf5_dispatches_to_the_right_subclass(self, make_nomsd,
                                                       make_phmsd, tmp_path):
        nomsd = tmp_path / 'nomsd.h5'
        phmsd = tmp_path / 'phmsd.h5'

        make_nomsd('closed', nelec=(3, 3)).to_hdf5(nomsd)
        make_phmsd().to_hdf5(phmsd)

        assert isinstance(Wavefunction.from_hdf5(nomsd), NOMSDWavefunction)
        assert isinstance(Wavefunction.from_hdf5(phmsd), PHMSDWavefunction)

    def test_a_subclass_refuses_the_other_representation(self, make_phmsd,
                                                        tmp_path):
        path = tmp_path / 'phmsd.h5'
        make_phmsd().to_hdf5(path)

        with pytest.raises(ValueError, match="NOMSDWavefunction does not read"):
            NOMSDWavefunction.from_hdf5(path)


class TestFileSemantics:
    """
    ``to_hdf5`` replaces the wavefunction in its target file, not the whole
    file, so a Hamiltonian and a wavefunction can share one file in either
    order.
    """

    @pytest.fixture
    def hamiltonian(self):
        return LatticeHamiltonian.from_dict({
            'lattice': dict(L1=2, L2=2, boundary1='pbc', boundary2='pbc'),
            'hamiltonian': dict(t=1.0, U=4.0, nelec=(2, 2)),
        })

    def test_the_file_is_created_when_absent(self, make_nomsd, tmp_path):
        path = tmp_path / 'fresh.h5'
        make_nomsd('closed', nelec=(3, 3)).orthonormalize().to_hdf5(path)

        assert path.exists()
        assert wavefunction_format(path) == 'nomsd'

    def test_a_hamiltonian_written_first_survives(self, hamiltonian,
                                                 make_nomsd, tmp_path):
        path = tmp_path / 'both.h5'
        hamiltonian.to_hdf5(path)
        make_nomsd('closed', nelec=(3, 3)).orthonormalize().to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert 'Hamiltonian/ModelHamiltonian/number_of_components' in fh5
            assert 'Wavefunction/NOMSD/dims' in fh5

    def test_a_hamiltonian_written_second_survives(self, hamiltonian,
                                                  make_nomsd, tmp_path):
        path = tmp_path / 'both.h5'
        make_nomsd('closed', nelec=(3, 3)).orthonormalize().to_hdf5(path)
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert 'Hamiltonian/ModelHamiltonian/number_of_components' in fh5
            assert 'Wavefunction/NOMSD/dims' in fh5

    def test_rewriting_replaces_the_wavefunction(self, make_nomsd, tmp_path):
        path = tmp_path / 'replace.h5'
        make_nomsd('collinear', nelec=(3, 2), ndets=3).orthonormalize() \
            .to_hdf5(path)
        make_nomsd('closed', nelec=(3, 3)).orthonormalize().to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            group = fh5['Wavefunction/NOMSD']
            # the three-determinant collinear wavefunction left six PsiT groups
            assert sorted(name for name in group if name.startswith('PsiT')) \
                == ['PsiT_0']
            assert 'Psi0_beta' not in group

        assert Wavefunction.from_hdf5(path).ndets == 1


class TestNoLengthBasedDispatch:
    """
    afqmctools' ``write_wfn`` decided the representation from ``len(wfn)`` — 2
    for NOMSD, 3 for PHMSD. The type split replaces that entirely.
    """

    def test_no_module_infers_a_representation_from_a_length(self):
        import re
        from pathlib import Path

        import safiretools.wavefunction as package

        # `len(x) == 2` / `== 3` on a wavefunction is the idiom being replaced
        idiom = re.compile(r'len\(\s*wfn\s*\)\s*==\s*[23]')
        offenders = []

        for source in Path(package.__file__).parent.glob('*.py'):
            if idiom.search(source.read_text()):
                offenders.append(source.name)

        assert offenders == []

    def test_a_single_determinant_still_needs_its_leading_axis(self,
                                                              orthonormal):
        # nothing sniffs a shape to guess what it was handed: a determinant
        #   array is always (ndets, nrows, ncols), and a bare matrix is an error
        with pytest.raises(ValueError, match="dets must have shape"):
            NOMSDWavefunction(coeffs=[1.0], dets=orthonormal(6, 3),
                              nelec=(3, 3), spin_symm='closed')

    def test_occupation_numbers_are_not_accepted_as_determinants(self):
        # afqmctools distinguished (coeffs, dets) from (coeffs, occa, occb) by
        #   tuple length; here the occupation numbers only reach PHMSD
        occa = np.array([[0, 1, 2]])
        occb = np.array([[0, 1]])

        with pytest.raises(ValueError, match="dets must have shape"):
            NOMSDWavefunction(coeffs=[1.0], dets=occa, nelec=(3, 2),
                              spin_symm='collinear')

        wavefunction = PHMSDWavefunction(coeffs=[1.0], occa=occa, occb=occb,
                                         nmo=6)
        assert wavefunction.nelec == (3, 2)

    def test_the_representations_write_to_distinct_groups(self, make_nomsd,
                                                         make_phmsd, tmp_path):
        assert NOMSDWavefunction._HDF5_GROUP == 'NOMSD'
        assert PHMSDWavefunction._HDF5_GROUP == 'PHMSD'

        path = tmp_path / 'wfn.h5'
        make_phmsd().to_hdf5(path)
        with h5.File(path, 'r') as fh5:
            assert list(fh5['Wavefunction']) == ['PHMSD']


class TestSpinSymmCoercion:

    @pytest.mark.parametrize('value, expected', [
        ('uhf', SpinSymm.COLLINEAR),
        ('rhf', SpinSymm.CLOSED),
        ('ghf', SpinSymm.NONCOLLINEAR),
        (2, SpinSymm.COLLINEAR),
        (SpinSymm.CLOSED, SpinSymm.CLOSED),
    ])
    def test_the_constructor_coerces_its_spin_symmetry(self, orthonormal, value,
                                                       expected):
        npol = 2 if expected is SpinSymm.NONCOLLINEAR else 1
        ncols = 3 if expected is SpinSymm.CLOSED else 6
        wavefunction = NOMSDWavefunction(
            coeffs=[1.0], dets=orthonormal(npol * 6, ncols)[np.newaxis],
            nelec=(3, 3), spin_symm=value, nmo=6)

        assert wavefunction.spin_symm is expected

    def test_a_polarized_wavefunction_is_collinear_with_no_beta_electrons(
            self, make_nomsd):
        # the replacement for what FULLYPOLARIZED used to encode
        wavefunction = make_nomsd('collinear', nelec=(3, 0), nmo=6)

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec_per_spin == (3, 0)
        assert wavefunction.nelec_on_disk == (3, 0)
        assert wavefunction.psi0[1].shape == (6, 0)

