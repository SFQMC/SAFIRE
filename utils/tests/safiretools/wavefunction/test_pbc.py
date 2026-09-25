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
`NOMSDWavefunction.from_pbc_scf`: the periodic construction path.
"""

import copy
import inspect

import h5py as h5
import numpy as np
import pytest

from safiretools import NOMSDWavefunction, PHMSDWavefunction, SpinSymm, Wavefunction
from safiretools.wavefunction.slater import spin_layout_shape

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


@pytest.fixture(scope='module')
def diamond_krks():
    """A 2x1x1 diamond KRKS calculation, small enough to run in-line."""
    from pyscf.pbc import dft, gto

    cell = gto.Cell()
    alat = 3.6
    cell.a = (np.ones((3, 3)) - np.eye(3)) * alat / 2.0
    cell.atom = (('C', 0, 0, 0), ('C', np.array([0.25, 0.25, 0.25]) * alat))
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [12] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kpts = cell.make_kpts([2, 1, 1])
    mf = dft.KRKS(cell, kpts=kpts)
    mf.kernel()
    return cell, mf, kpts


@pytest.fixture
def degenerate_kmf(diamond_krks):
    """A collinear reference with two partially occupied bands at one k-point."""
    _, mf, _ = diamond_krks
    degenerate = copy.copy(mf.to_uhf())
    occ = np.array(degenerate.mo_occ, dtype=float)
    occ[:, 0, 3] = 0.5
    occ[:, 0, 4] = 0.5
    degenerate.mo_occ = occ

    return degenerate


def datasets(path):
    found = {}
    with h5.File(path, 'r') as fh5:
        fh5.visititems(lambda name, obj: found.__setitem__(name, obj[...])
                       if isinstance(obj, h5.Dataset) else None)
    return found


class TestSingleDeterminant:

    @pytest.mark.parametrize('basis', [None, 'ortho_ao'])
    def test_a_closed_shell_reference(self, diamond_krks, basis):
        _, mf, _ = diamond_krks
        wavefunction = NOMSDWavefunction.from_pbc_scf(mf, basis=basis)

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.spin_symm is SpinSymm.CLOSED
        assert wavefunction.nelec == (8, 8)
        assert wavefunction.nmo == 16
        # a closed-shell determinant carries one spin block
        assert wavefunction.dets.shape == (1, 16, 8)

    def test_a_collinear_reference(self, diamond_krks):
        _, mf, _ = diamond_krks
        wavefunction = NOMSDWavefunction.from_pbc_scf(mf.to_uhf(),
                                                      basis='ortho_ao')

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (8, 8)
        assert spin_layout_shape(wavefunction.dets) == ((1, 16, 8), (1, 16, 8))

    def test_partial_occupancies_still_give_one_determinant(self,
                                                            degenerate_kmf):
        wavefunction = NOMSDWavefunction.from_pbc_scf(degenerate_kmf,
                                                      basis='ortho_ao')

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.ndets == 1

    def test_a_collinear_reference_requires_the_orthogonalized_basis(
            self, diamond_krks):
        _, mf, _ = diamond_krks
        with pytest.raises(ValueError, match="orthogonalized AO basis"):
            NOMSDWavefunction.from_pbc_scf(mf.to_uhf())

    def test_it_round_trips(self, diamond_krks, tmp_path):
        _, mf, _ = diamond_krks
        wavefunction = NOMSDWavefunction.from_pbc_scf(mf)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (8, 8)
        assert read_back.spin_symm is SpinSymm.CLOSED
        assert np.allclose(read_back.dets, wavefunction.dets)


class TestPartialOccupancies:
    """
    There is no multi-determinant route: **CoQuí is the supported path for
    solids.** A metallic PySCF reference still gives a single determinant, over
    the leading configuration of its partially occupied bands.
    """

    def test_the_leading_configuration_is_occupied(self, degenerate_kmf):
        wavefunction = NOMSDWavefunction.from_pbc_scf(degenerate_kmf,
                                                      basis='ortho_ao')

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.ndets == 1
        assert wavefunction.nelec == (8, 8)

    def test_there_is_no_phmsd_route(self):
        assert not hasattr(PHMSDWavefunction, 'from_pbc_scf') \
            or PHMSDWavefunction.from_pbc_scf.__func__ \
            is Wavefunction.from_pbc_scf.__func__

        from safiretools.wavefunction import pbc

        assert 'ndet_max' not in inspect.signature(pbc.from_pbc_scf).parameters

    def test_a_partially_occupied_closed_shell_reference_is_rejected(
            self, diamond_krks):
        _, mf, _ = diamond_krks
        smeared = copy.copy(mf)
        smeared.mo_occ = np.full_like(np.array(mf.mo_occ, dtype=float), 0.9)

        with pytest.raises(ValueError, match="closed-shell reference are"):
            NOMSDWavefunction.from_pbc_scf(smeared)

    def test_too_few_degenerate_bands_is_reported(self, diamond_krks):
        """
        Electrons sitting in bands below `low` still count toward the total but
        are not candidates to occupy, so a heavily smeared reference can leave
        more electrons to place than there are bands to place them in.
        """
        _, mf, _ = diamond_krks
        smeared = copy.copy(mf.to_uhf())
        occ = np.full_like(np.array(smeared.mo_occ, dtype=float), 0.09)
        occ[:, :, 0] = 0.9
        smeared.mo_occ = occ

        with pytest.raises(ValueError, match="remain to be placed over"):
            NOMSDWavefunction.from_pbc_scf(smeared, basis='ortho_ao')


class TestBaseClassDispatch:
    """
    `Wavefunction.from_pbc_scf` reaches the subclass through inheritance, and
    always produces an `NOMSDWavefunction`. See DESIGN.md "Every factory
    dispatches from the base class".
    """

    def test_the_base_factory_gives_an_nomsd(self, diamond_krks):
        _, mf, _ = diamond_krks
        assert isinstance(Wavefunction.from_pbc_scf(mf.to_uhf(),
                                                    basis='ortho_ao'),
                          NOMSDWavefunction)

    def test_the_subclass_alias_agrees(self, diamond_krks, layouts_close):
        _, mf, _ = diamond_krks
        kuhf = mf.to_uhf()
        through_base = Wavefunction.from_pbc_scf(kuhf, basis='ortho_ao')
        through_subclass = NOMSDWavefunction.from_pbc_scf(kuhf, basis='ortho_ao')

        assert layouts_close(through_base.dets, through_subclass.dets)

    def test_the_wrong_representation_is_refused(self, diamond_krks):
        _, mf, _ = diamond_krks
        with pytest.raises(ValueError, match='from_pbc_scf'):
            PHMSDWavefunction.from_pbc_scf(mf.to_uhf(), basis='ortho_ao')
