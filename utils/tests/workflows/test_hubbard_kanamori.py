# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pytest
import h5py as h5
import numpy as np

from afqmctools.utils.io import write_model_hamiltonian,read_one_body
from afqmctools.wavefunction.model import write_free_electron_wfn
from afqmctools.wavefunction.converter import read_wavefunction
from afqmctools.hamiltonian.model.ham_class import SpinSymm
from afqmctools.observables.greens import greens_1body

TEST_ROOT = Path(__file__).resolve().parent.parent

def _h5_are_same(fname1,fname2,datasets=None):
    """
    compare `datasets` (by name) in hdf5 files 
        fname1 and fname2
    """
    if not datasets:
        raise ValueError

    are_same = list()

    with h5.File(fname1,'r') as f1:
        with h5.File(fname2,'r') as f2:
            for dataset in datasets:
                data1 = f1[dataset][...]
                data2 = f2[dataset][...]
                
                if data1.shape != data2.shape:
                    are_same.append(False)
                    continue

                if (data1 != data2).any():
                    are_same.append(False)
                else:
                    are_same.append(True)

                print(f"Does dataset {dataset} match? {are_same[-1]}")

    return all(are_same)


class TestHubbardKanamori:
    """
    Workflow tests based on the Hubbard-Kanamori model
        in 1-D (a 6x1 lattice with PBCs) inspired by 
        the results in: 

    Phys. Rev. B 99, 235142 (preprint:https://arxiv.org/abs/1902.01463v1)
    
    Test goals:
    - generate the Hubbard-Kanamori (HK) Hamiltonian
        - check that it has the correct terms?
    - write the HK Hamiltonian and compare with a known good hdf5 file
    """
    @pytest.fixture(scope="session")
    def run_file(self,tmp_path_factory):
        return tmp_path_factory.mktemp("test_write_model") / f"afqmc.h5"

    @pytest.mark.parametrize(
        "component_key,field,expected",
        [
            ('tij','model_type','one_body'),
            ('tij','spin_symm',SpinSymm.COLLINEAR),
            ('Uij','model_type','hubbard_u'),
            ('Uij','spin_symm',SpinSymm.COLLINEAR),
            ('Jij','model_type','hubbard_j'),
            ('Jij','spin_symm',SpinSymm.COLLINEAR)
        ]
    )
    def test_components(self,hubbard_kanamori_6x1,component_key,field,expected):
        """
        Test that the Hamiltonian has the expected components
            and metadata. Matrix elements are checked against
            a pre-built file in a later test.
        """
        assert getattr(hubbard_kanamori_6x1[component_key][0],field) == expected

    @pytest.mark.parametrize(
        "component_key,field,expected",
        [
            ('Uij','hst_type','discrete_spin'),
            ('Jij','hst_type','continuous_spin')
        ]
    )
    def test_metadata(self,hubbard_kanamori_6x1,component_key,field,expected):
        assert hubbard_kanamori_6x1[component_key][0].metadata[field] == expected

    def test_write_hamiltonian(self,run_file,hubbard_kanamori_6x1):
        write_model_hamiltonian(
            hamiltonian=hubbard_kanamori_6x1,
            fname=run_file,
            nelec=(6,6)
        )

        prefixes = [
            '/Hamiltonian/ModelHamiltonian/ModelComponent_0/',
            '/Hamiltonian/ModelHamiltonian/ModelComponent_1/',
            '/Hamiltonian/ModelHamiltonian/ModelComponent_2/'
        ]
        model_types = [
            'tij/',
            'Jij/',
            'Uij/'
        ]
        model_datasets = [
            'model_type',
            'spin_type',
            'data_',
            'jdata_',
            'pointers_begin_',
            'pointers_end_',
        ]
        comparison_list = []
        for prefix,model_type in zip(prefixes,model_types):
            comparison_list.extend(
                [prefix + model_type + dataset if dataset.endswith('_') 
                    else prefix + dataset for dataset in model_datasets]
            )
            if model_type != 'tij/':
                comparison_list.append(prefix + 'hst_type')
            comparison_list.append(prefix + model_type + 'dims')

        assert _h5_are_same(
            run_file,
            (TEST_ROOT/'data'/'afqmc_hk_6x1.h5').as_posix(),
            datasets=comparison_list
        )


    def test_write_wavefunction(self,run_file):
        write_free_electron_wfn(
            hamiltonian_fname=run_file,
            nelec=(6,6),
            spin_symm='noncollinear'
        )

        wfn_test, psi0_test, nelec_test, _ = read_wavefunction(
            run_file
        )

        wfn_ref, psi0_ref, nelec_ref, _ = read_wavefunction(
            (TEST_ROOT/'data'/'afqmc_hk_6x1.h5').as_posix()
        )

        assert nelec_test == nelec_ref

        nalpha,nbeta = nelec_ref

        K = read_one_body(run_file).toarray()
        norbitals = K.shape[-1]
        Kup = K[:norbitals]
        Kdown = K[norbitals:]
        K_noncollinear = np.block(
            [[Kup,np.zeros_like(Kup)],
             [np.zeros_like(Kdown),Kdown]]
        )
        E = np.einsum(
            'ij,ji->',
            greens_1body(
            psi_left=wfn_test[1][0][:,:nalpha],
            psi_right=wfn_ref[1][0][:,:nalpha]
            ),
            K_noncollinear
        )

        assert np.isclose(E,-16.0)

