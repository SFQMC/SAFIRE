# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import pytest

pytest.importorskip("pyscf")
pytest.importorskip("mpi4py")
from mpi4py import MPI

from afqmctools.inputs.from_pyscf import pyscf_to_afqmc
from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.wavefunction.converter import read_wavefunction


@pytest.mark.xfail(reason="PySCF PBC interface has not been updated for new SlaterType enum")
@pytest.mark.mpi
@pytest.mark.parametrize("use_kpoint_sym",[True,False])
class TestDiamond:
    '''
    Workflow tests based on Diamond
    
    Attributes:
    - `test_files` : a dictionary for holding the 
                       the path where temporary 
                       test files are located. The
                       intention is to allow writing
                       and reading to occur in separate 
                       tests.

    Currently only tests that everythings runs without errors,
      we want to varify that we read the same stuff that we tried 
      to write.
    '''
    @pytest.fixture(scope="session")
    def run_dir(self,tmp_path_factory):
        return tmp_path_factory.mktemp(f"test_write_diamond")


    @pytest.fixture
    def run_file(self,run_dir,use_kpoint_sym):
        return run_dir / f"ukp{use_kpoint_sym}_afqmc.h5"


    def test_write(self,run_file,diamond_lda_k221_orthao,use_kpoint_sym):
        
        mf,_ = diamond_lda_k221_orthao

        pyscf_to_afqmc(
            chkfile=mf.chkfile,
            hamil_file=run_file,
            threshold=1e-4,
            comm=MPI.COMM_WORLD,
            ortho_ao=True,
            kpoint=use_kpoint_sym,
            verbose=True
            )
        assert True


    def test_read_hamiltonian(self,run_file):
        # now read it!!
        hamil = read_hamiltonian(run_file)
        assert hamil is not None


    def test_read_wavefunction(self,run_file):
        # now read it!!
        wfn = read_wavefunction(run_file)
        assert wfn is not None


