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
Defining custom pytest fixtures, setting up test logger
"""
import importlib
import os
import logging

import pytest
import numpy as np
import h5py as h5

# PySCF is officially an optional package
USE_PYSCF = importlib.util.find_spec('pyscf') or False
if USE_PYSCF:
    from pyscf import gto,scf,mcscf
    from afqmctools.utils.pyscf_utils import ci2chk

from afqmctools.utils.linalg import modified_cholesky_direct, get_ortho_ao
import afqmctools.systems.lattice as lat
from afqmctools.hamiltonian.model.director import HamiltonianDirector


def pytest_addoption(parser):
    parser.addoption("--afqmc-runmode",action="store",default=None,help="Runmode for AFQMC tests. (options are slurm_cpu, slurm_gpu, mpi_cpu, mpi_gpu)")
    parser.addoption("--afqmc-timeout",action="store",default=None,help="Timeout in minutes for AFQMC tests.")
    
@pytest.fixture(scope="session")
def afqmc_runmode(request):
    runmode = request.config.getoption("--afqmc-runmode")
    if runmode is None:
        runmode = "slurm_cpu"
    return runmode

@pytest.fixture(scope="session")
def timeout_mins(request):
    timeout = request.config.getoption("--afqmc-timeout")
    if timeout is None:
        timeout = 5
    return timeout

# TODO: configure CMake to configure the default AFQMC_EXEC and AFQMC_TEST_FILES
@pytest.fixture(scope='session')
def afqmc_exec():
    # TODO: get default location for AFQMC_EXEC
    return os.environ.get('AFQMC_EXEC')

@pytest.fixture(scope='session')
def test_files():
    # TODO: get default location for AFQMC_TEST_FILES
    return os.environ.get('AFQMC_TEST_FILES')

@pytest.fixture(scope='session')
def num_mpi_tasks():
    slurm_mpi_tasks = os.environ.get('SLURM_NTASKS')
    num_mpi_tasks = os.environ.get('NUM_MPI_TASKS')
    if num_mpi_tasks is not None:
        return int(num_mpi_tasks)
    elif slurm_mpi_tasks is not None:
        return int(slurm_mpi_tasks)
    else:
        return 1

@pytest.fixture(scope='session')
def my_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('pytest_logger.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)-8s] %(funcName)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class FakeComm:
    def __init__(self, size=1):
        self.size = size
        self.rank = 0

@pytest.fixture
def fake_comm():
    return FakeComm(16)


@pytest.fixture
def volatile_test_hdf5(tmp_path):
    '''
    a "factory as fixture" to generate temporary test HDF5 files
    that may (and most likely will) change between tests
    '''

    def _volatile_test_hdf5(name='test.h5',mode='w',data=None):

        my_hdf5 = tmp_path / name    

        with h5.File(my_hdf5, mode=mode) as f:
            if data is not None and mode in ['w','r+','a']:
                for key,val in data.items():                  
                    f.create_dataset(key,data=val)
        return my_hdf5

    return _volatile_test_hdf5


def random_hamiltonian(nmo=17,cplx=True,sym=8,seed=7):
    """
    Generate a random Hamiltonian for tests. Once generated, 
        the Hamiltonian instance will be the same across
        all tests in the TestConverter class. 
    """
    np.random.seed(seed)
    h1e = np.random.random((nmo,nmo))
    if cplx:
        h1e = h1e + 1j*np.random.random((nmo,nmo))
    eri = np.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    if cplx:
        eri = eri + 1j*np.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    # Restore symmetry to the integrals.
    if sym >= 4:
        # (ik|jl) = (jl|ik)
        # (ik|jl) = (ki|lj)*
        eri = eri + eri.transpose(2,3,0,1)
        eri = eri + eri.transpose(3,2,1,0).conj()
    if sym == 8:
        eri = eri + eri.transpose(1,0,2,3)
    # Construct hermitian matrix M_{ik,lj}.
    eri = eri.transpose((0,1,3,2))
    eri = eri.reshape((nmo*nmo,nmo*nmo))
    # Make positive semi-definite.
    eri = np.dot(eri,eri.conj().T)
    chol = modified_cholesky_direct(eri, tol=1e-4, verbose=False)
    chol = chol.reshape((-1,nmo,nmo))
    enuc = np.random.rand()
    return h1e, chol, enuc, eri


@pytest.fixture(scope='session')
def random_real_hamiltonian(my_logger):
    """
    Generate a random Hamiltonian for tests. Once generated, 
        the Hamiltonian instance will be the same across
        all tests in the TestConverter class. 
    """
    my_logger.info("making random real-valued Hamiltonian")
    return random_hamiltonian(
        nmo=17,
        cplx=False,
        sym=8,
        seed=7
        )


@pytest.fixture(scope='session')
def random_cplx_hamiltonian(my_logger):
    """
    Generate a random Hamiltonian for tests. Once generated, 
        the Hamiltonian instance will be the same across
        all tests in the TestConverter class. 
    """
    my_logger.info("making random complex-valued Hamiltonian")
    return random_hamiltonian(
        nmo=17,
        cplx=True,
        sym=4,
        seed=7
        )

# TODO: reorganize such that all PySCF-related tests live within their own subdir with their own conftest.
#         then we can simply use `pyscf = pytest.importskip(pyscf)` to skip the entire
@pytest.fixture(scope='session')
def neon_atom():
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return gto.M(atom='Ne 0 0 0', basis='sto-3g',parse_arg=False)


@pytest.fixture(scope='session')
def neon_atom_dz():
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return gto.M(atom='Ne 0 0 0', basis='aug-ccpvdz',parse_arg=False)


@pytest.fixture(scope='session')
def neon_atom_tz():
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return gto.M(atom='Ne 0 0 0', basis='aug-ccpvtz',parse_arg=False)


@pytest.fixture(scope='session')
def neon_rhf(tmp_path_factory,neon_atom):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    mf = scf.RHF(neon_atom)
    mf.chkfile = tmp_path_factory.mktemp('scf') / 'scf.chk'
    energy = mf.kernel()
    return mf, energy


@pytest.fixture(scope='session')
def neon_rhf_tz(tmp_path_factory,neon_atom_tz):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    mf = scf.RHF(neon_atom_tz)
    mf.chkfile = tmp_path_factory.mktemp('scf') / 'scf.chk'
    energy = mf.kernel()
    return mf, energy


@pytest.fixture(scope='session')
def neon_casscf(neon_rhf):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    mf,_ = neon_rhf
    mc = mcscf.CASSCF(mf, 4, (4,4))
    mc.kernel()

    ci2chk(mc.chkfile, ci=mc.ci)
    
    return mc


@pytest.fixture(scope='session')
def neon_eri(neon_atom):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return neon_atom.intor('int2e', aosym='s1')


@pytest.fixture(scope='session')
def neon_eri_dz(neon_atom_dz):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return neon_atom_dz.intor('int2e', aosym='s1')


@pytest.fixture(scope='session')
def diamond():
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")

    from pyscf.pbc import gto

    cell = gto.Cell()
    alat0 = 3.6
    cell.a = (np.ones((3,3))-np.eye(3))*alat0/2.0
    cell.atom = (('C',0,0,0),('C',np.array([0.25,0.25,0.25])*alat0))
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [12]*3  # 10 grids on postive x direction, => 21^3 grids in total
    cell.verbose = 0
    cell.build(parse_arg=False)
    return cell


@pytest.fixture(scope='session')
def diamond_lda(diamond,tmp_path_factory):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    from pyscf.pbc import dft

    cell = diamond
    nk = [2,1,1]
    kpts = cell.make_kpts(nk)

    mf = dft.KRKS(cell,kpts=kpts)
    mf.chkfile = tmp_path_factory.mktemp('data') /'scf.dump'
    mf.kernel()

    return mf,kpts


@pytest.fixture(scope='session')
def diamond_lda_k221(diamond,tmp_path_factory):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    from pyscf.pbc import dft

    cell = diamond
    nk = [2,2,1]
    kpts = cell.make_kpts(nk)

    mf = dft.KRKS(cell,kpts=kpts)
    mf.chkfile = tmp_path_factory.mktemp('data') /'scf.dump'
    mf.kernel()

    return mf,kpts


@pytest.fixture(scope='session')
def diamond_lda_k221_orthao(diamond_lda_k221):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    
    mf,kpts = diamond_lda_k221

    X, nmo_per_kpt = get_ortho_ao(mf.cell,kpts)
    hcore = mf.get_hcore()
    fock = hcore + mf.get_veff()

    with h5.File(mf.chkfile, 'a') as f:
        f['scf/orthoAORot'] = X
        f['scf/nmo_per_kpt'] = nmo_per_kpt
        f['scf/hcore'] = hcore
        f['scf/fock'] = fock

    return mf,kpts


@pytest.fixture(scope='session')
def oxygen():
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    return gto.M(
        atom="O 0. 0. 0.",
        spin=2,
        basis='ccpvdz',
        parse_arg=False
    )    


@pytest.fixture(scope='session')
def oxygen_rohf(oxygen,tmp_path_factory):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    mf = scf.ROHF(oxygen)
    mf.chkfile = tmp_path_factory.mktemp('scf') / 'rohf.chk'
    energy = mf.kernel()
    return mf, energy


@pytest.fixture(scope='session')
def oxygen_uhf(oxygen_rohf,tmp_path_factory):
    if not USE_PYSCF:
        pytest.skip("pyscf is not installed")
    mf,_ = oxygen_rohf
    mf = mf.to_uhf()
    mf.chkfile = tmp_path_factory.mktemp('scf') / 'uhf.chk'
    energy = mf.kernel()
    return mf, energy


@pytest.fixture(scope='session')
def hubbard_kanamori_6x1():
    return HamiltonianDirector(
        source={"hamiltonian" : dict(
            U=2.0,
            U1=1.5,
            U2=1.0,
            J=0.5,
            nbands=2
        )},
        lattice=lat.SquareLattice(L=(6,1),axis1_boundary=lat.PBCBoundary),
    ).build()

