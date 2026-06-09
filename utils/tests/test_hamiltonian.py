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
Tests for the utils/afqmctools/hamiltonian/* tooling.

Author: Kyle Eskridge
GitHub: bkesk

TODO: Break up tests into smaller ones
"""
import os

import numpy as np
import h5py as h5
import pytest 

pytest.importorskip("pyscf")

try:
    from mpi4py import MPI
    no_mpi = False
except ImportError:
    no_mpi = True


from afqmctools.hamiltonian.converter import (
        read_hamiltonian,
        read_fcidump,
        write_fcidump,
        read_hamil_type
        )
import afqmctools.hamiltonian.mol as mol
from afqmctools.utils.linalg import modified_cholesky_direct
from afqmctools.hamiltonian.io import write_sparse
import afqmctools.hamiltonian.kpoint as kp
import afqmctools.hamiltonian.supercell as sc
from afqmctools.utils.linalg import get_ortho_ao


class TestConverter:


    def test_convert_real(self,tmp_path,random_real_hamiltonian):
        nmo = 17
        nelec = (3,3)
        print(random_real_hamiltonian)
        h1e, chol, enuc, _ = random_real_hamiltonian
        write_sparse(h1e, chol.reshape((-1,nmo*nmo)).T.copy(),
                             nelec, nmo, e0=enuc, real_chol=True, filename=tmp_path/'hamiltonian.h5')
        hamil = read_hamiltonian(tmp_path/'hamiltonian.h5')
        write_fcidump(tmp_path/'FCIDUMP', hamil['hcore'], hamil['chol'], hamil['enuc'],
                      hamil['nmo'], hamil['nelec'], sym=8, cplx=False)
        h1e_r, eri_r, enuc_r, nelec_r = read_fcidump(tmp_path/'FCIDUMP',symmetry=8,verbose=False)
        dm = np.zeros((nmo,nmo))
        dm[(0,1,2),(0,1,2)] = 1.0
        eri_r = eri_r.transpose((0,1,3,2)).reshape((nmo*nmo,nmo*nmo))
        chol_r = modified_cholesky_direct(eri_r, tol=1e-8, verbose=False)
        chol_r = chol_r.reshape((-1,nmo,nmo))
        assert np.isclose(np.einsum('ij,ij->', dm, h1e-h1e_r).real, 0.0)
        assert np.isclose(np.einsum('ij,nij->', dm, chol-chol_r).real, 0.0)
        
        # Test integral only appears once in file.
        h1e_r, eri_r, enuc_r, nelec_r = read_fcidump(tmp_path/'FCIDUMP', symmetry=1,
                                                     verbose=False)
        i,j,k,l = (0,1,2,3)
        combs = [
            (i,j,k,l),
            (k,l,i,j),
            (j,i,l,k),
            (l,k,j,i),
            (j,i,k,l),
            (l,k,i,j),
            (i,j,l,k),
            (k,l,i,j),
            ]
        for c in combs:
            if abs(eri_r[c]) > 0:
                assert c == (l,k,j,i)


    def test_convert_cplx(self,tmp_path,random_cplx_hamiltonian):
        nmo = 17
        nelec = (3,3)
        h1e, chol, enuc, _ = random_cplx_hamiltonian
        write_sparse(h1e, chol.reshape((-1,nmo*nmo)).T.copy(),
                             nelec, nmo, e0=enuc, real_chol=False, filename=tmp_path/'hamiltonian.h5')
        hamil = read_hamiltonian(tmp_path/'hamiltonian.h5')
        write_fcidump(tmp_path/'FCIDUMP', hamil['hcore'], hamil['chol'], hamil['enuc'],
                      hamil['nmo'], hamil['nelec'], sym=4, cplx=True)
        h1e_r, eri_r, enuc_r, nelec_r = read_fcidump(tmp_path/'FCIDUMP', symmetry=4,
                                                     verbose=False)
        dm = np.zeros((nmo,nmo))
        dm[(0,1,2),(0,1,2)] = 1.0
        eri_r = eri_r.transpose((0,1,3,2)).reshape((nmo*nmo,nmo*nmo))
        chol_r = modified_cholesky_direct(eri_r, tol=1e-8, verbose=False)
        chol_r = chol_r.reshape((-1,nmo,nmo))
        assert np.isclose(np.einsum('ij,ij->', dm, h1e-h1e_r).real, 0.0)
        assert np.isclose(np.einsum('ij,nij->', dm, chol-chol_r).real, 0.0)
        # Test integral only appears once in file.
        h1e_r, eri_r, enuc_r, nelec_r = read_fcidump(tmp_path/'FCIDUMP', symmetry=1,
                                                     verbose=False)
        i,k,j,l = (1,0,0,0)
        ikjl = (i,k,j,l)
        jlik = (j,l,i,k)
        kilj = (k,i,l,j)
        ljki = (l,j,k,i)
        d1 = eri_r[ikjl] - eri_r[kilj].conj()
        d2 = eri_r[ikjl] - eri_r[jlik]
        d3 = eri_r[ikjl] - eri_r[ljki].conj()
        assert np.isclose(d1,0.0)
        assert np.isclose(d2,0.0)
        assert np.isclose(d3,-0.00254428836-0.00238852605j)


    @pytest.mark.parametrize(
            "test_input,expected",
                [
                    ['Hamiltonian/DenseFactorized/L','dense'],
                    ['Hamiltonian/Factorized/vals_0','sparse'],
                    ['Hamiltonian/KPFactorized/L0','kpoint'],
                    ['Hamiltonian/THC/Luv','thc'],
                    ['RandomName',None]
                ]
            )
    def test_read_hamil_type(self,volatile_test_hdf5,test_input,expected):
        name = volatile_test_hdf5(
            data = {test_input:0}
        )
        assert read_hamil_type(name) == expected


@pytest.mark.pyscf
class TestMol:


    def test_cholesky_direct(self,neon_atom):
        eri = neon_atom.intor('int2e', aosym='s1')
        assert eri.shape == (5,5,5,5)
        
        eri = eri.reshape(25,25)
        chol = modified_cholesky_direct(eri, 1e-5, cmax=20)
        Mrecon = np.dot(chol.T, chol)
        assert np.allclose(Mrecon, eri, atol=1e-8, rtol=1e-6)


    def test_chunked_cholesky(self,neon_atom_dz,neon_eri_dz):
        eri = neon_eri_dz
        assert eri.shape == (23,23,23,23)

        eri = eri.reshape(529,529)
        chol = mol.chunked_cholesky(neon_atom_dz, max_error=1e-5)
        assert chol.shape == (122,529)
        
        eri_loc = np.dot(chol.T, chol)
        assert np.allclose(eri, eri_loc, atol=1e-5, rtol=1e-3)


    def test_ao2mo_chol(self,neon_atom,neon_rhf,neon_eri):
        eri = neon_eri 
        mf, energy = neon_rhf
        assert np.isclose(energy, -126.60452499805)
        assert eri.shape == (5,5,5,5)
        eri = eri.reshape(25,25)
        chol = mol.chunked_cholesky(neon_atom, max_error=1e-5)
        mol.ao2mo_chol(chol, mf.mo_coeff)
        assert np.isclose(np.linalg.norm(chol), 3.52947146946)


    def test_ao2mo_chol_rect(self,random_real_hamiltonian):
        np.random.seed(7)
        _,chol,_,_ = random_real_hamiltonian
        X = np.random.random((17,15))
        nchol = chol.shape[0]
        chol_ = mol.ao2mo_chol(chol, X)
        assert chol_.shape == (nchol, 15*15)


    def test_frozen_core(self,neon_atom,neon_rhf,neon_casscf):
        mf,_ = neon_rhf
        C = mf.mo_coeff
        chol = mol.chunked_cholesky(neon_atom, max_error=1e-5)
        mol.ao2mo_chol(chol, C)
        hcore = mf.get_hcore()
        h1e = np.dot(C.T, np.dot(hcore, C))
        h1e, chol, efzc = mol.freeze_core(h1e, chol, 0, 1, 4, verbose=False)
        assert h1e.shape == (2,4,4)
        assert chol.shape == (15,16)
        
        # Check from CASSCF object with same core.
        mc = neon_casscf
        h1eff, ecore = mc.get_h1eff()
        assert np.isclose(efzc, ecore)
        assert np.allclose(h1eff, h1e, atol=1e-8, rtol=1e-5)


    def test_write_hamil_mol(self,tmp_path,neon_atom,neon_rhf):
        mf,_ = neon_rhf
        C = mf.mo_coeff
        scf_data = {
            'mo_coeff': C, 
            'mol': neon_atom, 
            'hcore': mf.get_hcore(),
            'isUHF': False,
            'walker_type' : 'closed'
            }
        h1e,chol,_,enuc,_ = mol.generate_hamiltonian(scf_data,walker_type='closed')

        mol.write_hamil_mol(scf_data, tmp_path/'ham.h5', 1e-5, verbose=False)
        hamil = read_hamiltonian(tmp_path/'ham.h5')

        assert np.allclose(hamil['hcore'], h1e, atol=1e-12, rtol=1e-8)
        assert np.allclose(np.array(hamil['chol']).real.T, chol, atol=1e-12, rtol=1e-8)
        assert np.isclose(enuc, hamil['enuc'])


@pytest.mark.pyscf
@pytest.mark.mpi
@pytest.mark.skipif(no_mpi,reason="requires mpi4py")
class TestKpoint:


    def test_momentum_maps(self,diamond,diamond_lda):
        cell = diamond
        _,kpts = diamond_lda

        qk, km = kp.construct_qk_maps(cell, kpts)
        assert qk.shape == (2,2)
        assert km.shape == (2,)
        assert qk[1,1] == 0
        assert qk[0,1] == 1
        assert km[1] == 1


    def test_file_handler(self,tmp_path):
        comm = MPI.COMM_WORLD
        handler = kp.FileHandler(comm, tmp_path/'ham.h5')
        assert not handler.error
        handler.close()


    def test_write_basic(self,tmp_path,diamond,diamond_lda):
        cell = diamond
        mf,kpts = diamond_lda

        comm = MPI.COMM_WORLD
        hcore = mf.get_hcore()
        X, nmo_pk = get_ortho_ao(cell, kpts)
        # Fake some data
        qk = np.zeros((2,2))
        km = np.ones((2,))
        with kp.FileHandler(comm, tmp_path/'ham.h5') as h5file:
            h5file.grp = h5file.create_group("Hamiltonian")
            kp.write_basic(comm, cell, kpts, hcore, h5file,
                        X, nmo_pk, qk, km)
            hcore_k1 = h5file.grp['H1_kp1'][:]
        assert hcore_k1.shape == (8,8,2)
        assert np.isclose(np.max(hcore_k1), 1.3715128636941056)


@pytest.mark.pyscf
@pytest.mark.mpi
@pytest.mark.skipif(no_mpi,reason="requires mpi4py")
class TestKPCholesky:


    @pytest.fixture(scope='class')
    def setup(self,diamond,diamond_lda):
        super().__init__()
        cell = diamond
        mf,kpts = diamond_lda

        qk, km = kp.construct_qk_maps(cell, kpts)
        X = [np.identity(C.shape[0]) for C in mf.mo_coeff]
        comm = MPI.COMM_WORLD
        nmo_pk = np.array([C.shape[-1] for C in mf.mo_coeff])
        chol = kp.KPCholesky(comm, cell, kpts, 20, nmo_pk, qk, km,
                                  verbose=False, gtol_chol=1e-4)

        return chol,X,comm
    

    def test_kpchol_constructor(self,setup):
        chol,_,_ = setup

        assert chol.maxvecs == 160
        assert chol.part.kkbounds[1] == 2
        assert chol.part.kkN == 2
        assert chol.ngs == 1728
        assert chol.gmap.shape == (27,1728)
        assert sum(chol.gmap[1]) == 1492128
        qtest = np.array([0.92358847, -0.92358847, -0.92358847])
        assert np.allclose(chol.Qi[4], qtest)
        assert chol.maxres_buff.shape == (5,)

    
    def test_orbital_products(self,setup):
        chol,X,_ = setup
        part = chol.part

        ngs = chol.ngs
        Xaoik = np.zeros((part.nkk,ngs,part.nij),
                             dtype=np.complex128)
        Xaolj = np.zeros((part.nkk,ngs,part.nij),
                            dtype=np.complex128)
        chol.generate_orbital_products(1, X, Xaoik, Xaolj)

        assert np.isclose(np.max(np.abs(Xaoik)),
                               0.01239256218954048)
        assert np.isclose(np.max(np.abs(Xaolj)),
                               23.93386691184637)

    def test_kpchol_solution(self,setup,tmp_path):
        chol,X,comm = setup
        with kp.FileHandler(comm, tmp_path/'test.h5') as h5file:
            h5file.grp = h5file.create_group("Hamiltonian")
            h5file.grp_v2 = h5file.create_group("Hamiltonian/KPFactorized")
            chol.run(comm, X, h5file)

@pytest.mark.pyscf
class TestSupercell:


    @pytest.fixture(scope='class')
    def setup(self,diamond_lda):
        mf,kpts = diamond_lda

        nmo_pk = np.array([C.shape[-1] for C in mf.mo_coeff])

        diamond_info = {
            'nmo_pk' : nmo_pk,
            'nkpts' : len(kpts),
            'nmo_max' : np.max(nmo_pk)
            }
        return diamond_info


    def test_basis_map(self,setup,diamond_lda):
        mf,_ = diamond_lda
        Xocc = mf.mo_occ

        ik2n, nmo_tot = sc.setup_basis_map(Xocc=Xocc,**setup)
        bench = [28, 92]

        assert np.allclose(sum(ik2n), bench)
        assert nmo_tot == 16


    def test_hcore(self,tmp_path,setup,diamond_lda):
        mf,_ = diamond_lda
        Xocc = mf.mo_occ
        hcore = mf.get_hcore()
        h5f = h5.File(tmp_path/'ham.h5', 'w')
        h5grp = h5f.create_group("Hamiltonian")
        ik2n, nmo_tot = sc.setup_basis_map(Xocc=Xocc,**setup)
        sc.write_one_body(
            hcore=hcore,
            X=mf.mo_coeff,
            ik2n=ik2n, 
            h5grp=h5grp,
            gtol=1e-16,
            **setup
            )


    def test_grid_shifts(self,diamond):
        gmap, Qi, ngs = sc.generate_grid_shifts(diamond)
        assert gmap[0,-1] == 1570
        assert np.allclose(Qi[1], [0,0,-1.8471769])
        assert ngs == 1728


    def test_gen_partition(self,setup,fake_comm):
        comm = fake_comm

        nmo_max = setup['nmo_max'] 
        nkpts = setup['nkpts']
        nmo_pk = setup['nmo_pk']

        nmo_tot = np.sum(nmo_pk)
        maxvecs = 20 * nmo_tot
        part = sc.Partition(
            comm=comm, 
            maxvecs=maxvecs,
            nmo_tot=nmo_tot,
            nmo_max=nmo_max, 
            nkpts=nkpts
            )
        
        assert part.ij0 == 0
        assert part.ijN == 8
        assert part.nij == 8
        assert part.nkk == 2
        assert part.n2k1[0] == 0
        assert part.n2k2[1] == 1
        assert part.kk0 == 0
        assert part.kkN == 2


    def test_gen_orbital_products(self,setup,fake_comm,diamond,diamond_lda):
        from pyscf.pbc import df
        cell = diamond
        mf,kpts = diamond_lda
        mydf = df.FFTDF(cell,kpts)
        comm = fake_comm
        
        nmo_max = setup['nmo_max'] 
        nkpts = setup['nkpts']
        nmo_pk = setup['nmo_pk']
        
        nmo_tot = np.sum(nmo_pk)
        maxvecs = 20 * nmo_tot
        part = sc.Partition(
            comm=comm, 
            maxvecs=maxvecs,
            nmo_tot=nmo_tot,
            nmo_max=nmo_max, 
            nkpts=nkpts
            )
        gmap, Qi, ngs = sc.generate_grid_shifts(cell)
        X = [np.identity(C.shape[0]) for C in mf.mo_coeff]
        xik, xlj = sc.gen_orbital_products(cell, mydf, X,
                                           nmo_pk, ngs, part, kpts,
                                           nmo_max)
        
        assert xik.shape == (2,1728,8)
        assert np.isclose(np.max(np.abs(xik)),0.01239256218954048)
        assert np.isclose(np.max(np.abs(xlj)),50.73269574297129)


    @pytest.mark.mpi
    @pytest.mark.skipif(no_mpi,reason="requires mpi4py")
    def test_modified_cholesky(self,setup,diamond,diamond_lda):
        from pyscf.pbc import df, tools
        cell = diamond
        mf,kpts = diamond_lda
        mydf = df.FFTDF(cell,kpts)
        comm = MPI.COMM_WORLD
        
        nmo_max = setup['nmo_max'] 
        nkpts = setup['nkpts']
        nmo_pk = setup['nmo_pk']

        nmo_tot = np.sum(nmo_pk)
        maxvecs = 20 * nmo_tot
        part = sc.Partition(
            comm=comm, 
            maxvecs=maxvecs,
            nmo_tot=nmo_tot,
            nmo_max=nmo_max, 
            nkpts=nkpts
            )
        gmap, Qi, ngs = sc.generate_grid_shifts(cell)
        X = [np.identity(C.shape[0]) for C in mf.mo_coeff]
        xik, xlj = sc.gen_orbital_products(cell, mydf, X,
                                           nmo_pk, ngs, part, kpts,
                                           nmo_max)
        kconserv = tools.get_kconserv(cell, kpts)
        solver = sc.Cholesky(part, kconserv, 1e-3, verbose=False)
        chol = solver.run(comm, xik, xlj, part, kpts,
                          nmo_pk, nmo_max, Qi, gmap)
        
        assert chol.shape == (4,64,53)
        assert np.isclose(np.max(np.abs(chol)),0.8077979869286759)
        assert len(chol[np.abs(chol)>1e-10]) == 6229

