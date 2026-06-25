////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <numeric>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/h5_utils.hpp"
#include "numerics/nda_functions.hpp"
#include "AFQMC/config.h"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "KPTHCHamiltonian.h"
#include "AFQMC/Utilities/wfn_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/HamiltonianOperations/detail/one_body.hpp"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{
// Coqui h5 structure
// H1: /System/H0:   [nspin,nkpts_ibz,nbnd,nbnd]
// X: /Interaction/collocation_matrix:  [nspins,nkpts,nbnd,Nu]
// L: /Interaction/factorized_coulomb_matrix:  [Nq][Nu][Nv]  
// Z: /Interaction/coulomb_matrix:  [Nq][Nu][Nv]  
// with half rotation:
// Xrot: /Interaction/collocation_matrix_half_rotated:  [nspins,nkpts,nbnd,Nu]
// Zrot: /Interaction/half_rotated_coulomb_matrix:      [Nq][Nu][Nv]

// PsiT[idet][ispin][ik] -> csr_mat( nel_up/nel_dn, NMO )
template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
KPTHCHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;

  std::string base_error(" Error in KPTHCHamiltonian::getHamiltonianOperations():\n   ");

  auto all = range::all;
  std::string format;  // only meaningful at root
  long nspin = (type == COLLINEAR?2:1);
  long npol = (type == NONCOLLINEAR?2:1);
  int nspin_in_PsiT = PsiT.extent(1);
  long ndet = PsiT.extent(0); 
  long nel_up = PsiT(0,0).extent(0);
  long NMO = PsiT(0,0).extent(1)/npol;
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(nspin_in_PsiT==nspin, "Size mismatch");
  utils::check(ndet==1, "Error: ndet > 1 not yet implemented in KPTHCHamiltonian::getHamiltonianOperations.");
  long nel_dn = (type == COLLINEAR ? PsiT(0,1).extent(0) : 0l);
  for(int i=0; i<ndet; ++i) 
    for(int ip=0; ip<npol; ++ip) {
      utils::check_shape(PsiT(i,0), "PsiT", nel_up, NMO);
      if(type == COLLINEAR)
        utils::check_shape(PsiT(i,nspin_in_PsiT-1), "PsiT", nel_dn, NMO);
    }
  utils::check(nel_up >= nel_dn, base_error + "nel_up:{} < nel_dn:{} not allowed.",nel_up,nel_dn);

  // THC variables
  long nspin_in_H1 = nspin;
  long npol_in_H1 = npol;
  // ISDF variables
  long nu = 1; // number of interpolating points/vectors
  long nv = 1; // number of columns of Vuv matrix
  long nu_rot = 1; // number of interpolating points/vectors
  long nv_rot = 1; // number of columns of Vuv matrix
  long have_rot_coul = 0;  
  // BZ variables
  long nkpts = 1;
  long nkpts_ibz = 1;
  long nqpts_ibz = 1;
  long nbnd = 1; 
  long Q0_index = 0;
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> kp_to_ibz; 
  nda::array<int,1> qp_to_ibz; 
  nda::array<bool,1> kp_trev;  
  nda::array<bool,1> qp_trev;  
  nda::array<int,1> kp_trev_pair;
  nda::array<double,2> qpoints;


  // only root reads
  h5::file file;
  std::optional<h5::group> grp, hgrp;
  if (mpi->comm.root())
  {
    file = h5::file(fileName,'r');
    grp = std::make_optional(h5::group(file));
    format = get_hamiltonian_format(*grp);
    // open subgroup
    utils::check(format == "coqui", base_error + " Only coqui format is allowed. Format found:{}",format);
    hgrp = std::make_optional(grp->open_group("System"));
    {
      int n;
      h5::group bz = hgrp->open_group("BZ");
      // read nkpts
      h5::h5_read_attribute(bz,"number_of_kpoints",n);
      nkpts = long(n); 
      h5::h5_read_attribute(bz,"number_of_kpoints_ibz",n);
      nkpts_ibz = long(n); 
      h5::h5_read_attribute(bz,"number_of_qpoints",n);
      utils::check(n==nkpts, base_error + "nqpts != nkpts, nqpts:{}, nkpts:{}",n,nkpts);
      h5::h5_read_attribute(bz,"number_of_qpoints_ibz",n);
      nqpts_ibz = long(n); 
      // nbnd
      h5::h5_read_attribute(*hgrp,"number_of_bands",n);  // per kpoint
      // check nbnd 
      utils::check(NMO%nkpts==0, base_error + "NMO%nkpts != 0, NMO:{}, nkpts:{}",NMO,nkpts);
      utils::check((NMO/nkpts)==n, base_error + "nbnd:{} differs from file n:{}",nbnd,n);
      nbnd = long(NMO/nkpts);
      // read nspin_in_H1
      h5::h5_read_attribute(*hgrp,"number_of_spins",n);
      nspin_in_H1 = long(n);
      // read npol_in_H1
//      h5::h5_read_attribute(bz,"number_of_polarizations",n);
//      npol_in_H1=long(n);
      utils::check(walkerDimsAreConvertible(nspin_in_H1, npol_in_H1, nspin, npol), "Hamiltonian with nspin: {}, npol: {} cannot be broadcasted to {}", nspin_in_H1, npol_in_H1, walkerTypeToString(type));
      minusq.resize(nkpts);
      nda::h5_read(bz,"qminus",minusq);
      qk_to_k2.resize(nkpts,nkpts);
      nda::h5_read(bz,"qk_to_k2",qk_to_k2);
      qp_to_ibz.resize(nkpts);
      nda::h5_read(bz,"qp_to_ibz",qp_to_ibz);
      kp_trev.resize(nkpts);
      nda::h5_read(bz,"kp_trev",kp_trev);
      kp_trev_pair.resize(nkpts);
      nda::h5_read(bz,"kp_trev_pair",kp_trev_pair);
      qp_trev.resize(nkpts);
      nda::h5_read(bz,"qp_trev",qp_trev);
      qpoints.resize(nkpts,3);
      nda::h5_read(bz,"qpoints",qpoints);
      Q0_index=-1;
      for(int i=0; i<nkpts; i++)
        if(nda::sum(qpoints(i,all)*qpoints(i,all)) < 1e-8) {
          utils::check(Q0_index<0, "Error: Multiple points with Q=0: {}, {}",Q0_index,i);
          Q0_index = i;
        }
      utils::check(Q0_index>=0, "Error: Problems finding Q=0");
    }
    // read from /Interaction
    {
      h5::group igrp = grp->open_group("Interaction");
      // check consistency
// MAM: fix this, integer types in h5 file are not consistent!!!
      long n;
      // read nspin_in_H1
      h5::h5_read_attribute(igrp,"number_of_spins",n);
      utils::check(nspin_in_H1==n,
                   base_error + " Incompatible nspin:{} in h5::/Interaction.",n);
      // read npol_in_H1
//      h5::h5_read_attribute(igrp,"number_of_polarizations",n);
//      utils::check(npol_in_H1==n,
//                   base_error + " Incompatible npol:{} in h5::/Interaction.",n);
      // nbnd 
      h5::h5_read_attribute(igrp,"number_of_bands",n);
      utils::check(nbnd==n,base_error + " Incompatible nbnd:{} in h5::/Interaction.",n);
      // nkpts
      h5::h5_read_attribute(igrp,"number_of_kpoints",n);
      utils::check(nkpts==n,base_error + " Incompatible nkpts:{} in h5::/Interaction.",n);
      // nqpts
      h5::h5_read_attribute(igrp,"number_of_qpoints",n);
      utils::check(nkpts==n,base_error + " Incompatible nqpts:{} in h5::/Interaction.",n);
      // now read dimensions
      auto lX = h5::array_interface::get_dataset_info(igrp,"collocation_matrix");
      nu = lX.lengths[3];
      utils::check_shape(lX, "collocation_matrix", nspin_in_H1, nkpts, nbnd, nu);
      auto lL = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      nv = lL.lengths[2];
      utils::check_shape(lL, "factorized_coulomb_matrix", nqpts_ibz, nu, nv);
      if(igrp.has_key("collocation_matrix_half_rotated")) {
        have_rot_coul = 1;
        auto lXr = h5::array_interface::get_dataset_info(igrp,"collocation_matrix_half_rotated");
        nu_rot = lXr.lengths[3];
        utils::check_shape(lXr, "collocation_matrix_half_rotated", nspin_in_H1, nkpts, nbnd, nu_rot);
        auto lZr = h5::array_interface::get_dataset_info(igrp,"half_rotated_coulomb_matrix");
        nv_rot = lZr.lengths[2];
        utils::check_shape(lZr, "half_rotated_coulomb_matrix", nqpts_ibz, nu_rot, nv_rot);
      } else {
        have_rot_coul = 0;
        nu_rot = nu;
        nv_rot = nv;
        auto lZ = h5::array_interface::get_dataset_info(igrp,"coulomb_matrix");
        utils::check_shape(lZ, "coulomb_matrix", nqpts_ibz, nu, nu);
        
      }
    }
  }
  mpi->comm.broadcast_value(nspin_in_H1);
  mpi->comm.broadcast_value(npol_in_H1);
  mpi->comm.broadcast_value(nu);
  mpi->comm.broadcast_value(nv);
  mpi->comm.broadcast_value(nu_rot);
  mpi->comm.broadcast_value(nv_rot);
  mpi->comm.broadcast_value(have_rot_coul);
  mpi->comm.broadcast_value(nbnd);
  mpi->comm.broadcast_value(nkpts);
  mpi->comm.broadcast_value(nkpts_ibz);
  mpi->comm.broadcast_value(nqpts_ibz);
  mpi->comm.broadcast_value(Q0_index);
  if(not mpi->comm.root()) {
    minusq.resize(nkpts);
    qk_to_k2.resize(nkpts,nkpts);
  }
  mpi->broadcast(minusq);
  mpi->broadcast(qk_to_k2);

  utils::check(nqpts_ibz==nkpts, "Error: Symmetry not yet implemented.");

  // reads a ComplexType dataset from /Interaction on the root only and shares
  // the result, in MEM space
  auto share_read = [&]<std::size_t N>(std::string name, std::array<long,N> shape) {
    return memory::share_from_root(*mpi, [&]() {
      nda::array<ComplexType,N> A(shape);
      h5::group igrp = grp->open_group("Interaction");
      utils::h5_read(igrp,name,A);
      return memory::to_memory_space<MEM>(std::move(A));
    });
  };

  // read H1.
  // H0: /System/H0:   [nspin][nkpts][npol*nbnd][npol*nbnd]   Only needed in host memory
  auto H1 = memory::share_from_root(*mpi, [&]() {
    nda::array<ComplexType,4> A(nspin_in_H1,nkpts,npol_in_H1*nbnd,npol_in_H1*nbnd);
    utils::h5_read(*hgrp,"H0",A);
    return A;
  });
  // X: /Interaction/collocation_matrix:  [nspins,nkpts,npol*nbnd,Nu]
  // L: /Interaction/factorized_coulomb_matrix:  [nqpts][Nu][Nv]
  // Z: /Interaction/coulomb_matrix:  [Nq][Nu][Nv]
  // with half rotation:
  // Xrot: /Interaction/collocation_matrix_half_rotated:  [nspins,nkpts,nbnd,Nu]
  // Zrot: /Interaction/half_rotated_coulomb_matrix:      [Nq][Nu][Nv]
  auto Xsiu = share_read("collocation_matrix",
                         std::array<long,4>{nspin_in_H1,nkpts,npol_in_H1*nbnd,nu});
  auto Luv = share_read("factorized_coulomb_matrix",std::array<long,3>{nqpts_ibz,nu,nv});
  std::optional<decltype(Luv)> Zuv = std::nullopt;
  // half-rotated
  std::optional<decltype(Xsiu)> Xsiu_rot = std::nullopt;
  std::optional<decltype(Luv)> Zuv_rot = std::nullopt;
  if(have_rot_coul) {
    Xsiu_rot = share_read("collocation_matrix_half_rotated",
                          std::array<long,4>{nspin_in_H1,nkpts,npol_in_H1*nbnd,nu_rot});
    Zuv_rot = share_read("half_rotated_coulomb_matrix",std::array<long,3>{nqpts_ibz,nu_rot,nu_rot});
  } else {
    Zuv = share_read("coulomb_matrix",std::array<long,3>{nqpts_ibz,nu,nu});
  }
  // kpoint dependent occupations: for each (spin,kpoint), the list of PsiT row
  // indices occupying that kpoint. Computed on every rank (depends only on PsiT,
  // which is replicated), so no broadcast of the ragged list-of-lists is needed.
  auto nocc = nocc_per_kpoint(type,nkpts,PsiT);
  auto nocc_max = max_nocc_per_kpoint(nocc);
  utils::check(nel_up == nelec_for_spin(nocc,0), "Error: Mismatch in number of electrons: nel_up:{} sum(nel_up(k)):{}",nel_up,nelec_for_spin(nocc,0));
  if(type == COLLINEAR)
    utils::check(nel_dn == nelec_for_spin(nocc,1), "Error: Mismatch in number of electrons: ndown:{} sum(ndown(k)):{}",nel_dn,nelec_for_spin(nocc,1));

  // Y = PsiT*conj(X): (since PsiT is already conjugated/transposed)
  auto Ydsau = memory::share_from_ranks<MEM,ComplexType,6,4>(*mpi,
      {ndet,nspin,npol,nkpts,nocc_max,nu},
      [&](std::array<long,4> idx, auto&& block) {
        auto [id,is,ip,ik] = idx;
        auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H1,npol_in_H1);
        auto const& rows = nocc(is,ik);
        int nel = int(rows.size());
        auto Aai = math::sparse::to_array<'N'>(PsiT(id,is),rows,range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
        auto Yau = block(range(nel),all);
        auto Xiu = Xsiu()(is_,ik,range(ip_*nbnd,(ip_+1)*nbnd),all);
        nda::tensor::contract(Aai,"ai", nda::conj(Xiu),"iu",Yau,"au");
      });
  std::optional<decltype(Ydsau)> Ydsau_rot = std::nullopt;
  // what to do with Ydsau_rot???

  // now calculate v0
  // Partition work over spin, pol, kpts.
  // v0(s,k,i,l) = -0.5*sum_k,q sum_j <i_k,j_k-q|j_k-q,l_k>
  //         = -0.5 sum_kq sum_j,u,v ma::conj(Piu(s,k,i,u)) ma::conj(Piu(s,k-q,j,v)) Mquv Piu(s,k-q,j,u) Piu(s,k,l,v)
  if(have_rot_coul) {
    utils::check(false, "Finish");
  }
  memory::buffered_array<MEM,ComplexType,2> Tuv(nu,nu);
  memory::buffered_array<MEM,ComplexType,2> Wuv(nu,nu);
  memory::buffered_array<MEM,ComplexType,2> vt(nbnd,nbnd);
  auto v0 = memory::share_from_ranks<HOST_MEMORY,ComplexType,4,2>(*mpi,
      std::array<long,4>{nspin_in_H1*npol_in_H1, nkpts, nbnd, nbnd},
      [&](std::array<long,2> idx, auto&& block) {
        auto [isp,ik] = idx;
        long is = isp/npol_in_H1;
        long ip = isp%npol_in_H1;
        // sum over q
        Tuv() = ComplexType(0.0);
        Wuv() = ComplexType(0.0);
        for(long iq=0; iq<nkpts; iq++) {
          int k2 = 0; // qp_to_k2(ik,iq);  // no symmetry yet!!!
          auto Xiu_k2 = Xsiu()(is,k2,range(ip*nbnd,(ip+1)*nbnd),all);
          auto Zu = (*Zuv)()(iq,all,all);
          nda::tensor::contract(Xiu_k2, "iu", nda::conj(Xiu_k2), "iv", Tuv, "uv");
          if constexpr (MEM==HOST_MEMORY)
            Tuv() = Zu() * Tuv();
          else
            nda::tensor::elementwise(ComplexType(1.0), Zu, "uv",
                                     ComplexType(1.0), Tuv, "uv", nda::tensor::op::MUL);
          nda::tensor::add(ComplexType(1.0),Tuv,"uv",ComplexType(1.0),Wuv,"uv");
        }
        auto Xiu = Xsiu()(is,ik,range(ip*nbnd,(ip+1)*nbnd),all);
        auto Tiv = Tuv(range(nbnd),all);
        nda::tensor::contract(nda::conj(Xiu), "iu", Wuv, "uv", Tiv, "iv");
        nda::tensor::contract(ComplexType(-0.5),Tiv, "iv", Xiu, "jv",
                              ComplexType(0.0),vt, "ij");
        block = nda::to_host(vt)();
      });

  // keep in full basis, better for batched
  // MAM: this uses a lot more memory/compute, but can be batched into a single call (energy eval)
  // time the two versions and change to k-dependent haj if the timing in gpu is not that different
  // You can also write a kernel that dispatches all the gemms from device side using the new library
  auto nel = std::to_array<long>({nel_up, (type == COLLINEAR ? nel_dn : 0l)});

  auto hfull = memory::share_from_root(*mpi, [&](){
    memory::array<MEM,ComplexType,3> hfull(nspin_in_H1, npol_in_H1*nkpts*nbnd, npol_in_H1*nkpts*nbnd);
    hfull() = 0;
    auto hfull7 = nda::reshape(hfull, nspin_in_H1, npol_in_H1, nkpts, nbnd, npol_in_H1, nkpts, nbnd);
    for(int ik = 0; ik < nkpts; ik++) {
      hfull7(all, all, ik, all, all, ik, all) = nda::reshape(H1(), nspin_in_H1, nkpts, npol_in_H1, nbnd, npol_in_H1, nbnd)(all, ik, all, all, all, all);
    }

    return hfull;
  });
  auto haj = half_rotate_hamiltonian<MEM>(*mpi, nel, nspin, npol, nspin_in_H1, npol_in_H1, NMO, PsiT(), hfull());

  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;
  return HamiltonianOperations<MEM>(KPTHCOps<MEM>(mpi,type,NMO,nel_up,nel_dn,nkpts,Q0_index,std::move(nocc),
     std::move(minusq),std::move(qk_to_k2),std::move(H1),std::move(haj),std::move(Xsiu),
     std::move(Ydsau),std::move(Luv),std::move(Zuv),std::move(Xsiu_rot),std::move(Ydsau_rot),
     std::move(Zuv_rot),std::move(v0),E0)); 
}

template HamiltonianOperations<HOST_MEMORY> 
  KPTHCHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY> 
  KPTHCHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc
