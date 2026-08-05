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
#include "numerics/operations/tensor.hpp"
#include "AFQMC/config.h"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "THCHamiltonian.h"
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

// PsiT[idet][ispin][ik] -> csr_mat( nup/ndn, NMO )
template<MEMORY_SPACE MEM, bool REAL> HamiltonianOperations<MEM> 
THCHamiltonian::getHamiltonianOperations_impl(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;

  std::string base_error(" Error in THCHamiltonian::getHamiltonianOperations():\n   ");

  auto all = range::all;
  std::string format;  // only meaningful at root
  long nspin = (type == COLLINEAR?2:1);
  long npol = (type == NONCOLLINEAR?2:1);
  long ndet = PsiT.extent(0); 
  long nup = PsiT(0,0).extent(0);
  long NMO = PsiT(0,0).extent(1)/npol;
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(PsiT.extent(1) == nspin, "Size mismatch");
  utils::check(ndet==1, "Error: ndet > 1 not yet implemented in THCHamiltonian::getHamiltonianOperations.");
  long ndn = (type == COLLINEAR ? PsiT(0,1).extent(0) : 0l);
  for(int i=0; i<ndet; ++i) 
    for(int ip=0; ip<npol; ++ip) {
      utils::check_shape(PsiT(i,0), "PsiT", nup, npol*NMO);
      if(type == COLLINEAR)
        utils::check_shape(PsiT(i,1), "PsiT", ndn, NMO);
    }
  utils::check(nup >= ndn, base_error + "nup:{} < ndn:{} not allowed.",nup,ndn);

  // THC variables
  long nspin_in_H1 = nspin;
  long npol_in_H1 = 1;
  long nu = 0; // number of interpolating points/vectors
  long nv = 0; // number of columns of Vuv matrix
  long nu_rot = 0; // number of interpolating points/vectors
  long nv_rot = 0; // number of columns of Vuv matrix
  bool have_rot_coul = false;  
    
  // only root reads
  ComplexType E0(0);
  h5::file file;
  std::optional<h5::group> grp, hgrp;
  if (mpi->comm.root())
  {
    file = h5::file(fileName,'r');
    grp = std::make_optional(h5::group(file));
    format = get_hamiltonian_format(*grp);
    E0 = read_energy_offset(*grp, format, type, nup, ndn);
    // open subgroup
    utils::check(format == "coqui", base_error + " Only coqui format is allowed. Format found:{}",format); 
    hgrp = std::make_optional(grp->open_group("System"));
    {
      int n;
      h5::h5_read_attribute(*hgrp,"number_of_bands",n);  // per kpoint
      // check NMO
      utils::check(NMO==n, base_error + "NMO:{} differs from file n:{}",NMO,n);
      h5::group bz = hgrp->open_group("BZ");
      // check nkpts
      h5::h5_read_attribute(bz,"number_of_kpoints",n);
      utils::check(n==1, base_error + "nkpts:1 differs from number_of_kpoints:{} in file",n);
      // read nspin_in_H1
      nspin_in_H1 = h5::h5_read_attribute<int>(*hgrp,"number_of_spins");
      // read npol_in_H1
      // h5::h5_read_attribute(bz,"number_of_polarizations",n);
      // npol_in_H1=long(n);
      utils::check(walkerDimsAreConvertible(nspin_in_H1, npol_in_H1, nspin, npol), "Hamiltonian with nspin: {}, npol: {} cannot be broadcasted to {}", nspin_in_H1, npol_in_H1, walkerTypeToString(type));
    }
    // read from /Interaction 
    {
      h5::group igrp = grp->open_group("Interaction");
      // check consistency
// MAM: fix this, integer types in h5 file are not consistent!!!
      long n;
      long nspin_in_H2= h5::read_attribute<long>(igrp,"number_of_spins");
      utils::check(nspin_in_H2 == nspin_in_H1, "Incompatible number_of_spins in Interaction {} and onebody Hamiltonian {}", nspin_in_H2, nspin_in_H1);
      
//      h5::h5_read_attribute(igrp,"number_of_polarizations",n);
//      utils::check(npol_in_H1==n,
//                   base_error + " Incompatible npol:{} in h5::/Interaction.",n);
      // NMO
      h5::h5_read_attribute(igrp,"number_of_bands",n);
      utils::check(NMO==n,base_error + " Incompatible NMO:{} in h5::/Interaction.",n);
      // nkpts
      h5::h5_read_attribute(igrp,"number_of_kpoints",n);
      utils::check(1==n,base_error + " Incompatible nkpts:{} in h5::/Interaction.",n);
      // nqpts
      h5::h5_read_attribute(igrp,"number_of_qpoints",n);
      utils::check(1==n,base_error + " Incompatible nqpts:{} in h5::/Interaction.",n);
      // now read dimensions
// MAM: right now only correct for Real==false, dimensions are assumed different for real case
      auto lX = h5::array_interface::get_dataset_info(igrp,"collocation_matrix");
      nu = lX.lengths[3];
      utils::check_shape(lX, "collocation_matrix", nspin_in_H1, 1, NMO, nu);
      auto lL = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      nv = lL.lengths[2];
      utils::check_shape(lL, "factorized_coulomb_matrix", 1, nu, nv);
      if(igrp.has_key("collocation_matrix_half_rotated")) {
        have_rot_coul = true;
        auto lXr = h5::array_interface::get_dataset_info(igrp,"collocation_matrix_half_rotated");
        nu_rot = lXr.lengths[3];
        utils::check_shape(lXr, "collocation_matrix_half_rotated", nspin_in_H1, 1, NMO, nu_rot);
        auto lZr = h5::array_interface::get_dataset_info(igrp,"half_rotated_coulomb_matrix");
        nv_rot = lZr.lengths[2];
        utils::check_shape(lZr, "half_rotated_coulomb_matrix", 1, nu_rot, nv_rot);
      } else {
        have_rot_coul = false;
        nu_rot = nu;
        nv_rot = nv;
        auto lZ = h5::array_interface::get_dataset_info(igrp,"coulomb_matrix");
        utils::check_shape(lZ, "coulomb_matrix", 1, nu, nu);
        
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
  mpi->comm.broadcast_value(E0);

  auto read_helper = [&](int reshape_type, h5::group& g, std::string name, auto && A)
  {
    utils::check(g.has_dataset(name), base_error + "Missing " + name + " dataset");
    auto l = h5::array_interface::get_dataset_info(g,name);
    if constexpr (REAL)
      utils::check(not l.has_complex_attribute,base_error+"Found complex " + name + " with REAL factory.");
    if (reshape_type==0) {
      utils::h5_read(g,name,A());
    } else if (reshape_type==1) {
      if constexpr (nda::get_rank<decltype(A())> == 3) {
        auto s = A().shape();
        auto h4d = nda::reshape(A(),std::array<long,4>{s[0],1l,s[1],s[2]});
        utils::h5_read(g,name,h4d);
      } else {
        utils::check(false,"Invalid use of read_helper");
      }
    } else if (reshape_type==2) {
      if constexpr (nda::get_rank<decltype(A())> == 2) {
        auto s = A().shape();
        auto h3d = nda::reshape(A(),std::array<long,3>{1l,s[0],s[1]});
        utils::h5_read(g,name,h3d);
      } else {
        utils::check(false,"Invalid use of read_helper");
      }
    }
  };

  // reads a ValueType dataset from /Interaction on the root only and shares the
  // result, in MEM space
  auto share_read = [&]<std::size_t N>(int reshape_type, std::string name, std::array<long,N> shape) {
    return memory::share_from_root(*mpi, [&]() {
      nda::array<ValueType,N> A(shape);
      h5::group igrp = grp->open_group("Interaction");
      read_helper(reshape_type,igrp,name,A);
      return memory::to_memory_space<MEM>(std::move(A));
    });
  };

  // read H1.
  // H0: /System/H0:   [nspin][npol*nbnd][npol*nbnd]   Only needed in host memory
  auto H1 = memory::share_from_root(*mpi, [&]() {
    nda::array<ComplexType,3> A(nspin_in_H1,npol_in_H1*NMO,npol_in_H1*NMO);
    read_helper(1,*hgrp,"H0",A);
    return A;
  });
  // X: /Interaction/collocation_matrix:  [nspins,npol*nbnd,Nu]
  // L: /Interaction/factorized_coulomb_matrix:  [Nu][Nv]
  // Z: /Interaction/coulomb_matrix:  [Nq][Nu][Nv]
  // with half rotation:
  // Xrot: /Interaction/collocation_matrix_half_rotated:  [nspins,nbnd,Nu]
  // Zrot: /Interaction/half_rotated_coulomb_matrix:      [Nq][Nu][Nv]
// MAM: right now only correct for Real==false, dimensions are assumed different for real case
  auto Xsiu = share_read(1,"collocation_matrix",
                         std::array<long,3>{nspin_in_H1,npol_in_H1*NMO,nu});
  auto Luv = share_read(2,"factorized_coulomb_matrix",std::array<long,2>{nu,nv});
  std::optional<decltype(Luv)> Zuv = std::nullopt;
  // half-rotated
  std::optional<decltype(Xsiu)> Xsiu_rot = std::nullopt;
  std::optional<decltype(Luv)> Zuv_rot = std::nullopt;
  if(have_rot_coul) {
    Zuv_rot = share_read(2,"half_rotated_coulomb_matrix",std::array<long,2>{nu_rot,nu_rot});
    Xsiu_rot = share_read(1,"collocation_matrix_half_rotated",
                          std::array<long,3>{nspin_in_H1,npol_in_H1*NMO,nu_rot});
  } else {
    Zuv = share_read(2,"coulomb_matrix",std::array<long,2>{nu,nu});
  }

  // Y = PsiT*conj(X): (since PsiT is already conjugated/transposed)
  auto Ydsau = memory::share_from_ranks<MEM,ComplexType,5,2>(*mpi,
      {ndet,nspin,npol,nup,nu},
      [&](std::array<long,2> idx, auto&& block) {
        using matrix_t = memory::buffered_array<MEM,ComplexType,2>;
        auto [id, is] = idx;
        long nel = (is==0 ? nup : ndn);
        auto Aai = math::sparse::to_array<'N'>(PsiT(id,is));
        // need to loop over npol since npol_in_H1 might be != than npol
        if constexpr (REAL) {
          auto Yau = matrix_t(nel,nu);
          auto Xiu = matrix_t(NMO,nu);
          for(long ip=0; ip<npol; ++ip) {
            auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H1,npol_in_H1);
            math::copy(Xsiu()(is_,range(ip_*NMO,(ip_+1)*NMO),all),Xiu);
            // for simplicity, make calculations with copies at full precision
            nda::tensor::contract(Aai(all,range(ip*NMO,(ip+1)*NMO)),"ai",
                            nda::conj(Xiu),"iu",Yau,"au");
            // now copy result
            math::copy(Yau,block(ip,range(nel),all));
          }
        } else {
          for(long ip=0; ip<npol; ++ip) {
            auto Yau = block(ip,range(nel),all);
            auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H1,npol_in_H1);
            auto Xiu = Xsiu()(is_,range(ip_*NMO,(ip_+1)*NMO),all);
            nda::tensor::contract(Aai(all,range(ip*NMO,(ip+1)*NMO)),"ai",
                            nda::conj(Xiu),"iu",Yau,"au");
          }
        }
      });
  std::optional<decltype(Ydsau)> Ydsau_rot = std::nullopt;
  // what to do with Ydsau_rot???

  // now calculate v0
  // Partition work over u. Replicate v for simplicity 
  // v0(s,i,l) = -0.5*sum_j <i,j|j,l>
  //         = -0.5 sum_j,u,v ma::conj(Piu(s,i,u)) ma::conj(Piu(s,j,v)) Muv Piu(s,j,u) Piu(s,l,v)
  //         = -0.5 sum_u,v ma::conj(Piu(s,i,u)) W(s,u,v) Piu(s,l,v), where
  // W(s,u,v) = Muv(u,v) * sum_j Piu(s,j,u) ma::conj(Piu(s,j,v))
  // Note: If this uses too much memory, distribute "u" axis over nodes, then construct
  // W(s,u,v) on shared memory and distribute calculation of v0 over node on a single array.
  if(have_rot_coul) {
    utils::check(false, "Finish");
  }
  auto v0 = memory::share_reduced<HOST_MEMORY,ComplexType,3>(*mpi,
      std::array<long,3>{nspin_in_H1*npol_in_H1, NMO, NMO},
      [&](auto&& partial) {
        auto [u0, u1] = itertools::chunk_range(0, nu, mpi->comm.size(), mpi->comm.rank());
        memory::buffered_array<MEM,ValueType,3> vt(partial.shape());
        vt() = ValueType(0.0);
        {
          memory::buffered_array<MEM,ValueType,2> Wuv((u1-u0),nu);
          memory::buffered_array<MEM,ValueType,2> Tiv(NMO,nu);
          Wuv() = ValueType(0.0);
          Tiv() = ValueType(0.0);
          for(long is=0, isp=0; is<nspin_in_H1; is++) {
            for(long ip=0; ip<npol_in_H1; ip++, isp++) {
              if(u0==u1) { continue; }
              auto Xiu = Xsiu()(is,range(ip*NMO,(ip+1)*NMO),range(u0,u1));
              auto Xiv = Xsiu()(is,range(ip*NMO,(ip+1)*NMO),all);
              auto Zu = (*Zuv)()(range(u0,u1),all);
              nda::tensor::contract(Xiu, "iu", nda::conj(Xiv), "iv", Wuv, "uv");
              nda::tensor::elementwise(ValueType(1.0), Zu, "uv", ValueType(1.0), Wuv, "uv",
                                       nda::tensor::op::MUL);
              nda::tensor::contract(nda::conj(Xiu), "iu", Wuv, "uv", Tiv, "iv");
              nda::tensor::contract(ValueType(-0.5),Tiv, "iv", Xiv, "jv",
                                    ValueType(0.0),vt(isp,all,all), "ij");
            }
          }
        }
        partial = nda::to_host(vt);
      });

  auto nel = std::to_array<long>({nup, (type == COLLINEAR ? ndn : 0l) });
  auto haj = half_rotate_hamiltonian<MEM>(*mpi, nel, nspin, npol, nspin_in_H1, npol_in_H1, NMO, PsiT(), H1());

  return HamiltonianOperations<MEM>(THCOps<MEM, REAL>(mpi,type,NMO,nup,ndn,
     std::move(H1),std::move(haj),std::move(Xsiu),std::move(Ydsau),std::move(Luv),std::move(Zuv),
     std::move(Xsiu_rot),std::move(Ydsau_rot),std::move(Zuv_rot),std::move(v0),E0)); 
}

template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
THCHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  bool Real = true; 

  // factorized_coulomb_matrix(q,u,n): required
  // collocation_matrix(s,k,i,u): required
  // Option 1: Provide L * dagger(L) where L=factorized_coulomb_matrix. Reuse collocation_matrix.
  // coulomb_matrix(q,u,n) 
  // Option 2: Independently generated THC factorization for half-rotated orbitals 
  // half_rotated_coulomb_matrix(q,u',v') 
  // collocation_matrix_half_rotated(s,k,i,u)
  if(mpi->comm.root()) { 

    h5::file file(fileName,'r');
    h5::group grp(file);
    h5::group igrp = grp.open_group("Interaction"); 
    { // check factorized_coulomb_matrix
      utils::check(igrp.has_dataset("factorized_coulomb_matrix"),"Missing dataset factorized_coulomb_matrix."); 
      auto l = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      if(l.has_complex_attribute) Real = false;
      utils::check((l.rank() == 2) or (l.rank() == 4), "Rank mismatch");
      if(l.has_complex_attribute or (l.rank() == 2)) Real = false;
    }
    { // check collocation_matrix 
      utils::check(igrp.has_dataset("collocation_matrix"),"Missing dataset collocation_matrix."); 
      auto l = h5::array_interface::get_dataset_info(igrp,"collocation_matrix");
      utils::check((l.rank() == 3) or (l.rank() == 5), "Rank mismatch");
      utils::check((Real and not (l.has_complex_attribute or (l.rank() == 5))) or 
                   (not Real and (l.has_complex_attribute or (l.rank() == 5))), "Incompatible datatypes in Interaction.");
    }
    // should I check the other ones???
  }
  mpi->comm.broadcast_n(&Real, 1, 0);

  if(Real) 
    return getHamiltonianOperations_impl<MEM,true>(type, mpi, PsiT); 
  else
    return getHamiltonianOperations_impl<MEM,false>(type, mpi, PsiT);
}

template HamiltonianOperations<HOST_MEMORY> 
  THCHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY> 
  THCHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc
