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
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "AFQMC/config.h"
#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "numerics/operations/small_mat_ops.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "numerics/device_kernels/kernels.h"

namespace sfqmc::afqmc
{

template<nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat>
void calculate_overlaps(int spin, ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,  T_t const& T, Mat&& ov)
{
  using nda::range;
  auto all = range::all;
  static_assert(memory::get_memory_space<T_t>() == memory::get_memory_space<Mat>(), "Memory space mismatch");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<Mat>();
  ov(0,all)=ComplexType(1.0); 
  int nw = ov.extent(1);
  utils::check(T.extent(0) == nw, "Size mismatch"); 
  if constexpr (MEM==HOST_MEMORY) {
    for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++) {
      // expanding some of them by hand for efficiency
      if (nex == 1) {
        for (auto it = abij.unique_begin(1)[spin]; it < abij.unique_end(1)[spin]; ++it, ++nd) {
          auto e = *it;
          ov(nd,all) = T(all,e[1],e[0]);
        }
      } else if (nex == 2) {
        for (auto it = abij.unique_begin(2)[spin]; it < abij.unique_end(2)[spin]; ++it, ++nd) {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw) 
            ov(nd,iw) = math::D2x2(T(iw,e[2],e[0]), T(iw,e[2],e[1]), T(iw,e[3],e[0]), T(iw,e[3],e[1]));
        }
      } else if (nex == 3) {
        for (auto it = abij.unique_begin(3)[spin]; it < abij.unique_end(3)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D3x3(T(iw,e[3],e[0]), T(iw,e[3],e[1]), T(iw,e[3],e[2]), T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]),
                            T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]));
        }
      } else if (nex == 4) {
        for (auto it = abij.unique_begin(4)[spin]; it < abij.unique_end(4)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D4x4(T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]), T(iw,e[4],e[3]), T(iw,e[5],e[0]), T(iw,e[5],e[1]),
                                   T(iw,e[5],e[2]), T(iw,e[5],e[3]), T(iw,e[6],e[0]), T(iw,e[6],e[1]), T(iw,e[6],e[2]), T(iw,e[6],e[3]),
                                   T(iw,e[7],e[0]), T(iw,e[7],e[1]), T(iw,e[7],e[2]), T(iw,e[7],e[3]));
        }
      } else if (nex == 5) {
        for (auto it = abij.unique_begin(5)[spin]; it < abij.unique_end(5)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D5x5(T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]), T(iw,e[5],e[3]), T(iw,e[5],e[4]), T(iw,e[6],e[0]),
                                   T(iw,e[6],e[1]), T(iw,e[6],e[2]), T(iw,e[6],e[3]), T(iw,e[6],e[4]), T(iw,e[7],e[0]), T(iw,e[7],e[1]),
                                   T(iw,e[7],e[2]), T(iw,e[7],e[3]), T(iw,e[7],e[4]), T(iw,e[8],e[0]), T(iw,e[8],e[1]), T(iw,e[8],e[2]),
                                   T(iw,e[8],e[3]), T(iw,e[8],e[4]), T(iw,e[9],e[0]), T(iw,e[9],e[1]), T(iw,e[9],e[2]), T(iw,e[9],e[3]),
                                   T(iw,e[9],e[4]));
        }
      } else {
        memory::buffered_array<MEM,ComplexType,2> Qwork(nex,nex); 
        memory::buffered_array<MEM,ComplexType,1> Qwork2(nex*nex); 
        memory::buffered_array<MEM,int,1> ipiv(nex); 
        for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
        {
          auto exct = *it;
          for(int iw=0; iw<nw; ++iw) {
            for (int p = 0; p < nex; p++)
              for (int q = 0; q < nex; q++) 
                Qwork(p,q) = T(iw,exct[p + nex],exct[q]);
            nda::lapack::getrf(Qwork,ipiv,Qwork2);
            math::log_determinant_from_getrf(Qwork,ipiv,ov(nd,iw)); 
            ov(nd,iw) = std::exp(ov(nd,iw));
          }
        }
      }
    } // nex
  } else { // MEM
#if defined(ENABLE_DEVICE)
    kernels::device::calculate_overlaps(spin,abij,T,ov);
#else
    utils::check(false, "Error in calculate_overlaps: DEVICE_MEMORY without device support.");
#endif
  }
}

// T(nw,nact,nel)
// weights(nd,nw)
// R(nw,nel,nact)
template<nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat, nda::MemoryArrayOfRank<3> R_t>
void calculate_R(int spin, ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,  T_t const& T, Mat&& weights, R_t && R)
{
  using nda::range;
  auto all = range::all;
  static_assert(::nda::mem::have_compatible_addr_space<T_t,Mat,R_t>, "Memory space mismatch");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<R_t>(); 
  int NEL = T.extent(2);
  int nw = T.extent(0);
  nda::array<int,1> orbs(NEL);
  ComplexType ov_a;
  R() = ComplexType(0);
  if constexpr (MEM==HOST_MEMORY) {
    {
      auto refc   = abij.reference_configuration(spin);
      // Wrong if ndown < nup!!! FIX FIX FIX
      for (int i = 0; i < NEL; ++i)
        R(all,i,refc[i]) += weights(0,all);
    }
    for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
    {
      memory::buffered_array<MEM,ComplexType,2> Q(nex,nex); 
      memory::buffered_array<MEM,int,1> ipiv(nex); 
      memory::buffered_array<MEM,ComplexType,1> work(nex*nex); 
      for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
      {
        auto e = *it;
        abij.get_configuration(spin, nd, orbs);
        for(int iw=0; iw<nw; ++iw) {
          if (nex == 1)
          {
            ov_a    = T(iw,e[1],e[0]);
            Q(0,0) = 1.0 / ov_a;
          }
          else if (nex == 2)
          {
            ov_a = math::I2x2(T(iw,e[2],e[0]), T(iw,e[2],e[1]), T(iw,e[3],e[0]), T(iw,e[3],e[1]), Q);
          }
          else if (nex == 3)
          {
            ov_a = math::I3x3(T(iw,e[3],e[0]), T(iw,e[3],e[1]), T(iw,e[3],e[2]), T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]),
                          T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]), Q);
          }
          else
          {
            for (int p = 0; p < nex; p++)
              for (int q = 0; q < nex; q++)
                Q(p,q) = T(iw,e[p + nex],e[q]);
            nda::lapack::getrf(Q,ipiv,work);
            math::log_determinant_from_getrf(Q,ipiv,ov_a);
            // how to handle cases where this is basically zero?
            ov_a = std::exp(ov_a);
            if(std::abs(ov_a) != 0.0) 
              nda::lapack::getri_or_zero(Q,ipiv,work);
          }
          ComplexType w(weights(nd,iw));
          if (std::abs(ov_a) != 0.0)
          {
            // add term coming from identity
            for (int i = 0; i < NEL; ++i)
              R(iw,i,orbs(i)) += w;
            for (int p = 0; p < nex; ++p)
            {
              auto Rp = R(iw,e[p],all);
              auto Ip = Q(p,all);
              for (int q = 0; q < nex; ++q)
              {
                auto Ipq = Ip(q);
                auto Tq  = T(iw,e[q + nex],all);
                for (int i = 0; i < NEL; ++i)
                  Rp(orbs(i)) -= w * Ipq * Tq(i);
                Rp(orbs(e[q])) += w * Ipq;
              }
            }
          }
        } // iw
      } // it
    } // nex
  } else {
#if defined(ENABLE_DEVICE)
    kernels::device::calculate_R(spin,abij,T,weights,R);
#else
    utils::check(false, "Error in calculate_overlaps: DEVICE_MEMORY without device support.");
#endif
  }
}

// R[nwalk,ndet,nex,nact]
// T(nwalk,nact,nelec]
template<MEMORY_SPACE MEM>
void get_compact_ph_R_matrices(nda::MemoryVector auto const& iexcit,
     nda::MemoryVector auto const& refc, nda::MemoryArrayOfRank<3> auto&& Tw, 
     nda::MemoryArrayOfRank<4> auto&& Rw) 
{
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(Tw,Rw);
  auto [nwalk,ndet,nex,nact] = Rw.shape();
  int nelec = Tw.extent(2);
  utils::check(Tw.extent(0) == nwalk and Tw.extent(1)==nact, "Size mismatch");
  if constexpr (MEM==HOST_MEMORY) {
    ComplexType ov_a;
    memory::buffered_array<MEM,ComplexType,2> Q(nex,nex); 
    memory::buffered_array<MEM,ComplexType,1> work(nex*nex); 
    memory::buffered_array<MEM,int,1> ipiv(nex); 
    memory::buffered_array<MEM,int,1> orbs(nelec); 
    for(int iw=0; iw<nwalk; iw++) 
    {  
      auto T = Tw(iw,all,all);
      for(int nd=0; nd<ndet; nd++) 
      {
        auto R = Rw(iw,nd,all,all); 
        auto e = iexcit.data()+2*nex*nd;
        for(int i=0; i<nelec; i++) {
          orbs(i) = refc(i);
          for(int p=0; p<nex; p++) {
             if(e[p]==i) {
              orbs(i) = e[nex+p];
              break;
            }  
          }  
        }
        if (nex == 1)
        {
          ov_a    = T(e[1],e[0]);
          Q(0,0) = 1.0 / ov_a;
        }
        else if (nex == 2)
        {
          ov_a = math::I2x2(T(e[2],e[0]), T(e[2],e[1]), T(e[3],e[0]), T(e[3],e[1]), Q);
        }
        else if (nex == 3)
        {
          ov_a = math::I3x3(T(e[3],e[0]), T(e[3],e[1]), T(e[3],e[2]), 
                          T(e[4],e[0]), T(e[4],e[1]), T(e[4],e[2]),
                          T(e[5],e[0]), T(e[5],e[1]), T(e[5],e[2]), Q);
        }
        else
        {
          for (int p = 0; p < nex; p++)
            for (int q = 0; q < nex; q++)
              Q(p,q) = T(e[p + nex],e[q]);
          nda::lapack::getrf(Q,ipiv,work);
          math::log_determinant_from_getrf(Q,ipiv,ov_a);
          // how to handle cases where this is basically zero?
          ov_a = std::exp(ov_a);
          if(std::abs(ov_a) != 0.0) nda::lapack::getri_or_zero(Q,ipiv,work);
        }
        if( std::abs(ov_a) != 0.0) {
          // compact notation:
          // R[p,nact] for p in [0, nex)
          // R[nex,nact] for diagonal term    
          for (int p = 0; p < nex; ++p)
          {
            for (int q = 0; q < nex; ++q)
            {
              auto Ipq = Q(p,q);
              auto Tq  = T(e[q + nex],all);
              for (int i = 0; i < nelec; ++i)
                R(p,orbs(i)) -= Ipq * Tq(i);
              R(p,orbs(e[q])) += Ipq;
            }
          }
        }
      } // nd
    } // iw
  } else {
#if defined(ENABLE_DEVICE)
    kernels::device::calculate_compact_ph_R(refc.data(),iexcit.data(),Tw,Rw);
#else
    utils::check(false, "Error in calculate_overlaps: DEVICE_MEMORY without device support.");
#endif
  }
}

// Calculates the Alpha/Alpha (XXX_first_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet,nwalk]
// T(nwalk,nact,nelec]
// E[nwalk,3]
// KE[ndet,nwalk,nke]
template<class Op, nda::MemoryMatrix Mat>
void ph_excited_energies_first_step(ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,
         Mat&& wgt, nda::MemoryArrayOfRank<3> auto T,
         nda::MemoryMatrix auto&& E, nda::MemoryArrayOfRank<3> auto&& KE, Op& HamOps, int ndet_block_size)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(T)>(); 
  memory::check_memory_space<MEM>(wgt,T,E,KE);
  auto [nwalk,nact,nelec] = T.shape();
  int nconfg = KE.extent(0);
  utils::check(E.extent(0) == nwalk and E.extent(1)==3, "Size mismatch");
  utils::check(wgt.extent(0) == nconfg and wgt.extent(1)==nwalk, "Size mismatch");
  utils::check(KE.extent(1) == nwalk, "Size mismatch");
  E() = ComplexType(0.0);
  
  int spin(0);
  auto refc=abij.get_reference_configuration_device(spin);
  // no use of blocks yet! 
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }

  // process each excitation shell in blocks of at most NDET_BLK determinants so the
  // per-block temporaries (R, and the ndet-proportional intermediates inside
  // ph_excited_energy) stay bounded regardless of the total determinant count.
  const int NDET_BLK = ndet_block_size;
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      for (int d0 = 0; d0 < ndet; d0 += NDET_BLK) {
        int nb = (ndet - d0 < NDET_BLK) ? (ndet - d0) : NDET_BLK;
        auto iexb = iexcit(range(long(d0)*2*nex, long(d0+nb)*2*nex));
        memory::buffered_array<MEM,ComplexType,4> R(nwalk, nb, nex, nact);
        R() = ComplexType(0.0);

        // generate R matrices for current block of the excitation shell
        get_compact_ph_R_matrices<MEM>(iexb, refc, T, R);

        // assumes no TG_local parallelization for now, needs TG for this
        // calculate E and KE for current block of excitations
        auto KEr=KE(range(idet+d0,idet+d0+nb),all,all);
        HamOps.ph_excited_energy(Alpha, nelec, iexb, refc, E,
                  wgt(range(idet+d0, idet+d0+nb),all), R, KEr, true);
      }
      idet += ndet;
    }
  }
}

// Calculates the Alpha/Alpha (XXX_second_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet,nwalk]
// T(nwalk,nact,nelec]
// E[nwalk,3]
// KE[ndet,nwalk,nke]
template<class Op, nda::MemoryMatrix Mat>
void ph_excited_energies_second_step(ph_excitations<int, ComplexType,
         memory::get_memory_space<Mat>()>& abij, Mat&& wgt, nda::MemoryArrayOfRank<3> auto T,
         nda::MemoryMatrix auto&& E, nda::MemoryArrayOfRank<3> auto&& KE, Op& HamOps, int ndet_block_size)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(T)>(); 
  memory::check_memory_space<MEM>(wgt,T,E,KE);
  auto [nwalk,nact,nelec] = T.shape();
  int nconfg = KE.extent(0);
  int nke = KE.extent(2);
  utils::check(E.extent(0) == nwalk and E.extent(1)==3, "Size mismatch");
  utils::check(wgt.extent(0) == nconfg and wgt.extent(1)==nwalk, "Size mismatch");
  utils::check(KE.extent(1) == nwalk, "Size mismatch");
  E() = ComplexType(0.0);

  int spin(1);
  auto refc=abij.get_reference_configuration_device(spin);
  // no use of blocks yet!
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }

  // process each excitation shell in blocks of at most NDET_BLK determinants (see
  // ph_excited_energies_first_step) to bound the per-block device temporaries.
  const int NDET_BLK = ndet_block_size;
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      for (int d0 = 0; d0 < ndet; d0 += NDET_BLK) {
        int nb = (ndet - d0 < NDET_BLK) ? (ndet - d0) : NDET_BLK;
        auto iexb = iexcit(range(long(d0)*2*nex, long(d0+nb)*2*nex));
        memory::buffered_array<MEM,ComplexType,4> R(nwalk, nb, nex, nact);
        memory::buffered_array<MEM,ComplexType,3> KEl(nb, nwalk, nke);
        R() = ComplexType(0.0);
        KEl() = ComplexType(0.0);

        // generate R matrices for current block of the excitation shell
        get_compact_ph_R_matrices<MEM>(iexb, refc, T, R);

        // calculate E and KE for current block of excitations
        HamOps.ph_excited_energy(Beta, nelec, iexb, refc, E,
                  wgt(range(idet+d0, idet+d0+nb),all), R, KEl, true);

        // eloc[iw] = sum_d_ke KE[d,iw,ke] KEl[d,iw,ke]
        nda::tensor::contract(ComplexType(1.0),KE(range(idet+d0,idet+d0+nb),all,all),"dwn",KEl,"dwn",
                              ComplexType(1.0),E(all,2),"w");
      }
      idet += ndet;
    }
  }
}

} // namespace sfqmc::afqmc

