/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <cassert>
#include "nda/nda.hpp"
#include "nda/tensor.hpp"

#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/device_kernels/to_view.hpp"
#endif


namespace sfqmc::afqmc 
{

// E[w] = sum_abpqdn w[d][w] T[w][i[p]][n][a] * T[w][i[q]][n][b] * Rwdpa * Rwdqb 
// KE[d][w][n] = sum_pa T[w][i[p]][a][n] * Rwdpa
// R is the compact version of the ph R matrix, where for each determinant there
// are nex rows
void ph_excited_2body_energy_dense_cholesky(nda::MemoryVector auto const& iexcit, nda::MemoryVector auto const& refc,
              nda::MemoryArrayOfRank<4> auto const& Twina, nda::MemoryArrayOfRank<4> auto const& R,
              nda::MemoryMatrix auto const& wgt, nda::MemoryVector auto&& EX,
              nda::MemoryVector auto&& EJ, nda::MemoryArrayOfRank<3> auto && KE)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(Twina)>();
  memory::check_memory_space<MEM>(iexcit,refc,Twina,R,wgt,EX,EJ,KE);
  auto [nwalk, ndet, nex, nact] = R.shape();
  long nelec = Twina.extent(1);
  long nchol = Twina.extent(2);

  utils::check(wgt.shape() == std::array<long,2>{ndet,nwalk}, "Size mismatch");
  utils::check(EX.shape() == std::array<long,1>{nwalk}, "Size mismatch");
  utils::check(EJ.shape() == std::array<long,1>{nwalk}, "Size mismatch");
  utils::check(KE.shape() == std::array<long,3>{ndet,nwalk,nchol}, "Size mismatch");
  utils::check(Twina.extent(0)==nwalk and Twina.extent(3)==nact, "Size mismatch");

  if constexpr (MEM==HOST_MEMORY) {
    nda::vector<int> occps(nelec);
    ComplexType zero(0.0), one(1.0), two(2.0), half(0.5);  
    memory::buffered_array<MEM,ComplexType,2> Fbuff(2,std::max(nact, nchol));
    nda::array_view<nda::get_value_t<decltype(iexcit)> const,3> iex(std::array<long,3>{ndet,2,nex}, iexcit.data());

    for(int iw=0; iw<nwalk; iw++) {
      for(int idet=0; idet<ndet; idet++) {

        auto Fa = Fbuff(0,range(nact));
        auto Fn = Fbuff(0,range(nchol));
        auto Fn1 = Fbuff(1,range(nchol));
        auto Kn = KE(idet,iw,all);
        auto Rxa = R(iw,idet,all,all); 
        Kn() = zero; 

        ComplexType eX(0.0);
        for(int i=0; i<nelec; i++) {
          occps(i) = refc(i);
          for(int ie1=0; ie1<nex; ie1++) {
            int ip = iex(idet,0,ie1);
            if(ip == i) {
              occps(i) = iex(idet,1,ie1); 
              break;
            }
          }
        }
        // spin-diagonal part of the kinetic energy of the reference configuration,
        // since the routine produces EJ-EJref (including only the spin-diagonal part) 
        // Fn = sum_i Twina(iw,i,n,refc(i))
        Fn = zero;
        for(int i=0; i<nelec; i++) 
          Fn += Twina(iw,i,range(nchol),refc(i));

        ComplexType eJ0 = nda::sum( Fn*Fn );

        for(int ie1=0; ie1<nex; ie1++) {
          int ip = iex(idet,0,ie1);

          // ie1==ie2 term
          nda::blas::gemv(one,Twina(iw,ip,all,all),Rxa(ie1,all),zero,Fn);
          eX += nda::sum( Fn*Fn );
          Kn() += Fn;

          // R[p]*R[q] terms
          for(int ie2=ie1+1; ie2<nex; ie2++) {
            int iq = iex(idet,0,ie2);
            // Twq[n][a] * Rwp[a] = Fn
            nda::blas::gemv(one,Twina(iw,iq,all,all),Rxa(ie1,all),zero,Fn);
            // Twp[n][b] * Rwq[b] = Fn
            nda::blas::gemv(one,Twina(iw,ip,all,all),Rxa(ie2,all),zero,Fn1);
            eX += two * nda::sum( Fn*Fn1 ); 
          }

          // R[p]*R[diagonal] term
          for(int j=0; j<nelec; j++) {
            int Oj = occps(j); 
            nda::blas::gemv(one,nda::transpose(Twina(iw,j,all,all)),Twina(iw,ip,all,Oj),zero,Fa);
            eX += two * nda::sum( Fa*Rxa(ie1,all) ); 
          }

        }  

        // Rdiag-Rdiag terms
        for(int i=0; i<nelec; i++) {
          int Oi = occps(i);
          int ri = refc(i);
          if( Oi!=ri )
            eX += nda::sum( Twina(iw,i,all,Oi)*Twina(iw,i,all,Oi) ) - nda::sum( Twina(iw,i,all,ri)*Twina(iw,i,all,ri) ); 

          for(int j=i+1; j<nelec; j++) {
            int Oj = occps(j);
            int rj = refc(j);
            // Rdiag-Rdiag terms
            if( Oi!=ri or Oj!=rj )
              eX += two * ( nda::sum( Twina(iw,i,all,Oj)*Twina(iw,j,all,Oi) ) - nda::sum( Twina(iw,i,all,rj)*Twina(iw,j,all,ri) )); 
          }
        }
        // R[diagonal]*R[diagonal] J-term
        for(int i=0; i<nelec; i++) 
          Kn() += Twina(iw,i,all,occps(i));

        EX(iw) -= half * wgt(idet,iw) * eX;
        EJ(iw) += half * wgt(idet,iw) * ( nda::sum( Kn*Kn )  - eJ0 );
      }
    }
  } else {
#if defined(ENABLE_DEVICE)
    using kernels::device::to_view;
    // the per-walker slicing is nda's job, so it stays on this side of the boundary
    auto EX_v = to_view(EX);
    auto EJ_v = to_view(EJ);
    for(long iw = 0; iw < nwalk; ++iw) {
      kernels::device::ph_excited_2body_energy_dense_cholesky_Tpna_walker(
          refc.data(), iexcit.data(), to_view(Twina(iw,nda::ellipsis{})),
          to_view(R(iw,nda::ellipsis{})), to_view(wgt(all,iw)), EX_v, EJ_v, iw,
          to_view(KE(all,iw,all)));
    }
#else
    sfqmc::utils::check(false,"Error: Missing device function.");
#endif
  }
}

void ph_excited_1body_energy(nda::MemoryVector auto const& iexcit, nda::MemoryVector auto const& refc, 
                  nda::MemoryArrayOfRank<3> auto const& S, nda::MemoryArrayOfRank<4> auto const& R, 
                  nda::MemoryMatrix auto const& wgt, nda::MemoryVector auto&& E)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(S)>(); 
  memory::check_memory_space<MEM>(iexcit,refc,S,R,wgt,R);
  auto [nwalk, ndet, nex, nact] = R.shape();
  int nelec = S.extent(1);

  utils::check(S.shape() == std::array<long,3>{nwalk,nelec,nact}, "Size mismatch");
  utils::check(wgt.shape() == std::array<long,2>{ndet,nwalk}, "Size mismatch");
  utils::check(E.shape() == std::array<long,1>{nwalk}, "Size mismatch");

  if constexpr (MEM==HOST_MEMORY) {
    nda::array_view<nda::get_value_t<decltype(iexcit)> const,3> iex(std::array<long,3>{ndet,2,nex}, iexcit.data());
    for (int iw = 0; iw < nwalk; iw++) {
      for (int d = 0; d < ndet; d++) {
        ComplexType e_(0.0);
        for (int p = 0; p < nex; p++) {
          int ip = iex(d,0,p); 
          e_ += nda::sum(S(iw,ip,all)*R(iw,d,p,all)) 
               + S(iw,ip,iex(d,1,p)) - S(iw,ip,refc(ip));
        }
        E(iw) += wgt(d,iw) * e_;
      }
    }
  } else {
#if defined(ENABLE_DEVICE)
    using kernels::device::to_view;
    kernels::device::ph_excited_1body_energy(refc.data(),iexcit.data(),to_view(S),to_view(R),
                                             to_view(wgt),to_view(E));
#else
    sfqmc::utils::check(false,"Error: Missing device function.");
#endif
  }
}

} // sfqmc::afqmc
