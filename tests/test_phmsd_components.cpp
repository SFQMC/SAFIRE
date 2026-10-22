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

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "configuration.hpp"

#include "utilities/h5_utils.hpp"
#include "test_common.hpp"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "test_utils.hpp"
#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void phmsd_components_ph_excited_energy_real_dense_cholesky() {
  

  int ndet = 50;
  int nelec = 5;
  int nact = 10;
  int nchol = 50;
  int nw = 1;

  nda::matrix<ComplexType> EXJ_ref = {
    {{-3.710389910208285,-666.8601775126258}, {-448.4580805193352,260.52361637250124}},
    {{257.47347099950423,1000.2224737398957}, {-823.4796831379458,-1004.1196065341555}},
    {{782.6224431929506,-1540.7376072631007}, {-183.51974621436153,85.49030799002679}},
    {{2023.4979464340804,-36.30962737644677}, {1682.3242532758773,-442.1588977259531}},
    {{314.60033183810424,-4993.354550714809}, {-1238.764790338722,-604.3447621988698}},
  };
  
  //for(int nex=1; nex<21; nex++) 
  for(int nex=1; nex<6; nex++) 
  {
//    std::cout<<" # excitations: " <<nex <<std::endl;
    nda::array<ComplexType, 4> Tna(nw, nelec, nchol, nact);
    nda::array<ComplexType, 4> R(nw, ndet, nex, nact);
    nda::array<ComplexType, 2> wgt(ndet, nw); 
    nda::array<int, 2> orbs(ndet,nelec);
    nda::array<int, 2> iexcit(ndet,2*nex);
    nda::array<int, 1> refc(nelec);

    for(int i=0; i<nelec; i++) {
      refc(i) = i;
    }
    
    {
      nda::vector<ComplexType> tmp( std::max(Tna.size(), R.size()) );
      for(int i=0; i<Tna.size(); ++i) {
        nda::flatten(Tna)(i) = ComplexType(sin(i), cos(i*i));
      }
      for(int i=0; i<R.size(); ++i) {
        nda::flatten(R)(i) = ComplexType(sin(i*3), cos(i+i*i));
      }
      for(int i=0; i<wgt.size(); ++i) {
        nda::flatten(wgt)(i) = ComplexType(sin(i*2), cos(i-2*i*i));
      }

      {  
        nda::vector<int> iocc(nelec, 0);
        for(int nd=0; nd<ndet; nd++) {
          iocc() = 0;
          for(int i=0; i<nex; i++) {
            int v = (i*i) % nelec;
            while(iocc(v) != 0) {
              v = (v + 1) % nelec;
            }
            iocc(v)=1;
            iexcit(nd,i) = v;
          }
        }
      }  
      {
        nda::vector<int> iocc(nact-nelec, 0);
        for(int nd=0; nd<ndet; nd++) {
          iocc() = 0;
          for(int i=0; i<nex; i++) {
            int v = (i+i*i) % (nact-nelec);
            while(iocc(v) != 0) {
              v = (v+1) % (nact-nelec);
            }
            iocc(v) = 1;
            iexcit(nd,i+nex) = v+nelec;
          }
        }
      }
    }
    memory::array<MEM, ComplexType, 3> KE(ndet, nw, nchol);
    memory::array<MEM,ComplexType,1> EJ(nw, 0.0);
    memory::array<MEM,ComplexType,1> EX(nw, 0.0);
    auto iexcit_d = memory::to_memory_space<MEM>(nda::flatten(iexcit));
    auto refc_d = memory::to_memory_space<MEM>(refc);
    auto Tna_d = memory::to_memory_space<MEM>(Tna);
    auto R_d = memory::to_memory_space<MEM>(R);
    auto wgt_d = memory::to_memory_space<MEM>(wgt);
    ph_excited_2body_energy_dense_cholesky(iexcit_d, refc_d, Tna_d, R_d, wgt_d, EX, EJ, KE);

    app_log(1, "{{{:15}, {:15}}}", nda::to_host(EX)[0], nda::to_host(EJ)[0]);
    
    CHECK_THAT(nda::to_host(EX)(0), utils::Approx(EXJ_ref(nex-1, 0)));
    CHECK_THAT(nda::to_host(EJ)(0), utils::Approx(EXJ_ref(nex-1, 1)));
    {
      // The exact integral-cache path must preserve complex (non-Hermitian)
      // contractions, walker strides and every excitation rank.
      constexpr int ntest=3;
      nda::array<ComplexType,4> Ttest(ntest,nelec,nchol,nact), Rtest(ntest,ndet,nex,nact);
      nda::matrix<ComplexType> wtest(ndet,ntest);
      nda::array<ComplexType,5> X(ntest,nelec,nact,nelec,nact);
      nda::array<ComplexType,3> Tan(nelec,nact,nchol), Ktest(ndet,ntest,nchol);
      nda::vector<ComplexType> extest(ntest,0.), ejtest(ntest,0.);
      auto all=nda::range::all;
      for(int iw=0; iw<ntest; ++iw) {
        ComplexType scale(1.+.1*iw,.07*iw), weight(1.-.1*iw,.03*iw);
        Ttest(iw,all,all,all)=scale*Tna(0,all,all,all);
        Rtest(iw,all,all,all)=R(0,all,all,all);
        wtest(all,iw)=weight*wgt(all,0);
        nda::tensor::add(ComplexType(1),Ttest(iw,all,all,all),"ina",ComplexType(0),Tan,"ian");
        auto tm=nda::reshape(Tan,std::array<long,2>{nelec*nact,nchol});
        auto xm=nda::reshape(X(iw,all,all,all,all),std::array<long,2>{nelec*nact,nelec*nact});
        nda::blas::gemm(ComplexType(1),tm,nda::transpose(tm),ComplexType(0),xm);
      }
      if constexpr(MEM==HOST_MEMORY) {
        ph_excited_2body_energy_integrals(iexcit_d,refc_d,Ttest,Rtest,wtest,extest,ejtest,Ktest,X);
      } else {
#if defined(ENABLE_DEVICE)
        using kernels::device::to_view;
        auto td=memory::to_memory_space<MEM>(Ttest), rd=memory::to_memory_space<MEM>(Rtest);
        auto xd=memory::to_memory_space<MEM>(X);
        auto wd=memory::to_memory_space<MEM>(wtest);
        memory::array<MEM,ComplexType,4> tn(ntest,nelec,nact,nchol);
        memory::array<MEM,ComplexType,3> f(ntest,nelec,nact), kt(ndet,ntest,nchol);
        memory::array<MEM,ComplexType,2> k0(ntest,nchol);
        memory::array<MEM,ComplexType,1> exd(ntest,0.), ejd(ntest,0.);
        nda::tensor::add(ComplexType(1),td,"wina",ComplexType(0),tn,"wian");
        kernels::device::phmsd_integral_reference(refc_d.data(),to_view(tn),to_view(xd),to_view(f),to_view(k0));
        kernels::device::phmsd_integral_energy(refc_d.data(),iexcit_d.data(),to_view(tn),to_view(xd),
            to_view(f),to_view(k0),to_view(rd),to_view(wd),to_view(exd),to_view(ejd),to_view(kt));
        extest=nda::to_host(exd); ejtest=nda::to_host(ejd); Ktest=nda::to_host(kt);
#endif
      }
      auto Kh=nda::to_host(KE);
      for(int iw=0; iw<ntest; ++iw) {
        ComplexType scale(1.+.1*iw,.07*iw), weight(1.-.1*iw,.03*iw);
        CHECK_THAT(extest(iw),utils::Approx(weight*scale*scale*EXJ_ref(nex-1,0)));
        CHECK_THAT(ejtest(iw),utils::Approx(weight*scale*scale*EXJ_ref(nex-1,1)));
        CHECK_THAT(Ktest(all,iw,all),utils::Approx(nda::make_regular(scale*Kh(all,0,all))));
      }
    }
    //    std::cout<<"    " <<tcpu1 <<" " <<tcpu2 <<std::endl;
  }  
}

TEST_CASE("phmsd_components: ph excited energy (real dense cholesky)", "[phmsd_components]") {
  phmsd_components_ph_excited_energy_real_dense_cholesky<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  phmsd_components_ph_excited_energy_real_dense_cholesky<DEVICE_MEMORY>();
#endif
}

} // namespace sfqmc
