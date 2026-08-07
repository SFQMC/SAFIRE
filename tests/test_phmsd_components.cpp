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
    Watch timer;
    ph_excited_2body_energy_dense_cholesky(iexcit_d, refc_d, Tna_d, R_d, wgt_d, EX, EJ, KE);
    timer.reset();
    

    app_log(1, "{{{:15}, {:15}}}", nda::to_host(EX)[0], nda::to_host(EJ)[0]);
    
    CHECK_THAT(nda::to_host(EX)(0), utils::Approx(EXJ_ref(nex-1, 0)));
    CHECK_THAT(nda::to_host(EJ)(0), utils::Approx(EXJ_ref(nex-1, 1)));
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
