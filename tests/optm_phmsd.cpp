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

//#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "IO/AppAbort.hpp"

#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "test_common.hpp"
#include "utilities/memory_utils.hpp"
#include "IO/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <algorithm>

#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "numerics/sparse/sparse.hpp"

// erase
#include "numerics/device_kernels/kernels.h"
#include "utilities/Timer.hpp"

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

// only meant for device for now, problems if run in host with DEVICE build
struct ph_excit
{
  ph_excit(int n, int nd, int nel, int nact): nex(n), ndet(nd),
    occ(2*nex*ndet),refc(nel)
#if defined(ENABLE_DEVICE)
    ,occ_d(2*ndet*nex),refc_d(nel)
#endif
  {
    using nda::range;
   
    for(int i=0,j=0,a=0; i<nex; ++i) {
      occ(i) = j;  // can use a randon number in [0,nel)
      occ(i+nex) = a+nel;
      j = (j+1)%nel;
      a = (a+1)%(nact-nel);
    }
    for(int i=0, j=0; i<ndet; ++i, j+=2*nex)
      occ(range(j,j+2*nex)) = occ(range(2*nex));
    for(int i=0; i<nel; ++i) refc(i)=i;
#if defined(ENABLE_DEVICE)
     refc_d() = refc(); 
    occ_d() = occ();
#endif
  }

  std::array<long, 2> maximum_excitation_number() const { return {nex+1,nex+1}; }

  std::array<long, 2> number_of_unique_excitations(int n) const
  {
    if(n==nex) return {ndet,ndet};
    return {0,0};
  }

  auto get_excitation_list_device(int ispin, int iex)
  {
    utils::check(iex == nex, "Error: iex > nex");
#if defined(ENABLE_DEVICE)
    return occ_d();
#else
    return occ();
#endif
  }

  auto get_reference_configuration_device(int ispin)
  {
#if defined(ENABLE_DEVICE)
    return refc_d();
#else
    return refc();
#endif
  }

  int nex;
  int ndet;
  nda::array<int,1> refc;
  nda::array<int,1> occ;
#if defined(ENABLE_DEVICE)
  nda::cuarray<int,1> refc_d;
  nda::cuarray<int,1> occ_d;
#endif
};

template<MEMORY_SPACE MEM>
void optm_phmsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi)
{
  sfqmc::TimerManager Timer;

  int max_nex = 8; 
  //int ndet = 10;
  int ndet = 1000;
  //int nel = 3;
  //int nact = 6;
  int nel = 32;
  int nact = 64;
  //int nwalk = 2;
  int nwalk = 32;

  memory::array<MEM,ComplexType,3> T(nwalk,nact,nel);
  memory::array<MEM,ComplexType,2> ov(ndet+1,nwalk);
  memory::array<MEM,ComplexType,3> R(nwalk,nel,nact);
  T() = ComplexType(0.0);
  ov() = ComplexType(0.0);

  { 
    ph_excit abij(max_nex,ndet,nel,nact);
    kernels::device::calculate_R(0,abij,T,ov,R);
    utils::resize_nda_static_allocator();
  }

  for(int nex=1; nex<=max_nex; nex++)
  {
    ph_excit abij(nex,ndet,nel,nact);

    kernels::device::calculate_overlaps(0,abij,T,ov);
    Timer.start("t0");
    kernels::device::calculate_overlaps(0,abij,T,ov);
    Timer.stop("t0");
    double t0 = Timer.elapsed("t0");

    kernels::device::calculate_R(0,abij,T,ov,R);
    Timer.start("t0");
    kernels::device::calculate_R(0,abij,T,ov,R);
    Timer.stop("t0");
    sfqmc::app_log(0,"Time: nex: {} ov: {} R: {}",nex,t0,Timer.elapsed("t0")); Timer.reset("t0");
  }
}

TEST_CASE("optm_phmsd", "[read_phmsd]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

#if defined(ENABLE_DEVICE)
  optm_phmsd<DEVICE_MEMORY>(mpi);
#endif
}

} // namespace sfqmc
