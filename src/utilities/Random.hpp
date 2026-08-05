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

#include <ctime>
#include <vector>
#include <random>
#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/operations/tensor.hpp"

#if defined(ENABLE_DEVICE)
#include "curand.h"
#endif

namespace sfqmc {
namespace utils {

using SeedType = unsigned long long;

SeedType make_seed(boost::mpi3::communicator& comm);
SeedType split_seed(int seed, boost::mpi3::communicator& comm);

struct HostRandomGenerator {
  std::mt19937 std_rng; // still used directly in popcontrol

  HostRandomGenerator(SeedType iseed) : std_rng{std::mt19937::result_type(iseed)} {};
  HostRandomGenerator() = default;
  
  template<nda::MemoryVector Vec>
  requires( nda::mem::on_host<Vec> )
  void sampleUniformFields(Vec && V) {
    using T = nda::get_value_t<Vec>;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for(long i=0; i<V.extent(0); i++)
      V(i) = T(distribution(std_rng));
  }
};  

#if defined(ENABLE_DEVICE)
class CurandRandomGenerator {
private:
  curandGenerator_t handle_ = nullptr;
public:
  CurandRandomGenerator(SeedType iseed);
  ~CurandRandomGenerator();

  CurandRandomGenerator(CurandRandomGenerator const&) = delete;
  CurandRandomGenerator& operator=(CurandRandomGenerator const&) = delete;
  // curandGenerator_t is a raw handle, so the defaulted moves would leave both
  // objects owning it and destroy it twice
  CurandRandomGenerator(CurandRandomGenerator&& other) noexcept;
  CurandRandomGenerator& operator=(CurandRandomGenerator&& other) noexcept;

  template<nda::MemoryVector Vec>
  requires(nda::mem::on_device<Vec>)
  void sampleUniformFields(Vec && V){
    using T = nda::get_value_t<Vec>;
    static_assert(std::is_same_v<T,double> or std::is_same_v<T,std::complex<double>>,
                  "sampleUniformFields: Type not implemented.");
    if constexpr (std::is_same_v<T,double>) {
      cuda::curand_check(curandGenerateUniformDouble(handle_, reinterpret_cast<double*>(V.data()), V.size()),
                           "curandGenerateUniformDouble");
    } else if constexpr (std::is_same_v<T,std::complex<double>>) {
      cuda::curand_check(curandGenerateUniformDouble(handle_, reinterpret_cast<double*>(V.data()), 2 * V.size()),
                           "curandGenerateUniformDouble");
      math::zero_imag(V);
    } 
    arch::synchronize_if_set();
  }
};
#endif


#if defined(ENABLE_DEVICE)
template<MEMORY_SPACE MEM = HOST_MEMORY>
using RandomGenerator_t = std::conditional_t<MEM==DEVICE_MEMORY,CurandRandomGenerator,HostRandomGenerator>; 
#else
template<MEMORY_SPACE MEM = HOST_MEMORY>
using RandomGenerator_t = HostRandomGenerator; 
#endif



}
}

