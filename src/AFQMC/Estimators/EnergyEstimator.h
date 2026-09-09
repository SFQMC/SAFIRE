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
#include <vector>

#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "EstimatorBase.h"
#include "Measurements.hpp"


namespace sfqmc
{
namespace afqmc
{

// use atan2 to enforce [-pi,pi] range of values
inline ComplexType mod2pi(ComplexType x){
  RealType x_r = std::real(x);
  RealType x_i = std::imag(x);
  return ComplexType(atan2(sin(x_r),cos(x_r)),
                     atan2(sin(x_i),cos(x_i))
                     );
}

template<MEMORY_SPACE MEM>
class EnergyEstimator : public EstimatorBase<MEM>
{
public:
  EnergyEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                  const EstimatorParameters& params,
                  int measure_interval_,
                  Wavefunction<MEM>& wfn)
      : wfn_{wfn},
        measure_interval_multiplier_{resolved(params.measure_interval_multiplier, "measure_interval_multiplier").at(0)}
  {
  }

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock, Measurements& meas, WalkerSet<MEM> &wset) override {
    if(measureBlock % measure_interval_multiplier_ != 0) {
      return;
    }

    auto energy_time = timers.energy.start();
    
    int nwalk = wset.size();
    
    memory::buffered_array<MEM,ComplexType,1> weights(wset.size());
    wset.getProperty(WEIGHT, weights);
    
    memory::buffered_array<MEM,ComplexType,2> localEnergy(nwalk,3);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nwalk);
    wfn_.Energy(wset, localEnergy, ovlp, wset.getTauStep());

    MeasurementOutput output{mpi, meas, "", weights};

    memory::buffered_array<MEM,ComplexType,1> avgLocalEnergy(3);
    nda::blas::gemv(nda::transpose(localEnergy), weights, avgLocalEnergy);

    auto avgLocalEnergy_h = nda::to_host(avgLocalEnergy);

    output.measure(mpi, "Energy", nda::sum(avgLocalEnergy_h));
    output.measure(mpi, "OnebodyEnergy", avgLocalEnergy_h(0));
    output.measure(mpi, "ExchangeEnergy", avgLocalEnergy_h(1));
    output.measure(mpi, "CoulombEnergy", avgLocalEnergy_h(2));
    output.measure(mpi, "Overlap", nda::blas::dot(ovlp, weights));
  }

private:
  Wavefunction<MEM>& wfn_;
  int measure_interval_multiplier_{};
};
} // namespace afqmc
} // namespace sfqmc

