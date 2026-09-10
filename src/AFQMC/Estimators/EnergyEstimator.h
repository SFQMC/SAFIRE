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

template<MEMORY_SPACE MEM>
class EnergyEstimator : public EstimatorBase<MEM>
{
public:
  /// `walkers_carry_energy` says whether the propagator already evaluated the local energy of
  /// this wavefunction and left its components on the walkers, in which case there is nothing
  /// to recompute here.
  EnergyEstimator(EnergyEstimatorParameters const& params, bool walkers_carry_energy, Wavefunction<MEM>& wfn)
      : wfn_{wfn},
        measure_interval_multiplier_{resolved(params.measure_interval_multiplier,
                                              "measure_interval_multiplier")},
        walkers_carry_energy_{walkers_carry_energy}
  {
  }

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock, Measurements& meas, WalkerSet<MEM> &wset) override {
    if(measureBlock % measure_interval_multiplier_ != 0) {
      return;
    }

    auto energy_time = timers.energy.start();

    auto all = nda::range::all;
    int nwalk = wset.size();

    memory::buffered_array<MEM,ComplexType,1> weights(wset.size());
    wset.getProperty(WEIGHT, weights);

    memory::buffered_array<MEM,ComplexType,2> localEnergy(nwalk,3);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nwalk);
    if(walkers_carry_energy_) {
      wset.getProperty(E1_, localEnergy(all, 0));
      wset.getProperty(EXX_, localEnergy(all, 1));
      wset.getProperty(EJ_, localEnergy(all, 2));
      wset.getProperty(OVLP, ovlp);
    } else {
      wfn_.Energy(wset, localEnergy, ovlp, wset.getTauStep());
    }

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
  bool walkers_carry_energy_{};
};
} // namespace afqmc
} // namespace sfqmc

