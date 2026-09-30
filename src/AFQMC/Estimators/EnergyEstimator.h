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
  /// to recompute here. `timestep` is the one the stage propagates with.
  EnergyEstimator(EnergyEstimatorParameters const& params, bool walkers_carry_energy, RealType timestep, Wavefunction<MEM>& wfn)
      : wfn_{wfn},
        measure_interval_{resolved(params.measure_interval, "measure_interval")},
        walkers_carry_energy_{walkers_carry_energy},
        timestep_{timestep}
  {
  }

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long step, Measurements& meas, WalkerSet<MEM> &wset) override {
    if(step % measure_interval_ != 0) {
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
    nda::tensor::scale(1.0, ovlp, nda::tensor::unary_op::EXP);
    output.measure(mpi, "Overlap", nda::blas::dot(ovlp, weights));

    WeightStatistics const weight_stats = weight_statistics(mpi, weights);
    if(mpi.comm.root()) {
      // the driver measures after propagating, so step 0 is one timestep into the stage. A
      // finite temperature sample is a whole sweep, which always ends at the same beta
      if(!wset.isFiniteTemperature()) {
        meas.measure("ProjectionTime", (step + 1) * timestep_);
      }
      meas.measure("EffectiveNumWalkers", weight_stats.effective_num_walkers);
      meas.measure("Phase", weight_stats.phase);
      meas.measure("TotalWeight", weight_stats.total_weight);
    }
  }

private:
  Wavefunction<MEM>& wfn_;

  // in units of steps
  int measure_interval_{};
  bool walkers_carry_energy_{};
  RealType timestep_{};
};
} // namespace afqmc
} // namespace sfqmc

