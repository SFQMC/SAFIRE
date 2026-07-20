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
#include <queue>
#include <string>
#include <iostream>
#include <fstream>

#include "nda/h5.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"


namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class BasicEstimator : public EstimatorBase<MEM>
{
public:
  BasicEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi, [[maybe_unused]] std::string title, ptree pt_in, bool impsamp_)
      : mpi(_mpi), nwfacts(0), importanceSampling(impsamp_), timers(false)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(1,"BasicEstimator input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    timers = pt.get<bool>("timers");
    nwfacts = pt.get<int>("nhist");
    int population_control_interval = pt.get<int>("_population_control_interval");
    measure_interval = pt.get<int>("measure_interval_multiplier") * population_control_interval;

    utils::check(nwfacts >= 0, "Error: nwfacts<0");
    weight_product = ComplexType(1.0, 0.0);
    for (int i = 0; i < nwfacts; i++)
      weight_factors.push(weight_product);

    app_log(1,"  BasicEstimator: Number of products in weight history: {}", nwfacts);

    data.resize(10);
    data2.resize(10);
    data3.resize(2);

    if (timers)
      AFQMCTimer.reset_all();

    enume          = 0.0;
    edeno          = 0.0;
    enume_sub      = 0.0;
    edeno_sub      = 0.0;
    enume2         = 0.0;
    edeno2         = 0.0;
    weight         = 0.0;
    weight_sub     = 0.0;
    nwalk          = 0;
    nwalk_good     = 0;
    nwalk_sub      = 0;
    ncalls         = 0;
    ncalls_substep = 0;
    nwalk_min      = 1000000;
    nwalk_max      = 0;
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool timers = pt0.get<bool>("timers", false);
    int nhist = pt0.get<int>("nhist", 0);
    int measure_interval_multiplier = pt0.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    int population_control_interval = pt0.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("timers", timers);
    pt1.put("nhist", nhist);
    pt1.put("measure_interval_multiplier", measure_interval_multiplier);
    pt1.put("_population_control_interval", population_control_interval);
    io::compare_known_keys("Basic Estimator",pt1, pt0);
    return pt1;
  }

  ~BasicEstimator() {}

  void accumulate_block([[maybe_unused]] double time, [[maybe_unused]] WalkerSet<MEM>& wset) {}


  //  curData:
  //  0: inverse of the factor used to rescale the weights
  //  1: 1/nW * sum_i w_i * Eloc_i   (where w_i is the normalized weight)
  //  2: 1/nW * sum_i w_i            (where w_i is the normalized weight)
  //  3: sum_i abs(w_i)       (where w_i is the normalized weight)
  //  4: 1/nW * sum_i abs(<psi_T|phi_i>)
  //  5: nW                          (total number of walkers)
  //  6: "healthy" nW                (total number of "healthy" walkers)
  void accumulate_step([[maybe_unused]] double time, WalkerSet<MEM>& wset, std::vector<ComplexType>& curData)
  {
    ncalls++;
    if (nwfacts > 0)
    {
      weight_product *= (curData[0] / weight_factors.front());
      weight_factors.pop();
      weight_factors.push(curData[0]);
    }
    else
      weight_product = ComplexType(1.0, 0.0);

    data2[0] = curData[1].real();
    data2[1] = curData[2].real();

    int nwlk = wset.size();
    if (nwlk > nwalk_max)
      nwalk_max = nwlk;
    if (nwlk < nwalk_min)
      nwalk_min = nwlk;
    enume += (curData[1] / curData[2]) * weight_product;
    edeno += weight_product;
    weight += curData[3].real();
    ovlp += curData[4].real();
    nwalk += static_cast<int>(std::floor(curData[5].real()));
    nwalk_good += static_cast<int>(std::floor(curData[6].real()));
  }

  void tags(std::ofstream& out)
  {
    if (mpi->comm.root())
    {
      if (nwfacts > 0)
      {
        out << "nWalkers weight Eloc_nume Eloc_deno ";
      }
      else
      {
        out << "nWalkers weight PseudoEloc ";
      }
      out << "LogOvlp ";
    }
  }

  void tags_timers(std::ofstream& out)
  {
    if (mpi->comm.root())
      if (timers)
        out << "PseudoEnergy_t vHS_t vbias_t G_t Propagate_t Energy_comm_t vHS_comm_t X_t popC_t ortho_t setup_t "
               "extra_t Block_t ";
  }

  int get_measurement_interval()
  {
    return measure_interval;
  }

  void print(std::ofstream& out, [[maybe_unused]] h5::file& file, [[maybe_unused]] WalkerSet<MEM>& wset)
  {
    if (ncalls ==0) 
      APP_ABORT("Estimator has no data but asked to print (ncalls=0), check settings");
    data[0] = enume.real() / ncalls;
    data[1] = edeno.real() / ncalls;

    if (mpi->comm.root())
    {
      out << std::setprecision(6) << nwalk / ncalls << " " << weight / ncalls << " " << std::setprecision(16);
      if (nwfacts > 0)
      {
        out << enume.real() / ncalls << " " << edeno.real() / ncalls << " ";
      }
      else
      {
        out << enume.real() / ncalls << " ";
      }
      out << ovlp / ncalls << " ";
    }

    enume      = 0.0;
    edeno      = 0.0;
    weight     = 0.0;
    enume2     = 0.0;
    edeno2     = 0.0;
    ncalls     = 0;
    nwalk      = 0;
    nwalk_good = 0;
    nwalk_min  = 1000000;
    nwalk_max  = 0;
    ovlp       = 0;
  }

  void print_timers(std::ofstream& out)
  {

    if (mpi->comm.root())
    {
      if (timers)
        out << std::setprecision(5) << AFQMCTimer.elapsed(pseudo_energy_timer) << " "
            << AFQMCTimer.elapsed(vHS_timer) << " " << AFQMCTimer.elapsed(vbias_timer) << " "
            << AFQMCTimer.elapsed(G_for_vbias_timer) << " " << AFQMCTimer.elapsed(propagate_timer) << " "
            << AFQMCTimer.elapsed(E_comm_overhead_timer) << " "
            << AFQMCTimer.elapsed(vHS_comm_overhead_timer) << " " << AFQMCTimer.elapsed(assemble_X_timer)
            << " " << AFQMCTimer.elapsed(popcont_timer) << " " << AFQMCTimer.elapsed(ortho_timer) << " "
            << AFQMCTimer.elapsed(setup_timer) << " " << AFQMCTimer.elapsed(extra_timer) << " "
            << AFQMCTimer.elapsed(block_timer) << " " << std::setprecision(16);
    }
    if (timers)
      AFQMCTimer.reset_all();
  }

  double getEloc() { return data[0] / data[1]; }

  double getEloc_step() { return data2[0] / data2[1]; }


private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  int nwfacts;

  [[maybe_unused]] bool importanceSampling;

  nda::array<double,1> data, data2, data3;

  std::queue<ComplexType> weight_factors;
  ComplexType weight_product = ComplexType(1.0, 0.0);

  ComplexType enume = 0.0, edeno = 0.0;
  ComplexType enume_sub = 0.0, edeno_sub = 0.0;
  ComplexType enume2 = 0.0, edeno2 = 0.0;
  RealType weight = 0.0, weight_sub = 0.0, ovlp = 0.0;
  int nwalk_good, nwalk, ncalls, ncalls_substep, nwalk_sub, nwalk_min, nwalk_max;

  // this is used for scheduling "accumulate_block" calls
  int measure_interval = 1;

  // optional
  bool timers;
};
} // namespace afqmc
} // namespace sfqmc

