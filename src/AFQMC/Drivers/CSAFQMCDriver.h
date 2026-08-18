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

#ifndef SFQMC_AFQMC_CSAFQMCDRIVER_H
#define SFQMC_AFQMC_CSAFQMCDRIVER_H

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "mpi3/communicator.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorHandler.h"

namespace sfqmc
{
namespace afqmc
{
class CSAFQMCDriver 
{
public:
  CSAFQMCDriver(boost::mpi3::communicator& comm,
              std::string& title,
              int mser,
              int blk0,
              int stp0,
              std::vector<double>&& eshft_,
              ptree pt_in,
              std::vector<std::reference_wrapper<AFQMCInfo>>&& info_,
              std::vector<std::reference_wrapper<Wavefunction>>&& wfn_,
              std::vector<std::reference_wrapper<Propagator>>&& prpg_,
              std::vector<EstimatorHandler>&& estim_)
     :  globalComm(comm),
        m_series(mser),
        project_title(title),
        block0(blk0),
        step0(stp0),
        info_ref(std::move(info_)),
        wfn_ref(std::move(wfn_)),
        prop_ref(std::move(prpg_)),
        estimators(std::move(estim_)),
        weight_reset_period(0.0),
        Eshift(std::move(eshft_))
  {
    if(info_ref.size() != wfn_ref.size() or
       info_ref.size() != prop_ref.size() or
       info_ref.size() != estimators.size() or
       info_ref.size() != Eshift.size() )
      APP_ABORT("Error in CSAFQMCDriver::CSAFQMCDriver(): Incompatible dimensions."); 
    name = "CSAFQMCDriver";
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2, "\nCSAFQMCDriver input:");
    app_log(2, "{}\n", io::to_string(pt));
    // initialize using verbose input
    hdf_write_restart = pt.get<std::string>("hdf_write_file");
    nBlock = pt.get<int>("blocks");
    nStep = pt.get<int>("steps");
    nSubstep = pt.get<int>("substeps");
    fix_bias = pt.get<int>("fix_bias");
    nStabilize = pt.get<int>("ortho");
    nCheckpoint = pt.get<int>("checkpoint");
    samplePeriod = pt.get<int>("sample_period");
    resetPeriod = pt.get<int>("reset_period");
    reset_reweight = pt.get<bool>("reset_reweight");
    combine_type = pt.get<std::string>("combine_type");
    weight_reset_period = pt.get<double>("weight_reset"); // in units of time
    dt = pt.get<double>("timestep");
    dShift = pt.get<double>("dshift");  // Etrial shift scale
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    auto hdf_write_file = pt0.get<std::string>("hdf_write_file", "");
    auto blocks        = pt0.get<int>("blocks", 100);
    auto steps         = pt0.get<int>("steps", 1);
    auto substeps      = pt0.get<int>("substeps", 1);
    auto fix_bias      = pt0.get<int>("fix_bias", 1);
    auto ortho         = pt0.get<int>("ortho", 1);
    auto checkpoint    = pt0.get<int>("checkpoint", -1);
    auto sample_period = pt0.get<int>("sample_period", -1);
    auto reset_period = pt0.get<int>("reset_period", -1);
    auto reset_reweight = pt0.get<bool>("reset_reweight", true);
    auto weight_reset = pt0.get<double>("weight_reset", 0.0);
    auto timestep     = pt0.get<double>("timestep", 0.01);
    auto dshift       = pt0.get<double>("dshift", 1.0);
    auto combine_type = pt0.get<std::string>("combine_type", "max");
    // validate inputs
    fix_bias = std::min(fix_bias, substeps);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("hdf_write_file", hdf_write_file);
    pt1.put("blocks", blocks);
    pt1.put("steps", steps);
    pt1.put("substeps", substeps);
    pt1.put("fix_bias", fix_bias);
    pt1.put("ortho", ortho);
    pt1.put("checkpoint", checkpoint);
    pt1.put("sample_period", sample_period);
    pt1.put("reset_period", reset_period);
    pt1.put("reset_reweight", reset_reweight);
    pt1.put("weight_reset", weight_reset);
    pt1.put("timestep", timestep);
    pt1.put("dshift", dshift);
    pt1.put("combine_type",combine_type);
    // check for unknown input keys
    std::unordered_set<std::string> pass_through_keys = {
      "walker_set",
      "wavefunction",
      "propagator",
      "estimator",
      "hamiltonian"
    };
    io::compare_known_keys("correlated-sampling (CS) AFQMC Driver",pt1, pt0, pass_through_keys);
    return pt1;
  }

  bool run(std::vector<std::reference_wrapper<WalkerSet>>&);

  bool checkpoint(WalkerSet&, int, int);

  bool clear();

protected:
  boost::mpi3::communicator& globalComm;

  std::string name;

  int m_series;
  std::string project_title;

  std::string hdf_write_restart;

  int nBlock;
  int nStep;
  int nSubstep;
  int fix_bias;

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  int block0, step0;
  std::string combine_type = "max";

  std::vector<std::reference_wrapper<AFQMCInfo>> info_ref;

  std::vector<std::reference_wrapper<Wavefunction>> wfn_ref;

  std::vector<std::reference_wrapper<Propagator>> prop_ref;

  std::vector<EstimatorHandler> estimators;

  bool writeSamples(WalkerSet&);

  int samplePeriod;
  int resetPeriod;
  bool reset_reweight;

  double weight_reset_period;

  RealType dShift;
  std::vector<RealType> Eshift;
  std::vector<RealType> Etav;
};

} // namespace afqmc
} // namespace sfqmc

#endif
