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

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Estimators/TimeEvolvedObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace detail
{

inline int get_number_of_averages(ptree pt)
{
  std::vector<int> nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt, "measure_interval_multiplier", {DEFAULT_MEASURE_INTERVAL_MULTIPLIER});
  return nback_prop_interval_multipliers.size();
}

}

/*
 * Implements back propagation by evolving operators forward in time.
 * Based on 10.1103/PhysRevA.100.023621.
 */
template<MEMORY_SPACE MEM>
class BPWithTimeEvolvedOperators : public EstimatorBase<MEM>
{

public:
  BPWithTimeEvolvedOperators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                          std::string name,
                          ptree pt_in,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop,
                          bool impsamp_ = true)
      : mpi(_mpi),
        walker_type(wset.getWalkerType()),
        nspin( (walker_type==COLLINEAR) ? 2 : 1 ),
        npol( (walker_type==NONCOLLINEAR) ? 2 : 1 ),
        observ0(mpi, name, pt_in, walker_type, detail::get_number_of_averages(pt_in), wfn.getNMO(), wfn),
        prop0(std::addressof(prop)),
        ncalls(0),
        path_restoration(true),
        importanceSampling(impsamp_),
        extra_path_restoration(true),
        first(true),
        X(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO()),
        Y(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO()),
        M(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO())
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(1, "\n  --   Back Propagation with Time Evolved Operators -- \n"); 

    // initialize using verbose input
    int equil_multiplier, _population_control_interval;
    _population_control_interval = pt.get<int>("_population_control_interval"); // only for computing nback_prop_steps!
    path_restoration = pt.get<bool>("path_restoration");
    extra_path_restoration = pt.get<bool>("extra_path_restoration");
    equil_multiplier  = pt.get<int>("equil_multiplier"); // units of population control interval
    std::vector<int> nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt, "measure_interval_multiplier", {DEFAULT_MEASURE_INTERVAL_MULTIPLIER}); // units of population control interval
    naverages = nback_prop_interval_multipliers.size();

    // allocate memory
    nback_prop_steps.reserve(naverages);
    for (int i = 0; i < naverages; i++){
      if (nback_prop_interval_multipliers[i] <= 0)
      utils::check(nback_prop_interval_multipliers[i]>0,
                   "BPWithTimeEvolvedOperators: measure_interval_multiplier values must be positive.");
      nback_prop_steps.push_back(nback_prop_interval_multipliers[i] * _population_control_interval);
      app_log(2, "BPWithTimeEvolvedOperators: nback_prop_steps[{}] = {} ( = measure_interval_multiplier[{}] * population_control_interval) \n", i, nback_prop_steps[i], i);
    }
    max_nback_prop = *std::max_element(nback_prop_steps.begin(), nback_prop_steps.end());

    utils::check((equil_multiplier * _population_control_interval) % max_nback_prop == 0,
                 "Error in BPWithTimeEvolvedOperators user input: 'equil_multiplier' must be evenly divisible by the maximum value in 'measure_interval_multiplier'");
    nblocks_equil = (equil_multiplier *_population_control_interval )/ max_nback_prop; // Note: nback_prop is in steps, so we have to convert equil_multiplier to steps by multiplying by _population_control_interval

    // MAM: In principle, this should be the MCD of the nback_prop_steps.
    //      But it gets complicated if nblocks_equil > 1, so setting this to this
    //      for simplicity. Should not cause serious performance issues.
    _measure_interval_for_handler = _population_control_interval;

    average_has_run.reserve(naverages);
    average_has_run.assign(naverages, false);

    // sort the requested number of steps
    std::sort(nback_prop_steps.begin(), nback_prop_steps.end());

    int ncv(prop0->number_of_cholesky_vectors());
    wset.resize_bp(max_nback_prop, ncv, 1);
    wset.setBPPos(0);
    // set SMN in case BP begins right away
    if (nblocks_equil == 0)
      reset(wset);

    if(extra_path_restoration)
      path_restoration       = true;

    if(extra_path_restoration)
      app_log(1," Using path restoration with modification to include extra time segment. "); 
    else if(path_restoration)
      app_log(1," Using path restoration. "); 
    else
      app_log(1," Path restoration is not used "); 
    app_log(1," Number of equilibration measurements: {}", nblocks_equil);
    app_log(1," Number of time measurements: {}", nback_prop_steps.size());
    app_log(1," Measuring at steps (relative to each BP start in units of population control interval): {}",nback_prop_steps);
  }

  ~BPWithTimeEvolvedOperators() {}

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool path_restoration, extra_path_restoration;
    int ortho, equil_multiplier, _population_control_interval;
    std::vector<int> nback_prop_interval_multipliers;
    path_restoration       = pt0.get<bool>("path_restoration", true);
    extra_path_restoration = pt0.get<bool>("extra_path_restoration", true);
    ortho         = pt0.get<int>("bp_walker_ortho_interval", 1);
    equil_multiplier = pt0.get<int>("equil_multiplier", 0);
     _population_control_interval = pt0.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL); // only for computing nback_prop_steps!

    // Use utility function to read either a single integer or vector of integers
    nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt0, "measure_interval_multiplier", 1);

    // check if empty vector was returned
    if (nback_prop_interval_multipliers.empty())
      nback_prop_interval_multipliers.push_back(DEFAULT_MEASURE_INTERVAL_MULTIPLIER);

    // validate inputs
    if (std::any_of(nback_prop_interval_multipliers.begin(), nback_prop_interval_multipliers.end(), [](int x) { return x <= 0; }))
      APP_ABORT("BPWithTimeEvolvedOperators: measure_interval_multiplier values must be positive.");
    // create verbose internal inputs
    ptree pt1;
    pt1.put("path_restoration", path_restoration);
    pt1.put("extra_path_restoration", extra_path_restoration);
    pt1.put("bp_walker_ortho_interval", ortho);
    pt1.put("equil_multiplier", equil_multiplier);
    pt1.put("_population_control_interval", _population_control_interval);
    ptree temp_tree;
    for (const auto& value : nback_prop_interval_multipliers) {
        ptree item;
        item.put("", value); // empty key for the value
        temp_tree.push_back(std::make_pair("", item));
    }
    pt1.add_child("measure_interval_multiplier", temp_tree);

    // check for unknown input keys
    std::unordered_set<std::string> pass_through_keys = {
      "name",
      "onerdm",
      "gfock",
      "genfock",
      "ekt",
      "diag2rdm",
      "twordm",
      "n2r",
      "ontop2rdm",
      "realspace_correlators",
      "correlators",
      "pair_correlators",
      "spinspin"
    };
    io::compare_known_keys("Back propagated estimator",pt1, pt0, pass_through_keys);
    return pt1;
  }

  void accumulate_step([[maybe_unused]] double time,
                       [[maybe_unused]] WalkerSet<MEM>& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet<MEM>& wset)
  {
    auto all = nda::range::all;
    // always set to false
    accumulated_in_last_block = false;
    int bp_step               = wset.getBPPos();
    int nwalk = wset.size();
    utils::check(bp_step>0," Error: Found bp_step <=0 in ~BPWithTimeEvolvedOperators::accumulate_block. ");
    utils::check(bp_step<=max_nback_prop, " Error: max_nback_prop in back propagation estimator must be commensurate with measure_interval.");
    utils::check(max_nback_prop <= wset.NumBackProp()," Error: max_nback_prop > wset.NumBackProp() ");

    // check if measurement is needed
    int iav(-1);
    if( auto it = std::find(nback_prop_steps.begin(), nback_prop_steps.end(), bp_step);
        it != nback_prop_steps.end() ) {
      iav = std::distance(nback_prop_steps.begin(),it);
      utils::check(iav==0 || average_has_run[iav-1],
          "Error: missed a measurement in BPWithTimeEvolvedOperators::accumulate_block.\n"
          "Use a number of steps in the back propagation estimator that is divisible\n"
          "by the measurement_interval defined in the execute block.");
    } else {
      return;
    }

    // 0. skip if requested
    if(ncalls < nblocks_equil) {
      average_has_run[iav] = true; // during equil, "running" means do nothing
      if (bp_step == max_nback_prop)
      {
        ncalls++;
        wset.setBPPos(0);
      }
      if( ncalls == nblocks_equil ) reset(wset);
      return;  
    }   

    // X,Y,M are already propagated until nback_prop_steps[iav-1]
    int nsteps = nback_prop_steps[iav] - (iav>0?nback_prop_steps[iav-1]:0); 

    // no time between measurement blocks for now 
    // skip is bp_step > max_nback_prop and 

    // We are within the BP measurement phase
    // 1. Propagate X, Y matrices forward and accumulate M 
    prop0->PropagateOperators(nsteps, wset, X, Y, M);

    // 2. weights factors if using path restoration
    nda::array<ComplexType,1> wgt(nwalk);
    wset.getProperty(WEIGHT, wgt);
    if (path_restoration)
    {
      auto factors = nda::to_host(wset.getWeightFactors());
      utils::check(factors.extent(0) == nwalk, "Size mismatch");
      int hpos(wset.getHistoryPos()); // position where next step goes... go bach in history...
      int maxpos(wset.HistoryBufferLength());
      int nbp = nback_prop_steps[iav] * (extra_path_restoration?2:1);
      for (int k = 0; k < nbp; k++)
      {
        // start going back since position is advanced for next step already
        hpos = ((hpos == 0) ? maxpos - 1 : hpos - 1); 
        wgt() *= factors(all,hpos);
      }
    }
    else if (!importanceSampling)
    {
      nda::array<ComplexType,1> phase(nwalk);
      wset.getProperty(PHASE, phase);
      // MAM: careful here, since convention keeps changing
      wgt() *= phase();
    }

    // 3. Calculate observables if needed
    // 3.a add walker weights at current time
    // accumulate expects Yc = conj(Y)
    nda::tensor::scale(ComplexType(1.0),Y,nda::tensor::op::CONJ); 
    observ0.accumulate(iav, wset, wgt, X, Y, M, importanceSampling);
    // conjugate Y back!
    nda::tensor::scale(ComplexType(1.0),Y,nda::tensor::op::CONJ); 
    average_has_run[iav] = true;

    if (bp_step == max_nback_prop) {
      // last measurement on this block, full reset 
      reset(wset);
      accumulated_in_last_block = true;
      // increase counter
      ncalls++;
    }
  }

  void tags([[maybe_unused]] std::ofstream& out)
  {
  }

  int get_measurement_interval()
  {
    // this is forced to be commensurate with population control interval; see constructor.
    return _measure_interval_for_handler;
  }

  void print([[maybe_unused]] std::ofstream& out, h5::file& file, [[maybe_unused]] WalkerSet<MEM>& wset)
  {
    if (accumulated_in_last_block)
    {
      if (mpi->comm.root())
      {
        h5::group grp(file);
        h5::group g1 = ( grp.has_key("Observables") ? grp.open_group("Observables") :
                                                      grp.create_group("Observables") );
        h5::group g2 = ( g1.has_key("BackPropagated") ? g1.open_group("BackPropagated") :
                                                g1.create_group("BackPropagated") );
        if(first) {
          first = false;
          if (write_metadata)
          {
            h5::group g3 = g2.create_group("Metadata"); // can this already exist???
            h5::h5_write(g3,"BackPropSteps",nback_prop_steps);
            h5::h5_write(g3, "NumAverages", nback_prop_steps.size());
            h5::h5_write(g3, "NumReferences", int(1));
            write_metadata = false;
          }
        }
        observ0.print(ncalls, &g2);
      } else {
        h5::group *g = nullptr;
        observ0.print(ncalls, g);
      }
    }
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  int nspin;
  int npol;
  int accumulated_in_last_block = 0;

  TimeEvolvedObsHandler<MEM> observ0;

  Propagator<MEM>* prop0;

  int max_nback_prop = 0;
  std::vector<int> nback_prop_steps;

  // this is for the EstimatorHandler
  int _measure_interval_for_handler = 1;

  int ncalls        = 0;
  int nblocks_equil = 0;

  // number of intervals to divide max_nback_prop into
  //   BP will be performed using each of these intervals for
  //   'm' starting from the same 'n'
  int naverages = 1;
  // stores True if an 'average' has already been run since the last BP reset
  std::vector<bool> average_has_run;

  // Whether to restore cosine projection and real local energy approximation for weights
  // along back propagation path.
  bool path_restoration = true;
  bool importanceSampling = true;
  bool extra_path_restoration = true;

  int first = true;

  bool write_metadata = true;

  // State matrices for the evolved operators. X->c^+, Y->c
  memory::array<MEM,ComplexType,4> X;  
  memory::array<MEM,ComplexType,4> Y;  
  // Accumulates the scalar terms coming from the stabilization procedure
  memory::array<MEM,ComplexType,4> M;  

  template<class WlkSet>
  void reset(WlkSet& wset)
  {
    average_has_run.assign(naverages, false);
    wset.setBPPos(0);

    // initialize X, Y, M
    // hard-wired for the native basis set. Add choices later...
    M() = ComplexType(0.0);
    math::set_identity(X);
    math::set_identity(Y);
  }

};
} // namespace afqmc
} // namespace sfqmc

