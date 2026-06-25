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
#include <string>
#include <iostream>
#include <fstream>

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Estimators/FullObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Top class for back propagated estimators. 
 * An instance of this class will manage a set of back propagated observables.
 * The main task of this class is to coordinate the generation of left-handed
 * states during the back propagation algorithm. The calculation and accumulation
 * of actual observables is handled by an object of the variant FullObsHandler.
 * BackPropagatedEstimator provides a list of back propagated references to 
 * FullObsHandler, whose job it is to calculate, accumulate and print quantities.
 */
template<MEMORY_SPACE MEM>
class BackPropagatedEstimator : public EstimatorBase<MEM>
{

  using EstimatorBase<MEM>::NMO;
  using EstimatorBase<MEM>::nup;
  using EstimatorBase<MEM>::ndown;

public:
  BackPropagatedEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                          AFQMCInfo& info,
                          std::string name,
                          ptree pt_in,
                          WALKER_TYPES wlk,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop,
                          bool impsamp_ = true)
      : EstimatorBase<MEM>(info),
        mpi(_mpi),
        walker_type(wlk),
        observ0(mpi, info, name, pt_in, wlk, wfn),
        wfn0(std::addressof(wfn)),
        prop0(std::addressof(prop)),
        max_nback_prop(10),
        nStabilize(10),
        path_restoration(true),
        importanceSampling(impsamp_),
        extra_path_restoration(true),
        first(true)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(1,"\nBPEstimator input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    int equil_multiplier, _population_control_interval;
    _population_control_interval = pt.get<int>("_population_control_interval"); // only for computing nback_prop_steps!
    path_restoration = pt.get<bool>("path_restoration");
    extra_path_restoration = pt.get<bool>("extra_path_restoration");
    nStabilize     = pt.get<int>("bp_walker_ortho_interval"); // units of steps!!
    equil_multiplier  = pt.get<int>("equil_multiplier"); // units of population control interval
    std::vector<int> nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt, "measure_interval_multiplier", {DEFAULT_MEASURE_INTERVAL_MULTIPLIER}); // units of population control interval
    naverages = nback_prop_interval_multipliers.size();

    // allocate memory
    nback_prop_steps.reserve(naverages);
    for (int i = 0; i < naverages; i++){
      if (nback_prop_interval_multipliers[i] <= 0)
        APP_ABORT("BackPropagatedEstimator: measure_interval_multiplier values must be positive.");
      nback_prop_steps.push_back(nback_prop_interval_multipliers[i] * _population_control_interval);
      app_log(2, "BackPropagatedEstimator: nback_prop_steps[{}] = {} ( = measure_interval_multiplier[{}] * population_control_interval) \n", i, nback_prop_steps[i], i);
    }
    max_nback_prop = *std::max_element(nback_prop_steps.begin(), nback_prop_steps.end());
    
    if ((equil_multiplier * _population_control_interval) % max_nback_prop != 0 )
      APP_ABORT("Error in BackPropagatedEstimator user input: 'equil_multiplier' must be evenly divisible by the maximum value in 'measure_interval_multiplier'");
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
    int number_of_references = wfn0->total_number_of_references();
    wset.resize_bp(max_nback_prop, ncv, number_of_references);
    wset.setBPPos(0);
    // set SMN in case BP begins right away
    if (nblocks_equil == 0)
      for (auto it = wset.begin(); it < wset.end(); ++it)
        it->setSlaterMatrixN();
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool path_restoration, extra_path_restoration;
    int ortho, equil_multiplier, _population_control_interval;
    std::vector<int> nback_prop_interval_multipliers;
    path_restoration       = pt0.get<bool>("path_restoration", true);
    extra_path_restoration = pt0.get<bool>("extra_path_restoration", true);
    ortho         = pt0.get<int>("bp_walker_ortho_interval", 10);
    equil_multiplier = pt0.get<int>("equil_multiplier", 0);
     _population_control_interval = pt0.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL); // only for computing nback_prop_steps!
    
    // Use utility function to read either a single integer or vector of integers
    nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt0, "measure_interval_multiplier", 1);
  
    // check if empty vector was returned
    if (nback_prop_interval_multipliers.empty())
      nback_prop_interval_multipliers.push_back(DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    
    // validate inputs
    if (std::any_of(nback_prop_interval_multipliers.begin(), nback_prop_interval_multipliers.end(), [](int x) { return x <= 0; }))
      APP_ABORT("BackPropagatedEstimator: measure_interval_multiplier values must be positive.");
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

  ~BackPropagatedEstimator() {}

  void accumulate_step([[maybe_unused]] double time, 
                       [[maybe_unused]] WalkerSet<MEM>& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet<MEM>& wset)
  {
    auto all = nda::range::all;
    accumulated_in_last_block = false;
    int bp_step               = wset.getBPPos();
    int nwalk = wset.size();
    int nel = nup + (walker_type == COLLINEAR ? ndown : 0);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    utils::check(bp_step>0," Error: Found bp_step <=0 in BackPropagate::accumulate_block. ");
    utils::check(bp_step<=max_nback_prop, " Error: max_nback_prop in back propagation estimator must be commensurate with measure_interval.");
    utils::check(max_nback_prop <= wset.NumBackProp()," Error: max_nback_prop > wset.NumBackProp() ");

    // check if measurement is needed
    int iav(-1);
    if( auto it = std::find(nback_prop_steps.begin(), nback_prop_steps.end(), bp_step); 
        it != nback_prop_steps.end() ) {
      iav = std::distance(nback_prop_steps.begin(),it);
      utils::check(iav==0 || average_has_run[iav-1],
          "Error: missed a measurement in BackPropagate::accumulate_block.\n"
          "Use a number of steps in the back propagation estimator that is divisible\n"
          "by the measurement_interval defined in the execute block.");
    } else {
      return;
    } 

    // 0. skip if requested
    if (iblock < nblocks_equil)
    {
      average_has_run[iav] = true; // during equil, "running" means do nothing
      if (bp_step == max_nback_prop)
      {
        if (iblock + 1 == nblocks_equil)
          for (auto it = wset.begin(); it < wset.end(); ++it)
            it->setSlaterMatrixN();
        iblock++;
        wset.setBPPos(0);
      }
      return;
    }

    AFQMCTimer.start(back_propagate_timer);

    // 1. allocate memory. Can loop over walkers if nrefs is too large 
    int number_of_references = wfn0->total_number_of_references();
    memory::buffered_array<MEM,ComplexType,4> Refs(nwalk, number_of_references, npol*NMO, nel);
    memory::buffered_array<MEM,ComplexType,2> logdetR(nwalk, number_of_references);

    // 2. setup back propagated references.
    // NOTE: getReferences may have side effects -- for a stochastic trial it performs a fresh
    // free-projection draw of the reference ensemble (mutating internal scratch + RNG state). It is meant
    // to be called once per BP block here (result copied into Refs, broadcast to all walkers, frozen for
    // the window). A repeated call within the SAME window is guarded to reuse the same draw (idempotent);
    // a forward step (begin_inner_step) opens the next window, which draws fresh.
    wfn0->getReferences(number_of_references, Refs(0,nda::ellipsis{}));
    for (int iw = 1; iw < nwalk; ++iw)
      Refs(iw,nda::ellipsis{}) = Refs(0,nda::ellipsis{});
    mpi->node_comm.barrier();

    //3. propagate backwards the references
    prop0->BackPropagate(bp_step, nStabilize, wset, Refs, logdetR);

    //4. calculate properties
    // adjust weights here is path restoration
    memory::buffered_array<HOST_MEMORY,ComplexType,1> wgt(nwalk);
    wset.getProperty(WEIGHT, wgt);
    if (path_restoration)
    {
      auto factors = nda::to_host(wset.getWeightFactors());
      int hpos(wset.getHistoryPos()); // position where next step goes... go bach in history...
      int maxpos(wset.HistoryBufferLength());
      int nbp(bp_step);
      if (extra_path_restoration)
        nbp *= 2;
      for (int k = 0; k < nbp; k++)
      {
        // start going back since position is advanced for next step already
        hpos =
            ((hpos == 0) ? maxpos - 1 : hpos - 1); 
        wgt(all) *= factors(all,hpos);
      }
    }
    else if (!importanceSampling)
    {
      memory::buffered_array<HOST_MEMORY,ComplexType,1> phase(nwalk);
      wset.getProperty(PHASE, phase);
      wgt() *= phase();
    }
    observ0.accumulate(iav, wset, Refs, wgt, logdetR, importanceSampling);
    average_has_run[iav] = true;

    if (bp_step == max_nback_prop)
    {
      // 5. setup for next block
      for (auto it = wset.begin(); it < wset.end(); ++it)
        it->setSlaterMatrixN();
      wset.setBPPos(0);
      average_has_run.assign(naverages, false);

      // 6. increase block counter
      iblock++;
      accumulated_in_last_block = true;
    }
    AFQMCTimer.stop(back_propagate_timer);
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
    // I doubt we will ever collect a billion blocks of data.
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
          int nave(nback_prop_steps.size());
          if (write_metadata)
          {
            h5::group g3 = g2.create_group("Metadata"); // can this already exist??? 
            h5::h5_write(g3,"BackPropSteps",nback_prop_steps);
            h5::h5_write(g3,"NumAverages",nave);
            int number_of_references = wfn0->total_number_of_references();
            h5::h5_write(g3,"NumReferences",number_of_references);
            write_metadata = false;
          }
        }

        observ0.print(iblock, &g2);
      } else {
        h5::group *g = nullptr;
        observ0.print(iblock, g);
      }
    }
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  bool accumulated_in_last_block = false;

  FullObsHandler<MEM> observ0;

  Wavefunction<MEM>* wfn0;

  Propagator<MEM>* prop0;

  int max_nback_prop = 0;
  std::vector<int> nback_prop_steps;
//  std::vector<int> nback_prop_interval_multipliers;

  // this is for the EstimatorHandler
  int _measure_interval_for_handler = 1; 

  int iblock       = 0;
  int nblocks_equil = 0;

  // Frequency of reorthogonalisation.
  int nStabilize = 1;
  // number of intervals to divide max_nback_prop into
  //   BP will be performed using each of these intervals for
  //   'm' starting from the same 'n'
  int naverages = 1;
  // stores True if an 'average' has already been run since the last BP reset
  std::vector<bool> average_has_run;

  // Whether to restore cosine projection and real local energy approximation for weights
  // along back propagation path.
  bool path_restoration = true, importanceSampling = true;
  bool extra_path_restoration = true;

  int first = true;

  bool write_metadata = true;
};
} // namespace afqmc
} // namespace sfqmc

