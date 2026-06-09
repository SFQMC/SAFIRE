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

#ifndef SFQMC_AFQMC_BACKPROPAGATEDESTIMATOR_HPP
#define SFQMC_AFQMC_BACKPROPAGATEDESTIMATOR_HPP

#include "AFQMC/config.h"
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "io/ptree/ptree_utilities.hpp"

#include "AFQMC/Estimators/FullObsHandler.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/ma_operations.hpp"
#include "Memory/buffer_managers.h"

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
class BackPropagatedEstimator : public EstimatorBase
{
  // allocators
  using Allocator = device_allocator<ComplexType>;

  // type defs
  using pointer       = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;

  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdCTensor     = boost::multi::array<ComplexType, 3>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;

  using stack_alloc_type = DeviceBufferManager::template allocator_t<ComplexType>;
  using StaticMatrix     = boost::multi::static_array<ComplexType, 2, stack_alloc_type>;

public:
  BackPropagatedEstimator(afqmc::TaskGroup_& tg_,
                          AFQMCInfo& info,
                          std::string name,
                          ptree pt_in,
                          WALKER_TYPES wlk,
                          WalkerSet& wset,
                          Wavefunction& wfn,
                          Propagator& prop,
                          bool impsamp_ = true)
      : EstimatorBase(info),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        Refs({0, 0, 0}, shared_allocator<ComplexType>{TG.TG_local()}),
        observ0(TG, info, name, pt_in, wlk, wfn),
        wfn0(wfn),
        prop0(prop),
        max_nback_prop(10),
        nStabilize(10),
        block_size(1),
        path_restoration(false),
        importanceSampling(impsamp_),
        extra_path_restoration(false),
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
    block_size     = pt.get<int>("block_size");
    equil_multiplier  = pt.get<int>("equil_multiplier"); // units of population control interval
    nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt, "measure_interval_multiplier", {DEFAULT_MEASURE_INTERVAL_MULTIPLIER}); // units of population control interval
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
    // Note: nback_prop is in steps, so we have to convert equil_multiplier to steps by multiplying by _population_control_interval
    nblocks_equil = (equil_multiplier *_population_control_interval )/ max_nback_prop;

    /* 
    BP uses "blocks" internally, but we want the input
        file to use "Steps". Convert here to keep the 
        user-facing side simple. "block_size" should be considered deprecated,
        and now the size of a block is given by "max_nback_prop". 
    */
    if (block_size != 1)
      app_warning("Explicit block_sizes other than 1 are deprecated in Back Propagtion.");

    average_has_run.reserve(naverages);
    average_has_run.assign(naverages, false);

    // sort the requested number of steps
    std::sort(nback_prop_steps.begin(), nback_prop_steps.end());

    int ncv(prop0.global_number_of_cholesky_vectors());
    nrefs = wfn0.number_of_references_for_back_propagation();
    wset.resize_bp(max_nback_prop, ncv, nrefs);
    wset.setBPPos(0);
    // set SMN in case BP begins right away
    if (nblocks_equil == 0)
      for (auto it = wset.begin(); it < wset.end(); ++it)
        it->setSlaterMatrixN();

    if (TG.getNCoresPerTG() > 1)
      APP_ABORT("ncores > 1 is broken with back propagation. Fix this.");
    writer = (TG.Global().rank() == 0);
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool path_restoration, extra_path_restoration;
    int ortho, block_size, equil_multiplier, _population_control_interval;
    std::vector<int> nback_prop_steps; // determined by nback_prop_interval_multipliers and nPopulation;
    std::vector<int> nback_prop_interval_multipliers;
    path_restoration       = pt0.get<bool>("path_restoration", false);
    extra_path_restoration = pt0.get<bool>("extra_path_restoration", false);
    ortho         = pt0.get<int>("bp_walker_ortho_interval", 1);
    equil_multiplier = pt0.get<int>("equil_multiplier", 0);
    block_size    = pt0.get<int>("block_size", 1);
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
    pt1.put("block_size", block_size);
    pt1.put("equil_multiplier", equil_multiplier);
    pt1.put("_population_control_interval", _population_control_interval);
    ptree temp_tree;
    for (const auto& value : nback_prop_interval_multipliers) {
        ptree item;
        item.put("", value); // empty key for the value
        temp_tree.push_back(std::make_pair("", item));
    }
    pt1.add_child("measure_interval_multiplier", temp_tree);

    // check for unkown input keys
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
                       [[maybe_unused]]  WalkerSet& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet& wset)
  {
    // MAM: BP will not work as written if steps in execute don't sync with steps in BP.
    //      Fix Fix Fix...
    // KE: safety blocks have been added. Would still be good to generalize such that
    //        commensurate steps are not required.
    accumulated_in_last_block = false;
    int bp_step               = wset.getBPPos();
    if (bp_step <= 0)
      APP_ABORT(" Error: Found bp_step <=0 in BackPropagate::accumulate_block. ");
    if (bp_step > max_nback_prop)
      APP_ABORT(" Error: max_nback_prop in back propagation estimator must be commensurate with measure_interval.");
    if (max_nback_prop > wset.NumBackProp())
      APP_ABORT(" Error: max_nback_prop > wset.NumBackProp() ");

    // check if measurement is needed
    int iav(-1);
    for (int i = 0; i < nback_prop_steps.size(); i++)
    {
      if (bp_step == nback_prop_steps[i])
      {
        iav = i;
        break;
      }
    }
    // check if we've missed a measurement
    {
      int previous_average(0);
      auto it = std::lower_bound(nback_prop_steps.begin(), nback_prop_steps.end(), bp_step);
      previous_average = std::distance(nback_prop_steps.begin(), it) - 1; // this is -1 if nothing is found
      if (previous_average > -1 && !average_has_run[previous_average])
      {
        APP_ABORT("Error: missed a measurement in BackPropagate::accumulate_block.\n"
          + std::string("Current bp_step = ") + std::to_string(bp_step) + "\n"
          + std::string("Previous average bp_step = ") + std::to_string(nback_prop_steps[previous_average]) + "\n"
          "Use a number of steps in the back propagation estimator that is divisible\n"
          "by the measurement_interval defined in the execute block." );
      }
    }

    if (iav < 0)
      return;

    using std::fill_n;
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
    int nrow(NMO * ((walker_type == NONCOLLINEAR) ? 2 : 1));
    int ncol(NAEA + ((walker_type == CLOSED) ? 0 : NAEB));
    int nx((walker_type == COLLINEAR) ? 2 : 1);

    // 1. check structures
    if (Refs.size(0) != wset.size() || Refs.size(1) != nrefs || Refs.size(2) != nrow * ncol)
      Refs = mpi3CTensor({wset.size(), nrefs, nrow * ncol}, Refs.get_allocator());
    DeviceBufferManager buffer_manager;
    StaticMatrix logdetR({wset.size(), nrefs * nx}, buffer_manager.get_generator().template get_allocator<ComplexType>());

    int n0, n1;
    std::tie(n0, n1) = FairDivideBoundary(TG.getLocalTGRank(), int(Refs.size(2)), TG.getNCoresPerTG());
    boost::multi::array_ref<ComplexType, 3> Refs_(raw_pointer_cast(Refs.origin()), Refs.extensions());

    // 2. setup back propagated references
    wfn0.getReferencesForBackPropagation(Refs_[0]);
    for (int iw = 1; iw < wset.size(); ++iw)
      for (int ref = 0; ref < nrefs; ++ref)
        copy_n(Refs_[0][ref].origin() + n0, n1 - n0, Refs_[iw][ref].origin() + n0);
    TG.TG_local().barrier();

    //3. propagate backwards the references
    prop0.BackPropagate(bp_step, nStabilize, wset, Refs_, logdetR);

    //4. calculate properties
    // adjust weights here is path restoration
    stdCVector wgt(iextensions<1u>{wset.size()});
    wset.getProperty(WEIGHT, wgt);
    if (path_restoration)
    {
      auto&& factors = *wset.getWeightFactors();
      int hpos(wset.getHistoryPos()); // position where next step goes... go bach in history...
      int maxpos(wset.HistoryBufferLength());
      int nbp(bp_step);
      if (extra_path_restoration)
        nbp *= 2;
      for (int k = 0; k < nbp; k++)
      {
        hpos =
            ((hpos == 0) ? maxpos - 1 : hpos - 1); // start going back since position is advanced for next step already
        for (int i = 0; i < wgt.size(); i++)
          wgt[i] *= factors[hpos][i];
      }
    }
    else if (!importanceSampling)
    {
      stdCVector phase(iextensions<1u>{wset.size()});
      wset.getProperty(PHASE, phase);
      for (int i = 0; i < wgt.size(); i++)
        wgt[i] *= phase[i];
    }
    observ0.accumulate(iav, wset, Refs_, wgt, logdetR, importanceSampling);
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
    // we want this to be the GCD of nback_prop_steps
    int gcd = nback_prop_steps[0];
    for (int i = 1; i < nback_prop_steps.size(); i++)
      gcd = std::gcd(gcd, nback_prop_steps[i]);
    return gcd;
  }

  void print([[maybe_unused]] std::ofstream& out, hdf_archive& dump, [[maybe_unused]] WalkerSet& wset)
  {
    // I doubt we will ever collect a billion blocks of data.
    if (accumulated_in_last_block)
    {
      if (writer && first)
      {
        first = false;
        int nave(nback_prop_steps.size());
        if (write_metadata)
        {
          dump.push("Observables");
          dump.push("BackPropagated");
          dump.push("Metadata");
          dump.write(nback_prop_steps, "BackPropSteps");
          dump.write(nave, "NumAverages");
          dump.write(nrefs, "NumReferences");
          dump.pop();
          dump.pop();
          dump.pop();
          write_metadata = false;
        }
      }
      if (writer)
      {
        dump.push("Observables");
        dump.push("BackPropagated");
      }
      observ0.print(iblock, dump);
      if (writer)
      {
        dump.pop();
        dump.pop();
      }
    }
  }

private:
  TaskGroup_& TG;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  bool writer = false;
  bool accumulated_in_last_block = false;

  mpi3CTensor Refs;

  FullObsHandler observ0;

  Wavefunction& wfn0;

  Propagator& prop0;

  int nrefs = 0;
  int max_nback_prop = 0;
  std::vector<int> nback_prop_steps;
  std::vector<int> nback_prop_interval_multipliers;

  int iblock       = 0;
  int nblocks_equil = 0;
  ComplexType zero = ComplexType(0.0, 0.0);
  ComplexType one  = ComplexType(1.0, 0.0);

  // Frequency of reorthogonalisation.
  int nStabilize = 1;
  // Block size over which RDM will be averaged.
  int block_size = 1;
  // number of intervals to divide max_nback_prop into
  //   BP will be peformed using each of these intervals for
  //   'm' starting from the same 'n'
  int naverages = 1;
  // stores True if an 'average' has already been run since the last BP reset
  std::vector<bool> average_has_run;

  // Whether to restore cosine projection and real local energy apprximation for weights
  // along back propagation path.
  bool path_restoration = true, importanceSampling = true;
  bool extra_path_restoration = true;

  int first = true;

  bool write_metadata = true;
};
} // namespace afqmc
} // namespace sfqmc

#endif
