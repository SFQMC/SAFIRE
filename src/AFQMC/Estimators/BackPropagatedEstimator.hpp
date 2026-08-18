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

#include "AFQMC/parameters.hpp"
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

public:
  /// The measurement and equilibration intervals of the input are multiples of
  /// population_control_interval, which is given in steps.
  BackPropagatedEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                          std::string name,
                          const EstimatorParameters& params,
                          int population_control_interval,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop,
                          bool impsamp_ = true)
      : mpi(_mpi),
        walker_type(wset.getWalkerType()),
        observ0(mpi, name, params, measure_interval_multipliers(params).size(), walker_type, wfn.getNMO(), wfn),
        wfn0(std::addressof(wfn)),
        prop0(std::addressof(prop)),
        max_nback_prop(10),
        nStabilize(params.bp_walker_ortho_interval), // units of steps!!
        path_restoration(params.path_restoration),
        importanceSampling(impsamp_),
        extra_path_restoration(params.extra_path_restoration),
        first(true)
  {
    const std::vector<int> multipliers = measure_interval_multipliers(params);
    naverages = multipliers.size();
    nback_prop_steps.reserve(naverages);
    for (int i = 0; i < naverages; i++){
      if (multipliers[i] <= 0)
        APP_ABORT("BackPropagatedEstimator: measure_interval_multiplier values must be positive.");
      nback_prop_steps.push_back(multipliers[i] * population_control_interval);
      app_log(2, "BackPropagatedEstimator: nback_prop_steps[{}] = {} ( = measure_interval_multiplier[{}] * population_control_interval) \n", i, nback_prop_steps[i], i);
    }
    max_nback_prop = *std::max_element(nback_prop_steps.begin(), nback_prop_steps.end());

    const int equil_steps = params.equil_multiplier * population_control_interval;
    if (equil_steps % max_nback_prop != 0 )
      APP_ABORT("Error in BackPropagatedEstimator user input: 'equil_multiplier' must be evenly divisible by the maximum value in 'measure_interval_multiplier'");
    nblocks_equil = equil_steps / max_nback_prop;

    // MAM: In principle, this should be the MCD of the nback_prop_steps.
    //      But it gets complicated if nblocks_equil > 1, so setting this to this
    //      for simplicity. Should not cause serious performance issues.
    _measure_interval_for_handler = population_control_interval;

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
      for(int iw = 0; iw < wset.size(); ++iw) {
        wset[iw].setSlaterMatrixN();
      }
  }

  void accumulate_step([[maybe_unused]] double time, 
                       [[maybe_unused]] WalkerSet<MEM>& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet<MEM>& wset)
  {
    auto all = nda::range::all;
    accumulated_in_last_block = false;
    int bp_step               = wset.getBPPos();
    int nwalk = wset.size();
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
          for(int iw = 0; iw < wset.size(); ++iw) {
            wset[iw].setSlaterMatrixN();
          }
        iblock++;
        wset.setBPPos(0);
      }
      return;
    }

    auto back_propagate_time = timers.back_propagate.start();

    // 1. allocate memory. Can loop over walkers if nrefs is too large
    int number_of_references = wfn0->total_number_of_references();

    memory::buffered_array<MEM,ComplexType,3> Ref0;
    wfn0->getReferences(Ref0);
    
    memory::buffered_array<MEM,ComplexType,4> Refs(nwalk, Ref0.extent(0), Ref0.extent(1), Ref0.extent(2));
    memory::buffered_array<MEM,ComplexType,2> logdetR(nwalk, number_of_references);

    // 2. setup back propagated references
    for (int iw = 0; iw < nwalk; ++iw)
      Refs(iw,nda::ellipsis{}) = Ref0();
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
      for(int iw = 0; iw < wset.size(); ++iw) {
        wset[iw].setSlaterMatrixN();
      }
      wset.setBPPos(0);
      average_has_run.assign(naverages, false);

      // 6. increase block counter
      iblock++;
      accumulated_in_last_block = true;
    }
    back_propagate_time.stop();
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
  bool extra_path_restoration = false;

  int first = true;

  bool write_metadata = true;
};
} // namespace afqmc
} // namespace sfqmc

