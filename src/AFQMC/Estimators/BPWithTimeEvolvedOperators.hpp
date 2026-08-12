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
#include <fstream>

#include "AFQMC/parameters.hpp"
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

/*
 * Implements back propagation by evolving operators forward in time.
 * Based on 10.1103/PhysRevA.100.023621.
 */
template<MEMORY_SPACE MEM>
class BPWithTimeEvolvedOperators : public EstimatorBase<MEM>
{

public:
  /// The measurement and equilibration intervals of the input are multiples of
  /// population_control_interval, which is given in steps.
  BPWithTimeEvolvedOperators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                          std::string name,
                          const EstimatorParameters& params,
                          int population_control_interval,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop,
                          bool impsamp_ = true)
      : mpi(_mpi),
        walker_type(wset.getWalkerType()),
        nspin( (walker_type==COLLINEAR) ? 2 : 1 ),
        npol( (walker_type==NONCOLLINEAR) ? 2 : 1 ),
        observ0(mpi, name, params, walker_type, measure_interval_multipliers(params).size(), wfn.getNMO(), wfn),
        prop0(std::addressof(prop)),
        ncalls(0),
        path_restoration(params.path_restoration),
        importanceSampling(impsamp_),
        extra_path_restoration(params.extra_path_restoration),
        first(true),
        X(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO()),
        Y(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO()),
        M(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO())
  {
    app_log(1, "\n  --   Back Propagation with Time Evolved Operators -- \n");

    const std::vector<int> multipliers = measure_interval_multipliers(params);
    naverages = multipliers.size();
    nback_prop_steps.reserve(naverages);
    for (int i = 0; i < naverages; i++){
      utils::check(multipliers[i] > 0,
                   "BPWithTimeEvolvedOperators: measure_interval_multiplier values must be positive.");
      nback_prop_steps.push_back(multipliers[i] * population_control_interval);
      app_log(2, "BPWithTimeEvolvedOperators: nback_prop_steps[{}] = {} ( = measure_interval_multiplier[{}] * population_control_interval) \n", i, nback_prop_steps[i], i);
    }
    max_nback_prop = *std::max_element(nback_prop_steps.begin(), nback_prop_steps.end());

    const int equil_steps = params.equil_multiplier * population_control_interval;
    utils::check(equil_steps % max_nback_prop == 0,
                 "Error in BPWithTimeEvolvedOperators user input: 'equil_multiplier' must be evenly divisible by the maximum value in 'measure_interval_multiplier'");
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
    nda::tensor::scale(1.0,Y,nda::tensor::unary_op::CONJ);
    observ0.accumulate(iav, wset, wgt, X, Y, M, importanceSampling);
    // conjugate Y back!
    nda::tensor::scale(1.0,Y,nda::tensor::unary_op::CONJ);
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
  bool extra_path_restoration = false;

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

