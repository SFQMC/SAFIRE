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
#include <algorithm>
#include <vector>

#include "AFQMC/parameters.hpp"
#include "EstimatorBase.h"
#include "Observables/Observable.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc::afqmc {

namespace detail {

template<MEMORY_SPACE MEM>
auto constructBPMeasurementInputs(nda::MemoryVector auto& weights, Wavefunction<MEM> const& wfn, WalkerSet<MEM> const& wset, nda::MemoryArrayOfRank<4> auto const& Refs, nda::MemoryArrayOfRank<2> auto& logdetR) {
  using nda::range;
  auto all = range::all;
  WALKER_TYPES walker_type = wfn.getWalkerType();
  auto [nspin, npol] = walkerTypeToDims(walker_type);
  int nwalk = wset.size();
  int nrefs = Refs.extent(1);
  utils::check_shape(logdetR, "logdetR", nwalk, nrefs);

  auto SMA = wset.SlaterMatricesN(Alpha);
  auto SMB = wset.SlaterMatricesN((walker_type == COLLINEAR) ? Beta : Alpha);

  int NMO = SMA.extent(1)/npol;

  int nup = SMA.extent(2); 
  int ndown = SMB.extent(2); 
  utils::check(SMB.extent(1) == npol*NMO, "Size mismatch");

  memory::buffered_array<MEM, ComplexType, 1> invTotalOverlaps(wset.size(), 0);
  memory::buffered_array<MEM, ComplexType, 1> singleRefOverlaps(wset.size());
  
  // add contribution from down electrons if CLOSED 
  if(walker_type == CLOSED) {
    nda::tensor::scale(2.0, logdetR);
  }

  // calculate all overlaps and accumulate denominator 
  // MAM: no reference overlap is being subtracted yet, 
  // find a suitable common reference at this stage
  for (int d = 0; d < nrefs; d++) {
    singleRefOverlaps() = 0.0;

    //1. Calculate Green functions
    det_ops::Log_Overlap(Refs(all,d,all,range(nup)), SMA, singleRefOverlaps);
    if (walker_type == COLLINEAR)
      det_ops::Log_Overlap(Refs(all,d,all,range(nup,nup+ndown)), SMB, singleRefOverlaps);

    // 2.accumulate totalOverlaps[m] += ci[n] * exp(logSingleRefOverlaps[n] + logR[n])
    if (walker_type == CLOSED) {
      nda::tensor::scale(2.0, singleRefOverlaps);
    }
    nda::tensor::add(1.0,nda::conj(logdetR(all,d)),"w",1.0,singleRefOverlaps,"w");
    nda::apply(std::conj(wfn.getReferenceWeight(d)),singleRefOverlaps,nda::tensor::unary_op::EXP);
    nda::tensor::add(1.0, singleRefOverlaps, 1.0, invTotalOverlaps);
  }
  nda::apply(1.0, invTotalOverlaps, nda::tensor::unary_op::RCP);

  // calculate GF and accumulate
  auto referenceLoop = [invTotalOverlaps = std::move(invTotalOverlaps),
                        singleRefOverlaps = std::move(singleRefOverlaps),
                        &logdetR, &Refs, &wfn, &wset](auto&& body) mutable {
    using nda::range;
    auto all = range::all;

    WALKER_TYPES walker_type = wfn.getWalkerType();
    auto [nspin, npol] = walkerTypeToDims(walker_type);
    
    auto SMA = wset.SlaterMatricesN(Alpha);
    auto SMB = wset.SlaterMatricesN(walker_type == COLLINEAR ? Beta : Alpha);
    int nup = SMA.extent(2); 
    int ndown = SMB.extent(2); 
    
    memory::buffered_array<MEM, ComplexType, 4> singleRefG(wset.size(), nspin, npol*wfn.getNMO(), npol*wfn.getNMO());

    for(int d = 0; d < wfn.total_number_of_references(); d++) {
      singleRefOverlaps() = 0.0;
      det_ops::MixedDensityMatrix(Refs(all,d,all,range(nup)), SMA, singleRefG(all,0,all,all), singleRefOverlaps, false);
      if(walker_type == COLLINEAR) {
        det_ops::MixedDensityMatrix(Refs(all,d,all,range(nup,nup+ndown)), SMB, singleRefG(all,1,all,all), singleRefOverlaps, false);
      }
      
      // the same combination the denominator above was built from, so that the reference
      // weights of a walker sum to one
      if(walker_type == CLOSED) {
        nda::tensor::scale(2.0, singleRefOverlaps);
      }
      nda::tensor::add(1.0,nda::conj(logdetR(all,d)),"w",1.0,singleRefOverlaps,"w");
      nda::apply(std::conj(wfn.getReferenceWeight(d)),singleRefOverlaps,nda::tensor::unary_op::EXP);
      nda::tensor::elementwise(1.0, invTotalOverlaps, 1.0, singleRefOverlaps, nda::tensor::binary_op::PROD);

      body(singleRefOverlaps, singleRefG);
    }  
  };

  return MeasurementInputs<MEM,decltype(referenceLoop)>{referenceLoop, weights(), walker_type, wfn.getNMO()};
}

} // namespace detail

template<MEMORY_SPACE MEM>
class BackPropEstimator : public EstimatorBase<MEM> {

public:
  /// Every back propagation length of the input is a number of propagation steps, and so is
  /// the `step` the estimator is measured at.
  BackPropEstimator(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                          const BackPropEstimatorParameters& params,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop)
      : wfn_(wfn),
        prop_(prop),
        observables_(mpi, params, wset.getWalkerType(), wfn.getNMO()),
        propagation_steps_{resolved(params.propagation_steps, "propagation_steps")},
        walker_ortho_interval_(resolved(params.walker_ortho_interval, "walker_ortho_interval")), // units of steps
        path_restoration(params.path_restoration),
        extra_path_restoration(params.extra_path_restoration) {

    for (int bpsteps : propagation_steps_) {
      utils::check(bpsteps > 0,"BackPropEstimator: propagation_steps values must be positive.");
    }
    std::ranges::sort(propagation_steps_);
    // a repeated length would measure the same window twice and leave the propagator with an
    // empty segment to cover
    utils::check(std::ranges::adjacent_find(propagation_steps_) == propagation_steps_.end(),
                 "BackPropEstimator: propagation_steps values must be distinct, but they are {}.",
                 propagation_steps_);

    int ncv(prop_.number_of_cholesky_vectors());
    int number_of_references = wfn_.total_number_of_references();
    wset.resize_bp(propagation_steps_.back(), ncv, number_of_references);
    setAnchor(wset, 0);
  }


  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long step, Measurements& meas, WalkerSet<MEM> &wset) override {
    long const bp_step = step - bp_pos_;
    utils::check(bp_step >= 0, " Error: Found bp_step < 0 in BackPropEstimator::measure. ");

    if(std::ranges::binary_search(propagation_steps_, bp_step)) {
      backPropagate(mpi, int(bp_step), meas, wset);
    }
    // the longest average has been taken over this window, so the next one starts here. A
    // step that overshoots the longest average without matching it re-anchors as well.
    if(bp_step >= propagation_steps_.back()) {
      setAnchor(wset, step);
    }
  }

  void equilibrated(long step, WalkerSet<MEM>& wset) override { setAnchor(wset, step); }

private:
  /// Back propagate the references over `bp_step` steps and measure the observables on them.
  void backPropagate(utils::mpi_context_t<boost::mpi3::communicator>& mpi, int bp_step,
                     Measurements& meas, WalkerSet<MEM>& wset) {
    auto all = nda::range::all;
    int nwalk = wset.size();

    auto back_propagate_time = timers.back_propagate.start();

    memory::buffered_array<MEM,ComplexType,3> Ref0;
    wfn_.getReferences(Ref0);
    
    memory::buffered_array<MEM,ComplexType,4> Refs(nwalk, Ref0.extent(0), Ref0.extent(1), Ref0.extent(2));
    memory::buffered_array<MEM,ComplexType,2> logdetR(nwalk, wfn_.total_number_of_references());

    // 2. setup back propagated references
    for(int iw = 0; iw < nwalk; ++iw)
      Refs(iw,nda::ellipsis{}) = Ref0();
    mpi.node_comm.barrier();

    //3. propagate backwards the references
    prop_.BackPropagate(bp_step, walker_ortho_interval_, wset, Refs, logdetR);

    // logdetR_shift[w] = (1/Nd) * sum_d logdetR[w][d]
    // apply shift: logdetR[w][d] = logdetR[w][d] - logdetR_shift[w]
    int const nrefs = wfn_.total_number_of_references();
    memory::buffered_array<MEM,ComplexType,1> ones(nrefs, 1.0);
    memory::buffered_array<MEM,ComplexType,1> shift(nwalk, 0);
    nda::blas::gemv(1.0/double(nrefs), logdetR, ones, 0.0, shift);
    nda::tensor::add(-1.0, shift, "w", 1.0, logdetR, "wd");

    //4. calculate properties
    // adjust weights here if path restoration
    memory::buffered_array<HOST_MEMORY,ComplexType,1> weights(nwalk);
    wset.getProperty(WEIGHT, weights);
    if(path_restoration)
    {
      auto factors = nda::to_host(wset.getWeightFactors());
      int hpos(wset.getHistoryPos()); // position where next step goes... go bach in history...
      int maxpos(wset.HistoryBufferLength());
      // the weight history is three times as long as the field ring, so the extra segment
      // still fits; before the run has produced that many steps the tail it reads is the
      // 1.0 resize_bp initialized it to
      int nbp(bp_step);
      if(extra_path_restoration) {
        nbp *= 2;
      }
      for(int k = 0; k < nbp; k++) {
        // start going back since position is advanced for next step already
        hpos = ((hpos == 0) ? maxpos - 1 : hpos - 1); 
        weights(all) *= factors(all,hpos);
      }
    }

    // the path restoration above runs on the host, while the observables consume the
    // weights wherever the walkers live
    decltype(auto) measurement_weights = memory::to_memory_space<MEM>(weights);

    auto inputs = detail::constructBPMeasurementInputs<MEM>(measurement_weights, wfn_, wset, Refs, logdetR);
    MeasurementOutput output{mpi, meas, std::format("BackPropEstimator/Steps={}", bp_step), weights};
    observables_.measure(mpi, output, inputs);

    back_propagate_time.stop();
  }

  /// Anchor back propagation at `step`, the number of propagation steps completed so far.
  /// The constructor anchors before the first step, at 0.
  void setAnchor(WalkerSet<MEM>& wset, long step) {
    for(int iw = 0; iw < wset.size(); ++iw) {
      wset[iw].setSlaterMatrixN();
    }
    bp_pos_ = step;
  }

  Wavefunction<MEM>& wfn_;
  Propagator<MEM>& prop_;
  Observables<MEM> observables_;

  // completed step count the anchor sits at, see setAnchor: at step s, s - bp_pos_ steps
  // have been propagated over since it
  long bp_pos_{};

  // the back propagation lengths, in steps, sorted
  std::vector<int> propagation_steps_;

  // Frequency of reorthogonalisation, in units of steps.
  int walker_ortho_interval_{1};

  // Whether to restore cosine projection and real local energy approximation for weights
  // along back propagation path.
  bool path_restoration = true;
  bool extra_path_restoration = false;
};

} // namespace sfqmc::afqmc

