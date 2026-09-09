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

#include <algorithm>
#include <array>
#include <format>
#include <ranges>
#include <vector>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"

#include "AFQMC/parameters.hpp"
#include "Measurements.hpp"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "numerics/operations/product.hpp"
#include "numerics/operations/tensor.hpp"

namespace sfqmc::afqmc {

namespace detail {

/*
 * Evolving the operators forward in time dresses the Green function of each reference as
 *   G_d -> M + T(Ref_d^H X) * (Gc_d * conj(Y)),
 * with Gc_d the compact mixed density matrix of reference d. The dressing is the same for
 * every reference, so the per-reference structure the observables consume is unchanged.
 */
template<MEMORY_SPACE MEM>
auto constructTimeEvolvedMeasurementInputs(nda::MemoryVector auto const& weights,
                                           Wavefunction<MEM>& wfn,
                                           WalkerSet<MEM>& wset,
                                           nda::MemoryArrayOfRank<3> auto const& Refs,
                                           nda::MemoryArrayOfRank<4> auto const& X,
                                           nda::MemoryArrayOfRank<4> auto const& Yc,
                                           nda::MemoryArrayOfRank<4> auto const& M) {
  using nda::range;
  auto all = range::all;

  WALKER_TYPES walker_type = wfn.getWalkerType();
  auto [nspin, npol] = walkerTypeToDims(walker_type);
  int nwalk   = wset.size();
  int npolNMO = npol * wfn.getNMO();
  int nrefs   = wfn.total_number_of_references();

  utils::check_shape(X, "X", nwalk, nspin, npolNMO, npolNMO);
  utils::check_shape(Yc, "Yc", nwalk, nspin, npolNMO, npolNMO);
  utils::check_shape(M, "M", nwalk, nspin, npolNMO, npolNMO);

  std::array<SpinTypes,2> const spins{Alpha, Beta};
  std::array<int,2> nel{int(wset.SlaterMatrices(Alpha).extent(2)),
                        (walker_type == COLLINEAR) ? int(wset.SlaterMatrices(Beta).extent(2)) : 0};
  int neltot = nel[0] + nel[1];
  utils::check_shape(Refs, "References", nrefs, npolNMO, neltot);

  memory::buffered_array<MEM,ComplexType,1> invTotalOverlaps(nwalk, 0);
  memory::buffered_array<MEM,ComplexType,1> singleRefOverlaps(nwalk);
  memory::buffered_array<MEM,ComplexType,1> logShift(nwalk);
  wset.getProperty(OVLP, logShift);

  // totalOverlaps[w] = sum_d conj(ci[d]) * exp(logOverlap[w][d] - logShift[w])
  for(int d = 0; d < nrefs; ++d) {
    singleRefOverlaps() = 0.0;
    for(int spin = 0, offset = 0; spin < nspin; ++spin) {
      det_ops::Log_Overlap(Refs(d, all, range(offset, offset + nel[spin])),
                           wset.SlaterMatrices(spins[spin]), singleRefOverlaps, false);
      offset += nel[spin];
    }
    if(walker_type == CLOSED) {
      nda::tensor::scale(2.0, singleRefOverlaps);
    }
    nda::tensor::add(-1.0, logShift, "w", 1.0, singleRefOverlaps, "w");
    nda::apply(std::conj(wfn.getReferenceWeight(d)), singleRefOverlaps, nda::tensor::unary_op::EXP);
    nda::tensor::add(1.0, singleRefOverlaps, 1.0, invTotalOverlaps);
  }
  nda::apply(1.0, invTotalOverlaps, nda::tensor::unary_op::RCP);

  auto referenceLoop = [invTotalOverlaps = std::move(invTotalOverlaps),
                        singleRefOverlaps = std::move(singleRefOverlaps),
                        logShift = std::move(logShift),
                        spins, nel, neltot, nrefs, nspin, npolNMO, nwalk, walker_type,
                        &wfn, &wset, &Refs, &X, &Yc, &M](auto&& body) mutable {
    using nda::range;
    auto all = range::all;

    memory::buffered_array<MEM,ComplexType,3> Gc(nwalk, neltot, npolNMO);
    memory::buffered_array<MEM,ComplexType,4> singleRefG(nwalk, nspin, npolNMO, npolNMO);

    for(int d = 0; d < nrefs; ++d) {
      singleRefOverlaps() = 0.0;
      Gc() = 0.0;
      singleRefG() = M();

      for(int spin = 0, offset = 0; spin < nspin; ++spin) {
        auto ref = Refs(d, all, range(offset, offset + nel[spin]));
        auto Gspin = Gc(all, range(offset, offset + nel[spin]), all);

        det_ops::MixedDensityMatrix(ref, wset.SlaterMatrices(spins[spin]), Gspin,
                                    singleRefOverlaps, true, false);

        memory::buffered_array<MEM,ComplexType,3> GYc(nwalk, nel[spin], npolNMO);
        memory::buffered_array<MEM,ComplexType,3> XRef(nwalk, nel[spin], npolNMO);

        math::product(Gspin, Yc(all, spin, all, all), GYc);
        math::product<'H'>(ref, X(all, spin, all, all), XRef);
        math::product<'T'>(ComplexType(1.0), XRef, GYc, ComplexType(1.0),
                           singleRefG(all, spin, all, all));

        offset += nel[spin];
      }
      if(walker_type == CLOSED) {
        nda::tensor::scale(2.0, singleRefOverlaps);
      }

      nda::tensor::add(-1.0, logShift, "w", 1.0, singleRefOverlaps, "w");
      nda::apply(std::conj(wfn.getReferenceWeight(d)), singleRefOverlaps, nda::tensor::unary_op::EXP);
      nda::tensor::elementwise(1.0, invTotalOverlaps, 1.0, singleRefOverlaps,
                               nda::tensor::binary_op::PROD);

      body(singleRefOverlaps, singleRefG);
    }
  };

  return MeasurementInputs<MEM,decltype(referenceLoop)>{referenceLoop, weights(), walker_type,
                                                        wfn.getNMO()};
}

} // namespace detail

/*
 * Implements back propagation by evolving operators forward in time.
 * Based on 10.1103/PhysRevA.100.023621.
 */
template<MEMORY_SPACE MEM>
class TimeEvolvedBPEstimator : public EstimatorBase<MEM> {

public:
  /// The measurement intervals of the input are multiples of population_control_interval.
  TimeEvolvedBPEstimator(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                             EstimatorParameters const& params,
                             int population_control_interval,
                             WalkerSet<MEM>& wset,
                             Wavefunction<MEM>& wfn,
                             Propagator<MEM>& prop)
      : wfn_(wfn),
        prop_(prop),
        observables_(mpi, params, wset.getWalkerType(), wfn.getNMO()),
        nback_prop_multipliers_{measure_interval_multipliers(params)},
        steps_per_interval_(population_control_interval),
        path_restoration_(params.path_restoration || params.extra_path_restoration),
        extra_path_restoration_(params.extra_path_restoration) {

    for(int bpsteps : nback_prop_multipliers_) {
      utils::check(bpsteps > 0,
                   "TimeEvolvedBPEstimator: measure_interval_multiplier values must be positive.");
    }
    std::ranges::sort(nback_prop_multipliers_);

    // no reference is back propagated, so a single slot is enough
    wset.resize_bp(nback_prop_multipliers_.back() * steps_per_interval_,
                   prop_.number_of_cholesky_vectors(), 1);

    auto [nspin, npol] = walkerTypeToDims(wset.getWalkerType());
    int npolNMO = npol * wfn_.getNMO();
    X_.resize(wset.size(), nspin, npolNMO, npolNMO);
    Y_.resize(wset.size(), nspin, npolNMO, npolNMO);
    M_.resize(wset.size(), nspin, npolNMO, npolNMO);
    setAnchor(-1);

    app_log(1, "\n  --   Back Propagation with Time Evolved Operators -- \n");
    if(extra_path_restoration_) {
      app_log(1, " Using path restoration with modification to include extra time segment. ");
    } else if(path_restoration_) {
      app_log(1, " Using path restoration. ");
    } else {
      app_log(1, " Path restoration is not used ");
    }
    app_log(1, " Measuring at steps (in units of the population control interval): {}",
            nback_prop_multipliers_);
  }

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock,
               Measurements& meas, WalkerSet<MEM>& wset) override {
    int const bp_step = int(measureBlock - bp_pos_);
    utils::check(bp_step >= 0, " Error: Found bp_step < 0 in TimeEvolvedBPEstimator::measure. ");

    // the operators are evolved one segment at a time, so a measurement needs the index of
    // the multiplier it belongs to and not just the fact that there is one
    auto const it = std::ranges::lower_bound(nback_prop_multipliers_, bp_step);
    if(it != nback_prop_multipliers_.end() && *it == bp_step) {
      evolveAndMeasure(mpi, bp_step, int(std::distance(nback_prop_multipliers_.begin(), it)),
                       meas, wset);
    }
    // the longest average has been taken over this block, so the next one starts here. A
    // block that overshoots the longest average without matching it re-anchors as well.
    if(bp_step >= nback_prop_multipliers_.back()) {
      setAnchor(measureBlock);
    }
  }

private:
  /// Evolve the operators over the segment since the previous measurement and measure the
  /// observables on the Green functions they dress. `iav` is the index of `bp_step` in
  /// nback_prop_multipliers_, which is where the previous segment ended.
  void evolveAndMeasure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, int bp_step,
                        int iav, Measurements& meas, WalkerSet<MEM>& wset) {
    auto all = nda::range::all;
    int nwalk = wset.size();

    auto back_propagate_time = timers.back_propagate.start();

    // 1. propagate X and Y forward and accumulate M. They are already propagated
    //    up to nback_prop_multipliers_[iav-1].
    int nsteps = (bp_step - ((iav > 0) ? nback_prop_multipliers_[iav - 1] : 0)) * steps_per_interval_;
    prop_.PropagateOperators(nsteps, wset, X_, Y_, M_);

    // 2. adjust the weights if using path restoration
    memory::buffered_array<HOST_MEMORY,ComplexType,1> weights(nwalk);
    wset.getProperty(WEIGHT, weights);
    if(path_restoration_) {
      auto factors = nda::to_host(wset.getWeightFactors());
      int hpos(wset.getHistoryPos()); // position where next step goes... go back in history...
      int maxpos(wset.HistoryBufferLength());
      int nbp = bp_step * steps_per_interval_ * (extra_path_restoration_ ? 2 : 1);
      for(int k = 0; k < nbp; k++) {
        // start going back since position is advanced for next step already
        hpos = ((hpos == 0) ? maxpos - 1 : hpos - 1);
        weights(all) *= factors(all, hpos);
      }
    }

    // 3. calculate properties
    memory::buffered_array<MEM,ComplexType,3> references;
    wfn_.getReferences(references);
    // the path restoration above runs on the host, while the observables consume the
    // weights wherever the walkers live
    decltype(auto) measurement_weights = memory::to_memory_space<MEM>(weights);

    auto inputs = detail::constructTimeEvolvedMeasurementInputs<MEM>(measurement_weights, wfn_, wset,
                                                                    references, X_, Y_, M_);
    MeasurementOutput output{mpi, meas, std::format("TimeEvolvedBP/Steps={}", bp_step), weights};

    // the reference loop expects conj(Y), while PropagateOperators keeps Y unconjugated
    nda::tensor::scale(1.0, Y_, nda::tensor::unary_op::CONJ);
    observables_.measure(mpi, output, inputs);
    nda::tensor::scale(1.0, Y_, nda::tensor::unary_op::CONJ);

    back_propagate_time.stop();
  }

  /// Anchor back propagation at `block`, the last block the evolved operators cover, and
  /// restart them from the identity. The constructor anchors before the first block is
  /// propagated, which is block -1.
  void setAnchor(long block) {
    M_() = 0.0;
    math::set_identity(X_);
    math::set_identity(Y_);
    bp_pos_ = block;
  }

  Wavefunction<MEM>& wfn_;
  Propagator<MEM>& prop_;
  Observables<MEM> observables_;

  int bp_pos_{};
  std::vector<int> nback_prop_multipliers_;

  // number of propagation steps per measurement block, which sets the units the
  // propagator and the walker set count back propagation in
  int steps_per_interval_{1};

  // Whether to restore cosine projection and real local energy approximation for weights
  // along back propagation path.
  bool path_restoration_{true};
  bool extra_path_restoration_{false};

  // State matrices for the evolved operators. X->c^+, Y->c
  memory::array<MEM,ComplexType,4> X_;
  memory::array<MEM,ComplexType,4> Y_;
  // Accumulates the scalar terms coming from the stabilization procedure
  memory::array<MEM,ComplexType,4> M_;
};

} // namespace sfqmc::afqmc
