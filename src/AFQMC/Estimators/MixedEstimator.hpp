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

#include "AFQMC/parameters.hpp"
#include "Measurements.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc::afqmc {

namespace detail {

/*
 * The mixed estimator measures the walkers where they are, so the Green function of a
 * reference is just the (non-compact) mixed density matrix between that reference and the
 * walker Slater matrix, weighted by the reference's share of the total overlap.
 */
template<MEMORY_SPACE MEM>
auto constructMixedMeasurementInputs(nda::MemoryVector auto const& weights,
                                    Wavefunction<MEM>& wfn,
                                    WalkerSet<MEM>& wset,
                                    nda::MemoryArrayOfRank<3> auto const& Refs) {
  using nda::range;
  auto all = range::all;

  WALKER_TYPES walker_type = wfn.getWalkerType();
  auto [nspin, npol] = walkerTypeToDims(walker_type);
  int nwalk   = wset.size();
  int npolNMO = npol * wfn.getNMO();
  int nrefs   = wfn.total_number_of_references();

  std::array<SpinTypes,2> const spins{Alpha, Beta};
  std::array<int,2> nel{int(wset.SlaterMatrices(Alpha).extent(2)),
                        (walker_type == COLLINEAR) ? int(wset.SlaterMatrices(Beta).extent(2)) : 0};
  utils::check_shape(Refs, "References", nrefs, npolNMO, nel[0] + nel[1]);

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
    nda::tensor::add(-1.0, logShift, "w", 1.0, singleRefOverlaps, "w");
    nda::apply(std::conj(wfn.getReferenceWeight(d)), singleRefOverlaps, nda::tensor::unary_op::EXP);
    nda::tensor::add(1.0, singleRefOverlaps, 1.0, invTotalOverlaps);
  }
  nda::apply(1.0, invTotalOverlaps, nda::tensor::unary_op::RCP);

  auto referenceLoop = [invTotalOverlaps = std::move(invTotalOverlaps),
                        singleRefOverlaps = std::move(singleRefOverlaps),
                        logShift = std::move(logShift),
                        spins, nel, nrefs, nspin, npolNMO, nwalk,
                        &wfn, &wset, &Refs](auto&& body) mutable {
    using nda::range;
    auto all = range::all;

    memory::buffered_array<MEM,ComplexType,4> singleRefG(nwalk, nspin, npolNMO, npolNMO);

    for(int d = 0; d < nrefs; ++d) {
      singleRefOverlaps() = 0.0;
      for(int spin = 0, offset = 0; spin < nspin; ++spin) {
        det_ops::MixedDensityMatrix(Refs(d, all, range(offset, offset + nel[spin])),
                                    wset.SlaterMatrices(spins[spin]),
                                    singleRefG(all, spin, all, all), singleRefOverlaps, false, false);
        offset += nel[spin];
      }

      nda::tensor::add(-1.0, logShift, "w", 1.0, singleRefOverlaps, "w");
      nda::apply(std::conj(wfn.getReferenceWeight(d)), singleRefOverlaps, nda::tensor::unary_op::EXP);
      nda::tensor::elementwise(1.0, invTotalOverlaps, 1.0, singleRefOverlaps,
                               nda::tensor::binary_op::PROD);

      body(singleRefOverlaps, singleRefG);
    }
  };

  return MeasurementInputs<MEM,decltype(referenceLoop)>{referenceLoop, weights(),
                                                       wfn.getWalkerType(), wfn.getNMO()};
}

} // namespace detail

template<MEMORY_SPACE MEM>
class MixedEstimator : public EstimatorBase<MEM> {
public:
  MixedEstimator(utils::mpi_context_t<boost::mpi3::communicator> &mpi,
                 EstimatorParameters const &params, WALKER_TYPES walker_type,
                 Wavefunction<MEM> &wfn)
      : wfn_{wfn},
        observables_{mpi, params, walker_type, wfn.getNMO()},
        measure_interval_multiplier_{resolved(
            params.measure_interval_multiplier, "measure_interval_multiplier").at(0)} {}

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock,
               Measurements& meas, WalkerSet<MEM> &wset) override {
    auto mixed_estimator_time = timers.mixed_estimator.start();

    if(measureBlock % measure_interval_multiplier_ != 0) {
      return;
    }

    memory::buffered_array<MEM,ComplexType,1> weights(wset.size());
    wset.getProperty(WEIGHT, weights);

    wfn_.getReferences(references_);
    auto inputs = detail::constructMixedMeasurementInputs<MEM>(weights, wfn_, wset, references_);
    MeasurementOutput output{mpi, meas, "MixedEstimator", weights};
    observables_.measure(mpi, output, inputs);

    mixed_estimator_time.stop();
  }
private:
  Wavefunction<MEM>& wfn_;
  Observables<MEM> observables_;

  int measure_interval_multiplier_{};

  // resized by getReferences, kept between measurements to avoid reallocating
  memory::buffered_array<MEM,ComplexType,3> references_;
};

}
