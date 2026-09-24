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

#include <array>
#include <functional>

#include <nda/nda.hpp>

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc {

namespace detail {

/// sum_i w_i * v_i / sum_i w_i over the walkers of all ranks.
ComplexType weightedAverage(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                         nda::array<ComplexType, 1> const& weight,
                         auto const& v) {
  std::array<ComplexType, 2> sums{nda::sum(weight * v), nda::sum(weight)};
  mpi.comm.all_reduce_in_place_n(sums.data(), sums.size(), std::plus<>{});

  return sums[0] / sums[1];
}

} // namespace detail

/// Growth estimator for the energy based on pseudo local energy of the walker population.
/// Rather than doing the naive average of Eloc, does the exponential average
///
/// log(Σ_i w_i exp(dt * Eloc_i) / Σ w)/dt ~ log(Σ w_old / Σ w)/dt ~ E
///
/// which seems to reduce the bias, especially in model systems and free projection.
///
/// `PSEUDO_ELOC_` is written by the propagator's walker update, and seeded from the trial energy
/// before the first step, so it is always on the same scale as the total energy.
template<MEMORY_SPACE MEM>
RealType averagePseudoEnergy(utils::mpi_context_t<boost::mpi3::communicator>& mpi, WalkerSet<MEM> const& wset, RealType timestep) {
  memory::buffered_array<HOST_MEMORY, ComplexType, 1> weight(wset.size()), eloc(wset.size());
  wset.getProperty(WEIGHT, weight);
  wset.getProperty(PSEUDO_ELOC_, eloc);
  for(auto& e : eloc) {
    e = std::exp(timestep*e);
  }

  return std::log(detail::weightedAverage(mpi, weight, eloc).real())/timestep;
}

/// Weight-averaged local energy of the walker population, taken from the components
/// `Wavefunction::Energy` leaves on the walkers, `E1_ + EXX_ + EJ_`.
template<MEMORY_SPACE MEM>
RealType averageEnergy(utils::mpi_context_t<boost::mpi3::communicator>& mpi, WalkerSet<MEM> const& wset) {
  memory::buffered_array<HOST_MEMORY, ComplexType, 1> weight(wset.size()), e1(wset.size()), exx(wset.size()), ej(wset.size());
  wset.getProperty(WEIGHT, weight);
  wset.getProperty(E1_, e1);
  wset.getProperty(EXX_, exx);
  wset.getProperty(EJ_, ej);

  return detail::weightedAverage(mpi, weight, e1 + exx + ej).real();
}

} // namespace sfqmc::afqmc
