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

/// Re(sum_i w_i * v_i) / Re(sum_i w_i) over the walkers of all ranks.
RealType weightedAverage(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                         nda::array<ComplexType, 1> const& weight,
                         auto const& v) {
  std::array<ComplexType, 2> sums{nda::sum(weight * v), nda::sum(weight)};
  mpi.comm.all_reduce_in_place_n(sums.data(), sums.size(), std::plus<>{});

  return sums[0].real() / sums[1].real();
}

} // namespace detail

/// Weight-averaged pseudo local energy of the walker population,
/// Re(sum_i w_i * Eloc_i) / Re(sum_i w_i), summed over all ranks.
///
/// `PSEUDO_ELOC_` is written by the propagator's walker update, so this is only meaningful after
/// the first propagation step. Before it, use `averageEnergy`.
template<MEMORY_SPACE MEM>
RealType averagePseudoEnergy(utils::mpi_context_t<boost::mpi3::communicator>& mpi, WalkerSet<MEM> const& wset) {
  nda::array<ComplexType, 1> weight(wset.size()), eloc(wset.size());
  wset.getProperty(WEIGHT, weight);
  wset.getProperty(PSEUDO_ELOC_, eloc);

  return detail::weightedAverage(mpi, weight, eloc);
}

/// Weight-averaged local energy of the walker population, taken from the components
/// `Wavefunction::Energy` leaves on the walkers, `E1_ + EXX_ + EJ_`.
template<MEMORY_SPACE MEM>
RealType averageEnergy(utils::mpi_context_t<boost::mpi3::communicator>& mpi, WalkerSet<MEM> const& wset) {
  nda::array<ComplexType, 1> weight(wset.size()), e1(wset.size()), exx(wset.size()), ej(wset.size());
  wset.getProperty(WEIGHT, weight);
  wset.getProperty(E1_, e1);
  wset.getProperty(EXX_, exx);
  wset.getProperty(EJ_, ej);

  return detail::weightedAverage(mpi, weight, e1 + exx + ej);
}

} // namespace sfqmc::afqmc
