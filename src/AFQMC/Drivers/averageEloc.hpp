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

/// Weight-averaged pseudo local energy of the walker population,
/// Re(sum_i w_i * Eloc_i) / Re(sum_i w_i), summed over all ranks. Walkers that
/// processWalkerData() found unusable carry zero weight and energy, so they drop out of
/// both sums.
template<MEMORY_SPACE MEM>
RealType averageEloc(utils::mpi_context_t<boost::mpi3::communicator>& mpi, WalkerSet<MEM> const& wset) {
  nda::array<ComplexType, 1> weight(wset.size()), eloc(wset.size());
  wset.getProperty(WEIGHT, weight);
  wset.getProperty(PSEUDO_ELOC_, eloc);

  std::array<ComplexType, 2> sums{nda::sum(weight * eloc), nda::sum(weight)};
  mpi.comm.all_reduce_in_place_n(sums.data(), sums.size(), std::plus<>{});

  return sums[0].real() / sums[1].real();
}

} // namespace sfqmc::afqmc
