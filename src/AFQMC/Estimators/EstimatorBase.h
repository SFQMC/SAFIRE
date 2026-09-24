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

#include <nda/h5.hpp>

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Measurements.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
class EstimatorBase
{
public:
  virtual ~EstimatorBase() {}

  /// Called once per propagation step, with `step` the number of steps completed so far. An
  /// estimator that does not measure at this step has to return before doing anything at all
  /// -- no timer, no allocation, no collective -- because this runs on every step of the run.
  virtual void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long step, Measurements& meas, WalkerSet<MEM> &wset) = 0;

  /// Called once, at the last equilibration step, before the first measure(). A back
  /// propagation estimator anchors here, because the anchor its constructor took describes
  /// the walkers before equilibration.
  virtual void equilibrated(long /*step*/, WalkerSet<MEM>& /*wset*/) {}
};

} // namespace sfqmc::afqmc

