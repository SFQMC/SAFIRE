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
#include "AFQMC/parameters.hpp"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Measurements.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
class EstimatorBase
{
public:
  virtual ~EstimatorBase() {}
  virtual void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, long measureBlock, Measurements& meas, WalkerSet<MEM> &wset) = 0;
};

} // namespace sfqmc::afqmc

