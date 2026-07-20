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
#include <iostream>
#include <fstream>

#include "nda/h5.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class EstimatorBase
{
public:
  virtual ~EstimatorBase() {}

  virtual void accumulate_block(double time, WalkerSet<MEM>& wlks) = 0;

  virtual void accumulate_step(double time, WalkerSet<MEM>& wlks, std::vector<ComplexType>& curData) = 0;

  virtual void print(std::ofstream& out, h5::file&, WalkerSet<MEM>& wlks) = 0;

  virtual void print_timers([[maybe_unused]] std::ofstream& out) {}

  virtual void tags(std::ofstream& out) = 0;

  virtual void tags_timers([[maybe_unused]] std::ofstream& out) {}

  virtual int get_measurement_interval() { return 1; }

  virtual double getEloc() { return 0; }

  virtual double getEloc_step() { return 0; }
};
} // namespace afqmc
} // namespace sfqmc

