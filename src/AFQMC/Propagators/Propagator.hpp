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

#include <fstream>
#include <variant>

#include "AFQMC/config.h"

#include "AFQMC/Propagators/AFQMCBasePropagator.h"
//#include "AFQMC/Propagators/AFQMCModelPropagator.h"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Propagator 
{
public:
  Propagator() = delete;

  explicit Propagator(AFQMCBasePropagator<MEM>&& other) : var(std::move(other)) {}
  explicit Propagator(AFQMCBasePropagator<MEM> const& other) : var(other) {} 

  template<class... Args>
  void Propagate(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Propagate(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void BackPropagate(Args&&... args)
  {
    std::visit([&](auto&& a) { a.BackPropagate(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void PropagateOperators(Args&&... args)
  {
    std::visit([&](auto&& a) { a.PropagateOperators(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void generateP1(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generateP1(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void Orthogonalize(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Orthogonalize(std::forward<Args>(args)...); }, var);
  }

  bool hybrid_propagation()
  {
    return std::visit([&](auto&& a) { return a.hybrid_propagation(); }, var);
  }

  bool free_propagation()
  {
    return std::visit([&](auto&& a) { return a.free_propagation(); }, var);
  }

  int number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<class... Args>
  void set_rng_block_size(Args&&... args)
  {
    std::visit([&](auto&& a) { a.set_rng_block_size(std::forward<Args>(args)...); }, var);
  }

  void printBoundStatistics()
  {
    std::visit([&](auto&& a) { a.printBoundStatistics(); }, var);
  }

  private:

  std::variant<AFQMCBasePropagator<MEM>> var;

};

} // namespace afqmc

} // namespace sfqmc

