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

#include <variant>

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/ModelComponents/Continuous_GeneralUJ.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/Discrete_GeneralUJ.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, bool REAL>
class ModelComponent 
{

public:
  ModelComponent(Continuous_GeneralUJ<MEM> const& other) : var(other) {}
  ModelComponent(Continuous_GeneralUJ<MEM>&& other) : var(std::move(other)) {}

  ModelComponent(Discrete_GeneralUJ<MEM,REAL> const& other) : var(other) {} 
  ModelComponent(Discrete_GeneralUJ<MEM,REAL>&& other) : var(std::move(other)) {}

  template<class... Args>
  void addOneBodyPropagatorMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.addOneBodyPropagatorMatrix(std::forward<Args>(args)...); },
                                var);
  }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void vHS(Args&&... args)
  {
    std::visit([&](auto&& s) { s.vHS(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void vbias(Args&&... args)
  {
    std::visit([&](auto&& s) { s.vbias(std::forward<Args>(args)...); }, var);
  }

  int number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  int number_of_ke_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_ke_vectors(); }, var);
  }

  void getFieldTypes(nda::MemoryVector auto && v) const 
  {
    std::visit([&](auto&& a) { a.getFieldTypes(v); }, var);
  }

  template<class... Args>
  void update(Args&&... args)
  {
    std::visit([&](auto&& a) { a.update(std::forward<Args>(args)...); }, var);
  }

  private:

  std::variant<Continuous_GeneralUJ<MEM>, Discrete_GeneralUJ<MEM,REAL>> var;

};

} // namespace afqmc

} // namespace sfqmc

