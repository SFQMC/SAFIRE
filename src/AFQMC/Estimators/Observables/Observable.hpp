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
#include <configuration.hpp>
#include <variant>
#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"

#include "AFQMC/Estimators/Observables/full1rdm.hpp"
#include "AFQMC/Estimators/Observables/full2rdm.hpp"
#include "AFQMC/Estimators/Observables/diagonal2rdm.hpp"
//#include "AFQMC/Estimators/Observables/n2r.hpp"
//#include "AFQMC/Estimators/Observables/realspace_correlators.hpp"
//#include "AFQMC/Estimators/Observables/atomcentered_correlators.hpp"
//#include "AFQMC/Estimators/Observables/generalizedFockMatrix.hpp"
//#include "AFQMC/Estimators/Observables/sk.hpp"
#include "AFQMC/Estimators/Observables/pair_correlators.hpp"
#include "AFQMC/Estimators/Observables/spinspin.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * Variant class for observables. 
 * Defines a common interface for all observable classes.
 */
template<MEMORY_SPACE MEM>
class Observable  
{
public:
  template<class Obs>
  explicit Observable(Obs&& other) : var(std::forward<Obs>(other)) {}

/*******   Interface for sum over independent references, e.g. NOMSD  *******/
  template<class... Args>
  void accumulate(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate(std::forward<Args>(args)...); }, var);
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/ 
  template<class... Args>
  void accumulate_reference_configuration(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_reference_configuration(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void accumulate_excited_configuration_first(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_excited_configuration_first(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void accumulate_excited_configuration_second(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_excited_configuration_second(std::forward<Args>(args)...); }, var);
  }

/*******   single print routine for all cases   *******/ 
  template<class... Args>
  void print(Args&&... args)
  {
    std::visit([&](auto&& a) { a.print(std::forward<Args>(args)...); }, var);
  }

private:

  std::variant<full1rdm, diagonal2rdm<MEM>, full2rdm<MEM>, pair_correlator, spinspinobs
    > var;
/*
                                         realspace_correlators,
                                         atomcentered_correlators,
                                         generalizedFockMatrix,
                                         n2r<shared_allocator<ComplexType>>
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
                                         ,
                                         n2r<device_allocator<ComplexType>>
#endif
                                         ,sk<true>
                                         ,sk<false>,
                                         >
*/

};


} // namespace afqmc

} // namespace sfqmc


