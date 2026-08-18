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

#include <variant>
#include <complex>

#include "configuration.hpp"
#include "utilities/macros.hpp"
#include "AFQMC/Walkers/WalkerSetBase.h"

namespace sfqmc
{
namespace afqmc
{

// MAM: use this as long as there is only 1 choice
template<MEMORY_SPACE MEM>
using WalkerSet = WalkerSetBase<MEM>;

// MAM: re-enable if another WalkerSet option is needed, e.g. PW, Finite-T, ...
/* 
template<MEMORY_SPACE MEM>
class WalkerSet
{

  public:

    WalkerSet() = default;

    explicit WalkerSet(WalkerSetBase<MEM> const& arg) : var(arg) {}
    explicit WalkerSet(WalkerSetBase<MEM> && arg) : var(std::move(arg)) {}

    WalkerSet& operator=(WalkerSetBase<MEM> const& arg) { var = arg; return *this; }
    WalkerSet& operator=(WalkerSetBase<MEM> && arg) { var = std::move(arg); return *this; }

    ~WalkerSet() = default;
    WalkerSet(WalkerSet const&) = default;
    WalkerSet(WalkerSet &&) = default;
    WalkerSet& operator=(WalkerSet const&) = default;
    WalkerSet& operator=(WalkerSet &&) = default;

    static ptree interpret_inputs(const ptree pt0) {
      // assuming that input/ptree interpretation does not depend on memory space
      return WalkerSetBase<HOST_MEMORY>::interpret_inputs(pt0);
    }

    // Using macros to simplify the implementation. 
    // Look at WalkerSetBase for description of routines. 
    VISITOR(get_memory_space,var,const)
    VISITOR(size,var,const)
    VISITOR(capacity,var,const)
    VISITOR(NumBackProp,var,const)
    VISITOR(NumCholVecs,var,const)
    VISITOR(HistoryBufferLength,var,const)
    VISITOR(getBPPos,var,const)
    VISITOR(getHistoryPos,var,const)
    VISITOR(clean,var,)
    VISITOR(getWalkerType,var,const)
    VISITOR(get_target_population,var,const)
    VISITOR(get_global_target_population,var,const)
    VISITOR(walker_dims,var,const)
    VISITOR(GlobalPopulation,var,const)
    VISITOR(GlobalWeight,var,const)
    VISITOR(get_mpi,var,const)
    VISITOR(single_walker_memory_usage,var,const)
    VISITOR(single_walker_size,var,const)
    VISITOR(single_walker_bp_memory_usage,var,const)
    VISITOR(single_walker_bp_size,var,const)
    VISITOR(population_control_parameters,var,const)
    VISITOR(walkerSizeIO,var,const)
    VISITOR(getRNG,var,)
    VISITOR(begin,var,)  
    VISITOR(begin,var,const)  
    VISITOR(end,var,)  
    VISITOR(end,var,const)  

    VOID_VISITOR(advanceBPPos,var,)
    VOID_VISITOR(advanceHistoryPos,var,)
    VOID_VISITOR(popControl,var,)
    VOID_VISITOR(scaleWeightsByOverlap,var,)
    VOID_VISITOR(resetWeights,var,)

    VOID_VISITOR_ARGS(resize,var,)
    VOID_VISITOR_ARGS(setBPPos,var,)
    VOID_VISITOR_ARGS(setHistoryPos,var,)
    VOID_VISITOR_ARGS(reserve,var,)
    VOID_VISITOR_ARGS(resize_bp,var,)
    VOID_VISITOR_ARGS(benchmark,var,)
    VOID_VISITOR_ARGS(processWalkerData,var,)
    VOID_VISITOR_ARGS(scaleWeight,var,)
    VOID_VISITOR_ARGS(popControl,var,)
    VOID_VISITOR_ARGS(copyToIO,var,const)
    VOID_VISITOR_ARGS(copyFromIO,var,)
    VOID_VISITOR_ARGS(getProperty,var,const)
    VOID_VISITOR_ARGS(setProperty,var,)
    VOID_VISITOR_ARGS(storeFields,var,)
    VOID_VISITOR_ARGS(loadBalance,var,)

    auto operator[](int i) { 
      return std::visit( [&](auto&& v) { return v[i]; }, var ); 
    } 
    auto operator[](int i) const { 
      return std::visit( [&](auto&& v) { return v[i]; }, var ); 
    } 

    VISITOR_ARGS(SlaterMatrices,var,)
    VISITOR_ARGS(SlaterMatrices,var,const)
    VISITOR_ARGS(UMatrices,var,)
    VISITOR_ARGS(UMatrices,var,const)
    VISITOR_ARGS(DMatrices,var,)
    VISITOR_ARGS(DMatrices,var,const)
    VISITOR_ARGS(VMatrices,var,)
    VISITOR_ARGS(VMatrices,var,const)
    VISITOR_ARGS(SlaterMatricesN,var,)
    VISITOR_ARGS(SlaterMatricesN,var,const)
    VISITOR_ARGS(getFields,var,)
    VISITOR(getWeightHistory,var,)
    VISITOR(getWeightFactors,var,)
    
  private:

    std::variant<WalkerSetBase<MEM>> var;

};
*/

} // namespace afqmc

} // namespace sfqmc

