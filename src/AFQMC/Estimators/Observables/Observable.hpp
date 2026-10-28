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

#include <optional>
#include <tuple>

#include "onerdm.hpp"
#include "twordm.hpp"
#include "diagonal_twordm.hpp"
#include "spincorr.hpp"
#include "paircorr.hpp"

namespace sfqmc::afqmc {

namespace detail {
/// Constructs the observable Obs if the input declared a block of parameters for it.
template<typename Obs, typename Params, typename... Args>
std::optional<Obs> observable_from_params(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                         std::optional<Params> const& params, Args&&... args) {
  if(!params) {
    return std::nullopt;
  }
  return std::optional<Obs>(std::in_place, mpi, *params, std::forward<Args>(args)...);
}
} // namespace detail

template<MEMORY_SPACE MEM>
class Observables {
public:
  Observables(utils::mpi_context_t<boost::mpi3::communicator>& mpi, auto const& params,
              WALKER_TYPES walker_type, int NMO)
      : observables_{detail::observable_from_params<OneRDM<MEM>>(mpi, params.onerdm, walker_type, NMO),
                     detail::observable_from_params<TwoRDM<MEM>>(mpi, params.twordm, walker_type, NMO),
                     detail::observable_from_params<DiagonalTwoRDM<MEM>>(mpi, params.diag_twordm, walker_type, NMO),
                     detail::observable_from_params<SpinCorr<MEM>>(mpi, params.spincorr, walker_type, NMO),
                     detail::observable_from_params<PairCorr<MEM>>(mpi, params.paircorr, walker_type,
                                                                   NMO)} {}

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output,
               MeasurementInputs<MEM, RefLoop>& inputs) {
    std::apply([&](auto&... obs) {
      auto measure_if_requested = [&](auto& o) {
        if(o) {
          o->measure(mpi, output, inputs);
        }
      };
      (measure_if_requested(obs), ...);
    }, observables_);
  }
private:
  std::tuple<
    std::optional<OneRDM<MEM>>,
    std::optional<TwoRDM<MEM>>,
    std::optional<DiagonalTwoRDM<MEM>>,
    std::optional<SpinCorr<MEM>>,
    std::optional<PairCorr<MEM>>
  > observables_;

};



} // namespace sfqmc::afqmc
