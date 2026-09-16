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

#include <array>
#include <span>
#include <variant>
#include <vector>

#include "configuration.hpp"
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc::afqmc {

/**
 * @brief Everything a walker set needs in order to size and populate its walkers.
 *
 * The payload is either a set of Slater matrices for ground-state walkers or a set of UDV slabs
 * for finite-temperature ones. The walker dimensions and the finite-temperature flag are derived
 * from whichever it holds, so they cannot disagree with the data they describe.
 *
 * The guess is produced when the trial wavefunction is read and consumed later, when the
 * walker set is built, so it owns its payload rather than viewing it.
 */
struct WalkerSetInitialGuess {
  /// Ground state: one (rows x naea) / (NMO x naeb) matrix per spin.
  using slater_guess = std::span<const nda::matrix<ComplexType>>;
  /// Finite temperature: {3, nspin, rows, naea}, the slabs being U, D (as a full matrix whose
  /// diagonal is used) and V.
  using udv_guess = memory::array_view<HOST_MEMORY, const ComplexType, 4>;

  /// The owning counterparts of the two views above. The finite-temperature guess lives in
  /// shared memory, so it is produced collectively and handed over by move.
  using slater_matrices = std::vector<nda::matrix<ComplexType>>;
  using udv_matrices    = memory::const_shared_array<HOST_MEMORY, ComplexType, 4>;

  WALKER_TYPES walker_type{UNDEFINED_WALKER_TYPE};
  std::variant<slater_matrices, udv_matrices> payload{};

  bool isFiniteTemperature() const { return std::holds_alternative<udv_matrices>(payload); }

  /// The Slater matrices of a ground-state guess, one per spin.
  slater_guess slater() const {
    auto const* M = std::get_if<slater_matrices>(&payload);
    utils::check(M != nullptr, "The initial guess is finite temperature, not ground state.");
    return *M;
  }

  /// The UDV slabs of a finite-temperature guess.
  udv_guess udv() const {
    auto const* UDV = std::get_if<udv_matrices>(&payload);
    utils::check(UDV != nullptr, "The initial guess is ground state, not finite temperature.");
    return (*UDV)();
  }

  /// Dimensions {rows, naea, naeb} of a walker set holding this guess. rows carries the
  /// 2*NMO factor for noncollinear; naeb is 0 unless the guess is collinear.
  std::array<int, 3> walker_dims() const {
    if(isFiniteTemperature()) {
      udv_guess UDV = udv();
      utils::check(UDV.extent(0) == 3, "Invalid finite-T initial guess.");
      int rows = int(UDV.extent(2));
      int naea = int(UDV.extent(3));
      // nspin == 2 signals collinear-ft, where both spins have the same width
      int naeb = (UDV.extent(1) == 2) ? naea : 0;
      return {rows, naea, naeb};
    }
    slater_guess M = slater();
    utils::check(M.size() == 1 || M.size() == 2, "Invalid initial guess.");
    return {int(M[0].extent(0)), int(M[0].extent(1)), M.size() > 1 ? int(M[1].extent(1)) : 0};
  }
};

} // namespace sfqmc::afqmc
