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

//
//

#pragma once

#include <array>

#include "utilities/Timer.hpp"

namespace sfqmc::afqmc {

struct AFQMCTimers {
  utils::Timer block{"Block"};
  utils::Timer pseudo_energy{"PseudoEnergy"};
  utils::Timer energy{"Energy"};
  utils::Timer vHS{"vHS"};
  utils::Timer assemble_X{"X"};
  utils::Timer vbias{"vbias"};
  utils::Timer G_for_vbias{"G_for_vbias"};
  utils::Timer propagate{"Propagate"};
  utils::Timer mixed_estimator{"MixedEstimator"};
  utils::Timer back_propagate{"BackPropagate"};
  utils::Timer popcontrol{"PopulationControl"};
  utils::Timer ortho{"WalkerOrthogonalization"};
  utils::Timer setup{"Setup"};
  utils::Timer extra{"Extra"};
  utils::Timer load_balance{"WalkerSet::loadBalance"};
  utils::Timer branching{"WalkerSet::branching"};

  static constexpr int ntimers = 16;

  std::array<utils::Timer*, ntimers> all() {
    return {&block,           &pseudo_energy,  &energy, &vHS,   &assemble_X,   &vbias,
            &G_for_vbias,     &propagate,      &mixed_estimator, &back_propagate,
            &popcontrol,      &ortho,          &setup,  &extra, &load_balance, &branching};
  }

  void reset_all() {
    for(auto* t : all()) {
      t->reset();
    }
  }

  void print_all() { utils::print_timers(all()); }
};

// every member of AFQMCTimers has the same type, hence no padding: a timer added without extending
// all() changes sizeof and is caught here rather than silently dropped from reset_all()/print_all()
static_assert(sizeof(AFQMCTimers) == AFQMCTimers::ntimers * sizeof(utils::Timer),
              "AFQMCTimers::all() is out of sync with the member list");

extern AFQMCTimers timers;

}

