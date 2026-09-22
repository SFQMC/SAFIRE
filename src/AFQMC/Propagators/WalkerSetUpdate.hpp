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

#include <algorithm>

#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/config.h"
#include "config.h"
#include "utilities/check.hpp"

namespace sfqmc {
namespace afqmc {

struct BoundStats {
  long total = 0;
  long upper = 0;
  long lower = 0;
  BoundStats &operator+=(BoundStats const &o) {
    total += o.total;
    upper += o.upper;
    lower += o.lower;
    return *this;
  }
};

/*
 * Clamps the real part of the local energy to Eshift +/- cutoff*sqrt(2/dt) and records how often
 * either side was hit. The bound keeps a diverging local energy from blowing up the weight of a
 * single walker; it biases the projection, so it only belongs on a constrained propagation.
 */
inline ComplexType bound_local_energy(ComplexType eloc, RealType dt, RealType Eshift,
                                      double lower_cutoff_scale, double upper_cutoff_scale,
                                      BoundStats &eloc_stats) {
  RealType hi = Eshift + upper_cutoff_scale * std::sqrt(2.0 / dt);
  RealType lo = Eshift - lower_cutoff_scale * std::sqrt(2.0 / dt);
  ++eloc_stats.total;
  if(eloc.real() > hi) {
    ++eloc_stats.upper;
  } else if(eloc.real() < lo) {
    ++eloc_stats.lower;
  }
  return ComplexType(std::clamp(eloc.real(), lo, hi), eloc.imag());
}

/*
 * Advances the walker weights by one propagation step.
 *
 * `hybrid` picks where the local energy comes from: the hybrid expression, reconstructed from the
 * mean-field factor and the step's overlap ratio, or the explicit E1+EXX+EJ that the wavefunction
 * evaluated. `free_projection` drops the phaseless constraint and keeps the full complex step
 * factor in the weight instead of splitting its phase off into PHASE; it also disables the
 * local-energy bound, which would bias an otherwise exact projection.
 */
template <class Wlk>
void walker_update(Wlk &w, bool hybrid, bool free_projection, bool use_cp_constraint,
                   RealType dt, RealType Eshift, RealType energy_offset,
                   nda::MemoryVector auto &&overlap,
                   nda::MemoryMatrix auto &&energies,
                   nda::MemoryVector auto &&meanfield_factor,
                   nda::MemoryVector auto &&hybrid_weight,
                   double lower_cutoff_scale, double upper_cutoff_scale,
                   bool debug_verbosity,
                   BoundStats &eloc_stats) {
  auto all = nda::range::all;
  int nwalk = w.size();
  bool BackProp = (w.NumBackProp() > 0);
  nda::range rng(nwalk);
  memory::buffered_array<HOST_MEMORY, ComplexType, 2> work(15, nwalk);
  auto weight = work(0, all);
  auto phase = work(1, all);
  auto pseudo_eloc = work(2, all);
  auto ovlp = work(3, all);
  auto weight_factor = work(4, all);
  auto theta = work(5, all);
  auto new_ovlp = work(6, all);
  auto mf_factor = work(7, all);
  auto hyb_weight = work(8, all);
  auto e1 = work(9, all);
  auto exx = work(10, all);
  auto ej = work(11, all);
  auto new_e1 = work(12, all);
  auto new_exx = work(13, all);
  auto new_ej = work(14, all);
  w.getProperty(WEIGHT, weight);
  w.getProperty(PHASE, phase);
  w.getProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.getProperty(OVLP, ovlp);
  new_ovlp = overlap(rng);
  mf_factor = meanfield_factor(rng);
  hyb_weight = hybrid_weight(rng);
  if(!hybrid) {
    w.getProperty(E1_, e1);
    w.getProperty(EXX_, exx);
    w.getProperty(EJ_, ej);
    new_e1 = energies(rng, 0);
    new_exx = energies(rng, 1);
    new_ej = energies(rng, 2);
  }

  for(int i = 0; i < nwalk; i++) {
    ComplexType old_ovlp = ovlp(i);
    ComplexType old_eloc = pseudo_eloc(i);
    ComplexType old_weight = weight(i);

    RealType scale = 1.0;
    RealType delta_theta = 0.0;
    if(!free_projection) {
      delta_theta = new_ovlp(i).imag() - old_ovlp.imag() - mf_factor(i).imag();
      if(use_cp_constraint) {
        scale = (std::cos(delta_theta) > 0.0 ? 1.0 : 0.0);
      } else {
        scale = std::max(0.0, std::cos(delta_theta));
      }
    }
    theta(i) = delta_theta;

    ComplexType eloc =
        hybrid ? (mf_factor(i) - hyb_weight(i) - (new_ovlp(i) - old_ovlp)) / dt + energy_offset
               : new_e1(i) + new_exx(i) + new_ej(i);
    ComplexType unbounded_eloc = eloc;

    if(!std::isfinite(eloc.real())) {
      // nothing sensible left to propagate; drop the walker and keep its last good energy
      weight(i) = 0.0;
      weight_factor(i) = 0.0;
      phase(i) = 0.0;
      eloc = old_eloc;
    } else {
      if(!free_projection) {
        eloc = bound_local_energy(eloc, dt, Eshift, lower_cutoff_scale, upper_cutoff_scale,
                                  eloc_stats);
      }
      // the hybrid expression is already the log of the exact step factor, while the local
      // energy needs a midpoint rule between the old and the new time slice
      ComplexType eloc_eff = hybrid ? eloc : 0.5 * (eloc + old_eloc);
      RealType amplitude = std::exp(-dt * (eloc_eff.real() - Eshift));
      ComplexType step_phase = std::exp(-ComplexType(0.0, dt) * eloc_eff.imag());
      if(std::abs(scale) > std::numeric_limits<RealType>::min()) {
        weight_factor(i) = step_phase / scale;
      } else {
        weight_factor(i) = 0.0;
      }
      // free projection carries the phase of the step in the weight itself; the phaseless
      // constraint drops it from the weight and only records it in PHASE
      weight(i) *= free_projection ? scale * amplitude * step_phase : scale * amplitude;
      phase(i) *= weight_factor(i);
    }

    if(debug_verbosity) {
      std::cout << " update: iw:       " << i << "\n"
                << "    eloc:          " << eloc << "\n"
                << "    eloc_:         " << unbounded_eloc << "\n"
                << "    ov:            " << new_ovlp(i) << "\n"
                << "    old_ov:        " << old_ovlp << "\n"
                << "    old_eloc:      " << old_eloc << "\n"
                << "    old_weight:    " << old_weight << "\n"
                << "    mf_factor:     " << mf_factor(i) << "\n"
                << "    hybrid_weight: " << hyb_weight(i) << "\n"
                << "    scale:         " << scale << "\n"
                << "    Eshift:        " << Eshift << "\n"
                << "    Theta:         " << theta(i) << "\n"
                << std::endl;
    }

    pseudo_eloc(i) = eloc;
    ovlp(i) = new_ovlp(i);
    if(!hybrid) {
      e1(i) = new_e1(i);
      exx(i) = new_exx(i);
      ej(i) = new_ej(i);
    }
  }

  w.setProperty(WEIGHT, weight);
  w.setProperty(PHASE, phase);
  w.setProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.setProperty(OVLP, ovlp);
  w.setProperty(THETA, theta);
  if(!hybrid) {
    w.setProperty(E1_, e1);
    w.setProperty(EXX_, exx);
    w.setProperty(EJ_, ej);
  }
  if(BackProp) {
    auto pos = w.getHistoryPos();
    auto WFac = w.getWeightFactors();
    WFac(all, pos) = weight_factor;
    auto WHis = w.getWeightHistory();
    WHis(all, pos) = weight;
  }
}

/*
 * Caps the magnitude of every walker weight at max(floor, fraction*N), where N is the global
 * target population. A walker over the bound is rescaled onto it and keeps its phase. This
 * keeps a single blown-up walker from dominating the population between two branching events.
 */
template <class Wlk>
void bound_walker_weights(Wlk &w, double weight_bound_floor,
                          double weight_bound_fraction,
                          BoundStats &weight_stats) {
  int nwalk = w.size();
  memory::buffered_array<HOST_MEMORY, ComplexType, 1> weight(nwalk);
  w.getProperty(WEIGHT, weight);

  const RealType max_weight =
      std::max(weight_bound_floor, weight_bound_fraction * w.get_global_target_population());

  for(int i = 0; i < nwalk; i++) {
    RealType abs_weight = std::abs(weight(i));
    ++weight_stats.total;
    if(abs_weight > max_weight) {
      ++weight_stats.upper;
      weight(i) *= max_weight / abs_weight;
    }
  }

  w.setProperty(WEIGHT, weight);
}

} // namespace afqmc

} // namespace sfqmc
