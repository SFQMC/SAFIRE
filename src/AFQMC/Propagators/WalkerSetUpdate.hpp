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

template <class Wlk>
void free_projection_walker_update(Wlk &w, RealType dt,
                                   nda::MemoryVector auto &&overlap,
                                   nda::MemoryVector auto &&MFfactor,
                                   RealType Eshift,
                                   nda::MemoryVector auto &&hybrid_weight,
                                   bool debug_verbosity) {
  auto all = nda::range::all;
  int nwalk = w.size();
  nda::range rng(nwalk);
  memory::buffered_array<HOST_MEMORY, ComplexType, 2> work(11, nwalk);
  auto weight = work(0, all);
  auto phase = work(1, all);
  auto pseudo_eloc = work(2, all);
  auto ovlp = work(3, all);
  auto new_ovlp = work(4, all);
  auto mf_factor = work(5, all);
  auto hyb_weight = work(6, all);
  w.getProperty(WEIGHT, weight);
  w.getProperty(PHASE, phase);
  w.getProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.getProperty(OVLP, ovlp);
  new_ovlp = overlap(rng);
  mf_factor = MFfactor(rng);
  hyb_weight = hybrid_weight(rng);

  for (int i = 0; i < nwalk; i++) {
    ComplexType old_ovlp = ovlp(i);
    ComplexType old_eloc = pseudo_eloc(i);
    ComplexType eloc;
    ComplexType ratioOverlaps = ComplexType(1.0, 0.0);
    eloc = mf_factor(i) / dt;
    ComplexType factor = std::exp(-dt * (0.5 * (eloc + old_eloc) - Eshift));

    if (debug_verbosity) {
      std::cout << " update: iw:       " << i << "\n"
                << "    eloc:          " << eloc << "\n"
                << "    ov:            " << new_ovlp(i) << "\n"
                << "    old_ov:        " << old_ovlp << "\n"
                << "    old_eloc:      " << old_eloc << "\n"
                << "    old_weight:    " << weight(i) << "\n"
                << "    ratio:         " << ratioOverlaps << "\n"
                << "    MFfactor:      " << mf_factor(i) << "\n"
                << "    hybrid_weight: " << hyb_weight(i) << "\n"
                << "    Eshift:         " << Eshift << "\n"
                << "    factor:         " << factor << "\n"
                << std::endl;
    }

    weight(i) *= std::abs(factor);
    phase(i) *= factor / std::abs(factor);
    pseudo_eloc(i) = eloc;
    ovlp(i) = new_ovlp(i);
  }

  w.setProperty(WEIGHT, weight);
  w.setProperty(PHASE, phase);
  w.setProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.setProperty(OVLP, ovlp);
}

template <class Wlk>
void hybrid_walker_update(Wlk &w, RealType dt, bool apply_constrain,
                          bool imp_sampl, RealType Eshift,
                          nda::MemoryVector auto &&overlap,
                          nda::MemoryVector auto &&MFfactor,
                          nda::MemoryVector auto &&hybrid_weight,
                          double lower_cutoff_scale, double upper_cutoff_scale,
                          bool symmetric_split,
                          bool step0, bool debug_verbosity,
                          bool use_cp_constraint,
                          BoundStats &eloc_stats) {
  auto all = nda::range::all;
  int nwalk = w.size();
  bool BackProp = (w.getBPPos() >= 0 && w.getBPPos() < w.NumBackProp());
  nda::range rng(nwalk);
  memory::buffered_array<HOST_MEMORY, ComplexType, 2> work(11, nwalk);
  auto weight = work(0, all);
  auto pseudo_eloc = work(1, all);
  auto ovlp = work(2, all);
  auto weight_factor = work(3, all);
  auto new_ovlp = work(4, all);
  auto mf_factor = work(5, all);
  auto hyb_weight = work(6, all);
  auto phase1 = work(7, all);
  auto phase2 = work(8, all);
  auto phase3 = work(9, all);
  auto theta = work(10, all);
  w.getProperty(WEIGHT, weight);
  w.getProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.getProperty(OVLP, ovlp);
  w.getProperty(PHASE1, phase1);
  w.getProperty(PHASE2, phase2);
  w.getProperty(PHASE3, phase3);
  new_ovlp = overlap(rng);
  mf_factor = MFfactor(rng);
  hyb_weight = hybrid_weight(rng);

  for (int i = 0; i < nwalk; i++) {
    ComplexType old_ovlp = ovlp(i);
    ComplexType old_eloc = pseudo_eloc(i);
    ComplexType eloc;
    RealType delta_theta;
    RealType scale = 1.0;
    ComplexType ratioOverlaps = ComplexType(1.0, 0.0);

    if (imp_sampl)
      ratioOverlaps = std::exp(new_ovlp(i) - old_ovlp);

    if (!std::isfinite(ratioOverlaps.real()) && apply_constrain && imp_sampl) {
      scale = 0.0;
      eloc = old_eloc;
    } else {
      // save constraint theta for sanity checks
      delta_theta = std::arg(ratioOverlaps) - mf_factor(i).imag();
      theta(i) = delta_theta;
      if (use_cp_constraint) {
        // if real part of ratioOverlaps is positive, scale is 1.0 otherwise,
        // scale is 0.0
        scale = (std::cos(delta_theta) > 0.0 ? 1.0 : 0.0);
        ratioOverlaps = std::real(ratioOverlaps); // is this needed?
      } else {
        scale = (apply_constrain ? std::max(0.0, std::cos(delta_theta)) : 1.0);
      }

      if (imp_sampl)
        eloc = (mf_factor(i) - hyb_weight(i) - (new_ovlp(i) - old_ovlp)) / dt;
      else
        eloc = mf_factor(i) / dt;
    }
    ComplexType eloc_ = eloc;

    if (!std::isfinite(eloc.real())) {
      scale = 0.0;
      eloc = old_eloc;
    } else {
      RealType hi = Eshift + upper_cutoff_scale * std::sqrt(2.0 / dt);
      RealType lo = Eshift - lower_cutoff_scale * std::sqrt(2.0 / dt);
      ++eloc_stats.total;
      if (eloc.real() > hi)
        ++eloc_stats.upper;
      else if (eloc.real() < lo)
        ++eloc_stats.lower;
      eloc = ComplexType(std::max(std::min(eloc.real(), hi), lo), eloc.imag());
    }

    if (debug_verbosity) {
      std::cout << " update: iw:       " << i << "\n"
                << "    eloc:          " << eloc << "\n"
                << "    eloc_:         " << eloc_ << "\n"
                << "    ov:            " << new_ovlp(i) << "\n"
                << "    old_ov:        " << old_ovlp << "\n"
                << "    old_eloc:      " << old_eloc << "\n"
                << "    old_weight:    " << weight(i) << "\n"
                << "    ratio:         " << ratioOverlaps << "\n"
                << "    MFfactor:      " << mf_factor(i) << "\n"
                << "    hybrid_weight: " << hyb_weight(i) << "\n"
                << "    scale:         " << scale << "\n"
                << "    Eshift:         " << Eshift << "\n"
                << "    Theta:         " << theta(i) << "\n"
                << std::endl;
    }

    if (symmetric_split) {
      if(step0)
          weight(i) *= ComplexType(
            scale *
                std::exp(-dt * (eloc.real() - Eshift)),
            0.0);
      else
        weight(i) *= ComplexType(
            scale *
                std::exp(-dt * (0.5 * (eloc.real() + old_eloc.real()) - Eshift)),
            0.0);
      }
    else
      weight(i) *=
          ComplexType(scale * std::exp(-dt * (eloc.real() - Eshift)), 0.0);
    pseudo_eloc(i) = eloc;
    ovlp(i) = new_ovlp(i);
    if (std::abs(scale) > std::numeric_limits<RealType>::min()) {
      weight_factor(i) = std::exp(-ComplexType(0.0, dt) *
                                  (0.5 * (eloc.imag() + old_eloc.imag()))) /
                         scale;
    }
    else
      weight_factor(i) = 0.0;
    phase1(i) *= weight_factor(i);
    phase2(i) = new_ovlp(i);
    phase3(i) =
        scale; // KE: this was originally the cumulative product of "scale"
               //    changed to just "scale" since this isn't used anywhere
  }
  w.setProperty(WEIGHT, weight);
  w.setProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.setProperty(OVLP, ovlp);
  w.setProperty(PHASE1, phase1);
  w.setProperty(PHASE2, phase2);
  w.setProperty(PHASE3, phase3);
  w.setProperty(THETA, theta);
  if (BackProp) {
    auto pos = w.getHistoryPos();
    auto WFac = w.getWeightFactors();
    WFac(all, pos) = weight_factor;
    auto WHis = w.getWeightHistory();
    WHis(all, pos) = weight;
  }
}

template <class Wlk>
void local_energy_walker_update(Wlk &w, RealType dt, bool apply_constrain,
                                RealType Eshift,
                                nda::MemoryVector auto &&overlap,
                                nda::MemoryMatrix auto &&energies,
                                nda::MemoryVector auto &&MFfactor,
                                double lower_cutoff_scale,
                                double upper_cutoff_scale,
                                BoundStats &eloc_stats) {
  auto all = nda::range::all;
  int nwalk = w.size();
  bool BackProp = (w.getBPPos() >= 0 && w.getBPPos() < w.NumBackProp());
  nda::range rng(nwalk);
  memory::buffered_array<HOST_MEMORY, ComplexType, 2> work(14, nwalk);
  auto weight = work(0, all);
  auto pseudo_eloc = work(1, all);
  auto ovlp = work(2, all);
  auto e1 = work(3, all);
  auto exx = work(4, all);
  auto ej = work(5, all);
  auto weight_factor = work(6, all);
  auto new_ovlp = work(7, all);
  auto mf_factor = work(8, all);
  auto new_e1 = work(9, all);
  auto new_exx = work(10, all);
  auto new_ej = work(11, all);
  auto phase = work(12, all);
  auto theta = work(13, all);
  w.getProperty(WEIGHT, weight);
  w.getProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.getProperty(OVLP, ovlp);
  w.getProperty(E1_, e1);
  w.getProperty(EXX_, exx);
  w.getProperty(EJ_, ej);
  w.getProperty(PHASE, phase);
  new_ovlp = overlap(rng);
  mf_factor = MFfactor(rng);
  new_e1 = energies(rng, 0);
  new_exx = energies(rng, 1);
  new_ej = energies(rng, 2);

  for (int i = 0; i < nwalk; i++) {
    ComplexType old_ovlp = ovlp(i);
    ComplexType old_eloc = pseudo_eloc(i);
    ComplexType eloc = new_e1(i) + new_exx(i) + new_ej(i);
    RealType scale = 1.0;
    ComplexType ratioOverlaps = std::exp(new_ovlp(i) - old_ovlp);

    if (!std::isfinite((ratioOverlaps * mf_factor(i)).real()) &&
        apply_constrain) {
      scale = 0.0;
      eloc = old_eloc;
    } else {
      theta(i) = std::arg(ratioOverlaps) - mf_factor(i).imag();
      scale =
          (apply_constrain ? (std::max(0.0, std::cos(std::arg(ratioOverlaps) -
                                                     mf_factor(i).imag())))
                           : 1.0);
    }
    if (!std::isfinite(eloc.real())) {
      scale = 0.0;
      eloc = old_eloc;
    } else {
      RealType hi = Eshift + upper_cutoff_scale * std::sqrt(2.0 / dt);
      RealType lo = Eshift - lower_cutoff_scale * std::sqrt(2.0 / dt);
      ++eloc_stats.total;
      if (eloc.real() > hi)
        ++eloc_stats.upper;
      else if (eloc.real() < lo)
        ++eloc_stats.lower;
      eloc = ComplexType(std::max(std::min(eloc.real(), hi), lo), eloc.imag());
    }

    weight(i) *= ComplexType(
        scale *
            std::exp(-dt * (0.5 * (eloc.real() + old_eloc.real()) - Eshift)),
        0.0);
    if (std::abs(scale) > std::numeric_limits<RealType>::min()) {
      weight_factor(i) = std::exp(-ComplexType(0.0, dt) *
                                (0.5 * (eloc.imag() + old_eloc.imag()))) /
                       scale;
    } else {
      weight_factor(i) = 0.0;
    }
    phase(i) *= weight_factor(i);
    pseudo_eloc(i) = eloc;
    ovlp(i) = new_ovlp(i);
    e1(i) = new_e1(i);
    exx(i) = new_exx(i);
    ej(i) = new_ej(i);
  }

  w.setProperty(WEIGHT, weight);
  w.setProperty(PSEUDO_ELOC_, pseudo_eloc);
  w.setProperty(OVLP, ovlp);
  w.setProperty(E1_, e1);
  w.setProperty(EXX_, exx);
  w.setProperty(EJ_, ej);
  w.setProperty(PHASE, phase);
  w.setProperty(THETA, theta);
  if (BackProp) {
    auto pos = w.getHistoryPos();
    auto WFac = w.getWeightFactors();
    WFac(all, pos) = weight_factor;
    auto WHis = w.getWeightHistory();
    WHis(all, pos) = weight;
  }
}

} // namespace afqmc

} // namespace sfqmc
