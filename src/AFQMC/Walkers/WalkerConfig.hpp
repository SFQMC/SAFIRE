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

#ifndef SFQMC_AFQMC_WALKERCONFIG_HPP
#define SFQMC_AFQMC_WALKERCONFIG_HPP

// basic definitions used by walker set classes
namespace sfqmc
{
namespace afqmc
{
// wlk_descriptor: [ nmo, naea, naeb, nback_prop, nCV, nRefs, nHist]
using wlk_descriptor = std::array<int, 8>;
using wlk_indices    = std::array<int, 24>;
enum walker_data
{
  SM,
  UR,
  DR,
  VR,
  WEIGHT,
  PHASE,
  PHASE1,
  PHASE2,
  PHASE3,
  PSEUDO_ELOC_,
  E1_,
  EXX_,
  EJ_,
  OVLP,
  LOGSCL_UP,
  LOGSCL_DN,
  IS_UNITARY,
  SMN,
  FIELDS,
  WEIGHT_FAC,
  WEIGHT_HISTORY,
  THETA,
  // Per-walker BOOKKEEPING scalar (stochastic conditioned trials only), NOT a physical walker quantity:
  // the pre-branch local slot index, used to realign a stochastic trial's conditioned inner ensemble with
  // the outer walkers after a population-control event. Stored as the real part of this ComplexType slot
  // because walker_buffer is a single homogeneous ComplexType array, and that homogeneity is exactly what
  // lets branch()/loadBalance() carry this index for free via whole-row copies -- a native int would have
  // to live in a separate parallel array mirrored by hand at every walker-move site. Integer values are
  // exact in the double mantissa. Appended after THETA so it stays outside the walkerSizeIO() checkpoint
  // window.
  SLOT_LINEAGE,
  // Per-walker BOOKKEEPING block (stochastic trials with a persistent field-sampled inner ensemble),
  // NOT a physical walker quantity: the auxiliary-field configurations that generate the walker's
  // tethered inner trial samples. Zero-width unless a wavefunction requests it via
  // resize_trial_fields(); real field values stored in the real parts. Living inside walker_buffer
  // means branch()'s whole-row copies clone the fields with the walker and the load-balance payload
  // ships them across ranks, so the inner samples can be reconstructed exactly wherever the walker
  // lands. Appended after SLOT_LINEAGE, outside the walkerSizeIO() checkpoint window.
  TRIAL_FIELDS,
};

} // namespace afqmc
} // namespace sfqmc

enum LOAD_BALANCE_ALGORITHM
{
  UNDEFINED_LOAD_BALANCE,
  SIMPLE,
  ASYNC
};
enum BRANCHING_ALGORITHM
{
  UNDEFINED_BRANCHING,
  PAIR,
  COMB,
  MIN_BRANCH,
  SERIAL_COMB
};

#endif
