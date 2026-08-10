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

// Helpers shared by StochasticWfn tests across five TUs. Kept out of test_common.hpp so non-stochastic
// tests do not pay for nda/tensor.hpp + nda/h5.hpp. Fixture flags stay in test_common.hpp.

#pragma once

#include <format>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/h5.hpp>

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"

namespace sfqmc {
namespace utils {

// Select the StochasticWfn branch of WavefunctionFactory. The NOMSD trial file is unchanged; only the
// input `type` differs, which is what makes a stochastic-vs-NOMSD parity comparison meaningful.
inline void mark_stochastic_wfn_input(afqmc::WavefunctionParameters& p) { p.type = afqmc::WavefunctionInputType::stochasticwfn; }

// Fill in the Hamiltonian-dependent parameter defaults that resolve_defaults() supplies in production.
//
// A unit test that hands a hand-built WavefunctionParameters / PropagatorParameters straight to a factory
// bypasses resolve_defaults, and reading an unresolved optional is a hard abort -- "The parameter
// 'dense_trial' was not resolved. Did resolve_defaults run?" (and 'vbias_bound' for the propagator). That
// abort is a REAL guard for production and must not be weakened; the caller that skipped resolution is
// what has to fix itself.
//
// Takes the built Hamiltonian and asks it for its type, which is upstream's own idiom
// (`apply_defaults(prop_params, ham.getHamType())` in propagator_factory: build) and avoids re-opening the
// integral file. Same escape hatch WavefunctionFactory uses for the inner propagator, which likewise never
// sees resolve_defaults.
inline void apply_wfn_defaults(afqmc::WavefunctionParameters& p, afqmc::Hamiltonian& ham)
{
  afqmc::apply_defaults(p, ham.getHamType());
}
inline void apply_prop_defaults(afqmc::PropagatorParameters& p, afqmc::Hamiltonian& ham)
{
  afqmc::apply_defaults(p, ham.getHamType());
}

// Deterministic, reproducible perturbation of the outer walker Slater matrices. Independently built
// walker sets come out bit-for-bit identical, which is what lets a parity check compare two
// wavefunctions rather than two walker populations; and it drives the walkers far enough off the anchor
// (|<psi|phi>| down to ~1e-9, |G| up to ~80) that the overlaps are non-trivial.
//
// Templated on the walker-set type so this header does not need AFQMC/Walkers/WalkerSet.hpp; callers
// still write perturb_stochastic_walkers<MEM>(wset, ...) and WlkSet is deduced.
template<MEMORY_SPACE MEM, class WlkSet>
void perturb_stochastic_walkers(WlkSet& wset, afqmc::WALKER_TYPES type, int NMO, int nup, int ndown)
{
  const int nspin = (type == afqmc::COLLINEAR) ? 2 : 1;
  const int npol  = (type == afqmc::NONCOLLINEAR) ? 2 : 1;
  const int nwalk = wset.size();
  std::array<int, 2> nels = {nup, ndown};
  for (int spin = 0; spin < nspin; spin++)
  {
    // A fully polarized system carried as COLLINEAR (the Li rohf_nomsd_polarized fixture) has
    // nels[Beta] == 0, so this block is EMPTY: there is nothing to perturb, and handing a zero-extent
    // operand to nda::tensor::add builds a cuTENSOR descriptor that fails CUTENSOR_STATUS_NOT_SUPPORTED
    // -- which aborts the whole process via MPI_Abort rather than failing one assertion.
    if (nels[spin] == 0)
      continue;
    nda::array<ComplexType, 1> p_h(long(nwalk) * npol * NMO * nels[spin]);
    for (long k = 0; k < p_h.size(); ++k)
    {
      double v = 0.1 * (k + 1);
      p_h[k] = {std::cos(v), std::sin(v * v)};
    }
    memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
    auto SM = wset.SlaterMatrices(static_cast<afqmc::SpinTypes>(spin));
    nda::tensor::add(p, "ijk", SM, "ijk");
  }
}

// Read back one back-propagated 1-RDM block from a .stat.h5 and assert it is usable: non-empty, nonzero
// denominator, every element finite. Shared by the estimator-handler and driver integration smokes,
// which assert finiteness only -- the accuracy of the back-propagated RDM is a separate question.
inline void require_finite_bp_one_rdm(h5::file const& file, std::string const& avg_path, int iblock)
{
  std::string suffix = std::format("{:09d}", iblock);
  nda::array<ComplexType, 1> read_data;
  ComplexType denom{};
  {
    h5::group root(file);
    utils::h5_read(root, avg_path + "/one_rdm_" + suffix, read_data);
    h5::read(root, avg_path + "/denominator_" + suffix, denom);
  }
  REQUIRE(read_data.size() > 0);
  REQUIRE(std::abs(denom) > 0.0);
  for (auto v : read_data)
  {
    REQUIRE(std::isfinite(real(v)));
    REQUIRE(std::isfinite(imag(v)));
  }
}

} // namespace utils
} // namespace sfqmc
