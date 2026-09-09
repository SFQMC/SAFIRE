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

#include "configuration.hpp"
#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"

#include "utilities/check.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/Measurements.hpp"

namespace sfqmc::afqmc {

/// Walker averaged 2 RDM, as [spinblock][i][k][j][l] with the spin blocks ordered
/// (a,a,a,a), (a,a,b,b), (b,b,b,b).
template<MEMORY_SPACE MEM>
class TwoRDM {
public:
  TwoRDM(utils::mpi_context_t<boost::mpi3::communicator>&, TwoRDMParameters const&, WALKER_TYPES walker_type,
         int /*NMO*/) {
    utils::check(walker_type == COLLINEAR, "TwoRDM is only implemented for collinear walkers, not {}",
                 walkerTypeToString(walker_type));
  }

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output,
               MeasurementInputs<MEM, RefLoop>& inputs) {
    using nda::ellipsis;
    auto all = nda::range::all;

    int nwalk = inputs.weights.size();
    int NMO   = inputs.NMO;

    memory::buffered_array<MEM, ComplexType, 1> weightedCoeff(nwalk);
    memory::buffered_array<MEM, ComplexType, 5> avg(3, NMO, NMO, NMO, NMO);
    avg() = 0.0;

    inputs.referenceLoop([&](memory::array_view<MEM, ComplexType, 1> refCoeff,
                             memory::array_view<MEM, ComplexType, 4> singleRefG) {
      weightedCoeff() = inputs.weights();
      nda::tensor::elementwise(1.0, refCoeff, 1.0, weightedCoeff, nda::tensor::binary_op::PROD);

      memory::buffered_array<MEM, ComplexType, 4> XwG(singleRefG.shape());
      nda::tensor::contract(weightedCoeff(), "w", singleRefG, "wsij", XwG, "wsij");

      // (ikjl) = G_ik G_jl - (same spin only) G_il G_jk
      for(int spin = 0; spin < 2; ++spin) {
        auto XwGs = XwG(all, spin, ellipsis{});
        auto Gs   = singleRefG(all, spin, ellipsis{});
        nda::tensor::contract(1.0, XwGs, "wik", Gs, "wjl", 1.0, avg(2 * spin, ellipsis{}), "ikjl");
        nda::tensor::contract(-1.0, XwGs, "wil", Gs, "wjk", 1.0, avg(2 * spin, ellipsis{}), "ikjl");
      }
      nda::tensor::contract(1.0, XwG(all, 0, ellipsis{}), "wik", singleRefG(all, 1, ellipsis{}), "wjl", 1.0,
                            avg(1, ellipsis{}), "ikjl");
    });

    output.measure(mpi, "TwoRDM", avg);
  }
};

} // namespace sfqmc::afqmc
