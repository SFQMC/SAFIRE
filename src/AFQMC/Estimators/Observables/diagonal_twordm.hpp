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

/// Walker averaged diagonal of the 2 RDM. The measurement is folded down over the i <-> j
/// symmetry before it is recorded, which halves the number of accumulated series.
template<MEMORY_SPACE MEM>
class DiagonalTwoRDM {
public:
  DiagonalTwoRDM(utils::mpi_context_t<boost::mpi3::communicator>&, DiagonalTwoRDMParameters const&,
                 WALKER_TYPES walker_type, int /*NMO*/) {
    utils::check(walker_type == CLOSED || walker_type == COLLINEAR || walker_type == NONCOLLINEAR,
                 "DiagonalTwoRDM is not implemented for {} walkers", walkerTypeToString(walker_type));
  }

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output,
               MeasurementInputs<MEM, RefLoop>& inputs) {
    using nda::ellipsis;
    auto all = nda::range::all;

    WALKER_TYPES walker_type = inputs.walkerType;
    auto [nspin, npol]       = walkerTypeToDims(walker_type);
    int nwalk                = inputs.weights.size();
    int M                    = npol * inputs.NMO;

    memory::buffered_array<MEM, ComplexType, 1> weightedCoeff(nwalk);
    memory::buffered_array<MEM, ComplexType, 3> avg(spin_blocks(walker_type), M, M);
    avg() = 0.0;

    inputs.referenceLoop([&](memory::array_view<MEM, ComplexType, 1> refCoeff,
                             memory::array_view<MEM, ComplexType, 4> singleRefG) {
      weightedCoeff() = inputs.weights();
      nda::tensor::elementwise(1.0, refCoeff, 1.0, weightedCoeff, nda::tensor::binary_op::PROD);

      memory::buffered_array<MEM, ComplexType, 4> XwG(singleRefG.shape());
      nda::tensor::contract(weightedCoeff(), "w", singleRefG, "wsij", XwG, "wsij");

      // the Hartree terms only read the diagonal of G. Taking it as a strided view rather
      // than as a repeated contraction index ("wii") keeps this expressible on the device:
      // cuTENSOR requires every mode to appear at most once per tensor.
      auto diag = [](auto&& a) { return memory::diagonal_view(a); };

      // (aaaa) and (bbbb)
      for(int spin = 0; spin < nspin; ++spin) {
        auto XwGs = XwG(all, spin, ellipsis{});
        auto Gs   = singleRefG(all, spin, ellipsis{});
        nda::tensor::contract(1.0, diag(XwGs), "wi", diag(Gs), "wj", 1.0, avg(2 * spin, ellipsis{}), "ij");
        nda::tensor::contract(-1.0, XwGs, "wij", Gs, "wji", 1.0, avg(2 * spin, ellipsis{}), "ij");
      }
      // (aabb) does not exist for noncollinear
      if(walker_type == CLOSED || walker_type == COLLINEAR) {
        auto XwGup = XwG(all, 0, ellipsis{});
        auto Gdn   = singleRefG(all, (walker_type == COLLINEAR) ? 1 : 0, ellipsis{});
        nda::tensor::contract(1.0, diag(XwGup), "wi", diag(Gdn), "wj", 1.0, avg(1, ellipsis{}), "ij");
      }
    });

    output.measure(mpi, "DiagonalTwoRDM", compress(avg, walker_type, inputs.NMO));
  }

private:
  /// Number of spin blocks: [(aaaa), (aabb), (bbbb)] for collinear, [(aaaa), (aabb)] for closed
  /// and [(aaaa)] for noncollinear, where the single noncollinear block spans 2*NMO.
  static int spin_blocks(WALKER_TYPES walker_type) {
    switch(walker_type) {
    case COLLINEAR: return 3;
    case CLOSED: return 2;
    default: return 1;
    }
  }

  /// Length of the folded measurement, see compress().
  static int compressed_size(WALKER_TYPES walker_type, int NMO) {
    int size = NMO * (2 * NMO - 1);
    if(walker_type == CLOSED) {
      size -= NMO * (NMO - 1) / 2;
    }
    return size;
  }

  /// Fold down the i <-> j symmetry: the same-spin blocks keep only their symmetrized upper
  /// triangle, while (aabb) is not symmetric and is kept whole. The result is laid out row by
  /// row - for every i, the (aaaa) row j > i followed by the whole (aabb) row - and the (bbbb)
  /// triangle follows at the end. Noncollinear has a single 2*NMO block and no (aabb).
  static auto compress(nda::MemoryArrayOfRank<3> auto const& full, WALKER_TYPES walker_type, int NMO) {
    auto host_full = nda::to_host(full);

    memory::buffered_array<HOST_MEMORY, ComplexType, 1> result(compressed_size(walker_type, NMO));
    int idx{};

    if(walker_type == CLOSED || walker_type == COLLINEAR) {
      for(int i = 0; i < NMO; ++i) {
        for(int j = i + 1; j < NMO; ++j, ++idx) {
          result(idx) = (host_full(0, i, j) + host_full(0, j, i)) / 2;
        }
        for(int j = 0; j < NMO; ++j, ++idx) {
          result(idx) = host_full(1, i, j);
        }
      }
      if(walker_type == COLLINEAR) {
        for(int i = 0; i < NMO; ++i) {
          for(int j = i + 1; j < NMO; ++j, ++idx) {
            result(idx) = (host_full(2, i, j) + host_full(2, j, i)) / 2;
          }
        }
      }
    } else {
      for(int i = 0; i < 2 * NMO; ++i) {
        for(int j = i + 1; j < 2 * NMO; ++j, ++idx) {
          result(idx) = (host_full(0, i, j) + host_full(0, j, i)) / 2;
        }
      }
    }

    utils::check(idx == result.extent(0), "DiagonalTwoRDM filled {} of {} compressed elements", idx,
                 result.extent(0));
    return result;
  }
};

} // namespace sfqmc::afqmc
