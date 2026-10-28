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

#include <format>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"

#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/Measurements.hpp"

namespace sfqmc::afqmc {

/// Walker averaged pair correlators P_ab(i,j). Every dataset in the `pairs` group defines one
/// orbital pair map and gives its name to the correlator; each ordered pair of maps is measured
/// as its own series.
template<MEMORY_SPACE MEM>
class PairCorr {
public:
  PairCorr(utils::mpi_context_t<boost::mpi3::communicator>&, PairCorrParameters const& params,
           WALKER_TYPES walker_type, int NMO) {
    utils::check(walker_type == COLLINEAR || walker_type == NONCOLLINEAR,
                 "PairCorr is not implemented for {} walkers, use collinear or noncollinear instead",
                 walkerTypeToString(walker_type));

    h5::file input(params.pairs.filename, 'r');
    h5::group group = h5::group{input}.open_group(params.pairs.group);

    names_ = group.get_all_dataset_names();
    utils::check(!names_.empty(), "the group '{}' of '{}' holds no orbital pair map", params.pairs.group,
                 params.pairs.filename);

    for(auto const& name : names_) {
      // the name becomes part of an h5 path below, which cannot carry a separator itself
      utils::check(name.find('/') == std::string::npos, "the pair map name '{}' contains a '/'", name);

      memory::host_array<int, 2> map;
      h5::read(group, name, map);
      utils::check_shape(map, name, NMO, 1);
      pair_maps_.push_back(std::move(map));
    }
  }

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output,
               MeasurementInputs<MEM, RefLoop>& inputs) {
    auto all = nda::range::all;

    WALKER_TYPES walker_type = inputs.walkerType;
    int nwalk                = inputs.weights.size();
    int NMO                  = inputs.NMO;
    int ncorr                = static_cast<int>(names_.size());

    memory::buffered_array<MEM, ComplexType, 1> weightedCoeff(nwalk);
    memory::buffered_array<HOST_MEMORY, ComplexType, 4> avg(ncorr, ncorr, NMO, NMO);
    avg() = 0.0;

    inputs.referenceLoop([&](memory::array_view<MEM, ComplexType, 1> refCoeff,
                             memory::array_view<MEM, ComplexType, 4> singleRefG) {
      weightedCoeff() = inputs.weights();
      nda::tensor::elementwise(1.0, refCoeff, 1.0, weightedCoeff, nda::tensor::binary_op::PROD);

      // scalar loops, so the Green function and the weights are needed on the host
      auto G_host = nda::to_host(singleRefG);
      auto Xw     = nda::to_host(weightedCoeff());

      // no parallelization over ncores for now, fix if needed
      for(int iw = 0; iw < nwalk; ++iw) {
        for(int alpha = 0; alpha < ncorr; ++alpha) {
          auto const& pair_map_i = pair_maps_[alpha];
          for(int beta = 0; beta < ncorr; ++beta) {
            auto const& pair_map_j = pair_maps_[beta];
            auto P                 = avg(alpha, beta, all, all);

            if(walker_type == COLLINEAR) {
              auto Gup = G_host(iw, 0, all, all);
              auto Gdn = G_host(iw, 1, all, all);
              for(int i = 0; i < NMO; ++i) {
                int ibar = pair_map_i(i, 0);
                if(ibar < 0) { // negative index means no valid pair!
                  continue;
                }
                for(int j = i + 1; j < NMO; ++j) {
                  int jbar = pair_map_j(j, 0);
                  if(jbar < 0) {
                    continue;
                  }

                  ComplexType c{};
                  c += Gup(i, j) * Gdn(ibar, jbar);
                  c += Gup(i, jbar) * Gdn(ibar, j);
                  c += Gdn(i, jbar) * Gup(ibar, j);
                  c += Gdn(i, j) * Gup(ibar, jbar);
                  P(i, j) += 0.5 * Xw(iw) * c;
                }
              }
            } else {
              auto G_ = G_host(iw, 0, all, all);
              for(int i = 0; i < NMO; ++i) {
                int ibar = pair_map_i(i, 0);
                if(ibar < 0) {
                  continue;
                }
                for(int j = 0; j < NMO; ++j) {
                  int jbar = pair_map_j(j, 0);
                  if(jbar < 0) {
                    continue;
                  }

                  ComplexType c{};
                  c += (G_(i, j) * G_(NMO + ibar, NMO + jbar) - G_(i, NMO + jbar) * G_(NMO + ibar, j));
                  c += (G_(i, jbar) * G_(NMO + ibar, NMO + j) - G_(i, NMO + j) * G_(NMO + ibar, jbar));
                  c += (G_(NMO + i, NMO + jbar) * G_(ibar, j) - G_(NMO + i, j) * G_(ibar, NMO + jbar));
                  c += (G_(NMO + i, NMO + j) * G_(ibar, jbar) - G_(NMO + i, jbar) * G_(ibar, NMO + j));
                  P(i, j) += 0.5 * Xw(iw) * c;
                }
              }
            }
          }
        }
      }
    });

    for(int alpha = 0; alpha < ncorr; ++alpha) {
      for(int beta = 0; beta < ncorr; ++beta) {
        output.measure(mpi, std::format("PairCorr/{}_{}", names_[alpha], names_[beta]),
                       avg(alpha, beta, all, all));
      }
    }
  }

private:
  std::vector<std::string> names_;
  std::vector<memory::host_array<int, 2>> pair_maps_;
};

} // namespace sfqmc::afqmc
