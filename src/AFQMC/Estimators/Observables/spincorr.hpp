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

#include "configuration.hpp"
#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"

#include "utilities/check.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/Measurements.hpp"

namespace sfqmc::afqmc {

/// Walker averaged spin-spin correlation <S_i . S_j>, split into an XY and a Z channel.
/// <S_i S_j> = <S_j S_i>, so only the upper triangle including the diagonal is measured.
template<MEMORY_SPACE MEM>
class SpinCorr {
public:
  SpinCorr(utils::mpi_context_t<boost::mpi3::communicator>&, SpinCorrParameters const&, WALKER_TYPES walker_type,
           int /*NMO*/) {
    utils::check(walker_type == CLOSED || walker_type == COLLINEAR || walker_type == NONCOLLINEAR,
                 "SpinCorr is not implemented for {} walkers", walkerTypeToString(walker_type));
  }

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output,
               MeasurementInputs<MEM, RefLoop>& inputs) {
    using nda::ellipsis;

    WALKER_TYPES walker_type = inputs.walkerType;
    int nwalk                = inputs.weights.size();
    int NMO                  = inputs.NMO;

    memory::buffered_array<MEM, ComplexType, 1> weightedCoeff(nwalk);
    memory::buffered_array<HOST_MEMORY, ComplexType, 2> avg(2, NMO * (NMO + 1) / 2);
    avg() = 0.0;

    auto avg_xy = avg(0, nda::range::all);
    auto avg_z  = avg(1, nda::range::all);

    inputs.referenceLoop([&](memory::array_view<MEM, ComplexType, 1> refCoeff,
                             memory::array_view<MEM, ComplexType, 4> singleRefG) {
      weightedCoeff() = inputs.weights();
      nda::tensor::elementwise(1.0, refCoeff, 1.0, weightedCoeff, nda::tensor::binary_op::PROD);

      // scalar loops, so the Green function and the weights are needed on the host
      auto G_host = nda::to_host(singleRefG);
      auto Xw     = nda::to_host(weightedCoeff());

      // no parallelization over ncores for now, fix if needed
      for(int iw = 0; iw < nwalk; ++iw) {
        int idx{};

        if(walker_type == CLOSED) {
          auto Gu_ = G_host(iw, 0, ellipsis{});
          for(int i = 0; i < NMO; ++i) {
            for(int j = i; j < NMO; ++j, ++idx) {
              if(j == i) {
                avg_z(idx) += Xw[iw] / 4.0 * (Gu_(i, i) * (1.0 - Gu_(j, j)) + Gu_(i, i) * (1.0 - Gu_(j, j)));
                avg_xy(idx) += Xw[iw] / 2.0 * (Gu_(i, j) * (1.0 - Gu_(j, i)) + Gu_(i, j) * (1.0 - Gu_(j, i)));
              } else {
                avg_z(idx) += Xw[iw] / 4.0 *
                              (Gu_(i, i) * (Gu_(j, j) - Gu_(j, j)) + Gu_(i, i) * (Gu_(j, j) - Gu_(j, j)) -
                               Gu_(i, j) * Gu_(j, i) // only U2 term
                               - Gu_(i, j) * Gu_(j, i));

                avg_xy(idx) += Xw[iw] / 2.0 * (Gu_(i, j) * (-Gu_(j, i)) + Gu_(i, j) * (-Gu_(j, i)));
              }
            }
          }
        } else if(walker_type == COLLINEAR) {
          auto Gu_ = G_host(iw, 0, ellipsis{});
          auto Gd_ = G_host(iw, 1, ellipsis{});

          for(int i = 0; i < NMO; ++i) {
            for(int j = i; j < NMO; ++j, ++idx) {
              if(j == i) {
                avg_z(idx) += Xw[iw] / 4.0 * (Gu_(i, i) * (1.0 - Gd_(j, j)) + Gd_(i, i) * (1.0 - Gu_(j, j)));
                avg_xy(idx) += Xw[iw] / 2.0 * (Gu_(i, j) * (1.0 - Gd_(j, i)) + Gd_(i, j) * (1.0 - Gu_(j, i)));
              } else {
                avg_z(idx) += Xw[iw] / 4.0 *
                              (Gu_(i, i) * (Gu_(j, j) - Gd_(j, j)) + Gd_(i, i) * (Gd_(j, j) - Gu_(j, j)) -
                               Gu_(i, j) * Gu_(j, i) // only U2 term
                               - Gd_(i, j) * Gd_(j, i));

                avg_xy(idx) += Xw[iw] / 2.0 * (Gu_(i, j) * (-Gd_(j, i)) + Gd_(i, j) * (-Gu_(j, i)));
              }
            }
          }
        } else {
          auto G_ = G_host(iw, 0, ellipsis{});
          for(int i = 0; i < NMO; ++i) {
            for(int j = i; j < NMO; ++j, ++idx) {
              if(j == i) {
                avg_z(idx) += Xw[iw] / 4.0 *
                              (G_(i, i) * (1.0 - G_(j + NMO, j + NMO)) + G_(i + NMO, i + NMO) * (1.0 - G_(j, j)) -
                               G_(i, j + NMO) * (-G_(j + NMO, i)) - G_(i + NMO, j) * (-G_(j, i + NMO)));

                avg_xy(idx) += Xw[iw] / 2.0 *
                               (G_(i, j) * (1.0 - G_(j + NMO, i + NMO)) + G_(i + NMO, j + NMO) * (1.0 - G_(j, i)) +
                                G_(j + NMO, j) * G_(i, i + NMO) + G_(j, j + NMO) * G_(i + NMO, i));
              } else {
                avg_z(idx) += Xw[iw] / 4.0 *
                              (G_(i, i) * (G_(j, j) - G_(j + NMO, j + NMO)) +
                               G_(i + NMO, i + NMO) * (G_(j + NMO, j + NMO) - G_(j, j)) -
                               G_(i, j) * G_(j, i) // only U2 term
                               - G_(i + NMO, j + NMO) * G_(j + NMO, i + NMO) - G_(i, j + NMO) * (-G_(j + NMO, i)) -
                               G_(i + NMO, j) * (-G_(j, i + NMO)));

                avg_xy(idx) += Xw[iw] / 2.0 *
                               (G_(i, j) * (-G_(j + NMO, i + NMO)) + G_(i + NMO, j + NMO) * -G_(j, i) +
                                G_(j + NMO, j) * G_(i, i + NMO) + G_(j, j + NMO) * G_(i + NMO, i));
              }
            }
          }
        }

        utils::check(idx == avg.extent(1), "SpinCorr filled {} of {} elements", idx, avg.extent(1));
      }
    });

    output.measure(mpi, "SpinCorr", avg);
  }
};

} // namespace sfqmc::afqmc
