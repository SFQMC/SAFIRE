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

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include <utilities/mpi_context.h>
#include <vector>
#include <string>

#include <nda/nda.hpp>
#include <nda/h5.hpp>

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged spin*spin correlation
 */
class pair_correlator
{
public:
   pair_correlator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, const PairCorrelatorParameters& params, WALKER_TYPES wlk, int NMO_, int nave_ = 1)
      : mpi{mpi},
        num_correlators{0},
        walker_type{wlk},
        NMO{NMO_}
  {
    app_log(1,"  --  Pair Correlator (PairCorr) estimator. -- ");
    const std::string& filename = params.filename;

    if (params.pair_type.empty())
      APP_ABORT("pair_correlator: No pair_type specified. Please specify at least one pair_type.");

    if (walker_type == CLOSED)
    {
      APP_ABORT("pair_correlator not yet implemented for CLOSED walkers. Use COLLINEAR or NONCOLLINEAR Walkers instead.");
    }

    {
      h5::file input(filename, 'r');
      app_log(1, "reading pair correlators from: {}", filename);
      h5::group group = h5::group{input}.open_group("PairCorrelator/orbital_map");

      for (const auto& correlator_name : params.pair_type) {
        memory::host_array<int,2> current_pair_map;
        h5::read(group, correlator_name, current_pair_map);

        pair_map.push_back(current_pair_map);
      }
    }
    mpi->node_comm.barrier();

    int x = walker_type == NONCOLLINEAR ? 2 : 1;
    int dm_size = x*NMO*(x*NMO-1)/2;

    pair_corr_average.resize(nave_, num_correlators, num_correlators, dm_size);
    nda::tensor::set(0, pair_corr_average);
  }

  /*******   Interface for sum over references, e.g. NOMSD ********/
  void accumulate(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryArrayOfRank<4> auto&& G_host, nda::MemoryVector auto&& Xw, [[maybe_unused]] bool impsamp)
  {
    // assumes G[nwalk][spin][M][M]
    using nda::range;
    ncalls++;

    // not implemented on GPU yet
    auto Xwhost = nda::to_host(Xw);

    // no parallelization over ncores for now, fix if needed
    for (int iw = 0; iw < G_host.shape(0); iw++)
    {
      for (int alpha = 0; alpha < num_correlators; alpha++)
      {
        const auto &pair_map_i = this->pair_map.at(alpha);
        for (int beta = 0; beta < num_correlators; beta++)
        {
          const auto &pair_map_j = this->pair_map.at(beta);

          auto avg = pair_corr_average(iav, alpha, beta, range::all);
          int idx{};

          if (walker_type == COLLINEAR)
          {
            auto Gup = G_host(iw, 0, range::all, range::all);
            auto Gdn = G_host(iw, 1, range::all, range::all);
            for (int i = 0; i < NMO; i++)
            {
              int ibar = pair_map_i(i,0);
              if (ibar < 0)  // negative index means no valid pair!
                continue;
              for (int j = i + 1; j < NMO; j++, idx++)
              {
                int jbar = pair_map_j(j,0);
                if (jbar < 0)  // negative index means no valid pair!
                  continue;

                ComplexType c{};
                c += Gup(i,j) * Gdn(ibar,jbar);
                c += Gup(i,jbar) * Gdn(ibar,j);
                c += Gdn(i,jbar) * Gup(ibar,j);
                c += Gdn(i,j) * Gup(ibar,jbar);
                avg(idx) += 0.5 * Xwhost(iw) * c;
              }
            }
          } else if (walker_type == NONCOLLINEAR) {
            auto G_ = G_host(iw, 0, range::all, range::all);
            for (int i = 0; i < NMO; i++)
            {
              int ibar = pair_map_i(i,0);
              if (ibar < 0)  // negative index means no valid pair!
                continue;
              for (int j = 0; j < NMO; j++, idx++)
              {
                int jbar = pair_map_j(j,0);
                if (jbar < 0)  // negative index means no valid pair!
                  continue;

                ComplexType c{};
                c += (G_(i,j) * G_(NMO + ibar,NMO + jbar) - G_(i,NMO + jbar) * G_(NMO + ibar,j));
                c += (G_(i,jbar) * G_(NMO + ibar,NMO + j) - G_(i,NMO + j) * G_(NMO + ibar,jbar));
                c += (G_(NMO + i,NMO + jbar) * G_(ibar,j) - G_(NMO + i,j) * G_(ibar,NMO + jbar));
                c += (G_(NMO + i,NMO + j) * G_(ibar,jbar) - G_(NMO + i,jbar) * G_(ibar,NMO + j));

                avg(idx) += 0.5 * Xwhost(iw) * c;
              }
            }
          }
        } // loop over beta
      } // loop over alpha
    } // loop over walkers 
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_second ");
  }
  
  void print(int iblock, h5::group *group, nda::Vector auto&& Wsum)
  {
    nda::tensor::scale(ComplexType(1.0 / double(ncalls)), pair_corr_average);
    mpi->reduce(pair_corr_average, std::plus<>(), 0);
    if (mpi->comm.root()) {
      assert(group);
      h5::group parent = ( group->has_key("PairCorrelator") ? 
                           group->open_group("PairCorrelator") : 
                           group->create_group("PairCorrelator") );
      for (int i = 0; i < pair_corr_average.shape(0); ++i) {
        std::string avg_name = std::format("Average_{}", i);
        h5::group obs_group = ( parent.has_key(avg_name) ? 
                                parent.open_group(avg_name) :
                                parent.create_group(avg_name) ); 
        
        std::string padded_iblock = std::format("{:09}",iblock);
        h5::write(obs_group, "P" + padded_iblock, nda::flatten(pair_corr_average(i, nda::ellipsis{})));
        h5::write(obs_group, "denominator_" + padded_iblock, Wsum(i));
      }
    }
    nda::tensor::set(0, pair_corr_average);
    ncalls = 0;
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;
  int ncalls = 0;

  int num_correlators{}; // number of correlators which are actually used! (vs number of correlators defined in the HDF5 input)

  WALKER_TYPES walker_type{};

  int NMO{};

  std::vector<memory::host_array<int,2>> pair_map;

  // (iave, α, β, npol*NMO*(npol*NMO-1)/2)
  memory::host_array<ComplexType,4> pair_corr_average;
};

} // namespace afqmc
} // namespace sfqmc
