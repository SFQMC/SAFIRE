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

#include "AFQMC/config.h"
#include <configuration.hpp>
#include <vector>
#include <memory>
#include <string>

#include <nda/nda.hpp>
#include <nda/h5.hpp>

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "utilities/check.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged diagonal of the 2 RDM 
 */
template<MEMORY_SPACE MEM>
class diagonal2rdm : public AFQMCInfo
{
public:
  diagonal2rdm(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1)
      : AFQMCInfo{info},
        mpi{mpi},
        walker_type{wlk}
  {
    app_log(1,"  --  Adding Diagonal 2RDM (Diag2RDM) estimator. -- ");
    
    // spin blocks per walker type:
    //   CLOSED       -> (aaaa), (aabb)                 : 2
    //   COLLINEAR    -> (aaaa), (aabb), (bbbb)         : 3
    //   NONCOLLINEAR -> (aaaa) over 2*NMO x 2*NMO       : 1
    int nspin = (walker_type == COLLINEAR or walker_type == COLLINEAR_FT ? 3 : (walker_type == CLOSED ? 2 : 1));
    int M     = (walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT ? 2 * NMO : NMO);
    dm_average.resize(nave_, nspin, M, M);
    nda::tensor::set(0, dm_average());
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  auto accumulate(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryArrayOfRank<4> auto&& G_host, nda::MemoryVector auto&& Xw, [[maybe_unused]] bool impsamp)
  {
    using nda::ellipsis;
    using nda::range;
    // assumes G[nwalk][spin][M][M]
    ncalls++;

    memory::buffered_array<MEM, ComplexType,4> XwG(G.shape());
    nda::tensor::contract(memory::to_memory_space<MEM>(Xw), "w", G, "wsij", XwG, "wsij");
    
    // (aaaa), (bbbb)
    for(int spin = 0; spin < G.shape(1); spin++) {
      nda::tensor::contract(1, XwG(range::all, spin, ellipsis{}), "wii", G(range::all, spin, ellipsis{}), "wjj", 1, dm_average(iav, 2*spin, ellipsis{}), "ij");
      nda::tensor::contract(-1, XwG(range::all, spin, ellipsis{}), "wij", G(range::all, spin, ellipsis{}), "wji", 1, dm_average(iav, 2*spin, ellipsis{}), "ij");
    }
    // (aabb) does not exist for noncollinear
    if(walker_type == CLOSED) {
      nda::tensor::contract(1, XwG(range::all, 0, ellipsis{}), "wii", G(range::all, 0, ellipsis{}), "wjj", 1, dm_average(iav, 1, ellipsis{}), "ij");
    } else if(walker_type == COLLINEAR) {
      nda::tensor::contract(1, XwG(range::all, 0, ellipsis{}), "wii", G(range::all, 1, ellipsis{}), "wjj", 1, dm_average(iav, 1, ellipsis{}), "ij");
    }
      
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

  auto print(int iblock, h5::group *group, const nda::Vector auto& Wsum)
  {
    nda::tensor::scale(1.0 / ncalls, dm_average);
    mpi->reduce(dm_average, std::plus<>(), 0);
    
    if(mpi->comm.root())
    {
      assert(group);
      auto compressed_average = nda::to_host(compress_dm_average(dm_average));
      
      h5::group parent = ( group->has_key("DiagTwoRDM") ? 
                           group->open_group("DiagTwoRDM") : 
                           group->create_group("DiagTwoRDM") );
      for (int i = 0; i < dm_average.shape(0); ++i)
      {
        std::string avg_name = std::format("Average_{}", i);
        h5::group obs_group = ( parent.has_key(avg_name) ? 
                                parent.open_group(avg_name) :
                                parent.create_group(avg_name) );
        std::string padded_iblock = std::format("{:09}", iblock);
        h5::write(obs_group, "diag_two_rdm_" + padded_iblock, compressed_average(i, nda::range::all));
        h5::write(obs_group, "denominator_" + padded_iblock, Wsum[i]);
      }
    }
    nda::tensor::set(0, dm_average);
    ncalls = 0;
  }


private:
  // fold down i <-> j symmetry
  auto compress_dm_average(const nda::MemoryArrayOfRank<4> auto& full_average) {
    auto host_full_average = nda::to_host(full_average);

    int out_size = NMO * (2 * NMO - 1);
    if (walker_type == CLOSED) out_size -= NMO * (NMO - 1) / 2;

    memory::host_array<ComplexType, 2> result(full_average.shape(0), out_size);
    for(int iav = 0; iav < result.shape(0); iav++) {
      int idx{};
      if (walker_type == CLOSED || walker_type == COLLINEAR || walker_type == COLLINEAR_FT) {
        for (int i = 0; i < NMO; i++)
        {
          for (int j = i + 1; j < NMO; j++, idx++) {
            result(iav, idx) = (host_full_average(iav, 0, i, j) + host_full_average(iav, 0, j, i)) / 2;
          }
          for (int j = 0; j < NMO; j++, idx++) {
            result(iav, idx) = host_full_average(iav, 1, i, j); 
          }
        }      
        if (walker_type == COLLINEAR || walker_type == COLLINEAR_FT) {
          for (int i = 0; i < NMO; i++) { 
            for (int j = i + 1; j < NMO; j++, idx++) {
              result(iav, idx) = (host_full_average(iav, 2, i, j) + host_full_average(iav, 2, j, i)) / 2;
            }
          }
        }
      } else if (walker_type == NONCOLLINEAR || walker_type == NONCOLLINEAR_FT) {
        for (int i = 0; i < 2 * NMO; i++) {
          for (int j = i + 1; j < 2 * NMO; j++, idx++) {
            result(iav, idx) = (host_full_average(iav, 0, i, j) + host_full_average(iav, 0, j, i)) / 2;
          }
        }
      } else {
        utils::check(false, "walker_type {} not implemented", walkerTypeToString(walker_type));
      }    
      assert(idx == result.shape(1));
    }

    return result;
  }
  
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  int ncalls = 0;
  WALKER_TYPES walker_type{};
  
  // dm_average (nave, spin, x*NMO, x*NMO)
  // x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  // spin = [(aaaa), (aabb), (bbbb)] for COLLINEAR
  // spin = [(aaaa), (aabb)] for CLOSED
  // spin = [(aaaa)] for NONCOLLINEAR
  memory::array<MEM, ComplexType,4> dm_average;
};

} // namespace afqmc
} // namespace sfqmc
