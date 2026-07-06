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
#include "nda/layout/range.hpp"
#include <mpi3/communicator.hpp>
#include <utilities/mpi_context.h>
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
class spinspinobs : public AFQMCInfo
{
public:
  spinspinobs(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1)
      : AFQMCInfo{info},
        mpi{mpi},
        walker_type{wlk}
  {
    app_log(1,"  --  Adding Spin*Spin (spinspinobs) estimator. -- ");

    using std::fill_n;

    // currently 2 * upper triangular since <SiSj> = <SjSi>
    int dm_size = NMO*(NMO+1)/2; 

    dm_average.resize(nave_, 2, dm_size);
    nda::tensor::set(0, dm_average);
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  auto accumulate(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryArrayOfRank<4> auto&& G_host, nda::MemoryVector auto&& Xw, [[maybe_unused]] bool impsamp)
  {
    ncalls++;

    // assumes G[nwalk][spin][M][M]
    // no parallelization over ncores for now, fix if needed

    auto avg_xy = dm_average(iav, 0, nda::range::all);
    auto avg_z = dm_average(iav, 1, nda::range::all);
    for (int iw = 0; iw < G_host.shape(0); iw++)
    {
      int idx{};

      if (walker_type == CLOSED)
      {
        // TODO: optimize this
        auto Gu_ = G_host(iw, 0, nda::ellipsis{});
        for (int ref = 0; ref < NMO; ref++ ){
          for (int j = ref; j < NMO; j++, idx++)
          {
            if(j==ref){
              avg_z(idx) += Xw[iw]/4.0 * (Gu_(ref,ref)*(1.0-Gu_(j,j))
                                    +Gu_(ref,ref)*(1.0-Gu_(j,j)));
              avg_xy(idx) += Xw[iw]/2.0 * (Gu_(ref,j)*(1.0-Gu_(j,ref))
                                    +Gu_(ref,j)*(1.0-Gu_(j,ref)));
            }
            else{
              avg_z(idx) += Xw[iw]/4.0 * (Gu_(ref,ref)*(Gu_(j,j)-Gu_(j,j))
                                +Gu_(ref,ref)*(Gu_(j,j)-Gu_(j,j))
                                -Gu_(ref,j)*Gu_(j,ref) // only U2 term
                                -Gu_(ref,j)*Gu_(j,ref));

              avg_xy(idx) += Xw[iw]/2.0 * (Gu_(ref,j)*(-Gu_(j,ref))
                                     +Gu_(ref,j)*(-Gu_(j,ref)));
            }
          }
        }
      }
      else if (walker_type == COLLINEAR or walker_type == COLLINEAR_FT)
      {
        auto Gu_ = G_host(iw,0,nda::ellipsis{});
        auto Gd_ = G_host(iw,1,nda::ellipsis{});

        for (int ref = 0; ref < NMO; ref++ ){
          for (int j = ref; j < NMO; j++, idx++)
          {
            if(j==ref){
              avg_z(idx) += Xw[iw]/4.0 * (Gu_(ref,ref)*(1.0-Gd_(j,j))
                                    +Gd_(ref,ref)*(1.0-Gu_(j,j)));
              avg_xy(idx) += Xw[iw]/2.0 * (Gu_(ref,j)*(1.0-Gd_(j,ref))
                                    +Gd_(ref,j)*(1.0-Gu_(j,ref)));
            }
            else{
              avg_z(idx) += Xw[iw]/4.0 * (Gu_(ref,ref)*(Gu_(j,j)-Gd_(j,j))
                                +Gd_(ref,ref)*(Gd_(j,j)-Gu_(j,j))
                                -Gu_(ref,j)*Gu_(j,ref) // only U2 term
                                -Gd_(ref,j)*Gd_(j,ref));

              avg_xy(idx) += Xw[iw]/2.0 * (Gu_(ref,j)*(-Gd_(j,ref))
                                     +Gd_(ref,j)*(-Gu_(j,ref)));
            }
          }
        }
      }
      else if (walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT)
      {
        auto G_ = G_host(iw,0,nda::ellipsis{});
        for (int ref = 0; ref < NMO; ref++){
          for (int j = ref; j < NMO; j++, idx++)
          {
            if(j==ref) {
              avg_z(idx) += Xw[iw]/4.0 * (G_(ref,ref)*(1.0-G_(j+NMO,j+NMO))
                                    +G_(ref+NMO,ref+NMO)*(1.0-G_(j,j))
                                    -G_(ref,j+NMO)*(-G_(j+NMO,ref))
                                    -G_(ref+NMO,j)*(-G_(j,ref+NMO)));

              avg_xy(idx) += Xw[iw]/2.0 * (G_(ref,j)*(1.0-G_(j+NMO,ref+NMO))
                                    +G_(ref+NMO,j+NMO)*(1.0-G_(j,ref))
                                    +G_(j+NMO,j)*G_(ref,ref+NMO)
                                    +G_(j,j+NMO)*G_(ref+NMO,ref)
                                    );
            }
            else {
              avg_z(idx) += Xw[iw]/4.0 * (G_(ref,ref)*(G_(j,j)-G_(j+NMO,j+NMO))
                                +G_(ref+NMO,ref+NMO)*(G_(j+NMO,j+NMO)-G_(j,j))
                                -G_(ref,j)*G_(j,ref) // only U2 term
                                -G_(ref+NMO,j+NMO)*G_(j+NMO,ref+NMO)
                                -G_(ref,j+NMO)*(-G_(j+NMO,ref))
                                -G_(ref+NMO,j)*(-G_(j,ref+NMO)));

              avg_xy(idx) += Xw[iw]/2.0 * (G_(ref,j)*(-G_(j+NMO,ref+NMO))
                                    +G_(ref+NMO,j+NMO)*-G_(j,ref)
                                    +G_(j+NMO,j)*G_(ref,ref+NMO)
                                    +G_(j,j+NMO)*G_(ref+NMO,ref)
                                    );
            }
          }
        }
      }
      else
      {
        APP_ABORT("walker_type {} not yet implemented", walkerTypeToString(walker_type));
      }

      assert(idx == dm_average.shape(2));
    }
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]]Args&&... args)
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
    nda::tensor::scale(1.0 / double(ncalls), dm_average);
    mpi->reduce(dm_average, std::plus<>(), 0);
    if(mpi->comm.root()) {
      assert(group);
      h5::group parent_group = ( group->has_key("SpinSpin") ? 
                                 group->open_group("SpinSpin") : 
                                 group->create_group("SpinSpin") );
      std::string padded_iblock = std::format("{:09}", iblock);
      for (int i = 0; i < dm_average.shape(0); ++i)
      {
        std::string avg_name = std::format("Average_{}", i);
        h5::group obs_group = ( parent_group.has_key(avg_name) ? 
                                parent_group.open_group(avg_name) :
                                parent_group.create_group(avg_name) );
        h5::write(obs_group, "spinspin_" + padded_iblock, nda::flatten(dm_average(i, nda::ellipsis{})));
        h5::write(obs_group, "denominator_" + padded_iblock,Wsum[i]);
      }
    }
    nda::tensor::set(0, dm_average());
    ncalls = 0;
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;
  int ncalls = 0;

  WALKER_TYPES walker_type;

  // dm_average (nave, [XY, Z], x*NMO*(x*NMO-1)/2 ), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  memory::host_array<ComplexType, 3> dm_average;
};

} // namespace afqmc
} // namespace sfqmc

