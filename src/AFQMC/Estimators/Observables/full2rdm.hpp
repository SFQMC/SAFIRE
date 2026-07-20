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
#include <string>
#include <h5/group.hpp>
#include <nda/layout/range.hpp>
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <mpi3/communicator.hpp>

#include "utilities/check.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged 2 RDM.
 * The resulting RDM will be [3*spin][i][k][j][l]  
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 * For collinear, the spin ordering is (a,a,a,a), (a,a,b,b), (b,b,b,b) 
 */
template<MEMORY_SPACE MEM>
class full2rdm
{
public:
  full2rdm(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi_, ptree pt, WALKER_TYPES wlk, int NMO_, int nave_ = 1)
      : mpi{mpi_},
        walker_type{wlk},
        NMO{NMO_},
        apply_rotation{false}
  {
    app_log(1,"  --  Adding 2RDM (TwoRDM) estimator. -- ");

    std::string rot_file, h5_path;
    rot_file = pt.get<std::string>("rotation", "");
    h5_path = pt.get<std::string>("path", "/");

    int dm_size{};
    if (rot_file != "")
    {
      apply_rotation = true;

      if (mpi->node_comm.root())
      {
        h5::file file(rot_file, 'r');
        h5::group grp = h5::group(file).create_group(h5_path);
        memory::array<MEM, ComplexType,2> R;
        h5::read(grp, "RotationMatrix", XRot);

        utils::check(XRot.shape(1) != NMO, "Rotation has wrong number of rows {} (expected {})", XRot.shape(1), NMO);
      }

      std::array<long int,2> dim = XRot.shape();
      mpi->node_comm.broadcast_n(dim.data(), dim.size(), 0);
      XRot.resize(dim);
      mpi->node_comm.broadcast_n(XRot.data(), XRot.size(), 0);

      dm_size = XRot.shape(0);
    }
    else
    {
      dm_size = NMO;
    }

    // (a,a,a,a), (a,a,b,b)
    int nspinblocks = 2;
    if (walker_type == COLLINEAR)
    {
      nspinblocks = 3; // (a,a,a,a), (a,a,b,b), (b,b,b,b)
    }
    else if (walker_type == NONCOLLINEAR)
      APP_ABORT(" Error: NONCOLLINEAR not yet implemented. \n\n");

    dm_average.resize(nave_, nspinblocks, dm_size, dm_size, dm_size, dm_size);
    nda::tensor::set(0,dm_average());
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  auto accumulate(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryArrayOfRank<4> auto&& G_host, nda::MemoryVector auto&& Xw, [[maybe_unused]] bool impsamp)
  {
    // assumes G[nwalk][spin][M][M]
    utils::check(G.shape(0) == Xw.shape(0), "G and Xw number of columns (walkers) mismatch: {} != {}", G.shape(0), Xw.shape(0));
    ncalls++;

    if (apply_rotation)
      acc_with_rotation(iav, G, Xw);
    else
      acc_no_rotation(iav, G, Xw);
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

  auto print(int iblock, h5::group *group, nda::Vector auto&& Wsum)
  {
    nda::tensor::scale(1.0 / double(ncalls), dm_average());
    mpi->reduce(dm_average, std::plus<>(), 0);
    if(mpi->comm.root())
    {
      assert(group);
      h5::group parent = ( group->has_key("FullTwoRDM") ? 
                           group->open_group("FullTwoRDM") : 
                           group->create_group("FullTwoRDM") );
      for (int i = 0; i < dm_average.shape(0); ++i)
      {
        std::string avg_name = std::format("Average_{}", i);
        h5::group obs_group = ( parent.has_key(avg_name) ? 
                                parent.open_group(avg_name) :
                                parent.create_group(avg_name) );
        std::string padded_iblock = std::format("{:09}", iblock);
        h5::write(obs_group, "two_rdm_" + padded_iblock, nda::to_host(nda::flatten(dm_average(i, nda::ellipsis{}))));
        h5::write(obs_group, "denominator_" + padded_iblock, Wsum[i]);
      }
    }
    nda::tensor::set(0, dm_average());
    ncalls = 0;
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO{};

  int ncalls = 0;
  bool apply_rotation{};

  memory::array<MEM, ComplexType,2> XRot;
  memory::array<MEM, ComplexType,1> Grot;

  // dm_average (nave, nspinblocks, x*NMO, x*NMO, x*NMO, xNMO), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  memory::array<MEM, ComplexType,6> dm_average;

  auto acc_no_rotation(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryVector auto&& Xw)
  {
    using nda::ellipsis;
    // (ikjl) = Gik * Gjl - (same spin) Gil Gjk
    if (walker_type == COLLINEAR)
    {
      memory::buffered_array<MEM,ComplexType,2> R(NMO * NMO, NMO * NMO);
      memory::buffered_array<MEM,ComplexType,2> Q(NMO, NMO * NMO * NMO);
    
      memory::buffered_array<MEM,ComplexType,4> XwG(G.shape());
      nda::tensor::contract(memory::to_memory_space<MEM>(Xw), "w", G, "wsij", XwG, "wsij");

      for(int ispin = 0; ispin < 2; ispin++) {
        auto XwGs = XwG(nda::range::all, ispin, ellipsis{});
        auto Gs = G(nda::range::all, ispin, ellipsis{});
        nda::tensor::contract(1, XwGs, "wik", Gs, "wjl", 1, dm_average(iav, 2*ispin, ellipsis{}), "ikjl");
        nda::tensor::contract(-1, XwGs, "wil", Gs, "wjk", 1, dm_average(iav, 2*ispin, ellipsis{}), "ikjl");
      }
      auto XwGu = XwG(nda::range::all, 0, ellipsis{});
      auto Gd = G(nda::range::all, 1, ellipsis{});
      nda::tensor::contract(1,XwGu, "wik", Gd, "wjl", 1, dm_average(iav, 1, ellipsis{}), "ikjl");
    }
    else
    {
      APP_ABORT("Error: Complete full2rdm.");
    }
  }

  template<class MatG, class CVec>
  void acc_with_rotation([[maybe_unused]] int iav, [[maybe_unused]] MatG&& G, [[maybe_unused]] CVec&& Xw)
  {
    APP_ABORT("Error: Complete full2rdm.");
  }
};

} // namespace afqmc
} // namespace sfqmc
