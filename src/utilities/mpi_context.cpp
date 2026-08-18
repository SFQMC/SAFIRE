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
////////////////////////////////////////////////////////////////////////////////


#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

namespace mpi3 = boost::mpi3;
namespace sfqmc {
namespace utils {

mpi_context_t<mpi3::communicator> make_mpi_context() {
  using mpi3::communicator;
  using mpi3::shared_communicator;
  utils::check(boost::mpi3::initialized(), "Uninitialized MPI3.");
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  auto internode_comm = world.split(node_comm.rank(), world.rank());
  return mpi_context_t<communicator,shared_communicator>(std::move(world),std::move(node_comm),std::move(internode_comm));
};

mpi_context_t<mpi3::communicator> make_mpi_context(mpi3::communicator& comm) {
  using mpi3::communicator;
  using mpi3::shared_communicator;
  utils::check(boost::mpi3::initialized(), "Uninitialized MPI3.");
  mpi3::communicator comm_copy(comm); 
  auto node_comm = comm.split_shared();
  auto internode_comm = comm.split(node_comm.rank(), comm.rank());
  return mpi_context_t<communicator,shared_communicator>(std::move(comm_copy),std::move(node_comm),std::move(internode_comm));
};

}
}
