#pragma once

#include <deque>
#include <mpi3/shared_communicator.hpp>
#include <memory>
#include <deque>

namespace sfqmc::utils {

struct shared_window {
  MPI_Win window{MPI_WIN_NULL};
  bool unused{false};
  void* base_ptr(int rank);
};

class shared_window_registry {
public:
  shared_window_registry(boost::mpi3::shared_communicator& comm) : comm_{comm} {};

  shared_window_registry(shared_window_registry const&) = delete;
  shared_window_registry& operator=(shared_window_registry const&) = delete;
  shared_window_registry(shared_window_registry&&) = default;
  shared_window_registry& operator=(shared_window_registry&&) = default;
  
  ~shared_window_registry();

  // open_window opens a MPI_shared window ala MPI_Win_alloc_shared.
  // This is a collective call on the shared communicator passed on construction of the registry.
  // The user is responsible for making sure the lifetime of the objects returned by this function is shorter than
  // the lifetime of the registry.
  std::shared_ptr<shared_window> open_window(long size);

  // collective_free_unused is a collective call on the shared communicator passed on construction of the registry.
  // It safely frees only those windows that are no longer in use on any node. 
  void collective_free_unused();

  // returns true if all windows allocated by this registry have been freed.
  bool isempty();
private:
  boost::mpi3::shared_communicator comm_;
  std::deque<shared_window> windows_;
};

}
