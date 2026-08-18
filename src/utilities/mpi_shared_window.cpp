#include "mpi_shared_window.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "mpi3/shared_communicator.hpp"
#include <algorithm>

namespace sfqmc::utils {

namespace {
  void shared_window_registry_deleter(shared_window* window) {
    window->unused = true;
  }
}

void *shared_window::base_ptr(int rank) {
  void *base{};
  boost::mpi3::size_t size{};
  int disp_unit{};
  int err = MPI_Win_shared_query(window, rank, &size, &disp_unit, &base);
  check(err == MPI_SUCCESS, "MPI_Win_shared_query failed with error code {}", err);
  return base;
}

shared_window_registry::~shared_window_registry() {
  for(const auto& w : windows_) {
    if(!w.unused) {
      sfqmc::app_warning("shared_window of was not deconstructed before its shared_window_registry. This is UB!");
    }
  }
}

std::shared_ptr<shared_window> shared_window_registry::open_window(long size) {
  void *base{};
  MPI_Win window{};
  int err = MPI_Win_allocate_shared(size, 1, MPI_INFO_NULL, comm_.get(), &base,
                                    &window);
  check(err == MPI_SUCCESS,
        "MPI_Win_allocate_shared of size {} failed with error code {}", size, err);

  auto free_it =
      std::find_if(windows_.begin(), windows_.end(),
                   [](const shared_window &w) { return w.window == MPI_WIN_NULL; });

  shared_window *result;
  if (free_it != windows_.end()) {
    *free_it = shared_window{window,  false};
    result = std::addressof(*free_it);
  } else {
    result = std::addressof(windows_.emplace_back(window, false));
  }

  return std::shared_ptr<shared_window>(result, shared_window_registry_deleter);
}

void shared_window_registry::collective_free_unused() {
  for(auto& w : windows_) {
    bool unused = comm_.all_reduce_value(w.unused, std::logical_and<>{});
    if(unused && w.window != MPI_WIN_NULL) {
      // MPI_Win_free resets w.window to MPI_WIN_NULL, marking the slot as free.
      int err = MPI_Win_free(&w.window);
      check(err == MPI_SUCCESS, "MPI_Win_free failed with error code {}", err);
    }
  }
}

bool shared_window_registry::isempty() {
  return std::all_of(windows_.begin(), windows_.end(), [](const shared_window& w) { return w.window == MPI_WIN_NULL; });
}

}
