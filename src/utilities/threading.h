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

#pragma once

namespace sfqmc::utils {

  /**
   * @brief Initialize the threading environment for the current process.
   *
   * @details
   * Because SAFIRE does not use shared-memory threading internally, each rank
   * is normally configured with a single thread (see `init_threading`).
   * However, a user may legitimately enable threading in a BLAS or
   * tensor-contraction backend (e.g. to exploit multi-threaded MKL on a node
   * with few MPI ranks).
   * Sets `OMP_NUM_THREADS` to 1 unless the user has
   * already placed those variables in the environment, and calls
   * `omp_set_num_threads(1)` when OpenMP is compiled in.  Emits a warning if
   * either variable is user-set to a value greater than 1.
   *
   * Must be called once during application startup, after loggers are
   * initialized and before any threading-sensitive library calls.  See
   * `check_thread_oversubscription` for the complementary MPI-aware check that
   * should be called once MPI context is available.
   *
   * The check is a best-effort safeguard; it is the users responsibility to 
   * ensure that the environment variables are set appropriately for their use case.  
   */
  void init_threading();

} // namespace sfqmc::utils
