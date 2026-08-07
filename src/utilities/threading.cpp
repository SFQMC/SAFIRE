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

#include "utilities/threading.h"

#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "IO/app_loggers.h"

namespace sfqmc::utils {

/// @copydoc sfqmc::utils::init_threading
void init_threading()
{
  const char* tblis_env = std::getenv("TBLIS_NUM_THREADS");
  const char* omp_env = std::getenv("OMP_NUM_THREADS");


  bool potential_problem{};
  
  if(!omp_env) {
    ::setenv("OMP_NUM_THREADS", "1", 1);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    app_log(1, "OMP_NUM_THREADS:   1 (default)", omp_env);
  } else {
    if(std::strtol(omp_env, nullptr, 10) != 1) {
      potential_problem = true;
    }
    app_log(1, "OMP_NUM_THREADS:   {} (user-provided)", omp_env);
  }

  if(!tblis_env) {
    app_log(1, "TBLIS_NUM_THREADS: (unset)", tblis_env);
  } else {
    if(std::strtol(tblis_env, nullptr, 10) != 1) {
      potential_problem = true;
    }
    app_log(1, "TBLIS_NUM_THREADS: {} (user-provided)", tblis_env);
  }

  if(potential_problem) {
    app_warning("OMP_NUM_THREADS or TBLIS_NUM_THREADS were set != 1 by the user. Be aware of potential oversubscription in MPI runs.");
  }
}
} // namespace sfqmc::utils
