/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#include "arch.h"

#include <nda/nda.hpp>

#include "config.h"
#include "utilities/mpi_context.h"
#include "utilities/threading.h"
#include "IO/app_loggers.h"


#if defined(ENABLE_CUDA)

#include "CUDA/cuda_init.h"

namespace sfqmc {
namespace arch
{
  void check_device_configuration() { cuda::check_device_configuration(); }
}
}

#else

namespace sfqmc {
namespace arch
{
  void check_device_configuration() {};
}
}

#endif

namespace sfqmc {
namespace arch
{

void init([[maybe_unused]] bool use_gpu)
{
  sfqmc::utils::init_threading();
#if defined(ENABLE_CUDA)
  if(use_gpu) { cuda::init(); }
#endif

  // setup shared memory, memory buffers, etc, etc
}
}
}

