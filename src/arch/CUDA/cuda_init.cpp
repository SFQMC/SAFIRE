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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>

#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

#include "IO/app_loggers.h"
#include "cuda_runtime.h" 
#include "curand.h"

#include "arch/CUDA/cuda_init.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/check.hpp"
#include "numerics/device_kernels/cuda/probe.hpp"


namespace sfqmc {
namespace cuda
{

namespace {
bool mpi_is_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) // Open MPI family
  return MPIX_Query_cuda_support() == 1;
#elif defined(MPICH_NUMVERSION) && MPICH_NUMVERSION >= 40000000 // MPICH family
  return MPIX_Query_cuda_support() == 1;
#else
  return false; // unknown/old: assume not
#endif
}
}

void cuda_check(cudaError_t success, std::string message)
{
  if (success != cudaSuccess) {
   app_error(" Cuda runtime error: {}",std::to_string(success));
   if(message != "")
     app_error(" message: {}",message);
   app_error(" cudaGetErrorName: {}",std::string(cudaGetErrorName(success)));
   app_error(" cudaGetErrorString: {}",std::string(cudaGetErrorString(success)));
   APP_ABORT(" Cuda runtime error"); 
  }
}

void curand_check(curandStatus_t success, std::string message)
{
  if (success != CURAND_STATUS_SUCCESS) {
   app_error(" Curand runtime error: {}",std::to_string(success));
   if(message != "")
     app_error(" message: {}",message);
   APP_ABORT(" Curand runtime error");
  }
}

void init() 
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  app_log(1, "\nRunning on node with {} GPUs.", num_devices);
  cudaDeviceProp dev;
  cuda_check(cudaGetDeviceProperties(&dev, 0), "cudaGetDeviceProperties");
  app_log(1, "CUDA compute capability: {}.{}", dev.major, dev.minor);
  app_log(1, "Device Name: {} ", dev.name);

  cuda_check(cudaSetDevice(node.rank()%num_devices), "cudaSetDevice()");
  int devn = 0;
  cuda_check(cudaGetDevice(&devn), "cudaGetDevice()");
  app_debug(3,"MPI world rank: {}, node rank: {}, cuda device number: {}",
	    world.rank(),node.rank(),devn);

  // explicit synchronization after every call is not useful
  nda::tensor::device::set_synchronization(false);

  check_probe_kernel();
  
  if(world.size() > 1 && !mpi_is_cuda_aware()) {
    utils::check(false, "Attempted to run GPU build on multiple ranks, but built with CUDA-unaware MPI!");
  }    
}

void check_device_configuration()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  cudaDeviceProp dev;
  cuda_check(cudaGetDeviceProperties(&dev, 0), "cudaGetDeviceProperties");
  if (dev.major <= 6 and world.root())
  {
    app_warning("Warning CUDA major compute capability < 6.0");
  }
  if (num_devices > node.size() and world.root())
  {
    app_warning("WARNING: Unused devices !!!!!!!!!!!!!! ");
    app_warning("         # tasks: {} ", node.size());
    app_warning("         # number of devices: {} ", num_devices);
  }
  utils::check(num_devices >= node.size(), "Error: # GPU < # tasks in node. \n # GPU: {} \n # MPI tasks: {}",num_devices,node.size());
}


} // cuda
} // sfqmc
