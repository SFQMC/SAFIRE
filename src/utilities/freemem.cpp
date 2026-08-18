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

#include "freemem.h"

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif
#include "IO/app_loggers.h"
#if defined(ENABLE_CUDA)
#include "cuda_runtime.h"
#endif

namespace sfqmc {
namespace utils {

std::size_t freemem()
{
#ifdef __linux__
  struct sysinfo si;
  sysinfo(&si);
  si.freeram += si.bufferram;
  return si.freeram >> 20;
#else
  return 0;
#endif
}

std::size_t freemem_device()
{
#if defined(ENABLE_CUDA)
  std::size_t free_, tot_;
  cudaMemGetInfo(&free_, &tot_);
  return std::size_t( double(free_) / 1024.0 / 1024.0 );  
#elif defined(ENABLE_HIP)
  std::size_t free_, tot_;
  hipMemGetInfo(&free_, &tot_);
  return std::size_t( double(free_) / 1024.0 / 1024.0 );  
#else
  return 0;
#endif
}

void memory_report(int io_lvl, std::string message)
{
  if(message.size() > 0) app_log(io_lvl,"  memory report: " + message); 
  app_log(io_lvl, "  --> CPU Memory Available: {} \n ", freemem());
#if defined(ENABLE_CUDA)
  std::size_t free_, tot_;
  cudaMemGetInfo(&free_, &tot_);
  app_log(io_lvl, "  --> GPU Memory Available: {},  Total in MB: {} ",
                  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 );
#elif defined(ENABLE_HIP)
  std::size_t free_, tot_;
  hipMemGetInfo(&free_, &tot_);
  app_log(io_lvl, "  --> GPU Memory Available: {},  Total in MB: {} ",
                  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 );
#endif
  app_log_flush();
}

}
}

