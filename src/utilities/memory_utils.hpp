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
#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace sfqmc::utils {
inline std::string format_bytes(double bytes) {
  constexpr std::array units = {
    "B  ", "KiB", "MiB", "GiB", "TiB",
  };

  int unit = 0;
  for(; unit + 1 < units.size() && bytes > 1024; unit++) {
    bytes /= 1024;
  }

  return std::format("{:6.1f} {}", bytes, units[unit]);
}

inline void resize_nda_static_allocator(double x = 0.05)
{
  utils::check(x >= 0.0, "Error in resize_nda_static_allocator: x<0.0.");
  auto static_alloc_host = memory::detail::static_allocator_t<HOST_MEMORY>{};
  {
    size_t oldsz = static_alloc_host.get_primary().size(); 
    size_t maxm  = static_alloc_host.get_primary().maximum_memory();
    if(maxm > oldsz) {
      size_t newsz = static_cast<size_t>(maxm * (1.0+x)); 
      app_log(2, "Increasing host buffer size: {} -> {}",
        format_bytes(oldsz), format_bytes(newsz));
      static_alloc_host.get_primary().resize(newsz);
    }
  } 
#if defined(ENABLE_DEVICE)
  auto static_alloc_dev = memory::detail::static_allocator_t<DEVICE_MEMORY>{};
  {
    size_t oldsz = static_alloc_dev.get_primary().size();
    size_t maxm  = static_alloc_dev.get_primary().maximum_memory();
    if(maxm > oldsz) {
      size_t newsz = size_t( double(maxm) * (1.0+x) ); 
      app_log(2, "Increasing device buffer size: {} -> {}",
        format_bytes(oldsz), format_bytes(newsz));
      static_alloc_dev.get_primary().resize(newsz);
    }
  }
#endif
}

/*
 * Releases the buffer pools of the static allocators.
 *
 * Call this from an explicit shutdown path, once every buffered_array is gone. The allocators
 * outlive every other object with static storage duration and never release their pools on their
 * own: a static destructor runs after the device runtime has torn down its context, so freeing
 * device memory from there fails and aborts the process.
 */
inline void release_nda_static_allocator()
{
  auto release = [](auto &alloc, std::string_view space) {
    if(alloc.size() == 0) { return; }
    if(!alloc.idle()) {
      app_warning("The {} buffer was still in use when the simulation is already over. "
        "This is probably a bug.", space);
      return;
    }
    app_log(2, "Releasing {} buffer: {}", space, format_bytes(alloc.size()));
    alloc.release_pool();
  };
  release(memory::detail::static_allocator_t<HOST_MEMORY>{}.get_primary(), "host");
#if defined(ENABLE_DEVICE)
  release(memory::detail::static_allocator_t<DEVICE_MEMORY>{}.get_primary(), "device");
#endif
}

} // namespace sfqmc::utils
