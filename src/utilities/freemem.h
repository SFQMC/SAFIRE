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

#include <cstdlib>
#include <string>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif
namespace sfqmc {
namespace utils {

std::size_t freemem();
std::size_t freemem_device();
void memory_report(int io_lvl = 3, std::string message = {});

}
}
