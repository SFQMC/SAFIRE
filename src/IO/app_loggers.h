/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include "config.h"
#if defined(ENABLE_SPDLOG)
#include "spdlog/spdlog.h"
#else
#include <format>
#endif
#include "IO/fmt_extensions.hpp"
#include "IO/AppAbort.hpp"

namespace sfqmc {

extern int __app_debug_level__; 
extern int __app_output_level__; 

// currently using 2 separate loggers
// app_log: uses "std_console" with a clean format only on Global().root()
// app_warning/app_error/app_critical/app_debug: use "err_console" shared by everyone
// consider later using app_debug as a separate logger to file per mpi rank 

void setup_loggers(bool root=true, int output_level=2, int debug_level=0);
void set_output_level(bool root, int output_level);
void set_debug_level(bool root, int debug_level);
void set_stacktrace(bool stk);

template<class... Args>
void app_log(int level, const std::string_view string_format, Args&&... args)
{
  if(__app_output_level__ > 0 and level <= __app_output_level__) {
#if defined(ENABLE_SPDLOG)
    auto l = spdlog::get("std_console");
    if(l) 
      l->info(string_format,std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_log used uninitialized."); 
#else
    if constexpr (sizeof...(Args) > 0)
      std::cout<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
    else
      std::cout<<string_format <<"\n";
#endif
  }
}

template<class... Args>
void app_warning(const std::string_view string_format, Args&&... args)
{ 
  if(__app_output_level__ > 0) { 
#if defined(ENABLE_SPDLOG)
    auto l = spdlog::get("err_console");
    if(l)
      l->warn(string_format,std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_warning used uninitialized.");
#else
    if constexpr (sizeof...(Args) > 0)
      std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
    else
      std::cerr<<string_format <<"\n";
#endif
  }
}

template<class... Args>
void app_error(const std::string_view string_format, Args&&... args)
{ 
#if defined(ENABLE_SPDLOG)
  auto l = spdlog::get("err_console");
  if(l) { 
    l->error(string_format,std::forward<Args>(args)...);
    l->flush();
  } else
    APP_ABORT(" Error: app_error used uninitialized.");
#else
  if constexpr (sizeof...(Args) > 0)
    std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
  else
    std::cerr<<string_format <<"\n";
#endif
}

template<class... Args>
void app_debug(int level, const std::string_view string_format, Args&&... args)
{ 
  if(__app_debug_level__ > 0 and level <= __app_debug_level__) {
#if defined(ENABLE_SPDLOG)
    auto l = spdlog::get("err_console");
    if(l)
      l->debug(string_format,std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_debug used uninitialized.");
#else
    if constexpr (sizeof...(Args) > 0)
      std::cerr<<std::vformat(string_format,std::make_format_args(args...)) <<"\n";
    else
      std::cerr<<string_format <<"\n";
#endif
  }
}

inline void app_log_flush() 
{
  if(__app_output_level__ > 0) {
#if defined(ENABLE_SPDLOG)
    auto l = spdlog::get("std_console");
    if(l)
      l->flush();
    else
      APP_ABORT(" Error: app_warning used uninitialized.");
#else
    std::cout.flush();
#endif
  }
}

inline void app_error_flush() 
{
#if defined(ENABLE_SPDLOG)
  auto l = spdlog::get("err_console");
  if(l)
    l->flush();
  else
    APP_ABORT(" Error: app_debug used uninitialized.");
#else
    std::cerr.flush();
#endif
}

}
