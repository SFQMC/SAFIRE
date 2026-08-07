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

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <mpi.h>
#include <source_location>
#include <string_view>
#include <concepts>
#include <cstdint>
#if defined(ENABLE_CPPTRACE)     
#include <cpptrace/cpptrace.hpp>
#endif
#if defined(ENABLE_SPDLOG)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#else
#include <format>
#endif

namespace sfqmc {

extern bool __app_stacktrace__;

class IostreamLogBackend {
public:
  void error(std::string_view fmt) const{
    std::cerr << fmt << "\n";
  }
  template<typename... Args>
  void error(std::string_view fmt, Args&&... args) const{
    std::cerr << std::vformat(fmt, std::make_format_args(args...)) << "\n";
  }
  void flush() const {
    std::cerr.flush();
  }
};
static constexpr IostreamLogBackend _iostream_log_backend;

inline auto get_abort_log_backend() {
#if defined(ENABLE_SPDLOG)
  auto l = spdlog::get("err_console");
  if(!l) {
    return spdlog::stdout_color_mt("err_console");
  }
  return l;
#else
  return &_iostream_log_backend;
#endif
}

/// Exception thrown by APP_ABORT when the program is running in test mode.
struct AppAbortException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

template<typename... Args>
[[noreturn]] void common_abort(std::optional<std::source_location> loc, Args&&... args) {
  auto log_backend = get_abort_log_backend();
  
  log_backend->error("**********************************************");
  log_backend->error("        APPLICATION ABORT: Fatal Error.");
  log_backend->error("**********************************************");
  if(loc) {
    log_backend->error("file:     {}:{}:{}",loc->file_name(), loc->line(), loc->column());
    log_backend->error("function: {}",loc->function_name());
    log_backend->error("**********************************************");
  }
  log_backend->error(std::forward<Args>(args)...);
  log_backend->error("**********************************************");

  // how to make cpptrace interact with spdlog???
  if(__app_stacktrace__) {
#if defined(ENABLE_CPPTRACE)
    cpptrace::generate_trace().print();
#else
    log_backend->error("For stack trace, compile with -DENABLE_CPPTRACE=ON.");
#endif
  } else {
    log_backend->error("For stack trace, rerun with --verbosity 2 or higher.");
  }
  log_backend->error("**********************************************");
  log_backend->flush();
  // Abort
  throw AppAbortException("APP_ABORT triggered (see error log for details)");
}



/// Format string that also captures the caller's source location, so that
/// APP_ABORT reports where it was called from without the caller passing a
/// std::source_location explicitly. A trailing defaulted parameter cannot be
/// used for this because it would follow a deduced parameter pack.
struct abort_format {
  std::string_view fmt;
  std::source_location loc;

  template<class T>
    requires std::convertible_to<T const&, std::string_view>
  abort_format(T const& f, const std::source_location& l = std::source_location::current())
      : fmt(f), loc(l) {}
};

template<class... Args>
[[noreturn]] void APP_ABORT(abort_format f, Args&&... args) {
  common_abort(f.loc, f.fmt, std::forward<Args>(args)...);
}

template<class... Args>
[[noreturn]] void APP_ABORT_with_source(const std::source_location& loc = std::source_location::current(), Args&&... args) {
  common_abort(loc, std::forward<Args>(args)...);
}

} // sfqmc
