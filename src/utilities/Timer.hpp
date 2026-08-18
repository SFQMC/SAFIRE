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

#include <chrono>
#include <span>
#include <string>
#include "IO/banner.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace sfqmc::utils {

/**
 * Accumulates the wall-clock time spent in a region of code, over any number of visits.
 *
 * A region is timed through the handle returned by start(), which is stopped at the end of the
 * region.
 */
struct Timer {
  std::string name;
  int ncalls{};
  double total_time{};

  /**
   * Times the region between its construction and stop(); the destructor stops a handle that
   * still runs, so an early return or a thrown exception is still accounted for.
   *
   * Neither copyable nor movable, so a handle cannot outlive the scope that started it.
   */
  class Handle {
    std::chrono::steady_clock::time_point start_;
    Timer* timer_;

  public:
    explicit Handle(Timer& t) : start_(std::chrono::steady_clock::now()), timer_(&t) {}
    Handle(Handle const&) = delete;
    Handle(Handle&&) = delete;
    Handle& operator=(Handle const&) = delete;
    Handle& operator=(Handle&&) = delete;
    ~Handle() {
      if(timer_ != nullptr) {
        stop();
      }
    }

    void stop() {
      check(timer_ != nullptr, "Timer::Handle::stop() called on a handle that is not timing");
      timer_->total_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
      timer_->ncalls++;
      timer_ = nullptr;
    }
  };

  [[nodiscard]] Handle start() { return Handle{*this}; }

  void reset() {
    total_time = 0.0;
    ncalls = 0;
  }
};

/// Prints elapsed, per-call and call-count for each timer as a table.
inline void print_timers(std::span<Timer* const> timers) {
  app_log(1, hrule());
  app_log(1, "{:>30}:{:>16}{:>16}{:>9}", "Timer Name", "Elapsed (s)", "Averaged (s)", "# calls");
  app_log(1, hrule());
  for(auto* t : timers) {
    app_log(1, "{:>30}:{:16.8g}{:16.8g}{:9d}", t->name, t->total_time,
            (t->ncalls > 0 ? t->total_time / double(t->ncalls) : 0.0), t->ncalls);
  }
  app_log(1, hrule());
  app_log_flush();
}

/// Returns the time in seconds taken by a single call to f.
inline double function_timer(auto&& f) {
  auto start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}
