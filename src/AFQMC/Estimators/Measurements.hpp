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
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <format>
#include <map>
#include <memory>
#include <ranges>
#include <string>

#include "AFQMC/config.h"
#include <nda/nda.hpp>
#include "utilities/mpi_context.h"
#include "utilities/h5_utils.hpp"
#include "Accumulator.hpp"

namespace sfqmc::afqmc {


namespace detail {
template<typename A, bool = nda::Array<std::remove_cvref_t<A>>>
struct sample_type {
  using type = std::remove_cvref_t<A>;
};
template<typename A>
struct sample_type<A, true> {
  using type = nda::array<std::remove_const_t<nda::get_value_t<std::remove_cvref_t<A>>>,
                          nda::get_rank<std::remove_cvref_t<A>>>;
};

/// The type an Accumulator stores for a sample of type A: arrays lose their view-ness
/// and their allocator, everything else is taken as is.
template<typename A>
using sample_t = typename sample_type<A>::type;
}

class Measurements {
public:
  explicit Measurements(std::string prefix = {}, long binsize = 1) : prefix_{std::move(prefix)}, binsize_{binsize} {
    utils::check(binsize_ > 0, "binsize has to be positive, but it is {}", binsize_);
  }

  void measure(std::string_view name, auto const &sample) {
    using Acc = Accumulator<detail::sample_t<decltype(sample)>>;
    auto [it, inserted] = observables_.try_emplace(std::string{name}, std::make_unique<Acc>(binsize_));

    auto* acc = dynamic_cast<Acc*>(it->second.get());
    utils::check(acc != nullptr, "observable '{}' was registered with a different sample type", name);
    acc->add(sample);
  }

  void write(h5::group& out) {
    h5::group base = prefix_.empty() ? out : utils::h5_open_or_create(out, prefix_);
    for(auto& [name, obs] : observables_) {
      // an observable name is a '/'-separated path, which has to be created one component at a
      // time because H5Gcreate2 does not create intermediate groups
      h5::group group = base;
      for(auto const component : std::views::split(name, '/')) {
        group = utils::h5_open_or_create(group, std::string(component.begin(), component.end()));
      }
      obs->write(group);
    }
  }

private:
  std::string prefix_{};
  long binsize_{1};
  std::map<std::string, std::unique_ptr<AccumulatorBase>> observables_{};
};

template<MEMORY_SPACE MEM, typename ReferenceLoop>
struct MeasurementInputs {
  ReferenceLoop referenceLoop;
  memory::array_view<MEM, ComplexType const, 1> weights;
  WALKER_TYPES walkerType{};
  int NMO{};
};

class MeasurementOutput {
public:
  MeasurementOutput(utils::mpi_context_t<boost::mpi3::communicator>& mpi, Measurements& meas, std::string_view prefix, nda::MemoryVector auto const& weights)
    : prefix_{prefix.empty() ? std::string{} : std::format("{}/", prefix)}, meas_{meas} {
    // nda::sum folds with host element access, so the weights have to be on the host
    denominator_ = nda::sum(nda::to_host(weights));
    denominator_ = mpi.comm.reduce_value(denominator_);
  }

  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, std::string_view name, auto sample) {
    std::string obs_name{std::format("{}{}", prefix_, name)};
    using Arr = decltype(sample);
    if constexpr(nda::Array<Arr>) {
      memory::buffered_array<HOST_MEMORY, nda::get_value_t<Arr>, nda::get_rank<Arr>> buffer(sample);

      mpi.reduce(buffer, std::plus{});

      if(mpi.comm.root()) {
        buffer() /= denominator_;
        meas_.measure(obs_name, buffer);
      }
    } else {
      auto m = mpi.comm.reduce_value(sample);
      if(mpi.comm.root()) {
        meas_.measure(obs_name, m / denominator_);
      }
    }
  }
private:
  std::string prefix_{};
  ComplexType denominator_{};
  Measurements& meas_;
};

}
