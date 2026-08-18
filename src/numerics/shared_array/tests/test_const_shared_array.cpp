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

#include "catch2/catch_test_macros.hpp"

#include <array>
#include <optional>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "test_common.hpp"

namespace bdft_tests {

namespace mpi3 = boost::mpi3;

// Build an mpi_context_t whose node_comm is artificially restricted to
// `node_size` ranks. All ranks of a unit-test run live on the same physical
// node, so world.split_shared() yields a single node and the internode
// broadcast path of const_shared_array never fires. By splitting the real node
// communicator into smaller, still-shared sub-communicators we simulate several
// nodes on one host: a subset of ranks on a host is still a valid shared-memory
// domain for MPI_Win_allocate_shared, so both the shared-window allocation and
// the cross-node broadcast get exercised.
sfqmc::utils::mpi_context_t<mpi3::communicator>
make_small_node_context(int node_size) {
  auto world      = mpi3::environment::get_world_instance();    // copy of world
  auto real_node  = world.split_shared();                       // all ranks on this host
  int  color      = real_node.rank() / node_size;
  auto small_node = real_node.split(color, real_node.rank());   // shared_communicator
  auto internode  = world.split(small_node.rank(), world.rank());
  return {std::move(world), std::move(small_node), std::move(internode)};
}


TEST_CASE("const_shared_array", "[shared_array][mpi]") {
  using base_t = nda::array<double, 2>;
  std::array<long, 2> const shape{7, 5};

  // Deterministic reference, materialized identically on every rank for checks.
  base_t ref(shape);
  for(long i = 0; i < ref.size(); ++i) {
    ref.data()[i] = double(i);
  }

  // node_size = 1 makes every rank its own node (pure internode broadcast);
  // node_size = 2 gives genuine intra-node sharing plus internode broadcast.
  for(int node_size : {1, 2}) {
    auto mpi = make_small_node_context(node_size);

    // Source array is supplied on the global root only; nullopt elsewhere.
    std::optional<base_t> value;
    if(mpi.comm.root()) {
      value = ref;
    }

    memory::const_shared_array<HOST_MEMORY, base_t::value_type, nda::get_rank<base_t>> csa(mpi, value);

    // Shape and data must be visible and correct on every rank.
    CHECK(csa.shape() == shape);
    CHECK_THAT(csa(), sfqmc::utils::Approx(ref));

    // Same coverage through the root-only-callable builder, which never
    // materializes the source array on non-root ranks.
    auto csa2 = memory::share_from_root(mpi, [&]() {
      base_t a(shape);
      for(long i = 0; i < a.size(); ++i) {
        a.data()[i] = double(i);
      }
      return a;
    });
    CHECK(csa2.shape() == shape);
    CHECK_THAT(csa2(), sfqmc::utils::Approx(ref));

#if defined(ENABLE_DEVICE)
    // Device arrays cannot live in a shared window; share_from_root broadcasts
    // them instead and returns a plain device array on every rank.
    auto da = memory::share_from_root(mpi, [&]() { return nda::to_device(ref); });
    static_assert(nda::MemoryArray<decltype(da)> && nda::mem::on_device<decltype(da)>,
                  "device path must return the device array itself, not a const_shared_array");
    CHECK(da.shape() == shape);
    CHECK_THAT(nda::to_host(da), sfqmc::utils::Approx(ref));
#endif
  }
}

TEST_CASE("share_from_ranks", "[shared_array][mpi]") {
  std::array<long, 3> const shape{6, 4, 3};

  // Deterministic reference, materialized identically on every rank.
  nda::array<double, 3> ref(shape);
  for(long i = 0; i < shape[0]; ++i) {
    for(long j = 0; j < shape[1]; ++j) {
      for(long k = 0; k < shape[2]; ++k) {
        ref(i, j, k) = 100.0 * i + 10.0 * j + k + 1.0;
      }
    }
  }

    
  for(int node_size : {1, 2}) {
    auto mpi = make_small_node_context(node_size);

    // K=1: items are the leading index, blocks the trailing matrices.
    auto a1 = memory::share_from_ranks<HOST_MEMORY, double, 3, 1>(mpi, shape,
        [&](std::array<long, 1> idx, auto&& block) {
      auto [i] = idx;
      block = ref(i, nda::ellipsis{});
    });
    CHECK(a1.shape() == shape);
    CHECK_THAT(a1(), sfqmc::utils::Approx(ref));

    // K=2 with partially-written blocks: untouched elements must stay zero.
    nda::array<double, 3> ref_partial(ref);
    ref_partial(nda::range::all, nda::range::all, shape[2] - 1) = 0.0;
    auto a2 = memory::share_from_ranks<HOST_MEMORY, double, 3, 2>(mpi, shape,
        [&](std::array<long, 2> idx, auto&& block) {
      auto [i, j] = idx;
      for(long k = 0; k < shape[2] - 1; ++k) {
        block(k) = ref(i, j, k);
      }
    });
    CHECK_THAT(a2(), sfqmc::utils::Approx(ref_partial));

    // More ranks than items: with a single item along the leading axis, every
    // rank but one gets an empty chunk and must still participate in the
    // gather with a zero count.
    std::array<long, 3> const small_shape{1, shape[1], shape[2]};
    nda::array<double, 3> const ref_small(ref(nda::range(0, 1), nda::ellipsis{}));
    auto a3 = memory::share_from_ranks<HOST_MEMORY, double, 3, 1>(mpi, small_shape,
        [&](std::array<long, 1> idx, auto&& block) {
      auto [i] = idx;
      block = ref(i, nda::ellipsis{});
    });
    CHECK(a3.shape() == small_shape);
    CHECK_THAT(a3(), sfqmc::utils::Approx(ref_small));

#if defined(ENABLE_DEVICE)
    // Device path returns a plain device array assembled with a full all_reduce.
    auto d1 = memory::share_from_ranks<DEVICE_MEMORY, double, 3, 1>(mpi, shape,
        [&](std::array<long, 1> idx, auto&& block) {
      auto [i] = idx;
      block = nda::array<double, 2>(ref(i, nda::ellipsis{}));
    });
    static_assert(nda::MemoryArray<decltype(d1)> && nda::mem::on_device<decltype(d1)>,
                  "device path must return the device array itself, not a const_shared_array");
    CHECK_THAT(nda::to_host(d1), sfqmc::utils::Approx(ref));

    // More ranks than items on the device path: surplus ranks contribute only
    // zeros to the assembling all_reduce.
    auto d3 = memory::share_from_ranks<DEVICE_MEMORY, double, 3, 1>(mpi, small_shape,
        [&](std::array<long, 1> idx, auto&& block) {
      auto [i] = idx;
      block = nda::array<double, 2>(ref(i, nda::ellipsis{}));
    });
    CHECK_THAT(nda::to_host(d3), sfqmc::utils::Approx(ref_small));
#endif
  }
}

TEST_CASE("share_reduced", "[shared_array][mpi]") {
  std::array<long, 2> const shape{5, 3};

  for(int node_size : {1, 2}) {
    auto mpi = make_small_node_context(node_size);

    // Every rank contributes the same overlapping partial; the result is the
    // sum over all ranks.
    auto r = memory::share_reduced<HOST_MEMORY, double, 2>(mpi, shape,
        [&](auto&& partial) {
      for(long i = 0; i < shape[0]; ++i) {
        for(long j = 0; j < shape[1]; ++j) {
          partial(i, j) = 10.0 * i + j + 1.0;
        }
      }
    });

    nda::array<double, 2> expected(shape);
    for(long i = 0; i < shape[0]; ++i) {
      for(long j = 0; j < shape[1]; ++j) {
        expected(i, j) = mpi.comm.size() * (10.0 * i + j + 1.0);
      }
    }
    CHECK(r.shape() == shape);
    CHECK_THAT(r(), sfqmc::utils::Approx(expected));
  }
}

TEST_CASE("const_shared_array_default", "[shared_array][mpi]") {
  // Default-constructed arrays are empty and usable as nda container elements
  // until assigned from a properly constructed instance.
  auto mpi = make_small_node_context(1);
  std::array<long, 2> const shape{3, 4};

  nda::array<memory::const_shared_array<HOST_MEMORY, double, 2>, 1> arr(2);
  CHECK(arr(0).size() == 0);

  for(int n = 0; n < 2; ++n) {
    arr(n) = memory::share_from_ranks<HOST_MEMORY, double, 2, 1>(mpi, shape,
        [&](std::array<long, 1> idx, auto&& block) {
      auto [i] = idx;
      block = double(n * 100 + i);
    });
  }
  nda::array<double, 2> expected(shape);
  for(int n = 0; n < 2; ++n) {
    for(long i = 0; i < shape[0]; ++i) {
      expected(i, nda::range::all) = double(n * 100 + i);
    }
    CHECK_THAT(arr(n)(), sfqmc::utils::Approx(expected));
  }
}

} // namespace bdft_tests
