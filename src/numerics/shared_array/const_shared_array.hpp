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

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>

#include <nda/nda.hpp>
#include <mpi3/shared_window.hpp>

#include "configuration.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"

namespace memory {

/**
 * @brief An immutable, node-shared nda array as a safe alternative to mpi3::shared_array.
 *
 * The data is supplied on the global root only; it is copied into the
 * node-local shared-memory segment, broadcast to every node, and thereafter
 * exposed read-only through operator().
 *
 * @tparam ValueType Element type of the array.
 * @tparam Rank Rank (number of dimensions) of the array.
 */
template<typename ValueType, int Rank>
class host_const_shared_array {
public:
  using view_type  = nda::array_const_view<ValueType, Rank>;

  /**
   * @brief Empty state: zero shape, no window.
   *
   * Allows deferred initialization and use as the element type of nda
   * containers; access the data only after assignment from a properly
   * constructed instance.
   */
  host_const_shared_array() = default;

  /**
   * @brief Construct from an array supplied on the global root.
   *
   * Collective on mpi.comm. The value is copied into the node-local
   * shared-memory segment and broadcast to the remaining nodes.
   *
   * @tparam Array nda::Array type convertible to the element type.
   * @param mpi MPI context providing the communicators and shared windows.
   * @param value Source array; must hold a value on the global root and be
   * empty on all other ranks.
   */
  template<nda::Array Array>
  host_const_shared_array(sfqmc::utils::mpi_context_t<mpi3::communicator>& mpi,
                          std::optional<Array> value) {
    
    if(mpi.comm.root()) {
      sfqmc::utils::check(value.has_value(),
        "host_const_shared_array must be constructed with a value on the root process");
      shape_ = value->shape();
    } else {
      sfqmc::utils::check(!value.has_value(),
        "host_const_shared_array must not be given a value on non-root processes");
    }
    mpi.comm.broadcast_n(shape_.data(), shape_.size(), 0);
    
    auto const size = std::reduce(shape_.begin(), shape_.end(),
                                  mpi3::size_t{1}, std::multiplies<>{});

    window_ = std::move(mpi.shared_windows.open_window(
        mpi.node_comm.root() ? size * mpi3::size_t(sizeof(ValueType)) : 0));

    if(mpi.comm.root()) {
      nda::array_view<ValueType, Rank> dst(shape_, static_cast<ValueType *>(window_->base_ptr(0)));
      dst = value.value();
    }

    // Propagate from node 0 to the remaining node roots. Chunked to stay below
    // the MPI count limit.
    if(mpi.node_comm.root() and size > 0) {
      constexpr mpi3::size_t chunk = mpi3::size_t(1e9);
      ValueType* base = static_cast<ValueType*>(window_->base_ptr(0));
      for(mpi3::size_t shift = 0; shift < size; shift += chunk) {
        mpi3::size_t count = (shift + chunk < size) ? chunk : size - shift;
        mpi.internode_comm.broadcast_n(base + shift, count, 0);
      }
    }
    mpi.node_comm.barrier();
  }

  /// @brief Read-only view of the shared data.
  view_type operator()() const {
    sfqmc::utils::check(bool(window_),
      "data access on a default-constructed (empty) instance");
    return view_type(shape_, static_cast<ValueType *>(window_->base_ptr(0)));
  }

  /// @brief Shape of the array.
  auto const& shape() const { return shape_; }
  /// @brief Extent along dimension `rank`.
  long extent(int rank) const { return shape_.at(rank); }
  /// @brief Total number of elements.
  long size() const {
    return std::reduce(shape_.begin(), shape_.end(), 1l, std::multiplies<>{});
  }

private:
  std::array<long, Rank> shape_{};
  std::shared_ptr<sfqmc::utils::shared_window> window_{};
};

/**
 * @brief Memory-space-generic immutable shared array.
 *
 * Resolves to host_const_shared_array in host memory; device (or unified)
 * arrays cannot live in an MPI shared window, so they resolve to a plain,
 * per-rank memory::array instead.
 */
template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
using const_shared_array = std::conditional_t<MEM == HOST_MEMORY,
                                              host_const_shared_array<T, N>,
                                              memory::array<MEM, T, N, Layout>>;

/**
 * @brief Share an array built by a callable that is only evaluated on the
 * global root, so non-root ranks never materialize the source array.
 *
 * Host arrays are returned as a node-shared host_const_shared_array; device
 * (or unified) arrays cannot live in an MPI shared window, so they are
 * broadcast to every rank and returned by value instead. Collective on
 * mpi.comm.
 *
 * @code
 * // read the integrals from file once, on the global root only
 * auto h0 = memory::share_from_root(mpi, [&] {
 *   return read_h0(filename);  // nda::array<double, 2>
 * });
 * auto e0 = h0()(0, 0);  // read-only access on every rank
 * @endcode
 *
 * @tparam RootOnlyFunc Callable returning an nda array; invoked on the global
 * root only.
 * @param mpi MPI context providing the communicators and shared windows.
 * @param func Callable building the source array.
 * @return const_shared_array over the memory space of the callable's result.
 */
template<typename RootOnlyFunc>
auto share_from_root(
    sfqmc::utils::mpi_context_t<mpi3::communicator>& mpi, RootOnlyFunc func) {
  using array_t = std::decay_t<decltype(func())>;
  using value_type = array_t::value_type;
  constexpr int rank = nda::get_rank<array_t>;
  
  static_assert(nda::Array<array_t>,
                "share_from_root: callable must return an nda array");

  if constexpr(nda::mem::on_host<array_t>) {
    std::optional<array_t> value{};
    if(mpi.comm.root()) {
      value = func();
    }
    return host_const_shared_array<value_type, rank>(mpi, std::move(value));
  } else {
    std::array<long, rank> shape{};
    array_t a;
    if(mpi.comm.root()) {
      a = func();
      shape = a.shape();
    }
    mpi.comm.broadcast_n(shape.data(), shape.size(), 0);
    if(!mpi.comm.root()) {
      a = array_t(shape);
    }
    mpi.broadcast(a);
    return a;
  }
}

/**
 * @brief Build an immutable array from blocks computed independently on each
 * rank.
 *
 * The array is split along its leading K axes into items; fill(idx, block) is
 * invoked once for every item assigned to this rank, where idx is the item's
 * multi-index over the leading axes and block is a PRIVATE, zero-initialized
 * view of the trailing Rank-K dimensions. Each item is computed by exactly one
 * rank; ranks own contiguous chunks of the item range. On the host, the chunks
 * are gathered into a full array materialized on the global root only, which
 * is then node-shared through the regular constructor. fill never touches
 * shared memory, so it needs no synchronization of its own. Collective on
 * mpi.comm.
 *
 * @code
 * // V(q, i, j): the q slices are distributed over the ranks
 * auto V = memory::share_from_ranks<HOST_MEMORY, ComplexType, 3, 1>(
 *     mpi, {nq, n, n}, [&](auto idx, auto&& block) {
 *       block = coulomb_matrix(idx[0]);  // block is the (n, n) slice at q = idx[0]
 *     });
 * auto v = V()(iq, i, j);  // read-only access on every rank
 * @endcode
 *
 * @tparam MEM Memory space of the result.
 * @tparam ValueType Element type of the array.
 * @tparam Rank Rank of the full array.
 * @tparam K Number of leading axes to split into items; 0 < K < Rank.
 * @tparam Fill Callable with signature fill(std::array<long, K> idx, block).
 * @param mpi MPI context providing the communicators and shared windows.
 * @param shape Shape of the full array.
 * @param fill Callable writing one item's trailing-axes block.
 * @return const_shared_array<MEM, ValueType, Rank> holding the assembled data.
 */
template<MEMORY_SPACE MEM, typename ValueType, int Rank, int K, typename Fill>
auto share_from_ranks(sfqmc::utils::mpi_context_t<mpi3::communicator>& mpi,
                      std::array<long, Rank> shape, Fill&& fill) {
  static_assert(K > 0 and K < Rank,
                "share_from_ranks: K must split the shape into leading and trailing axes");

  std::array<long, K> item_shape{};
  for(int d = 0; d < K; ++d) {
    item_shape[d] = shape[d];
  }
  auto const nitems = std::reduce(item_shape.begin(), item_shape.end(),
                                  1l, std::multiplies<>{});
  // A zero extent among the trailing axes makes every block empty. There is
  // nothing for fill to write, and a zero extent also zeroes the strides of the
  // enclosing axes, so the block slice itself would violate nda's stride order.
  bool const empty_blocks = std::any_of(shape.begin() + K, shape.end(),
                                        [](long n) { return n == 0; });
  auto decode = [&](long i) {
    std::array<long, K> idx{};
    for(int d = K - 1; d >= 0; --d) {
      idx[d] = i % item_shape[d];
      i /= item_shape[d];
    }
    return idx;
  };

  if constexpr(MEM == HOST_MEMORY) {
    // This rank's contiguous chunk of items, filled into a private buffer that
    // holds only this rank's share of the array.
    long const i0 = nitems * mpi.comm.rank() / mpi.comm.size();
    long const i1 = nitems * (mpi.comm.rank() + 1) / mpi.comm.size();

    std::array<long, Rank - K + 1> local_shape{};
    local_shape[0] = i1 - i0;
    for(int d = K; d < Rank; ++d) {
      local_shape[d - K + 1] = shape[d];
    }
    nda::array<ValueType, Rank - K + 1> local(local_shape);
    local() = ValueType{};
    if(not empty_blocks) {
      for(long i = i0; i < i1; ++i) {
        fill(decode(i), local(i - i0, nda::ellipsis{}));
      }
    }

    // Chunks are gathered in rank order, which is item order, so they land
    // directly in the final layout on the global root.
    sfqmc::utils::check(local.size() <= long(std::numeric_limits<int>::max()),
                        "share_from_ranks: per-rank chunk too large for gatherv");
    std::optional<nda::array<ValueType, Rank>> value{};
    if(mpi.comm.root()) {
      value.emplace(shape);
    }
    mpi.comm.gatherv_n(local.data(), local.size(),
                       mpi.comm.root() ? value->data() : static_cast<ValueType*>(nullptr));
    return host_const_shared_array<ValueType, Rank>(mpi, std::move(value));
  } else {
    // Device arrays are private per rank anyway: write the blocks in place
    // over a zeroed array and assemble by summation.
    memory::array<MEM, ValueType, Rank> a(shape);
    a() = ValueType{};
    if(not empty_blocks) {
      for(long i = mpi.comm.rank(); i < nitems; i += mpi.comm.size()) {
        auto idx = decode(i);
        std::apply([&](auto... is) { fill(idx, a(is..., nda::ellipsis{})); }, idx);
      }
    }
    mpi.all_reduce(a(), std::plus<>{});
    return a;
  }
}

/**
 * @brief Build an immutable array as the elementwise sum of per-rank
 * contributions.
 *
 * fill receives a PRIVATE, zero-initialized array of the full shape on every
 * rank; contributions may overlap arbitrarily (unlike share_from_ranks, this
 * expresses a reduction over a contracted index rather than a disjoint
 * partition of the output). Each rank transiently holds a full-size private
 * copy, so this is meant for small arrays. Collective on mpi.comm.
 *
 * @code
 * // S(i, j) = sum over all walkers, which are distributed over the ranks
 * auto S = memory::share_reduced<HOST_MEMORY, ComplexType, 2>(
 *     mpi, {n, n}, [&](auto&& acc) {
 *       for(auto const& w : local_walkers) {
 *         acc += overlap_matrix(w);
 *       }
 *     });
 * auto s = S()(i, j);  // read-only access on every rank
 * @endcode
 *
 * @tparam MEM Memory space of the result.
 * @tparam ValueType Element type of the array.
 * @tparam Rank Rank of the array.
 * @tparam Fill Callable taking a writable view of the full array.
 * @param mpi MPI context providing the communicators and shared windows.
 * @param shape Shape of the array.
 * @param fill Callable adding this rank's contribution into its private copy.
 * @return const_shared_array<MEM, ValueType, Rank> holding the reduced data.
 */
template<MEMORY_SPACE MEM, typename ValueType, int Rank, typename Fill>
auto share_reduced(sfqmc::utils::mpi_context_t<mpi3::communicator>& mpi,
                   std::array<long, Rank> shape, Fill&& fill) {
  memory::array<MEM, ValueType, Rank> partial(shape);
  partial() = ValueType{};
  fill(partial());

  if constexpr(MEM == HOST_MEMORY) {
    // only the global root feeds the shared window, so reduce instead of
    // all_reduce
    mpi.reduce(partial(), std::plus<>{});
    std::optional<memory::array<MEM, ValueType, Rank>> value{};
    if(mpi.comm.root()) {
      value = std::move(partial);
    }
    return host_const_shared_array<ValueType, Rank>(mpi, std::move(value));
  } else {
    mpi.all_reduce(partial(), std::plus<>{});
    return partial;
  }
}

} // namespace memory
