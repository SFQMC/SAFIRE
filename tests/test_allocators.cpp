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

#include "utilities/allocators.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using bucket_t   = memory::dynamic_bucket<nda::mem::Host>;
using fallback_t = nda::mem::mallocator<nda::mem::Host>;

constexpr size_t A = alignof(std::max_align_t);

static_assert(nda::mem::Allocator<bucket_t>);

// number of bytes the bucket carves out of the pool for a request of s bytes
size_t aligned(size_t s) { return ((s + (A - 1)) / A) * A; }

// A request the pool cannot serve is still accounted for, because the peak it contributes to is
// what tells resize() how large the pool would have to be. This models what static_fallback does
// with such a request: it hands out memory of its own, and releases it through the bucket again.
nda::mem::blk_t serve_elsewhere(bucket_t &bucket, size_t s) {
  auto declined = bucket.allocate(s);
  REQUIRE(declined.ptr == nullptr);
  return fallback_t::allocate(s);
}

void release_elsewhere(bucket_t &bucket, nda::mem::blk_t b) {
  REQUIRE(!bucket.owns(b));
  bucket.deallocate(b);
  fallback_t::deallocate(b);
}

} // namespace

TEST_CASE("dynamic_bucket without a pool", "[allocators]")
{
  bucket_t bucket;
  REQUIRE(bucket.size() == 0);
  REQUIRE(bucket.maximum_memory() == 0);

  std::array<char, 8> dummy{};
  REQUIRE(!bucket.owns({dummy.data(), dummy.size()}));

  // nothing can be served, but the demand is recorded
  release_elsewhere(bucket, serve_elsewhere(bucket, 100));
  REQUIRE(bucket.maximum_memory() == aligned(100));

  // and it is accounted exactly: repeating the cycle may not inflate the peak
  for(int i = 0; i < 10; ++i) {
    release_elsewhere(bucket, serve_elsewhere(bucket, 100));
  }
  REQUIRE(bucket.maximum_memory() == aligned(100));

  // an empty request is never served, by the pool or by anyone else
  auto empty = bucket.allocate(0);
  REQUIRE(empty.ptr == nullptr);
  REQUIRE(empty.s == 0);
  REQUIRE(bucket.maximum_memory() == aligned(100));
}

TEST_CASE("dynamic_bucket allocate and deallocate", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(16 * A);
  size_t const cap = bucket.size();
  REQUIRE(cap >= 16 * A);

  auto a = bucket.allocate(3 * A);
  auto b = bucket.allocate(2 * A + 1);
  auto c = bucket.allocate(A);
  for(auto const &blk : {a, b, c}) {
    REQUIRE(blk.ptr != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(blk.ptr) % A == 0);
    REQUIRE(bucket.owns(blk));
  }
  // the requested size is returned unrounded, and the segments do not overlap
  REQUIRE(a.s == 3 * A);
  REQUIRE(b.s == 2 * A + 1);
  REQUIRE(b.ptr == a.ptr + aligned(a.s));
  REQUIRE(c.ptr == b.ptr + aligned(b.s));

  size_t const peak = aligned(a.s) + aligned(b.s) + aligned(c.s);
  REQUIRE(bucket.maximum_memory() == peak);

  // a freed segment is handed out again
  bucket.deallocate(b);
  auto b2 = bucket.allocate(2 * A + 1);
  REQUIRE(b2.ptr == b.ptr);
  REQUIRE(bucket.maximum_memory() == peak);

  bucket.deallocate(a);
  bucket.deallocate(b2);
  bucket.deallocate(c);

  // a request larger than the pool is declined, and leaves the pool untouched
  release_elsewhere(bucket, serve_elsewhere(bucket, 2 * cap));
  auto whole = bucket.allocate(cap);
  REQUIRE(whole.ptr != nullptr);
  REQUIRE(whole.s == cap);
  bucket.deallocate(whole);
}

TEST_CASE("dynamic_bucket declines a size it cannot round up", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(8 * A);

  // a negative array extent reaches the allocator as a size near SIZE_MAX. Rounding that up
  // overflows, so it has to be declined outright rather than wrapping to a satisfiable size
  for(size_t s : {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() - 3}) {
    auto declined = bucket.allocate(s);
    REQUIRE(declined.ptr == nullptr);
    REQUIRE(declined.s == 0);
  }
  // and it may neither be counted nor disturb the pool
  REQUIRE(bucket.maximum_memory() == 0);
  auto whole = bucket.allocate(bucket.size());
  REQUIRE(whole.ptr != nullptr);
  REQUIRE(whole.s == bucket.size());
  bucket.deallocate(whole);
}

TEST_CASE("dynamic_bucket allocate_zero", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(8 * A);

  // dirty the memory first, so that the zeroing is actually observable
  auto dirty = bucket.allocate(4 * A);
  REQUIRE(dirty.ptr != nullptr);
  std::fill(dirty.ptr, dirty.ptr + dirty.s, char(0xff));
  bucket.deallocate(dirty);

  auto clean = bucket.allocate_zero(4 * A);
  REQUIRE(clean.ptr == dirty.ptr);
  REQUIRE(std::all_of(clean.ptr, clean.ptr + clean.s, [](char v) { return v == 0; }));
  bucket.deallocate(clean);
}

TEST_CASE("dynamic_bucket merges adjacent segments", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(16 * A);
  size_t const cap = bucket.size();
  size_t const s   = (cap / 4 / A) * A;

  std::array<nda::mem::blk_t, 4> blk;
  for(auto &b : blk) {
    b = bucket.allocate(s);
    REQUIRE(b.ptr != nullptr);
  }
  REQUIRE(blk[1].ptr == blk[0].ptr + s);
  REQUIRE(blk[2].ptr == blk[1].ptr + s);
  REQUIRE(blk[3].ptr == blk[2].ptr + s);

  // Free the four adjacent segments in every interesting order. Whatever the order, the pool has
  // to end up as a single segment again, which is what the final full-pool request proves.
  std::vector<size_t> order;
  SECTION("ascending") { order = {0, 1, 2, 3}; }
  SECTION("descending") { order = {3, 2, 1, 0}; }
  SECTION("merge with the preceding segment") { order = {1, 0, 3, 2}; }
  SECTION("merge with the following segment") { order = {2, 3, 0, 1}; }
  SECTION("merge both neighbours at once") { order = {0, 2, 1, 3}; }
  SECTION("isolated holes first") { order = {1, 3, 0, 2}; }
  SECTION("interleaved") { order = {2, 0, 3, 1}; }

  for(auto i : order) {
    bucket.deallocate(blk[i]);
  }

  auto whole = bucket.allocate(cap);
  REQUIRE(whole.ptr != nullptr);
  REQUIRE(whole.ptr == blk[0].ptr);
  bucket.deallocate(whole);
}

TEST_CASE("dynamic_bucket resize", "[allocators]")
{
  bucket_t bucket;

  // the pool can only be replaced while nothing is carved out of it
  bucket.resize(4 * A);
  auto live = bucket.allocate(A);
  REQUIRE(live.ptr != nullptr);
  REQUIRE_THROWS_AS(bucket.resize(8 * A), std::bad_alloc);
  bucket.deallocate(live);
  REQUIRE_NOTHROW(bucket.resize(8 * A));
  REQUIRE(bucket.size() >= 8 * A);

  // a live block that the pool declined does not block a resize
  auto served_elsewhere = serve_elsewhere(bucket, 1024 * A);
  REQUIRE_NOTHROW(bucket.resize(4 * A));
  release_elsewhere(bucket, served_elsewhere);
}

TEST_CASE("dynamic_bucket release_pool", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(8 * A);
  REQUIRE(bucket.idle());

  // just like a resize, releasing the pool is only allowed while nothing is carved out of it
  auto live = bucket.allocate(A);
  REQUIRE(!bucket.idle());
  REQUIRE_THROWS_AS(bucket.release_pool(), std::bad_alloc);
  bucket.deallocate(live);

  REQUIRE(bucket.idle());
  bucket.release_pool();
  REQUIRE(bucket.size() == 0);
  REQUIRE(bucket.idle());
  REQUIRE(!bucket.owns(live));
  // without a pool every request is declined again, and the peak is left alone
  size_t const peak = bucket.maximum_memory();
  release_elsewhere(bucket, serve_elsewhere(bucket, A));
  REQUIRE(bucket.maximum_memory() == peak);

  // releasing twice, and releasing a pool that was never created, are both harmless
  REQUIRE_NOTHROW(bucket.release_pool());
  REQUIRE_NOTHROW(bucket_t{}.release_pool());

  // resize(0) is the same thing, and a pool can be created again afterwards
  bucket.resize(8 * A);
  REQUIRE(bucket.size() == 8 * A);
  bucket.resize(0);
  REQUIRE(bucket.size() == 0);
  bucket.resize(8 * A);
  auto blk = bucket.allocate(4 * A);
  REQUIRE(blk.ptr != nullptr);
  bucket.deallocate(blk);
}

TEST_CASE("dynamic_bucket peak drives the pool size", "[allocators]")
{
  bucket_t bucket;
  bucket.resize(4 * A);

  // over-subscribe the pool: the excess has to be served elsewhere, but is counted
  auto served_here      = bucket.allocate(bucket.size());
  auto served_elsewhere = serve_elsewhere(bucket, 4 * A);
  REQUIRE(served_here.ptr != nullptr);
  size_t const peak = bucket.maximum_memory();
  REQUIRE(peak == served_here.s + 4 * A);

  bucket.deallocate(served_here);
  release_elsewhere(bucket, served_elsewhere);

  // the reported peak is what the pool has to grow to for the fallback to become unnecessary
  bucket.resize(peak);
  auto first  = bucket.allocate(served_here.s);
  auto second = bucket.allocate(4 * A);
  REQUIRE(first.ptr != nullptr);
  REQUIRE(second.ptr != nullptr);
  bucket.deallocate(first);
  bucket.deallocate(second);
  REQUIRE(bucket.maximum_memory() == peak);
}
