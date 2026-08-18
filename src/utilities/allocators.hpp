#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <nda/nda.hpp>

namespace memory {

template<typename Primary>
class static_fallback
{
  using Secondary = nda::mem::mallocator<Primary::address_space>;

  inline static Primary primary;
public:
  static constexpr auto address_space = Primary::address_space;

  Primary& get_primary() { return primary; }

  nda::mem::blk_t allocate(std::size_t s) noexcept
  {
    nda::mem::blk_t b = primary.allocate(s);
    if (b.ptr) return b;
    return Secondary::allocate(s);
  }

  nda::mem::blk_t allocate_zero(std::size_t s) noexcept
  {
    nda::mem::blk_t b = this->allocate(s);
    if (b.ptr and b.s > 0) nda::mem::memset<address_space>(b.ptr, 0, b.s);
    return b;
  }

  void deallocate(nda::mem::blk_t b) noexcept
  {
    // the primary accounts for every release, but reclaims the block only if it owns it
    primary.deallocate(b);
    if (!primary.owns(b)) Secondary::deallocate(b);
  }
};

/*
 * Arena allocator over a single, resizable pool obtained from mallocator<AdrSp>.
 *
 * A request that does not fit in the pool returns a null block, leaving it to the caller (see
 * static_fallback) to serve it elsewhere. Such a request still counts towards maximum_memory(),
 * which reports the peak amount of memory that was outstanding at any one time. That is the
 * number used to grow the pool until the fallback is no longer needed, see
 * sfqmc::utils::resize_nda_static_allocator.
 */
template <nda::mem::AddressSpace AdrSp = nda::mem::Host, size_t Alignment = alignof(std::max_align_t)>
class dynamic_bucket {
private:
  // auxiliary allocator
  using Auxiliary = nda::mem::mallocator<AdrSp>;

  /// alignment of the segments handed out
  constexpr static size_t align_ = std::max(size_t(1), Alignment);
  static_assert((align_ & (align_ - 1)) == 0, "dynamic_bucket: Alignment must be a power of two.");
  // the pool itself needs no adjustment: mallocator hands back memory aligned for any type with
  // a fundamental alignment requirement, so the pool starts out align_-aligned already
  static_assert(align_ <= alignof(std::max_align_t), "dynamic_bucket: Alignment is too large.");

  /// pool of memory. pool_.s is the size of the pool
  nda::mem::blk_t pool_{};

  /// available memory segments, sorted by address and fully merged
  std::vector<nda::mem::blk_t> avail_ = {};

  /// amount of memory currently handed out, rounded up to align_
  size_t in_use_ = 0;

  /// high water mark of in_use_
  size_t maximum_needed_ = 0;

  /// largest request that can be rounded up without overflowing
  constexpr static size_t max_request_ = std::numeric_limits<size_t>::max() - (align_ - 1);

  /// smallest multiple of align_ that is >= s, for s <= max_request_
  static constexpr size_t round_up(size_t s) { return (s + (align_ - 1)) & ~(align_ - 1); }

  public:
  /// nda::mem::AddressSpace in which the memory is allocated.
  static constexpr auto address_space = AdrSp;

  dynamic_bucket() = default;

  /// Destructor
  ~dynamic_bucket() {
    if(pool_.ptr != nullptr) { Auxiliary::deallocate(pool_); }
  }

  dynamic_bucket(dynamic_bucket const &)            = delete;
  dynamic_bucket &operator=(dynamic_bucket const &) = delete;
  dynamic_bucket(dynamic_bucket &&)                 = delete;
  dynamic_bucket &operator=(dynamic_bucket &&)      = delete;

  /// true if no segment is currently carved out of the pool, so that it may be replaced
  bool idle() const {
    auto add_size = [](size_t a, auto const &b) { return a + b.s; };
    return std::accumulate(avail_.begin(), avail_.end(), size_t(0), add_size) == pool_.s;
  }

  /*
   * Releases the pool. Requires that no segment is in use, see idle(). Subsequent requests are
   * declined until the next resize(), which leaves the caller to serve them.
   */
  void release_pool() {
    if(!idle()) { throw std::bad_alloc{}; }
    if(pool_.ptr != nullptr) { Auxiliary::deallocate(pool_); }
    pool_ = {};
    avail_.clear();
  }

  /// Replaces the pool with one of at least s bytes, or releases it for s == 0. Requires that no
  /// segment is in use.
  void resize(size_t s) {
    release_pool();
    if(s == 0) { return; }
    // round up, so that a request for the whole pool can be served
    pool_ = Auxiliary::allocate(round_up(s));
    // mallocator reports the requested size even when the underlying allocation failed
    if(pool_.ptr == nullptr) {
      pool_ = {};
      throw std::bad_alloc{};
    }
    avail_.reserve(64);
    avail_.emplace_back(pool_);
  }

  /// size of the pool
  size_t size() const { return pool_.s; }

  size_t maximum_memory() const { return maximum_needed_; }

  nda::mem::blk_t allocate(size_t s) noexcept {
    // a size that cannot be rounded up is a caller error, no pool could ever hold it. Decline
    // it without accounting for it, so that it cannot distort the peak
    if(s == 0 || s > max_request_) { return {nullptr, 0}; }
    size_t const aligned_s = round_up(s);
    // account for the request before knowing whether the pool can serve it: what is needed
    // here is the peak demand, including whatever had to be served by the fallback
    in_use_ += aligned_s;
    maximum_needed_ = std::max(maximum_needed_, in_use_);
    // first available segment with enough space
    auto b = std::find_if(avail_.begin(), avail_.end(), [=](auto const &a) { return a.s >= aligned_s; });
    if(b == avail_.end()) { return {nullptr, 0}; }
    char *p = b->ptr;
    if(aligned_s == b->s) {
      avail_.erase(b);
    } else {
      b->ptr += aligned_s;
      b->s -= aligned_s;
    }
    return {p, s};
  }

  nda::mem::blk_t allocate_zero(size_t s) noexcept {
    nda::mem::blk_t b = allocate(s);
    if(b.ptr && b.s > 0) { nda::mem::memset<address_space>(b.ptr, 0, b.s); }
    return b;
  }

  void deallocate(nda::mem::blk_t b) noexcept {
    if(b.ptr == nullptr || b.s == 0) { return; }
    // allocate() returns the requested size, so rounding it up again recovers the size of the
    // segment that was carved out of the pool
    size_t const aligned_s = round_up(b.s);
    EXPECTS(in_use_ >= aligned_s);
    in_use_ -= std::min(in_use_, aligned_s);
    // a block outside the pool was served by the fallback: only the accounting applies to it
    if(owns(b)) { release({b.ptr, aligned_s}); }
  }

  bool owns(nda::mem::blk_t b) const noexcept {
    return (pool_.ptr != nullptr) && (b.ptr >= pool_.ptr) && (b.ptr < pool_.ptr + pool_.s);
  }

  private:
  /// returns a segment to avail_, merging it with the adjacent available segments
  void release(nda::mem::blk_t b) {
    EXPECTS_WITH_MESSAGE((b.ptr >= pool_.ptr) && (b.ptr + b.s <= pool_.ptr + pool_.s),
       "Error in memory::dynamic_bucket::release: Segment is not contained in the pool.")
    auto it = std::ranges::lower_bound(avail_, b.ptr, std::less<>{}, &nda::mem::blk_t::ptr);
    EXPECTS_WITH_MESSAGE((it == avail_.end()) || (b.ptr + b.s <= it->ptr),
       "Error in memory::dynamic_bucket::release: Segment overlaps an available segment (double free?).")
    EXPECTS_WITH_MESSAGE((it == avail_.begin()) || ((it - 1)->ptr + (it - 1)->s <= b.ptr),
       "Error in memory::dynamic_bucket::release: Segment overlaps an available segment (double free?).")
    if((it != avail_.end()) && (b.ptr + b.s == it->ptr)) {
      // merge with the following segment
      it->ptr = b.ptr;
      it->s += b.s;
    } else {
      it = avail_.insert(it, b);
    }
    if(it != avail_.begin()) {
      auto prev = it - 1;
      if(prev->ptr + prev->s == it->ptr) {
        // merge with the preceding segment. Together with the merge above this also covers the
        // case where b joins both of its neighbours into a single segment
        prev->s += it->s;
        avail_.erase(it);
      }
    }
  }
};

}
