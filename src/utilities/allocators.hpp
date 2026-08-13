#pragma once

#include <utility>

#include <nda/nda.hpp>

namespace memory {

template<typename Primary>
class static_fallback
{
  inline static Primary alloc = {};
  using Secondary = nda::mem::mallocator<Primary::address_space>;

public:
  static constexpr auto address_space = Primary::address_space;

  static_fallback()                                  = default;
  static_fallback(static_fallback const&)            = delete;
  static_fallback(static_fallback&&)                 = default;
  static_fallback& operator=(static_fallback const&) = delete;
  static_fallback& operator=(static_fallback&&)      = default;

  auto get_primary()       { return std::addressof(alloc); }
  auto get_primary() const { return std::addressof(alloc); }

  nda::mem::blk_t allocate(std::size_t s) noexcept
  {
    nda::mem::blk_t b = alloc.allocate(s);
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
    if (alloc.owns(b)) {
      alloc.deallocate(b);
    } else {
      // account the release in the primary's counters, then free via the secondary
      alloc.deallocate(b);
      Secondary::deallocate(b);
    }
  }
};

namespace detail {
  // returns the next address aligned to "align"
  inline char *align_up(char *ptr, std::size_t align = alignof(std::max_align_t)) {
    std::uintptr_t align_(align);
    uintptr_t ptr_i = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<char *>(align_ * ((ptr_i + (align_ - 1)) / align_));
  };
} // namespace detail

template <nda::mem::AddressSpace AdrSp = nda::mem::Host, size_t Alignment = alignof(std::max_align_t)>
class dynamic_bucket {
private:
  // auxiliary allocator
  using Auxiliary = nda::mem::mallocator<AdrSp>;

  /// alignment
  constexpr static size_t align_ = std::max(size_t(1), Alignment);

  // size of the pool
  size_t size_ = 0;

  /// maximum amount of memory needed
  size_t maximum_needed_ = 0;

  /// total amount of memory requested
  size_t total_requested_ = 0;

  /// total amount of memory released
  size_t total_released_ = 0;

  // pool of memory
  nda::mem::blk_t pool_;

  // aligned start of pool_
  char *p0 = nullptr;

  // list of available memory segments
  std::vector<nda::mem::blk_t> avail_;

  // list of allocated memory segments
  std::vector<nda::mem::blk_t> segments_;

  public:
  /// Default constructor.
  dynamic_bucket(size_t s = 8000) : size_(s + align_), pool_{Auxiliary::allocate(size_)} {
    // first alligned memory location
    p0 = detail::align_up(pool_.ptr, align_);
    avail_.reserve(10);
    segments_.reserve(10);
    // initialize avail_
    avail_.emplace_back(nda::mem::blk_t{p0, size_t(std::distance(p0, pool_.ptr + pool_.s))});
  }

  /// Destructor
  ~dynamic_bucket() {
    if (pool_.s > 0 and pool_.ptr != nullptr) Auxiliary::deallocate(pool_);
  }

  dynamic_bucket(dynamic_bucket const &) = delete;
  dynamic_bucket &operator=(dynamic_bucket const &) = delete;

  dynamic_bucket(dynamic_bucket &&other) noexcept
     : size_(std::exchange(other.size_, 0)),
       maximum_needed_(std::exchange(other.maximum_needed_, 0)),
       total_requested_(std::exchange(other.total_requested_, 0)),
       total_released_(std::exchange(other.total_released_, 0)),
       pool_(std::exchange(other.pool_, nda::mem::blk_t{})),
       p0(std::exchange(other.p0, nullptr)),
       avail_(std::move(other.avail_)),
       segments_(std::move(other.segments_)) {
    // a moved-from vector is valid but unspecified; make the empty state explicit
    other.avail_.clear();
    other.segments_.clear();
  }

  dynamic_bucket &operator=(dynamic_bucket &&other) noexcept {
    if(this != &other) {
      if(pool_.s > 0 && pool_.ptr != nullptr) Auxiliary::deallocate(pool_);
      size_            = std::exchange(other.size_, 0);
      maximum_needed_  = std::exchange(other.maximum_needed_, 0);
      total_requested_ = std::exchange(other.total_requested_, 0);
      total_released_  = std::exchange(other.total_released_, 0);
      pool_            = std::exchange(other.pool_, nda::mem::blk_t{});
      p0               = std::exchange(other.p0, nullptr);
      avail_           = std::move(other.avail_);
      segments_        = std::move(other.segments_);
      other.avail_.clear();
      other.segments_.clear();
    }
    return *this;
  }

  /// nda::mem::AddressSpace in which the memory is allocated.
  static constexpr auto address_space = AdrSp;

  void resize(size_t s) {
    if (segments_.size() > 0) throw std::bad_alloc{};
    size_ = s + align_;
    Auxiliary::deallocate(pool_);
    pool_ = Auxiliary::allocate(size_);
    p0    = detail::align_up(pool_.ptr, align_);
    avail_.clear();
    avail_.emplace_back(nda::mem::blk_t{p0, size_t(std::distance(p0, pool_.ptr + pool_.s))});
  }

  auto size() const {
    return std::distance(p0, pool_.ptr + pool_.s);
  }

  auto maximum_memory() const { return maximum_needed_; }

  nda::mem::blk_t allocate(size_t s) noexcept {
    // round up to closest multiple of align
    size_t aligned_s = ((s + (align_ - 1)) / align_) * align_;
    total_requested_ += aligned_s;
    maximum_needed_ = std::max(maximum_needed_, total_requested_ - total_released_);
    // find available block with enough space
    auto b = std::find_if(avail_.begin(), avail_.end(), [&](auto const &a) { return a.s >= aligned_s; });
    if (b != avail_.end()) {
      auto p_s = b->ptr;
      // add segment to list, keeping list unsorted
      segments_.push_back({p_s, aligned_s});
      // remove segment from avail_
      if (aligned_s == b->s)
        avail_.erase(b);
      else {
        b->ptr += aligned_s;
        b->s -= aligned_s;
      }
      return {p_s, s};
    }
    return {nullptr, 0};
  }

  nda::mem::blk_t allocate_zero(size_t s) noexcept {
    nda::mem::blk_t b = allocate(s);
    if (b.ptr and b.s > 0) memset<address_space>(b.ptr, 0, b.s);
    return b;
  }

  void deallocate(nda::mem::blk_t b) noexcept {
    if (segments_.size() == 0 or size_ == 0) {
      total_released_ += b.s;
      return;
    }
    // 1. find blk in segments_
    auto it = std::find_if(segments_.begin(), segments_.end(), [&](auto const &a) { return std::distance(a.ptr, b.ptr) == 0; });
    if (it != segments_.end()) {
      total_released_ += it->s;
      move_blk_to_avail(*it);
      segments_.erase(it);
      return;
    }
    // 2. Deallocates owned memory.
    EXPECTS_WITH_MESSAGE(
       (std::distance(b.ptr, p0) > 0) or (std::distance(pool_.ptr + pool_.s, b.ptr) > 0),
       "Error in nda::mem::dynamic_bucket::deallocate: Deallocating memory within the pool of the allocator, yet not registered as a segment.")
    // signal that memory is not owned by this allocator
    total_released_ += b.s;
  }

  bool owns(nda::mem::blk_t b) const noexcept {
    if (segments_.size() == 0 or size_ == 0) return false;
    return (std::distance(p0, b.ptr) >= 0) and (std::distance(b.ptr, pool_.ptr + pool_.s) > 0);
  }

  private:
  void move_blk_to_avail(nda::mem::blk_t b) {
    if (avail_.size() == 0) {
      avail_.emplace_back(b);
      return;
    }
    // 0. find location of b.ptr in list
    auto it = std::lower_bound(avail_.begin(), avail_.end(), b, [&](auto &&i, auto &&j) { return std::distance(i.ptr, j.ptr) > 0; });
    // add in front
    if (it == avail_.begin()) {
      auto it_beg = avail_.begin();
      if (std::distance(b.ptr + b.s, it_beg->ptr) == 0) {
        //merge
        it_beg->ptr = b.ptr;
        it_beg->s += b.s;
      } else {
        // add
        avail_.insert(it_beg, b);
      }
      return;
    }
    // add in back
    if (it == avail_.end()) {
      auto it_last = avail_.end() - 1;
      if (std::distance(it_last->ptr + it_last->s, b.ptr) == 0) {
        //merge
        it_last->s += b.s;
      } else {
        // add
        avail_.emplace_back(b);
      }
      return;
    }
    // General scenario: 4 cases to consider
    auto prev   = it - 1;
    auto d_prev = std::distance(prev->ptr + prev->s, b.ptr);
    auto d_next = std::distance(b.ptr + b.s, it->ptr);
    if ((d_prev == 0) and (d_next == 0)) {
      // a. b is contiguous with previous and next element in avail_: merge into a single segment
      prev->s += b.s + it->s;
      avail_.erase(it);
    } else if (d_prev == 0) {
      // b. b is contiguous with previous element, not with next: merge with previous
      prev->s += b.s;
    } else if (d_next == 0) {
      // c. b is contiguous with next element, not with previous: merge with next
      it->ptr = b.ptr;
      it->s += b.s;
    } else {
      // d. b is completely isolated: add new segment
      avail_.insert(it, b);
    }
  }
};

}
