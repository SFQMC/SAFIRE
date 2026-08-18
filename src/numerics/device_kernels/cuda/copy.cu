#include "numerics/device_kernels/cuda/cast_pairs.cuh"
#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

template<typename A, typename B, int R>
struct copy_fn
{
  view<A const, R> a;
  view<B, R>       b;

  __device__ void operator()(auto... idx) const { b(idx...) = native_t<B>(a(idx...)); }
};

template<typename A, typename B, int R>
void copy(view<A const, R> a, view<B, R> b)
{
  for_each(a, copy_fn<A, B, R>{a, b});
}

// The view is layout-erased, so one instantiation per (pair, rank) covers every stride order and
// both device and unified memory.
#define _inst_(A, B, R) template void copy<A, B, R>(view<A const, R>, view<B, R>);

static_assert(max_copy_rank == 6, "_inst_ranks_ must cover every rank up to max_copy_rank");

#define _inst_ranks_(A, B)                                                                         \
  _inst_(A, B, 1) _inst_(A, B, 2) _inst_(A, B, 3) _inst_(A, B, 4) _inst_(A, B, 5) _inst_(A, B, 6)

_for_each_pair_(_inst_ranks_)

} // namespace kernels::device
