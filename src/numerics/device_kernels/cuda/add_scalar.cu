#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

template<typename T, int R>
struct add_scalar_fn
{
  scalar_arg<T> alpha;
  native_t<T>   beta;
  view<T, R>    a;

  __device__ void operator()(auto... idx) const
  {
    native_t<T> s = alpha.address ? *alpha.address : alpha.value;
    // beta == 0 is a fill and must not read a: the caller may be initializing it
    a(idx...) = (beta == native_t<T>{}) ? s : s + beta * a(idx...);
  }
};

template<typename T, int R>
void add_scalar(scalar_arg<T> alpha, T beta, view<T, R> a)
{
  for_each(a, add_scalar_fn<T, R>{alpha, to_native(beta), a});
}

// The view is layout-erased, so one instantiation per (type, rank) covers every stride order and
// both device and unified memory.
#define _inst_(T, R) template void add_scalar<T, R>(scalar_arg<T>, T, view<T, R>);

static_assert(max_add_scalar_rank == 4,
              "_inst_ranks_ must cover every rank up to max_add_scalar_rank");

#define _inst_ranks_(T) _inst_(T, 1) _inst_(T, 2) _inst_(T, 3) _inst_(T, 4)

_inst_ranks_(double)
_inst_ranks_(std::complex<double>)

} // namespace kernels::device
