#include <cuda/std/cmath>

#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"

namespace kernels::device
{

template<typename T, int R, unary_op Op>
struct apply_fn
{
  native_t<T> alpha;
  view<T, R>  a;

  __device__ void operator()(auto... idx) const
  {
    auto& x = a(idx...);
    if constexpr(Op == unary_op::SQRT) {
      x = alpha * ::cuda::std::sqrt(x);
    } else if constexpr(Op == unary_op::ABS) {
      x = alpha * ::cuda::std::abs(x);
    } else if constexpr(Op == unary_op::RCP) {
      x = alpha / x;
    } else if constexpr(Op == unary_op::EXP) {
      x = alpha * ::cuda::std::exp(x);
    } else {
      x = alpha * ::cuda::std::log(x);
    }
  }
};

template<typename T, int R>
void apply(T alpha, view<T, R> a, unary_op op)
{
  auto alpha_d = to_native(alpha);
  switch(op) {
    case unary_op::SQRT:
      for_each(a, apply_fn<T, R, unary_op::SQRT>{alpha_d, a});
      break;
    case unary_op::ABS:
      for_each(a, apply_fn<T, R, unary_op::ABS>{alpha_d, a});
      break;
    case unary_op::RCP:
      for_each(a, apply_fn<T, R, unary_op::RCP>{alpha_d, a});
      break;
    case unary_op::EXP:
      for_each(a, apply_fn<T, R, unary_op::EXP>{alpha_d, a});
      break;
    case unary_op::LOG:
      for_each(a, apply_fn<T, R, unary_op::LOG>{alpha_d, a});
      break;
  }
}

#define _inst_(T)                                                                                  \
  template void apply<T, 1>(T, view<T, 1>, unary_op);                                              \
  template void apply<T, 2>(T, view<T, 2>, unary_op);                                              \
  template void apply<T, 3>(T, view<T, 3>, unary_op);

_inst_(double)
_inst_(std::complex<double>)

} // namespace kernels::device
