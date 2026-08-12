#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"
#include "numerics/operations/add_diagonal_impl.hpp"

namespace kernels::device
{

namespace detail
{
  template<typename V>
  void apply_impl(nda::get_value_t<V> alpha, V& a, nda::tensor::unary_op oper);
}

template<typename V, nda::MemoryArray A>
requires(std::decay_t<A>::is_stride_order_C())
void apply(V alpha, A&& a, nda::tensor::unary_op oper = nda::tensor::unary_op::IDENTITY) {
  using T = nda::get_value_t<V>;
  if(a.is_contiguous() and nda::get_rank<A> > 1) {
    kernels::device::apply(T(alpha),nda::flatten(a),oper);
    return;
  }
  auto a_b = to_basic_layout(a());
  detail::apply_impl(T(alpha),a_b,oper);
}

} // namespace kernels::device
