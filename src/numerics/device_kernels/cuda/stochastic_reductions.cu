////////////////////////////////////////////////////////////////////////////////
// Device kernels for SAFIRE StochasticWfn inner-ensemble reductions (PR-3 GPU port).
// See stochastic_reductions.cuh for the operation contracts.
////////////////////////////////////////////////////////////////////////////////

#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "nda/nda.hpp"
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"

namespace kernels::device::detail
{

// E[w, j] += S[w] * In[w, j]   (contiguous C-layout [nwalk, ncol]; S is [nwalk])
template<typename VS, typename MI, typename ME>
void row_accumulate_impl(VS const& s, MI const& in, ME& e)
{
  static_assert(nda::get_rank<VS> == 1 and nda::get_rank<MI> == 2 and nda::get_rank<ME> == 2, "rank");
  auto s_d  = to_cuda_std_mdspan(s);
  auto in_d = to_cuda_std_mdspan(in);
  auto e_d  = to_cuda_std_mdspan(e);
  long const ncol = in.extent(1);
  long const sz   = in.size();
  auto f = [=] __device__(long n) {
    long w = n / ncol;
    long j = n - w * ncol;
    e_d(w, j) += s_d(w) * in_d(w, j);
  };
  cub::DeviceFor::Bulk(sz, f);
}

// E[w, j] = E[w, j] / D[w]
template<typename VD, typename ME>
void row_divide_impl(VD const& d, ME& e)
{
  static_assert(nda::get_rank<VD> == 1 and nda::get_rank<ME> == 2, "rank");
  auto d_d = to_cuda_std_mdspan(d);
  auto e_d = to_cuda_std_mdspan(e);
  long const ncol = e.extent(1);
  long const sz   = e.size();
  auto f = [=] __device__(long n) {
    long w = n / ncol;
    long j = n - w * ncol;
    e_d(w, j) = e_d(w, j) / d_d(w);
  };
  cub::DeviceFor::Bulk(sz, f);
}

// Per-walker phase reduction (non-leapfrog, non-log-aggregate):
//   lin = exp(ov[w]); m = |lin|; s = (m>0) ? lin/m : 0;
//   Sp[w] = s;  Ov[w] += inv_P * lin;  D[w] += s;
template<typename VOVc, typename VSP, typename VOV, typename VD>
void inner_scalar_reduce_impl(VOVc const& ov, VSP& sp, VOV& Ov, VD& D, double inv_P)
{
  static_assert(nda::get_rank<VOVc> == 1 and nda::get_rank<VSP> == 1 and nda::get_rank<VOV> == 1 and
                    nda::get_rank<VD> == 1,
                "rank");
  auto ov_d = to_cuda_std_mdspan(ov);
  auto sp_d = to_cuda_std_mdspan(sp);
  auto Ov_d = to_cuda_std_mdspan(Ov);
  auto D_d  = to_cuda_std_mdspan(D);
  long const sz = ov.size();
  auto f = [=] __device__(long w) {
    auto lin = cuda::std::exp(ov_d(w));
    double m = cuda::std::abs(lin);
    auto s   = (m > 0.0) ? (lin / m) : cuda::std::complex<double>(0.0, 0.0);
    sp_d(w) = s;
    Ov_d(w) += inv_P * lin;
    D_d(w)  += s;
  };
  cub::DeviceFor::Bulk(sz, f);
}

// A[w] = log(A[w])
template<typename V>
void elementwise_log_impl(V& a)
{
  static_assert(nda::get_rank<V> == 1, "rank");
  auto a_d = to_cuda_std_mdspan(a);
  long const sz = a.size();
  auto f = [=] __device__(long w) { a_d(w) = cuda::std::log(a_d(w)); };
  cub::DeviceFor::Bulk(sz, f);
}

// Kl[w, c] += sum_a T[w, a, a, c]   (per-walker diagonal trace over the two orbital axes)
template<typename MT, typename MK>
void diag_trace_accumulate_impl(MT const& T, MK& K)
{
  static_assert(nda::get_rank<MT> == 4 and nda::get_rank<MK> == 2, "rank");
  auto T_d = to_cuda_std_mdspan(T);
  auto K_d = to_cuda_std_mdspan(K);
  long const NMO = T.extent(1);
  long const nCV = T.extent(3);
  long const sz  = K.size(); // nwalk * nCV
  auto f = [=] __device__(long n) {
    long w = n / nCV;
    long c = n - w * nCV;
    cuda::std::complex<double> acc(0.0, 0.0);
    for (long a = 0; a < NMO; ++a)
      acc += T_d(w, a, a, c);
    K_d(w, c) += acc;
  };
  cub::DeviceFor::Bulk(sz, f);
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T, V)                                                                                        \
  template void row_accumulate_impl(V<const T, 1, basic_layout_t<1>> const&,                                \
                                    V<const T, 2, basic_layout_t<2>> const&, V<T, 2, basic_layout_t<2>>&);  \
  template void row_divide_impl(V<const T, 1, basic_layout_t<1>> const&, V<T, 2, basic_layout_t<2>>&);      \
  template void inner_scalar_reduce_impl(V<const T, 1, basic_layout_t<1>> const&,                           \
                                         V<T, 1, basic_layout_t<1>>&, V<T, 1, basic_layout_t<1>>&,          \
                                         V<T, 1, basic_layout_t<1>>&, double);                              \
  template void elementwise_log_impl(V<T, 1, basic_layout_t<1>>&);                                          \
  template void diag_trace_accumulate_impl(V<const T, 4, basic_layout_t<4>> const&,                         \
                                           V<T, 2, basic_layout_t<2>>&);

_inst_(std::complex<double>, device_array_view)

} // namespace kernels::device::detail
