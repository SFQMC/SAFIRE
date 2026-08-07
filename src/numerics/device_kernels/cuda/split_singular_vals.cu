#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>
#include "numerics/operations/split_singular_vals_impl.hpp"

namespace kernels::device::detail
{

template<typename V1, typename V2, typename V3, typename V4, typename V5> 
void splitDmatrix_impl(V1 const& A, V2& B, V3& C, V4& res, V5 const& scl)
{
  long N = A.extent(0);
  if(N==0) return;
  auto A_d = to_cuda_std_mdspan(A);
  auto B_d = to_cuda_std_mdspan(B);
  auto C_d = to_cuda_std_mdspan(C);
  auto res_d = to_cuda_std_mdspan(res);
  auto scl_d = to_cuda_std_mdspan(scl);
  auto F = math::detail::splitDmatrix_impl<decltype(A_d),decltype(B_d),decltype(C_d),decltype(res_d),decltype(scl_d)>{A_d,B_d,C_d,res_d,scl_d};

  cub::DeviceFor::Bulk(N,F);
}
  
using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

template void splitDmatrix_impl(
    device_array_view<const std::complex<double>,2,basic_layout_t<2>> const&,
    device_array_view<std::complex<double>,2,basic_layout_t<2>>&,
    device_array_view<std::complex<double>,2,basic_layout_t<2>>&,
    device_array_view<std::complex<double>,1,basic_layout_t<1>>&,
    device_array_view<const std::complex<double>,1,basic_layout_t<1>> const&);


} //kernels::device::detail
