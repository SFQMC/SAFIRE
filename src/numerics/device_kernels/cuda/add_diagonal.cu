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
#include "numerics/operations/add_diagonal_impl.hpp"

namespace kernels::device::detail
{

template<typename V1, typename V2> 
//void add_diagonal_impl(V1 alpha, V2 const& A)
void add_diagonal_impl(V1 alpha, V2& A)
{
  long N = A.extent(0);
  if(N==0) return;
  auto alpha_d = cuda_std_value_cast(alpha);
  auto A_d = to_cuda_std_mdspan(A);

  auto F = math::detail::add_diagonal_impl<decltype(alpha_d),decltype(A_d)>{alpha_d,A_d};

  cub::DeviceFor::Bulk(N,F);
}
  
using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

template void add_diagonal_impl(std::complex<double>,
    device_array_view<std::complex<double>,3,basic_layout_t<3>> & );


} //kernels::device::detail
