#include "numerics/device_kernels/cuda/launch.cuh"
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/operations/determinants_impl.hpp"

namespace kernels::device
{

void log_determinant_from_getrf(view<std::complex<double> const, 3> a, view<int const, 2> pivot,
                                view<std::complex<double>, 1> res)
{
  math::detail::log_determinant_from_getrf_impl<decltype(a), decltype(pivot), decltype(res)> f{
      a, pivot, res};
  bulk(a.extent(0), f);
}

void log_determinant_from_geqrf(view<std::complex<double> const, 3> a,
                                view<std::complex<double>, 2> scl,
                                view<std::complex<double>, 1> res)
{
  math::detail::log_determinant_from_geqrf_impl<decltype(a), decltype(scl), decltype(res)> f{a, scl,
                                                                                            res};
  bulk(a.extent(0), f);
}

} // namespace kernels::device
