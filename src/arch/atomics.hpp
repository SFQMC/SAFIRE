#pragma once
#include<atomic>
#include <complex>
#include "host_device.h"

// wrappers for atomic that support complex numbers and are device agnostic
namespace sfqmc::arch {

namespace detail {
// make public to replace is_complex_v?
template<class T>
  concept ComplexLike = requires(T t) {
    typename T::value_type;
    { t.real() } -> std::convertible_to<typename T::value_type>;
    { t.imag() } -> std::convertible_to<typename T::value_type>;
};
}

template<typename T>
__host__ __device__
void atomic_add(T* dest, std::type_identity_t<T> a) {
  if constexpr (detail::ComplexLike<T>) {
    auto* p = reinterpret_cast<T::value_type*>(dest);
    atomic_add(&p[0], a.real());
    atomic_add(&p[1], a.imag());
  } else {
#if defined(__CUDA_ARCH__)
  atomicAdd(dest, a);
#else
  std::atomic_ref<T>(*dest).fetch_add(a, std::memory_order_relaxed);
#endif
  }
}

}
