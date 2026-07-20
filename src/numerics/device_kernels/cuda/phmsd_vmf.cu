#include <complex>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/arch.h"
#include "arch/atomics.hpp"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include <cuda/std/complex>

namespace kernels::device::detail
{

// GPU evaluation of the off-diagonal contribution to the PHMSD mean-field Green's
// function (PHMSD::vMF). Replaces the O(sum nnz^2) host loop over excitation pairs.
//
// Load-balanced formulation: all upper-triangular pairs (n1<n2) of every coupling row
// are flattened into a single global index space and processed with a grid-stride loop,
// so the work is spread evenly across the whole device (the per-row nnz is highly skewed
// -- a few reference configurations couple to thousands of others -- so one-block-per-row
// starves the GPU). For a global pair index gp:
//   * binary-search the per-row pair prefix-sum `pair_off` to find the coupling row nd,
//   * triangular-unrank the local pair index to (n1,n2),
//   * merge the two sorted configs; if they differ by a single orbital (r removed, in c1
//     not c2; a added, in c2 not c1) accumulate
//         G[a][:] += w        * conj( Orbs[spin][:,r] )
//         G[r][:] += conj(w)  * conj( Orbs[spin][:,a] )
//     with w = coup_val[n1] * conj(coup_val[n2]). Sign ignored (as in the host code).
//
//   pair_off : prefix sum of nnz_nd*(nnz_nd-1)/2, length nrows+1 (device).
//   configs  : sorted occupied-orbital lists, row-major [nconfig][nelec] (device).
//   O        : dense orbitals view, shape (npol*NMO, nact).
//   G        : output Green's function slice, shape (nact, npol*NMO).
template<typename O_t, typename G_t>
__global__ void kernel_vmf_offdiag(long total_pairs, int nrows, int rank, int size, int nelec,
                                   long const* pair_off,
                                   int const* coup_rbegin, int const* coup_rend,
                                   int const* coup_jdet,
                                   cuda::std::complex<double> const* coup_val,
                                   int const* configs, O_t const O, G_t G)
{
  using Type = cuda::std::complex<double>;
  long ncol = long(G.extent(1)); // npol*NMO
  long gid    = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (long gp = gid; gp < total_pairs; gp += stride)
  {
    // locate coupling row nd: largest nd with pair_off[nd] <= gp
    int lo = 0, hi = nrows;
    while (lo + 1 < hi)
    {
      int mid = (lo + hi) >> 1;
      if (pair_off[mid] <= gp) lo = mid; else hi = mid;
    }
    int nd = lo;
    if (nd % size != rank)
      continue;
    int  b   = coup_rbegin[nd];
    long nnz = long(coup_rend[nd] - b);
    long lp  = gp - pair_off[nd]; // local pair index in [0, nnz*(nnz-1)/2)
    // triangular unrank lp -> (n1,n2), n1<n2 ; base(k) = k*nnz - k*(k+1)/2
    double d = 2.0 * double(nnz) - 1.0;
    d = d * d - 8.0 * double(lp);
    if (d < 0.0) d = 0.0;
    long n1 = long((2.0 * double(nnz) - 1.0 - sqrt(d)) * 0.5);
    if (n1 < 0) n1 = 0;
    if (n1 > nnz - 2) n1 = nnz - 2;
    while (n1 > 0 && (n1 * nnz - n1 * (n1 + 1) / 2) > lp) --n1;
    while ((n1 + 1) * nnz - (n1 + 1) * (n1 + 2) / 2 <= lp) ++n1;
    long base = n1 * nnz - n1 * (n1 + 1) / 2;
    long n2 = lp - base + n1 + 1;

    int const* c1 = configs + (long)coup_jdet[b + n1] * nelec;
    int const* c2 = configs + (long)coup_jdet[b + n2] * nelec;
    // merge two sorted configs -> removed (in c1 not c2), added (in c2 not c1)
    int ia = 0, ib = 0, nrem = 0, nadd = 0, removed = -1, added = -1;
    while (ia < nelec && ib < nelec)
    {
      int x = c1[ia], y = c2[ib];
      if (x == y) { ia++; ib++; }
      else if (x < y) { removed = x; nrem++; ia++; if (nrem > 1) break; }
      else { added = y; nadd++; ib++; if (nadd > 1) break; }
    }
    if (nrem > 1 || nadd > 1)
      continue;
    nrem += (nelec - ia);
    nadd += (nelec - ib);
    if (nrem != 1 || nadd != 1)
      continue;
    if (ia < nelec) removed = c1[ia];
    if (ib < nelec) added = c2[ib];
    int r = removed, a = added;
    Type w  = coup_val[b + n1] * cuda::std::conj(coup_val[b + n2]);
    Type wc = cuda::std::conj(w);
    for (long j = 0; j < ncol; ++j)
      sfqmc::arch::atomic_add(&G(a, j), w * cuda::std::conj(O(j, r)));
    for (long j = 0; j < ncol; ++j)
      sfqmc::arch::atomic_add(&G(r, j), wc * cuda::std::conj(O(j, a)));
  }
}

template<typename O_t, typename G_t>
void vmf_offdiag_impl(long total_pairs, int nrows, int rank, int size, int nelec,
                      long const* pair_off,
                      int const* coup_rbegin, int const* coup_rend, int const* coup_jdet,
                      std::complex<double> const* coup_val,
                      int const* configs, O_t const& O, G_t& G)
{
  if (total_pairs <= 0)
    return;
  auto O_d = to_cuda_std_mdspan(O);
  auto G_d = to_cuda_std_mdspan(G);
  int  nthreads = 256;
  long nblk = (total_pairs + nthreads - 1) / nthreads;
  if (nblk > 200000) nblk = 200000; // cap; grid-strides over the remainder
  dim3 grid((unsigned)nblk, 1, 1);
  kernel_vmf_offdiag<<<grid, nthreads>>>(total_pairs, nrows, rank, size, nelec,
                                         pair_off, coup_rbegin, coup_rend, coup_jdet,
                                         cuda_std_ptr_cast(coup_val),
                                         configs, O_d, G_d);
  sfqmc::arch::synchronize_if_set();
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T, V) \
template void vmf_offdiag_impl(long, int, int, int, int, long const*, int const*, int const*, int const*, T const*, int const*, \
    V<const T, 2, basic_layout_t<2>> const&, V<T, 2, basic_layout_t<2>>&);

_inst_(std::complex<double>, device_array_view)

} // namespace kernels::device::detail
