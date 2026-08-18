
#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

// Device kernels for the SAFIRE StochasticWfn inner-ensemble reductions (PR-3 GPU port). These implement
// the per-outer-walker (per-row) scaled reductions that nda forbids as expression templates on device
// arrays. All arrays are contiguous C-layout; the row index w is the slow (outer) index.
//   row_accumulate(S, In, E) : E[w, :] += S[w] * In[w, :]          (S rank-1 [nwalk], In/E rank-2 [nwalk, ncol])
//   row_divide(D, E)         : E[w, :]  = E[w, :] / D[w]           (D rank-1, E rank-2)
//   inner_scalar_reduce(...) : per w -> lin=exp(ov[w]); s=lin/|lin| (0 if |lin|==0);
//                              Sp[w]=s; Ov[w]+=inv_P*lin; D[w]+=s   (non-leapfrog, non-log-aggregate path;
//                              leapfrog/log-aggregate require inner_nsteps>0, which is CPU-only on device)
//   elementwise_log(A)       : A[w] = log(A[w])                    (rank-1)

namespace kernels::device
{

namespace detail
{

template<typename VS, typename MI, typename ME>
void row_accumulate_impl(VS const& s, MI const& in, ME& e);

template<typename VD, typename ME>
void row_divide_impl(VD const& d, ME& e);

template<typename VOVc, typename VSP, typename VOV, typename VD>
void inner_scalar_reduce_impl(VOVc const& ov, VSP& sp, VOV& Ov, VD& D, double inv_P);

template<typename V>
void elementwise_log_impl(V& a);

template<typename MT, typename MK>
void diag_trace_accumulate_impl(MT const& T, MK& K);

} // namespace detail

/*  E[w, :] += S[w] * In[w, :]   (per-row scaled accumulate) */
template<nda::MemoryArray VS_t, nda::MemoryArray MI_t, nda::MemoryArray ME_t>
void row_accumulate(VS_t const& S, MI_t const& In, ME_t&& E)
{
  static_assert(nda::get_rank<VS_t> == 1 and nda::get_rank<MI_t> == 2 and nda::get_rank<ME_t> == 2,
                "row_accumulate: expected rank-1 scale vector and rank-2 matrices");
  sfqmc::utils::check(In.shape() == E.shape() and S.extent(0) == In.extent(0),
                      "row_accumulate: shape mismatch");
  auto S_b  = to_basic_layout(S());
  auto In_b = to_basic_layout(In());
  auto E_b  = to_basic_layout(E());
  detail::row_accumulate_impl(S_b, In_b, E_b);
}

/*  E[w, :] = E[w, :] / D[w]     (per-row divide) */
template<nda::MemoryArray VD_t, nda::MemoryArray ME_t>
void row_divide(VD_t const& D, ME_t&& E)
{
  static_assert(nda::get_rank<VD_t> == 1 and nda::get_rank<ME_t> == 2,
                "row_divide: expected rank-1 divisor vector and rank-2 matrix");
  sfqmc::utils::check(D.extent(0) == E.extent(0), "row_divide: shape mismatch");
  auto D_b = to_basic_layout(D());
  auto E_b = to_basic_layout(E());
  detail::row_divide_impl(D_b, E_b);
}

/*  Per-walker phase reduction (non-leapfrog, non-log-aggregate). */
template<nda::MemoryArray VOVc_t, nda::MemoryArray VSP_t, nda::MemoryArray VOV_t, nda::MemoryArray VD_t>
void inner_scalar_reduce(VOVc_t const& ov, VSP_t&& Sp, VOV_t&& Ov, VD_t&& D, double inv_P)
{
  static_assert(nda::get_rank<VOVc_t> == 1 and nda::get_rank<VSP_t> == 1 and nda::get_rank<VOV_t> == 1 and
                    nda::get_rank<VD_t> == 1,
                "inner_scalar_reduce: expected rank-1 vectors");
  sfqmc::utils::check(ov.extent(0) == Sp.extent(0) and ov.extent(0) == Ov.extent(0) and
                          ov.extent(0) == D.extent(0),
                      "inner_scalar_reduce: shape mismatch");
  auto ov_b = to_basic_layout(ov());
  auto Sp_b = to_basic_layout(Sp());
  auto Ov_b = to_basic_layout(Ov());
  auto D_b  = to_basic_layout(D());
  detail::inner_scalar_reduce_impl(ov_b, Sp_b, Ov_b, D_b, inv_P);
}

/*  A[w] = log(A[w])   (elementwise complex log) */
template<nda::MemoryArray V_t>
void elementwise_log(V_t&& A)
{
  static_assert(nda::get_rank<V_t> == 1, "elementwise_log: expected rank-1 vector");
  auto A_b = to_basic_layout(A());
  detail::elementwise_log_impl(A_b);
}

/*  Kl[w, c] += sum_a T4D[w, a, a, c]   (per-walker diagonal trace over the two orbital axes; EJ Coulomb
    vector for the un-rotated full-G energy). T4D is [nwalk, NMO, NMO, nCV], Kl is [nwalk, nCV]. */
template<nda::MemoryArray MT_t, nda::MemoryArray MK_t>
void diag_trace_accumulate(MT_t const& T4D, MK_t&& Kl)
{
  static_assert(nda::get_rank<MT_t> == 4 and nda::get_rank<MK_t> == 2,
                "diag_trace_accumulate: expected rank-4 T4D and rank-2 Kl");
  sfqmc::utils::check(T4D.extent(0) == Kl.extent(0) and T4D.extent(3) == Kl.extent(1) and
                          T4D.extent(1) == T4D.extent(2),
                      "diag_trace_accumulate: shape mismatch");
  auto T_b = to_basic_layout(T4D());
  auto K_b = to_basic_layout(Kl());
  detail::diag_trace_accumulate_impl(T_b, K_b);
}

} // namespace kernels::device
