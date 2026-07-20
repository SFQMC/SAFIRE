
#pragma once

#include <complex>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace kernels::device
{

namespace detail
{

template<typename T_t, typename Ov_t> 
void phmsd_det_impl(int nex, int const* confg, T_t const& T, Ov_t& ov); 

template<typename T_t, typename R_t>
void phmsd_compact_R_impl(int nex, int const* refc, int const* iex, T_t const& T, R_t &R);

template<typename W_t, typename Rb_t, typename R_t>
void phmsd_reduce_R_impl(int nex, int const* refc, int const* iex, W_t const& wgt, Rb_t const& Rbuff, R_t &R); 

template<typename S_t, typename R_t, typename W_t, typename E_t>
void ph_excited_1body_energy_impl(int const* refc, int const* iexcit, S_t const& S, R_t const& R, W_t const& w, E_t &E);

template<typename T_t, typename R_t, typename W_t, typename EX_t, typename EJ_t, typename KE_t>
void ph_excited_2body_energy_dense_cholesky_Tpna_impl(int const* refc, int const* iexcit, T_t const& T, R_t const& R, W_t const& w, EX_t &EX, EJ_t& EJ, KE_t &KE);

template<typename O_t, typename G_t>
void vmf_offdiag_impl(long total_pairs, int nrows, int rank, int size, int nelec,
                      long const* pair_off,
                      int const* coup_rbegin, int const* coup_rend, int const* coup_jdet,
                      std::complex<double> const* coup_val,
                      int const* configs, O_t const& O, G_t& G);

}

/*
 *   Evaluates phmsd determinants 
 */
template<typename abij_t, nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat>
void calculate_overlaps(int spin, abij_t& abij,  T_t const& T, Mat&& ov)
{
  using nda::range;
  auto all = range::all;
  // ov(ndet,nwalk)
  // T(nw,nact,nel)
  sfqmc::utils::check(ov.extent(1) == T.extent(0), "Shape mismatch");
  auto T_b = to_basic_layout(T());

  for (int nex = 1, idet = 1; nex < abij.maximum_excitation_number()[spin]; nex++) {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex); 
      auto ov_b = to_basic_layout(ov(range(idet,idet+ndet),all));
      detail::phmsd_det_impl(nex,iexcit.data(),T_b,ov_b);
      idet += ndet;
    }
  }
}

/*
 *   Evaluates R matrix 
 */
template<typename abij_t, nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat, nda::MemoryArrayOfRank<3> R_t>
void calculate_R(int spin, abij_t& abij,  T_t const& T, Mat const& wgt, R_t &R)
{
  using nda::range;
  auto all = range::all;
  // T(nw,nact,nel)
  // weights(nd,nw)
  // R(nw,nel,nact)
  auto [nwalk,nact,nel] = T.shape(); 
  sfqmc::utils::check(wgt.extent(1) == nwalk, "Shape mismatch");
  sfqmc::utils::check(R.extent(0) == nwalk, "Shape mismatch");
  sfqmc::utils::check(R.extent(2) == nact, "Shape mismatch");
  sfqmc::utils::check(T.extent(2) == R.extent(1), "Shape mismatch");
  auto T_b = to_basic_layout(T());
  auto R_b = to_basic_layout(R());

  auto refc=abij.get_reference_configuration_device(spin);
  auto refc_h = nda::to_host(refc);

  // add ontribution from reference determinant
  // R[w][i][ refc[i] ] += weights[w]
  for(int i=0; i<nel; ++i)
    nda::tensor::add(ComplexType(1.0),wgt(0,all),"w",ComplexType(1.0),R(all,i,refc_h(i)),"w");

  // add contributions from ph expansion
  for (int nex = 1, idet = 1; nex < abij.maximum_excitation_number()[spin]; nex++) {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      auto wgt_b = to_basic_layout(wgt(range(idet,idet+ndet),all));
      // - loop over blocks of determinants to control memory usage
      memory::buffered_array<DEVICE_MEMORY,ComplexType,4> Rbuff(nwalk,ndet,nex,nact);
      auto Rbuff_b = to_basic_layout(Rbuff());
      detail::phmsd_compact_R_impl(nex,refc.data(),iexcit.data(),T_b,Rbuff_b);
      detail::phmsd_reduce_R_impl(nex,refc.data(),iexcit.data(),wgt_b,Rbuff_b,R_b);
      idet += ndet;
    }
  }
}

/*
 *   Evaluates R matrix 
 */
template<nda::MemoryArrayOfRank<3> T_t, nda::MemoryArrayOfRank<4> R_t>
void calculate_compact_ph_R(int const* refc, int const* iexcit, T_t const& T, R_t &R)
{
  using nda::range;
  // T(nw,nact,nel)
  // R(nw,ndet,nex,nact)
  auto [nwalk,ndet,nex,nact] = R.shape(); 
  if(ndet > 0) {
    // - loop over blocks of determinants to control memory usage
    auto T_b = to_basic_layout(T());
    auto R_b = to_basic_layout(R());
    detail::phmsd_compact_R_impl(nex,refc,iexcit,T_b,R_b);
  }
}

/*
 *   Accumulate contributions to the 1-body energy from determinants in given excitation shell 
 */
template<nda::MemoryArrayOfRank<3> S_t, nda::MemoryArrayOfRank<4> R_t, nda::MemoryArrayOfRank<2> W_t, nda::MemoryArrayOfRank<1> E_t>
void ph_excited_1body_energy(int const* refc, int const* iexcit, S_t const& S, R_t const& R, W_t const& w, E_t &E)
{
  auto S_b = to_basic_layout(S());
  auto R_b = to_basic_layout(R());
  auto w_b = to_basic_layout(w());
  auto E_b = to_basic_layout(E());
  detail::ph_excited_1body_energy_impl(refc,iexcit,S_b,R_b,w_b,E_b);
}

template<nda::MemoryArrayOfRank<4> T_t, nda::MemoryArrayOfRank<4> R_t, nda::MemoryArrayOfRank<2> W_t, nda::MemoryArrayOfRank<1> EX_t, nda::MemoryArrayOfRank<1> EJ_t, nda::MemoryArrayOfRank<3> KE_t>
void ph_excited_2body_energy_dense_cholesky_Tpna(int const* refc, int const* iexcit, T_t const& T, R_t const& R, W_t const& w, EX_t &EX, EJ_t& EJ, KE_t &KE)
{
  auto T_b = to_basic_layout(T());
  auto R_b = to_basic_layout(R());
  auto w_b = to_basic_layout(w());
  auto EX_b = to_basic_layout(EX());
  auto EJ_b = to_basic_layout(EJ());
  auto KE_b = to_basic_layout(KE());
  detail::ph_excited_2body_energy_dense_cholesky_Tpna_impl(refc,iexcit,T_b,R_b,w_b,EX_b,EJ_b,KE_b);
}

/*
 *   Off-diagonal contribution to the PHMSD mean-field Green's function (PHMSD::vMF).
 *   Replaces the O(sum nnz^2) host loop over excitation pairs with a GPU kernel.
 *     coup_(indptr,jdet,val) : device CSR of OpSpinDetCouplings[other_spin]
 *     configs                : device sorted occ-orbital lists [nconfig][nelec] for `spin`
 *     O                      : dense orbitals (npol*NMO, nact) for `spin`
 *     G                      : output Green's function slice (nact, npol*NMO)
 */
template<nda::MemoryArrayOfRank<2> O_t, nda::MemoryArrayOfRank<2> G_t>
void vmf_offdiag(long total_pairs, int nrows, int rank, int size, int nelec,
                 long const* pair_off,
                 int const* coup_rbegin, int const* coup_rend, int const* coup_jdet,
                 std::complex<double> const* coup_val,
                 int const* configs, O_t const& O, G_t& G)
{
  auto O_b = to_basic_layout(O());
  auto G_b = to_basic_layout(G());
  detail::vmf_offdiag_impl(total_pairs, nrows, rank, size, nelec, pair_off, coup_rbegin, coup_rend, coup_jdet, coup_val, configs, O_b, G_b);
}

} // namespace kernels::device

