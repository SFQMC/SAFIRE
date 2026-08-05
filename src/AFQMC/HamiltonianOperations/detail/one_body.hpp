#pragma once

#include <utility>

#include "AFQMC/config.h"
#include "AFQMC/Utilities/wfn_utils.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"
#include "numerics/shared_array/const_shared_array.hpp"
#include "nda/tensor.hpp"

namespace sfqmc::afqmc {

// broadcasts a onebody operator `src` of shape [nspin][npol][NMO][npol][NMO] to `dest` with shape [nspin'][npol'][NMO][npol'][NMO]
// while taking the proper spin symmetry into account. Only conversions to lesser symmetries are allowed (closed -> collinear -> noncollinear)
void broadcast_one_body(nda::ArrayOfRank<5> auto const& src, nda::ArrayOfRank<5> auto&& dest) {
  int src_nspin = src.extent(0);
  int src_npol = src.extent(1);
  int dest_nspin = dest.extent(0);
  int dest_npol = dest.extent(1);
  int NMO = src.extent(2);

  WALKER_TYPES src_walker_type = walkerTypeFromDims(src_nspin, src_npol);
  WALKER_TYPES dest_walker_type = walkerTypeFromDims(dest_nspin, dest_npol);
  
  utils::check(walkerTypeIsConvertible(src_walker_type, dest_walker_type),
               "Tried disallowed conversion from walker_type {} to {}",
               walkerTypeToString(src_walker_type),
               walkerTypeToString(dest_walker_type));

  utils::check_shape(src, "src", src_nspin, src_npol, NMO, src_npol, NMO); 
  utils::check_shape(dest, "dest", dest_nspin, dest_npol, NMO, dest_npol, NMO); 
  
  auto all = nda::range::all;

  dest() = 0;

  if(src_walker_type != NONCOLLINEAR && dest_walker_type == NONCOLLINEAR) {
    for(int pol1 = 0; pol1 < dest_npol; pol1++) {
        dest(0, pol1, all, pol1, all) = src(pol1%src_nspin, 0, all, 0, all);
    }
  } else {
    for(int spin = 0; spin < dest_nspin; spin++) {
      int spin_ = spin % src_nspin;
      for(int pol1 = 0; pol1 < dest_npol; pol1++) {
        int pol1_ = pol1 % src_npol;
        for(int pol2 = 0; pol2 < dest_npol; pol2++) {
          int pol2_ = pol2 % src_npol;
          dest(spin, pol1, all, pol2, all) = src(spin_, pol1_, all, pol2_, all);
        }
      }
    }
  }
}

// Maps a walker (spin is, pol ip) to the (spin, pol) block of an interaction tensor stored
// with native symmetry (nspin_in_H, npol_in_H). Folds collinear spin blocks into
// noncollinear polarization blocks, consistent with broadcast_one_body() above.
inline std::array<int,2> interaction_block(int is, int ip, int npol, int nspin_in_H, int npol_in_H) {
  int c = (is*npol + ip) % (nspin_in_H * npol_in_H);
  return {c / npol_in_H, c % npol_in_H};
}

// comuptes H' = H + meanfield_shift + vexx. The meanfield shift was created as vHS(vMF) so it already includes dt.
void add_one_body_shifts(RealType dt, nda::ArrayOfRank<3> auto const& hij, nda::ArrayOfRank<4> auto const& meanfield_shift, nda::ArrayOfRank<3> auto const& vexx, nda::ArrayOfRank<5> auto&& dest) {
  auto all = nda::range::all;
  int nspin = dest.extent(0);
  int npol = dest.extent(1);
  int NMO = dest.extent(2);

  // the strange amalgamation of nspin and npol in the interaction
  int nspinpol_inter = vexx.extent(0);

  utils::check_shape(hij, "hij", nspin, npol*NMO, npol*NMO);
  utils::check_shape(vexx, "vexx", nspinpol_inter, NMO, NMO);
  utils::check_shape(dest, "dest", nspin, npol, NMO, npol, NMO);

  reshape(dest(), nspin, npol*NMO, npol*NMO) = dt * hij;
  auto v = nda::reshape(meanfield_shift, nspinpol_inter, NMO, NMO);
  for(int is = 0; is < nspin; is++) {
    for(int p1 = 0; p1 < npol; p1++) {
      dest(is, p1, all, p1, all) += dt * vexx((is * npol + p1) % nspinpol_inter, all, all) + v((is * npol + p1) % nspinpol_inter, all, all);
    }
  }
}

void kpoint_add_one_body_shifts(RealType dt, nda::ArrayOfRank<4> auto const& hij, nda::ArrayOfRank<4> auto const& meanfield_shift, nda::ArrayOfRank<4> auto const& vexx, nda::ArrayOfRank<7> auto&& dest) {
  auto all = nda::range::all;
  int nspin = dest.extent(0);
  int npol = dest.extent(1);
  int nkpts = dest.extent(2);
  int nbnd = dest.extent(3);
  int NMO = nkpts * nbnd;

  // the strange amalgamation of nspin and npol in the interaction
  int nspinpol_inter = vexx.extent(0);
  
  utils::check_shape(hij, "hij", nspin, nkpts, npol*nbnd, npol*nbnd);
  utils::check_shape(vexx, "vexx", nspinpol_inter, nkpts, nbnd, nbnd);
  utils::check_shape(dest, "dest", nspin, npol, nkpts, nbnd, npol, nkpts, nbnd);
  
  
  // hij is k diagonal, but pol off-diagonal
  auto hij6 = reshape(hij, nspin, nkpts, npol, nbnd, npol, nbnd);
  for(int ik = 0; ik < nkpts; ik++) {
    dest(all, all, ik, all, all, ik, all) += dt * hij6(all, ik, all, all, all, all);
  }

  auto v = nda::reshape(meanfield_shift, nspinpol_inter, NMO, NMO);
  for(int is = 0; is < nspin; is++) {
    for(int p1 = 0; p1 < npol; p1++) {
      // meanfield_shift is k off-diagonal but pol diagonal 
      nda::reshape(dest, nspin, npol, NMO, npol, NMO)(is, p1, all, p1, all) += v((is * npol + p1) % nspinpol_inter, all, all); 
    
      for(int ik = 0; ik < nkpts; ik++) {
        // vexx is diagonal in both
        dest(is, p1, ik, all, p1, ik, all) += dt * vexx((is * npol + p1) % nspinpol_inter, ik, all, all);
      }
    }
  }
}

template<MEMORY_SPACE MEM, std::size_t N>
auto half_rotate_hamiltonian(utils::mpi_context_t<mpi3::communicator>& mpi, std::array<long,N> nel, int nspin, int npol, int nspin_in_H1, int npol_in_H1, int NMO, nda::ArrayOfRank<2> auto const &PsiT, nda::ArrayOfRank<3> auto const& H1) {
  using nda::range;
  auto all = range::all;
  int ndet = PsiT.extent(0);

  memory::const_shared_array<MEM, ComplexType, 3> haj;
  {
    // temporary broadcast H1 to walker_type
    auto hc = memory::share_from_root(mpi, [&]() {
      memory::array<MEM, ComplexType, 5> hc{nspin, npol, NMO, npol, NMO};
      broadcast_one_body(nda::reshape(H1(), nspin_in_H1, npol_in_H1, NMO, npol_in_H1, NMO), hc);
      return hc;
    });
  
    haj = std::move(memory::share_from_ranks<MEM,ComplexType,3,1>(mpi,
        {ndet, nel[0]+nel[1], npol*NMO},
        [&](std::array<long,1> idx, auto&& block) {
      auto [id] = idx;
    
      for(long is=0; is<nspin; ++is) {
        auto Aai = math::sparse::to_array<'N'>(PsiT(id,is));
        auto h_ = block(range(is*nel[0],nel[0]+is*nel[1]),all);
        nda::blas::gemm(Aai,nda::reshape(hc(), nspin, npol*NMO, npol*NMO)(is, all, all), h_);
      }
    }));
  }
  mpi.shared_windows.collective_free_unused();
  return haj;
}

// Half-rotates the KP momentum-transfer Cholesky factors LQ by the trial PsiT,
// producing (Lank, Lbnk) in the layout KP3IndexFactorization expects. Shared by
// KPFactorizedHamiltonian and the unit tests so there is a single implementation.
//   Lank(Q)(idet,ispin,ik,a,n,ip*nbnd+j) = sum_i PsiT(idet,ispin)(a, ip*NMO+ik*nbnd+i)
//                                                 * LQ(Q)(ispin*ip,ik,i,j,n)
// LQ is stored only for Q<=minusq(Q); Lbnk holds the self-inverse (Q==minusq(Q))
// blocks, indexed by Qmap(Q). nspin_in_H1/npol_in_H1 are the Hamiltonian's intrinsic
// spin/pol dims, which can differ from the walker's (nspin,npol) derived from `type`.
template<MEMORY_SPACE MEM>
std::pair<nda::array<memory::const_shared_array<MEM,ComplexType,6>,1>,
          nda::array<memory::const_shared_array<MEM,ComplexType,6>,1>>
kpoint_half_rotate_cholesky(utils::mpi_context_t<mpi3::communicator>& mpi,
    WALKER_TYPES type, int nspin_in_H1, int npol_in_H1, long NMO, int nbnd,
    nda::array<PsiT_Matrix<MEM>,2> const& PsiT,
    nda::array<nda::array<int,1>,2> const& nocc,
    nda::array<int,1> const& minusq, nda::array<int,2> const& qk_to_k2,
    nda::array<int,1> const& Qmap, nda::array<int,1> const& nchol,
    nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> const& LQ) {
  using nda::range;
  auto all = range::all;
  ComplexType zero(0.0), one(1.0);
  long nspin = (type == COLLINEAR ? 2 : 1);
  long npol  = (type == NONCOLLINEAR ? 2 : 1);
  long ndet  = PsiT.extent(0);
  int nkpts  = nocc.extent(1);
  int nocc_max = max_nocc_per_kpoint(nocc);
  int number_of_symmetric_Q = 0;
  for(int Q=0; Q<nkpts; ++Q) {
    if(Q == minusq(Q)) {
      ++number_of_symmetric_Q;
    }
  }

  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> Lank(nkpts);
  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> Lbnk(number_of_symmetric_Q);
  for(int Q=0; Q<nkpts; ++Q) {
    int Qm = minusq(Q);
    Lank(Q) = memory::share_from_ranks<MEM,ComplexType,6,3>(mpi,
        {ndet,nspin,nkpts,nocc_max,nchol(Q),npol*nbnd},
        [&](std::array<long,3> idx, auto&& block) {
      auto [id,is,ik] = idx;
      auto const& rows = nocc(is,ik);
      int nk = int(rows.size());
      for(long ip=0; ip<npol; ++ip) {
        auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H1,npol_in_H1);
        if(Q <= Qm) {
          // L[Q,k,k2=k-Q]
          auto Aai = math::sparse::to_array<'N'>(PsiT(id,is),rows,
                                                 range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
          auto Lijn = LQ(Q)()(is_,ip_,ik,all,all,all);
          auto L_ = block(range(nk),all,range(ip*nbnd,(ip+1)*nbnd));
          utils::check(Lijn.extent(2)==L_.extent(1), "Size mismatch.");
          nda::tensor::contract(one,Aai,"ai",Lijn,"ijn",zero,L_,"anj");
        } else {
          // L[Q,k,k2=k-Q]
          int k2 = qk_to_k2(Q,ik);
          auto Abj = math::sparse::to_array<'N'>(PsiT(id,is),rows,
                                                 range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
          auto Lljn = LQ(Qm)()(is_,ip_,k2,all,all,all);
          auto L_ = block(range(nk),all,range(ip*nbnd,(ip+1)*nbnd));
          utils::check(Lljn.extent(2)==L_.extent(1), "Size mismatch.");
          nda::tensor::contract(one,Abj,"bj",nda::conj(Lljn),"ljn",zero,L_,"bnl");
        }
      } // ip
    });
    if(Q==Qm) {
      // Lbnk is indexed by k2 = qk_to_k2(Q,ik): invert the map so the fill for
      // item k2 knows the source kpoint ik of LQ
      nda::array<int,1> k2_to_k(nkpts);
      for(int ik=0; ik<nkpts; ++ik) {
        k2_to_k(qk_to_k2(Q,ik)) = ik;
      }
      Lbnk(Qmap(Q)) = memory::share_from_ranks<MEM,ComplexType,6,3>(mpi,
          {ndet,nspin,nkpts,nocc_max,nchol(Q),npol*nbnd},
          [&](std::array<long,3> idx, auto&& block) {
        auto [id,is,k2] = idx;
        int ik = k2_to_k(k2);
        auto const& rows = nocc(is,k2);
        int nb = int(rows.size());
        for(long ip=0; ip<npol; ++ip) {
          auto [is_,ip_] = interaction_block(is,ip,npol,nspin_in_H1,npol_in_H1);
          // conj(L[Q,k,k2](lj,n)) * A[k2]bj
          auto Abj = math::sparse::to_array<'N'>(PsiT(id,is),rows,
                                                 range(ip*NMO+k2*nbnd,ip*NMO+(k2+1)*nbnd));
          auto Lljn = LQ(Q)()(is_,ip_,ik,all,all,all);
          auto L_ = block(range(nb),all,range(ip*nbnd,(ip+1)*nbnd));
          utils::check(Lljn.extent(2)==L_.extent(1), "Size mismatch.");
          nda::tensor::contract(one,Abj,"bj",nda::conj(Lljn),"ljn",zero,L_,"bnl");
        }
      });
    }
  }  // Q
  return {std::move(Lank), std::move(Lbnk)};
}

}
