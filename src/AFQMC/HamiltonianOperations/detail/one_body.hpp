#pragma once

#include "AFQMC/config.h"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"
#include "numerics/shared_array/const_shared_array.hpp"

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

}
