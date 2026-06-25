////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
////////////////////////////////////////////////////////////////////////////////

// Helpers for the self-contained HamiltonianOperations cross-implementation test.
//
// Builds a 2D Hubbard model directly in memory and produces every
// HamiltonianOperations variant of the same physical Hamiltonian via its
// lowest-level constructor. The Cholesky / THC tensors are hand-derived from
// the diagonal structure of the onsite Hubbard interaction.
//
// Two independent axes are spanned:
//   * the intrinsic spin symmetry of the one-body matrix T (`h_symmetry`):
//       - CLOSED       : T is (1, NMO, NMO)         -> nspin_in_H=1, npol_in_H=1
//       - COLLINEAR    : T is (2, NMO, NMO)         -> nspin_in_H=2, npol_in_H=1
//       - NONCOLLINEAR : T is (1, 2*NMO, 2*NMO)     -> nspin_in_H=1, npol_in_H=2
//   * the target walker type (`walker_type`), any type reachable from
//     `h_symmetry` under walkerTypeIsConvertible (CLOSED<=COLLINEAR<=NONCOLLINEAR).
//
// The two-body (onsite Hubbard-U) interaction is spin-independent (closed) in
// every case; only the one-body piece carries the spin structure. The factors
// are therefore derived from the closed onsite interaction, broadcast into the
// representation required by the target walker.

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AFQMC/config.h"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/KPTHCOps.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/sparse/sparse.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc::tests {

struct HubbardSpec {
  int Lx{4};
  int Ly{4};
  RealType t{1.0};
  RealType U{4.0};
  int nup{4};
  int ndown{4};
  // Intrinsic spin symmetry of the one-body matrix T.
  WALKER_TYPES h_symmetry{CLOSED};
  // Target walker type set in the HamiltonianOperations (must be convertible
  // from h_symmetry).
  WALKER_TYPES walker_type{COLLINEAR};
  // Symmetry-breaking amplitudes used to make the non-closed T genuinely
  // distinct from a closed one (so the new code paths are actually exercised):
  //   dz: spin-dependent onsite potential added to the beta block (COLLINEAR T)
  //   hx: transverse (spin-flip) onsite hopping (NONCOLLINEAR T)
  RealType dz{0.5};
  RealType hx{0.3};

  int NMO() const { return Lx * Ly; }

  // intrinsic Hamiltonian spin structure
  int nspin_in_H() const { return (h_symmetry == COLLINEAR) ? 2 : 1; }
  int npol_in_H() const { return (h_symmetry == NONCOLLINEAR) ? 2 : 1; }

  // walker spin structure
  int walker_nspin() const { return (walker_type == COLLINEAR) ? 2 : 1; }
  int walker_npol() const { return (walker_type == NONCOLLINEAR) ? 2 : 1; }

  // electron counts handed to the variant constructors. NONCOLLINEAR carries
  // every electron in the spinor "alpha" channel with ndown==0.
  int walker_nup() const { return (walker_type == NONCOLLINEAR) ? (nup + ndown) : nup; }
  int walker_ndown() const { return (walker_type == COLLINEAR) ? ndown : 0; }
  int walker_nel() const { return walker_nup() + walker_ndown(); }

  // electrons in walker-spin channel `is` (only is==0 for CLOSED/NONCOLLINEAR)
  int nel_for_spin(int is) const { return is == 0 ? walker_nup() : walker_ndown(); }
};

/// Bare nearest-neighbour hopping for a 2D Hubbard model with PBC, shape (NMO,NMO).
inline nda::array<RealType, 2> base_hopping(HubbardSpec const& s) {
  int N = s.NMO();
  nda::array<RealType, 2> h(N, N);
  h() = RealType(0);
  for(int rx = 0; rx < s.Lx; ++rx) {
    for(int ry = 0; ry < s.Ly; ++ry) {
      int i  = rx * s.Ly + ry;
      int jx = ((rx + 1) % s.Lx) * s.Ly + ry;
      int jy = rx * s.Ly + ((ry + 1) % s.Ly);
      h(i, jx) -= s.t;
      h(jx, i) -= s.t;
      h(i, jy) -= s.t;
      h(jy, i) -= s.t;
    }
  }
  return h;
}

/// Intrinsic one-body matrix hij in the stored layout
/// (nspin_in_H, npol_in_H*NMO, npol_in_H*NMO):
///   CLOSED       -> (1, NMO, NMO)
///   COLLINEAR    -> (2, NMO, NMO), beta block carries a +dz Zeeman shift
///   NONCOLLINEAR -> (1, 2*NMO, 2*NMO), diagonal spin blocks = hopping,
///                   off-diagonal (spin-flip) blocks = hx * I
inline nda::array<ComplexType, 3> build_hij_intrinsic(HubbardSpec const& s) {
  using nda::range;
  auto all = range::all;
  long NMO = s.NMO();
  auto h   = base_hopping(s);
  int nsH  = s.nspin_in_H();
  int npH  = s.npol_in_H();
  nda::array<ComplexType, 3> hij(nsH, npH * NMO, npH * NMO);
  hij() = ComplexType(0);
  if(s.h_symmetry == CLOSED) {
    hij(0, all, all) = h;
  }
  else if(s.h_symmetry == COLLINEAR) {
    hij(0, all, all) = h;
    hij(1, all, all) = h;
    for(long i = 0; i < NMO; ++i) {
      hij(1, i, i) += s.dz;
    }
  }
  else { // NONCOLLINEAR
    for(int p = 0; p < 2; ++p) {
      hij(0, range(p * NMO, (p + 1) * NMO), range(p * NMO, (p + 1) * NMO)) = h;
    }
    for(long i = 0; i < NMO; ++i) {
      hij(0, i, NMO + i) += s.hx;
      hij(0, NMO + i, i) += s.hx;
    }
  }
  return hij;
}

/// Physically-correct (block-diagonal in polarization) embedding of T into the
/// walker basis, shape (walker_nspin, walker_npol*NMO, walker_npol*NMO). Used to
/// build the half-rotated one-body haj. The same haj is handed to every variant
/// so the one-body energy E1 is cross-consistent by construction.
inline nda::array<ComplexType, 3> build_T_walker(HubbardSpec const& s,
                                                 nda::array<ComplexType, 3> const& hij) {
  using nda::range;
  auto all = range::all;
  long NMO = s.NMO();
  int wns  = s.walker_nspin();
  int wnp  = s.walker_npol();
  nda::array<ComplexType, 3> T(wns, wnp * NMO, wnp * NMO);
  T() = ComplexType(0);
  if(s.walker_type == CLOSED) {
    T(0, all, all) = hij(0, all, all);
  } else if(s.walker_type == COLLINEAR) {
    for(int is = 0; is < 2; ++is) {
      T(is, all, all) = hij(is % hij.extent(0), all, all);
    }
  } else { // NONCOLLINEAR walker, block-diagonal embedding
    if(s.h_symmetry == NONCOLLINEAR) {
      T(0, all, all) = hij(0, all, all);
    } else { // CLOSED or COLLINEAR intrinsic
      T(0, range(0, NMO), range(0, NMO))             = hij(0, all, all);
      T(0, range(NMO, 2 * NMO), range(NMO, 2 * NMO)) = hij(s.h_symmetry == COLLINEAR ? 1 : 0, all, all);
    }
  }
  return T;
}

/// Deterministic orthonormal trial in the walker basis, shape
/// (walker_npol*NMO, walker_nel). Columns are an orthonormal QR factor of a
/// Hilbert matrix. For CLOSED/COLLINEAR walkers the alpha and beta blocks reuse
/// the first columns (closed-shell convention); NONCOLLINEAR uses a genuine
/// spinor trial on the 2*NMO spin-orbital space.
inline nda::array<RealType, 2> build_trial_orbitals(HubbardSpec const& s) {
  using nda::range;
  long NMO  = s.NMO();
  int wnp   = s.walker_npol();
  long dim  = wnp * NMO;
  nda::array<RealType, 2> M(dim, dim);
  for(long i = 0; i < dim; ++i) {
    for(long j = 0; j < dim; ++j) {
      M(i, j) = RealType(1) / RealType(i + j + 1);
    }
  }
  nda::array<RealType, 1> tau(dim);
  // Following test_phmsd.cpp: act on transpose so M ends up with orthonormal columns.
  nda::lapack::geqrf(nda::transpose(M), tau);
  nda::lapack::gqr(nda::transpose(M), tau);

  nda::array<RealType, 2> Psi(dim, s.walker_nel());
  if(wnp == 1) {
    int nup = s.walker_nup();
    Psi(range::all, range(0, nup)) = M(range::all, range(0, nup));
    if(s.walker_type == COLLINEAR) {
      int ndn = s.walker_ndown();
      Psi(range::all, range(nup, nup + ndn)) = M(range::all, range(0, ndn));
    }
  } else { // NONCOLLINEAR spinor trial
    int nel = s.walker_nel();
    Psi(range::all, range(0, nel)) = M(range::all, range(0, nel));
  }
  return Psi;
}

/// (NMO, nel_for_spin(is)) block of the walker trial associated with walker-spin
/// `is` and polarization `ip`. For npol==1 walkers `ip` is always 0 and the
/// alpha/beta blocks are column-ranges of a single NMO-row matrix. For the
/// NONCOLLINEAR walker `is`==0 and the block is the ip-polarization component
/// (rows ip*NMO..(ip+1)*NMO) of every spinor orbital.
inline auto psi_block(HubbardSpec const& s, nda::array<RealType, 2> const& Psi, int is, int ip) {
  using nda::range;
  long NMO = s.NMO();
  if(s.walker_type == NONCOLLINEAR) {
    return Psi(range(ip * NMO, (ip + 1) * NMO), range(0, s.walker_nel()));
  }
  long c0 = (is == 0) ? 0 : s.walker_nup();
  long n  = s.nel_for_spin(is);
  return Psi(range(0, NMO), range(c0, c0 + n));
}

namespace detail {

/// Produce the immutable const_shared_array used by the ported
/// HamiltonianOperations. src is replicated on every rank, so the root-only
/// callable just hands over a copy.
template<MEMORY_SPACE MEM, typename T, int N>
auto to_const_shared(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                     nda::array<T, N> const& src) {
  return memory::share_from_root(*mpi, [&]() { return memory::to_memory_space<MEM>(src); });
}

/// Half-rotated one-body matrix haj of shape (1, walker_nel, walker_npol*NMO):
///   haj[0, a, P] = sum_Q Psi(Q, a) * T_walker(Q, P)
/// computed per walker-spin block for npol==1 walkers and as a single spinor
/// contraction for NONCOLLINEAR. Psi is real, so no conjugation is required.
inline nda::array<ComplexType, 3> host_haj(HubbardSpec const& s,
                                           nda::array<ComplexType, 3> const& T_walker,
                                           nda::array<RealType, 2> const& Psi) {
  using nda::range;
  auto all = range::all;
  long NMO = s.NMO();
  int wns  = s.walker_nspin();
  int wnp  = s.walker_npol();
  long nel = s.walker_nel();
  nda::array<ComplexType, 2> PsiC(Psi);
  nda::array<ComplexType, 3> haj(1, nel, wnp * NMO);
  haj() = ComplexType(0);
  if(wnp == 1) {
    for(int is = 0; is < wns; ++is) {
      long c0  = (is == 0) ? 0 : s.walker_nup();
      long n   = s.nel_for_spin(is);
      auto Pb  = PsiC(all, range(c0, c0 + n)); // (NMO, n)
      nda::array<ComplexType, 2> blk(n, NMO);
      nda::blas::gemm(nda::transpose(Pb), T_walker(is, all, all), blk);
      haj(0, range(c0, c0 + n), all) = blk;
    }
  } else { // NONCOLLINEAR
    nda::blas::gemm(nda::transpose(PsiC), T_walker(0, all, all), haj(0, all, all));
  }
  return haj;
}

/// Diagonal exchange counterterm v[m, i, i] = -0.5 U replicated over `nH`
/// leading slices, in the stored Hamiltonian-symmetry layout.
template<typename T>
nda::array<T, 3> host_vexx_diagonal(long NMO, RealType U, int nH) {
  nda::array<T, 3> v(nH, NMO, NMO);
  v() = T(0);
  for(int m = 0; m < nH; ++m) {
    nda::diagonal(v(m, nda::range::all, nda::range::all)) = T(-0.5) * T(U);
  }
  return v;
}

/// Single-kpoint variant of host_vexx_diagonal, shape (nH, 1, NMO, NMO).
template<typename T>
nda::array<T, 4> host_vexx_diagonal_kp(long NMO, RealType U, int nH) {
  nda::array<T, 4> v(nH, 1, NMO, NMO);
  v() = T(0);
  for(int m = 0; m < nH; ++m) {
    nda::diagonal(v(m, 0, nda::range::all, nda::range::all)) = T(-0.5) * T(U);
  }
  return v;
}

/// Trivial Brillouin-zone tables for the single-kpoint lift (nkpts=1).
struct SingleQTables {
  nda::array<int, 2> nocc;     // (nspin, nkpts) — occupations per spin per k
  nda::array<int, 1> minusq;   // q -> index of -q
  nda::array<int, 2> qk_to_k2; // (q, k) -> k+q index
  nda::array<int, 1> qmap;     // q -> index in symmetric-q list (KP3IndexFactorization)
};

inline SingleQTables make_single_q_tables(HubbardSpec const& s) {
  return SingleQTables{
      .nocc     = nda::array<int, 2>{{s.walker_nup()}, {s.walker_ndown()}},
      .minusq   = nda::array<int, 1>{0},
      .qk_to_k2 = nda::array<int, 2>{{0}},
      .qmap     = nda::array<int, 1>{0},
  };
}

} // namespace detail

/// Real3IndexFactorization for the 2D Hubbard model. The onsite interaction is
/// closed, so the Cholesky vectors are diagonal: L_n[i,j] = sqrt(U) δ_ni δ_nj.
/// They are replicated over the nH = nspin_in_H*npol_in_H leading slices that
/// the variant uses to size the auxiliary-field potential for the target walker.
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_real3index(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s) {
  using nda::range;
  auto all     = range::all;
  long NMO     = s.NMO();
  long nCV     = NMO;
  RealType sqU = std::sqrt(s.U);
  int wns      = s.walker_nspin();
  int wnp      = s.walker_npol();
  int nH       = s.nspin_in_H() * s.npol_in_H();

  auto hij_h    = build_hij_intrinsic(s);
  auto T_walker = build_T_walker(s, hij_h);
  auto Psi      = build_trial_orbitals(s);
  auto haj_h    = detail::host_haj(s, T_walker, Psi);
  auto vexx_h   = detail::host_vexx_diagonal<RealType>(NMO, s.U, nH);

  // Likn(nH, NMO, NMO, nCV) with L_n[i,j] = sqU δ_ni δ_nj, replicated over nH.
  nda::array<RealType, 4> Likn_h(nH, NMO, NMO, nCV);
  Likn_h() = RealType(0);
  for(int m = 0; m < nH; ++m) {
    for(long n = 0; n < nCV; ++n) {
      Likn_h(m, n, n, n) = sqU;
    }
  }

  // Lnak(is)(0, ip, n, a, k) = sqU * Psi_block(is,ip)(n,a) δ_{k,n}.
  nda::array<memory::const_shared_array<MEM, ComplexType, 5>, 1> Lnak(wns);
  for(int is = 0; is < wns; ++is) {
    long n_is = s.nel_for_spin(is);
    nda::array<ComplexType, 5> Lnak_h(1, wnp, nCV, n_is, NMO);
    Lnak_h() = ComplexType(0);
    for(int ip = 0; ip < wnp; ++ip) {
      auto Pb = psi_block(s, Psi, is, ip);
      for(long a = 0; a < n_is; ++a) {
        nda::diagonal(Lnak_h(0, ip, all, a, all)) = sqU * Pb(all, a);
      }
    }
    Lnak(is) = detail::to_const_shared<MEM>(mpi, Lnak_h);
  }

  return HamiltonianOperations<MEM>(Real3IndexFactorization<MEM>(
      mpi, s.walker_type, NMO, s.walker_nup(), s.walker_ndown(),
      detail::to_const_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_const_shared<MEM>(mpi, haj_h),
      detail::to_const_shared<MEM>(mpi, Likn_h),
      std::move(Lnak),
      detail::to_const_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// THCOps (REAL=true). The X factor is the identity on each (spin,pol) orbital
/// block (sized by the Hamiltonian symmetry of hij), Y holds the walker-basis
/// trial, L = sqrt(U) I (factorized Coulomb) and Z = U I (Coulomb matrix).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_thc(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s) {
  using nda::range;
  auto all     = range::all;
  long NMO     = s.NMO();
  long nu      = NMO;
  RealType sqU = std::sqrt(s.U);
  int wns      = s.walker_nspin();
  int wnp      = s.walker_npol();
  int nsH      = s.nspin_in_H();
  int npH      = s.npol_in_H();

  auto hij_h    = build_hij_intrinsic(s);
  auto T_walker = build_T_walker(s, hij_h);
  auto Psi      = build_trial_orbitals(s);
  auto haj_h    = detail::host_haj(s, T_walker, Psi);
  auto vexx_h   = detail::host_vexx_diagonal<ComplexType>(NMO, s.U, nsH * npH);

  // X(nsH, npH*NMO, nu) = identity on each (spin,pol) orbital block.
  nda::array<RealType, 3> X_h(nsH, npH * NMO, nu);
  X_h() = RealType(0);
  for(int is = 0; is < nsH; ++is) {
    for(int ip = 0; ip < npH; ++ip) {
      nda::diagonal(X_h(is, range(ip * NMO, (ip + 1) * NMO), all)) = RealType(1);
    }
  }

  // Y(1, wns, wnp, walker_nup, nu): Y[0, is, ip, a, u] = Psi_block(is,ip)(u,a).
  nda::array<ComplexType, 5> Y_h(1, wns, wnp, s.walker_nup(), nu);
  Y_h() = ComplexType(0);
  for(int is = 0; is < wns; ++is) {
    for(int ip = 0; ip < wnp; ++ip) {
      auto Pb = psi_block(s, Psi, is, ip);
      Y_h(0, is, ip, range(Pb.extent(1)), all) = nda::transpose(Pb);
    }
  }

  nda::array<RealType, 2> L_h(nu, nu);
  L_h() = RealType(0);
  nda::diagonal(L_h) = sqU;

  nda::array<RealType, 2> Z_h(nu, nu);
  Z_h() = RealType(0);
  nda::diagonal(Z_h) = s.U;

  return HamiltonianOperations<MEM>(THCOps<MEM, /*REAL=*/true>(
      mpi, s.walker_type, NMO, s.walker_nup(), s.walker_ndown(),
      detail::to_const_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_const_shared<MEM>(mpi, haj_h),
      detail::to_const_shared<MEM>(mpi, X_h),
      detail::to_const_shared<MEM>(mpi, Y_h),
      detail::to_const_shared<MEM>(mpi, L_h),
      std::optional{detail::to_const_shared<MEM>(mpi, Z_h)},
      /*X_rot=*/std::nullopt,
      /*Y_rot=*/std::nullopt,
      /*Z_rot=*/std::nullopt,
      detail::to_const_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// KP3IndexFactorization, single-kpoint lift (nkpts=1, nbnd=NMO).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_kp3index(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s) {
  using nda::range;
  auto all      = range::all;
  long NMO      = s.NMO();
  long nbnd     = NMO;
  long nkpts    = 1;
  long nCV      = NMO;
  long q0       = 0;
  RealType sqU  = std::sqrt(s.U);
  int wns       = s.walker_nspin();
  int wnp       = s.walker_npol();
  int nsH       = s.nspin_in_H();
  int npH       = s.npol_in_H();
  long nocc_max = std::max(s.walker_nup(), s.walker_ndown());

  auto hij3     = build_hij_intrinsic(s);
  auto T_walker = build_T_walker(s, hij3);
  auto Psi      = build_trial_orbitals(s);
  auto haj_h    = detail::host_haj(s, T_walker, Psi);
  auto vexx_h   = detail::host_vexx_diagonal_kp<ComplexType>(NMO, s.U, nsH * npH);
  auto bz       = detail::make_single_q_tables(s);

  // Lift hij to (nsH, nkpts=1, npH*NMO, npH*NMO).
  nda::array<ComplexType, 4> hij_h(nsH, 1, npH * NMO, npH * NMO);
  hij_h(all, 0, all, all) = hij3;

  // LQ(nsH, npH, nkpts, nbnd, nbnd, nCV): sqU diagonal in (i,j,n), replicated.
  nda::array<ComplexType, 6> LQ_h(nsH, npH, nkpts, nbnd, nbnd, nCV);
  LQ_h() = ComplexType(0);
  for(int is = 0; is < nsH; ++is) {
    for(int ip = 0; ip < npH; ++ip) {
      for(long n = 0; n < nCV; ++n) {
        LQ_h(is, ip, 0, n, n, n) = ComplexType(sqU);
      }
    }
  }

  // Lank/Lbnk(1, wns, nkpts, nocc_max, nCV, wnp*nbnd):
  //   diagonal over (n,k) of the (·, ip*NMO+·) slice = sqU * Psi_block(is,ip)(:,a).
  nda::array<ComplexType, 6> Lank_h(1, wns, nkpts, nocc_max, nCV, wnp * NMO);
  Lank_h() = ComplexType(0);
  for(int is = 0; is < wns; ++is) {
    for(int ip = 0; ip < wnp; ++ip) {
      auto Pb = psi_block(s, Psi, is, ip);
      for(long a = 0; a < Pb.extent(1); ++a) {
        nda::diagonal(Lank_h(0, is, 0, a, all, range(ip * NMO, (ip + 1) * NMO))) = sqU * Pb(all, a);
      }
    }
  }
  nda::array<ComplexType, 6> Lbnk_h(Lank_h);

  nda::array<memory::const_shared_array<MEM, ComplexType, 6>, 1> LQ(nkpts);
  nda::array<memory::const_shared_array<MEM, ComplexType, 6>, 1> Lank(nkpts);
  nda::array<memory::const_shared_array<MEM, ComplexType, 6>, 1> Lbnk(nkpts);
  LQ(0)   = detail::to_const_shared<MEM>(mpi, LQ_h);
  Lank(0) = detail::to_const_shared<MEM>(mpi, Lank_h);
  Lbnk(0) = detail::to_const_shared<MEM>(mpi, Lbnk_h);

  return HamiltonianOperations<MEM>(KP3IndexFactorization<MEM>(
      mpi, s.walker_type, nbnd, q0, std::move(bz.nocc), std::move(bz.minusq),
      std::move(bz.qk_to_k2), std::move(bz.qmap),
      detail::to_const_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_const_shared<MEM>(mpi, haj_h),
      std::move(LQ), std::move(Lank), std::move(Lbnk),
      detail::to_const_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// KPTHCOps, single-kpoint lift (nkpts=1, nbnd=NMO).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_kpthc(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s) {
  using nda::range;
  auto all     = range::all;
  long NMO     = s.NMO();
  long nu      = NMO;
  long nkpts   = 1;
  long q0      = 0;
  RealType sqU = std::sqrt(s.U);
  int wns      = s.walker_nspin();
  int wnp      = s.walker_npol();
  int nsH      = s.nspin_in_H();
  int npH      = s.npol_in_H();
  long nocc_max = std::max(s.walker_nup(), s.walker_ndown());

  auto hij3     = build_hij_intrinsic(s);
  auto T_walker = build_T_walker(s, hij3);
  auto Psi      = build_trial_orbitals(s);
  auto haj_h    = detail::host_haj(s, T_walker, Psi);
  auto vexx_h   = detail::host_vexx_diagonal_kp<ComplexType>(NMO, s.U, nsH * npH);
  auto bz       = detail::make_single_q_tables(s);

  nda::array<ComplexType, 4> hij_h(nsH, 1, npH * NMO, npH * NMO);
  hij_h(all, 0, all, all) = hij3;

  // X(nsH, nkpts, wnp*nbnd, nu) = identity on each (spin,pol) block.
  nda::array<ComplexType, 4> X_h(nsH, nkpts, wnp * NMO, nu);
  X_h() = ComplexType(0);
  for(int is = 0; is < nsH; ++is) {
    for(int ip = 0; ip < wnp; ++ip) {
      nda::diagonal(X_h(is, 0, range(ip * NMO, (ip + 1) * NMO), all)) = ComplexType(1);
    }
  }

  // Y(1, wns, wnp, nkpts, nocc_max, nu): Y[0, is, ip, 0, a, u] = Psi_block(u,a).
  nda::array<ComplexType, 6> Y_h(1, wns, wnp, nkpts, nocc_max, nu);
  Y_h() = ComplexType(0);
  for(int is = 0; is < wns; ++is) {
    for(int ip = 0; ip < wnp; ++ip) {
      auto Pb = psi_block(s, Psi, is, ip);
      Y_h(0, is, ip, 0, range(Pb.extent(1)), all) = nda::transpose(Pb);
    }
  }

  nda::array<ComplexType, 3> L_h(1, nu, nu);
  nda::array<ComplexType, 3> Z_h(1, nu, nu);
  L_h() = ComplexType(0);
  Z_h() = ComplexType(0);
  nda::diagonal(L_h(0, all, all)) = ComplexType(sqU);
  nda::diagonal(Z_h(0, all, all)) = ComplexType(s.U);

  return HamiltonianOperations<MEM>(KPTHCOps<MEM>(
      mpi, s.walker_type, NMO, s.walker_nup(), s.walker_ndown(), nkpts, q0,
      std::move(bz.nocc), std::move(bz.minusq), std::move(bz.qk_to_k2),
      detail::to_const_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_const_shared<MEM>(mpi, haj_h),
      detail::to_const_shared<MEM>(mpi, X_h),
      detail::to_const_shared<MEM>(mpi, Y_h),
      detail::to_const_shared<MEM>(mpi, L_h),
      std::optional{detail::to_const_shared<MEM>(mpi, Z_h)},
      /*X_rot=*/std::nullopt,
      /*Y_rot=*/std::nullopt,
      /*Z_rot=*/std::nullopt,
      detail::to_const_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// Subclass that promotes the protected helpers needed to assemble a
/// ModelHamOps directly from in-memory sparse matrices, without an HDF5
/// backing file. We pick `shift_1body=true` so the -0.5 U diagonal
/// counterterm lands in H1, matching the Real3 / THC / KP* convention.
class HubbardModelHamOpsAccess : public ModelHamOpsGenerator {
public:
  explicit HubbardModelHamOpsAccess()
      : ModelHamOpsGenerator(dummy_pt(), ComplexType(0), ComplexType(0)) {}

  using ModelHamOpsGenerator::addComponent;
  using ModelHamOpsGenerator::find_occupied_pairs;
  using ModelHamOpsGenerator::make_SparseEnergy;

private:
  static ptree dummy_pt() {
    ptree pt;
    pt.put("name", "test_hubbard");
    pt.put("filename", "/dev/null");
    pt.put("shift_1body", true);
    return pt;
  }
};

/// ModelHamOps for the 2D Hubbard model, onsite-U through the continuous-charge
/// HS. The intrinsic one-body matrix is built in its native spin layout and
/// converted to the walker layout via the same closed/collinear/noncollinear
/// sparse promotions used by ModelHamOpsGenerator::spin_to_walker_type. CLOSED
/// walkers are not supported by Model Hamiltonians and must not reach here.
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_modelhamops(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s) {
  using nda::range;
  long NMO = s.NMO();
  using csrM = math::sparse::csr_matrix<RealType, HOST_MEMORY, int, int>;
  utils::check(s.walker_type != CLOSED,
               "build_modelhamops: Model Hamiltonians do not support CLOSED walkers.");

  auto Psi = build_trial_orbitals(s);
  auto h   = base_hopping(s);

  // Build the intrinsic one-body matrix in its native sparse layout.
  csrM hij_walker = [&]() -> csrM {
    if(s.h_symmetry == CLOSED) {
      csrM hc({long(NMO), long(NMO)}, NMO);
      for(long i = 0; i < NMO; ++i) {
        for(long j = 0; j < NMO; ++j) {
          if(std::abs(h(i, j)) > 1e-12) {
            hc.add({int(i), int(j)}, h(i, j));
          }
        }
      }
      hc.remove_empty_spaces();
      if(s.walker_type == COLLINEAR) {
        return math::sparse::closed_to_collinear(hc);
      }
      return math::sparse::closed_to_noncollinear(hc);
    } else if(s.h_symmetry == COLLINEAR) {
      // (2*NMO, NMO): alpha rows [0,NMO), beta rows [NMO,2NMO) with +dz Zeeman.
      csrM hcol({long(2 * NMO), long(NMO)}, NMO);
      for(long i = 0; i < NMO; ++i) {
        for(long j = 0; j < NMO; ++j) {
          if(std::abs(h(i, j)) > 1e-12) {
            hcol.add({int(i), int(j)}, h(i, j));
            hcol.add({int(NMO + i), int(j)}, h(i, j));
          }
        }
      }
      for(long i = 0; i < NMO; ++i) {
        hcol.add({int(NMO + i), int(i)}, s.dz);
      }
      hcol.remove_empty_spaces();
      if(s.walker_type == COLLINEAR) {
        return hcol;
      }
      return math::sparse::collinear_to_noncollinear(hcol);
    } else { // NONCOLLINEAR intrinsic: (2*NMO, 2*NMO), diag blocks=h, off-diag=hx*I
      csrM hnc({long(2 * NMO), long(2 * NMO)}, 2 * NMO);
      for(int p = 0; p < 2; ++p) {
        for(long i = 0; i < NMO; ++i) {
          for(long j = 0; j < NMO; ++j) {
            if(std::abs(h(i, j)) > 1e-12) {
              hnc.add({int(p * NMO + i), int(p * NMO + j)}, h(i, j));
            }
          }
        }
      }
      for(long i = 0; i < NMO; ++i) {
        hnc.add({int(i), int(NMO + i)}, s.hx);
        hnc.add({int(NMO + i), int(i)}, s.hx);
      }
      hnc.remove_empty_spaces();
      return hnc;
    }
  }();

  // Onsite Hubbard U on (2*NMO, NMO): opposite-spin diagonal in the upper
  // block; same-spin onsite vanishes by Pauli. Layout is identical for
  // COLLINEAR and NONCOLLINEAR walkers.
  csrM U_combined({long(2 * NMO), long(NMO)}, 1);
  for(long i = 0; i < NMO; ++i) {
    U_combined.add({int(i), int(i)}, s.U);
  }
  U_combined.remove_empty_spaces();

  // ModelHamOpsGenerator consumes per-HS-channel U/J vectors in fixed slots:
  // [0]=continuous_charge, [1]=continuous_spin, [2]=discrete_charge,
  // [3]=discrete_spin. Onsite-U through continuous_charge populates only [0].
  std::vector<csrM> Uvec;
  Uvec.reserve(4);
  Uvec.emplace_back(std::move(U_combined));
  for(int i = 1; i < 4; ++i) {
    Uvec.emplace_back(csrM({long(2 * NMO), long(NMO)}, 0));
  }
  std::vector<csrM> Jvec;
  Jvec.reserve(3);
  for(int i = 0; i < 3; ++i) {
    Jvec.emplace_back(csrM({long(NMO), long(NMO)}, 0));
  }

  HubbardModelHamOpsAccess gen;

  nda::array<long, 1> n2IJ =
      gen.find_occupied_pairs<RealType>(s.walker_type, Uvec, Jvec);

  using map_t = std::unordered_map<long, int>;
  map_t IJ2n;
  IJ2n.reserve(n2IJ.size());
  for(long n = 0; n < n2IJ.size(); ++n) {
    IJ2n.insert(std::make_pair(n2IJ[n], int(n)));
  }

  // Ordering is load-bearing: addComponent only reads its U/J inputs whereas
  // make_SparseEnergy moves them. Building the component first keeps Uvec[0]
  // valid for the SparseEnergy call.
  std::vector<ModelComponent<MEM, /*REAL=*/true>> Hams;
  Hams.reserve(1);
  gen.addComponent<MEM, /*REAL=*/true, RealType, map_t>(
      s.walker_type, ContinuousChargePropagator, mpi, Uvec[0], Jvec[0], Hams,
      n2IJ, IJ2n);
  utils::check(Hams.size() == 1, "Expected exactly one ModelComponent for onsite Hubbard U.");

  auto ET = gen.make_SparseEnergy<MEM, /*REAL=*/true, RealType>(
      mpi, s.walker_type, hij_walker, Uvec[0], Jvec[0], ComplexType(0));

  // PsiC(1, wns, wnp*NMO, walker_nup): per-(spin,block) Slater matrices.
  int wns = s.walker_nspin();
  int wnp = s.walker_npol();
  nda::array<ComplexType, 4> PsiC_h(1, wns, wnp * NMO, s.walker_nup());
  PsiC_h() = ComplexType(0);
  if(wnp == 1) {
    for(int is = 0; is < wns; ++is) {
      auto Pb = psi_block(s, Psi, is, 0);
      PsiC_h(0, is, range(NMO), range(Pb.extent(1))) = Pb;
    }
  } else { // NONCOLLINEAR spinor trial
    PsiC_h(0, 0, range(wnp * NMO), range(s.walker_nel())) = Psi;
  }

  return HamiltonianOperations<MEM>(ModelHamOps<MEM, /*REAL=*/true>(
      mpi, s.walker_type, s.walker_nup(), s.walker_ndown(),
      detail::to_const_shared<MEM>(mpi, PsiC_h),
      std::move(ET), std::move(Hams), n2IJ));
}

} // namespace sfqmc::afqmc::tests
