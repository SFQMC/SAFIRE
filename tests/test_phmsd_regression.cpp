// Distributed under the Apache License, Version 2.0. See LICENSE.
#include "test_common.hpp"
#include "utilities/check_shape.hpp"
#include "AFQMC/Wavefunctions/phmsd_helpers.hpp"
#include <algorithm>
#include <complex>
#include <vector>

namespace sfqmc {
namespace {
using C = std::complex<double>;

// Pivoted elimination provides independent determinant and inverse references.
std::pair<C, nda::array<C, 2>> direct_inverse(nda::array<C, 2> a) {
  int n = a.extent(0);
  nda::array<C, 2> inv(n, n);
  inv() = C(0);
  for(int i=0; i<n; ++i) inv(i,i) = 1.;
  C det = 1.;
  for(int k=0; k<n; ++k) {
    int p = k;
    for(int i=k+1; i<n; ++i)
      if(std::abs(a(i,k)) > std::abs(a(p,k))) p = i;
    REQUIRE(std::abs(a(p,k)) > 1.e-12);
    if(p != k) {
      for(int j=0; j<n; ++j) {
        std::swap(a(p,j),a(k,j));
        std::swap(inv(p,j),inv(k,j));
      }
      det = -det;
    }
    C pivot = a(k,k);
    det *= pivot;
    for(int j=0; j<n; ++j) { a(k,j) /= pivot; inv(k,j) /= pivot; }
    for(int i=0; i<n; ++i) if(i != k) {
      C x = a(i,k);
      for(int j=0; j<n; ++j) {
        a(i,j) -= x*a(k,j);
        inv(i,j) -= x*inv(k,j);
      }
    }
  }
  return {det, std::move(inv)};
}

template<MEMORY_SPACE MEM>
void phmsd_regression(int nex, bool unit_determinant, int spin, bool swap_rows) {
  using nda::range;
  auto all = range::all;
  constexpr int nup=10, ndown=9, nact=23, nw=3, nd=2;
  int nel = spin == 0 ? nup : ndown;
  CAPTURE(MEM, nex, unit_determinant, spin, swap_rows);
  std::vector<long> ca(nex+1,0), cb(nex+1,0);
  ca[0]=nup; cb[0]=ndown;
  ca[nex]=cb[nex]=2*nex*nd;
  afqmc::ph_excitations<int,C,MEM> abij(nd+1,nup,ndown,ca,cb);
  std::vector<int> ra(nup), rb(ndown);
  for(int i=0; i<nup; ++i) ra[i]=2*i;
  for(int i=0; i<ndown; ++i) rb[i]=2*i;
  abij.add_reference(ra,rb);
  for(int d=0; d<nd; ++d) {
    std::vector<int> ex(2*nex);
    for(int p=0; p<nex; ++p) { ex[p]=p; ex[nex+p]=2*(p+d)+1; }
    abij.add_alpha(nex,ex.begin());
    abij.add_beta(nex,ex.begin());
  }
  nda::array<C,3> T(nw,nact,nel), Rref(nw,nel,nact);
  nda::array<C,2> w(nd+1,nw), ovref(nd+1,nw);
  T()=C(0); Rref()=C(0); ovref(0,all)=C(1);
  for(int iw=0; iw<nw; ++iw) {
    for(int i=0; i<nel; ++i) T(iw,2*i,i)=1.;
    for(int a=1; a<nact; a+=2)
      for(int i=0; i<nel; ++i)
        T(iw,a,i)=C(.07*std::sin(1+a+3*i+iw),.05*std::cos(2*a+i+2*iw));
    // Two overlapping excitation lists; generic complex blocks with different
    // walkers exercise pivoting, phases and reuse of output buffers.
    for(int p=0; p<nel; ++p) {
      T(iw,2*p+1,p) += C(1.2+.1*iw,.15);
      if(p>0) T(iw,2*p+1,p-1) += C(1.1,-.1);
    }
    if(unit_determinant) {
      // The first excitation block is exactly unit triangular, with nonzero
      // spectator columns. log(det)==0 must not mark its inverse as singular.
      for(int p=0; p<nex; ++p)
        for(int q=0; q<nex; ++q)
          T(iw,2*p+1,q)=p==q ? C(1) : (q>p ? C(.125,.0625) : C(0));
    }
    if(swap_rows && nex>1)
      for(int i=0; i<nel; ++i) std::swap(T(iw,1,i),T(iw,2*nex-1,i));
    for(int d=0; d<=nd; ++d) w(d,iw)=C(.2+.1*d,.03*(iw-d));
    for(int d=0; d<=nd; ++d) {
      std::vector<int> orbs(nel);
      abij.get_configuration(spin,d,orbs);
      nda::array<C,2> S(nel,nel);
      for(int i=0; i<nel; ++i)
        for(int j=0; j<nel; ++j) S(i,j)=T(iw,orbs[i],j);
      auto [det,inv]=direct_inverse(std::move(S));
      ovref(d,iw)=det;
      for(int i=0; i<nel; ++i)
        for(int j=0; j<nel; ++j) Rref(iw,i,orbs[j]) += w(d,iw)*inv(i,j);
    }
  }
  auto Td=memory::to_memory_space<MEM>(T);
  auto wd=memory::to_memory_space<MEM>(w);
  memory::array<MEM,C,2> ov(nd+1,nw);
  memory::array<MEM,C,3> R(nw,nel,nact);
  // Repeated calls must overwrite determinant ratios, not accumulate logs.
  for(int repeat=0; repeat<2; ++repeat) {
    ov()=C(.37,-.21);
    afqmc::calculate_overlaps(spin,abij,Td,ov);
    CHECK_THAT(nda::to_host(ov),utils::Approx(ovref));
    afqmc::calculate_R(spin,abij,Td,wd,R);
    CHECK_THAT(nda::to_host(R),utils::Approx(Rref));
  }
  // Independently check the compact inverse used by the energy routine.
  auto ex=abij.get_excitation_list_device(spin,nex);
  auto ref=abij.get_reference_configuration_device(spin);
  memory::array<MEM,C,4> compact(nw,nd,nex,nact);
  compact()=C(0);
  afqmc::get_compact_ph_R_matrices<MEM>(ex,ref,Td,compact);
  auto ch=nda::to_host(compact);
  nda::array<C,4> cref(nw,nd,nex,nact);
  cref()=C(0);
  for(int iw=0; iw<nw; ++iw) for(int d=0; d<nd; ++d) {
    std::vector<int> orbs(nel);
    abij.get_configuration(spin,d+1,orbs);
    nda::array<C,2> S(nel,nel);
    for(int i=0; i<nel; ++i) for(int j=0; j<nel; ++j) S(i,j)=T(iw,orbs[i],j);
    auto [det,inv]=direct_inverse(std::move(S));
    for(int p=0; p<nex; ++p) {
      for(int j=0; j<nel; ++j) cref(iw,d,p,orbs[j])=inv(p,j);
      cref(iw,d,p,orbs[p])-=1.;
    }
  }
  CHECK_THAT(ch,utils::Approx(cref));
}
template<MEMORY_SPACE MEM>
void phmsd_density_scattered_holes(int spin, int batch)
{
  constexpr int nup=6, ndown=5, nact=14, nw=3, nd=4;
  int nel=spin==0 ? nup : ndown;
  std::vector<long> ca{nup,0,8,6},cb{ndown,0,8,6};
  afqmc::ph_excitations<int,C,MEM> abij(nd,nup,ndown,ca,cb);
  std::vector<int> ra{10,2,8,0,6,4},rb{10,2,8,0,6};
  abij.add_reference(ra,rb);
  for(auto ex : {std::vector<int>{1,4,1,9},std::vector<int>{0,3,5,11},
                 std::vector<int>{0,2,4,3,7,13}}) {
    abij.add_alpha(ex.size()/2,ex.begin());
    abij.add_beta(ex.size()/2,ex.begin());
  }
  auto ref=abij.reference_configuration(spin);
  nda::array<C,3> T(nw,nact,nel),expected(nw,nel,nact);
  nda::array<C,2> weights(nd,nw);
  T()=C(0); expected()=C(0);
  for(int iw=0; iw<nw; ++iw) {
    for(int i=0; i<nel; ++i) T(iw,ref[i],i)=C(1);
    for(int a=1; a<nact; a+=2)
      for(int j=0; j<nel; ++j)
        T(iw,a,j)=C(std::sin((a+1)*(j+2)+iw),.2*std::cos((a+2)*(j+1)+2*iw));
    for(int d=0; d<nd; ++d) {
      weights(d,iw)=C(.15+.1*d,.03*(d-iw));
      std::vector<int> occ(nel);
      abij.get_configuration(spin,d,occ);
      nda::array<C,2> S(nel,nel);
      for(int i=0; i<nel; ++i)
        for(int j=0; j<nel; ++j) S(i,j)=T(iw,occ[i],j);
      auto [det,inverse]=direct_inverse(std::move(S));
      for(int i=0; i<nel; ++i)
        for(int j=0; j<nel; ++j) expected(iw,i,occ[j])+=weights(d,iw)*inverse(i,j);
    }
  }
  auto t=memory::to_memory_space<MEM>(T),w=memory::to_memory_space<MEM>(weights);
  memory::array<MEM,C,3> R(nw,nel,nact);
  for(int repeat=0; repeat<2; ++repeat) {
    R()=C(.3,-.7);
    afqmc::calculate_R(spin,abij,t,w,R,batch);
    CHECK_THAT(nda::to_host(R),utils::Approx(expected));
  }
}
template<int N>
void phmsd_host_inverse_scaling()
{
  nda::array<C,2> base(N,N),matrix(N,N),expected(N,N);
  base()=C(0);
  for(int i=0; i<N; ++i)
    for(int j=i; j<N; ++j) base(i,j)=i==j ? C(1) : C(.125,.0625);
  auto [det,inverse]=direct_inverse(base);
  for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) {
    double row_scale=i<N-N%2 ? (i%2==0 ? 1.e40 : 1.e-40) : 1.;
    double col_scale=j<N-N%2 ? (j%2==0 ? 1.e40 : 1.e-40) : 1.;
    matrix(i,j)=row_scale*base(i,j);
    expected(i,j)=inverse(i,j)/col_scale;
  }
  for(int j=0; j<N; ++j) {
    std::swap(matrix(0,j),matrix(N-1,j));
    std::swap(expected(j,0),expected(j,N-1));
  }
  auto found=afqmc::detail::phmsd_small_inverse<N>(matrix);
  CHECK_THAT(found,utils::Approx(-det));
  CHECK_THAT(matrix,utils::Approx(expected));
  matrix()=C(0);
  CHECK(afqmc::detail::phmsd_small_inverse<N>(matrix)==C(0));
  nda::array<C,2> zero(N,N);
  zero()=C(0);
  CHECK_THAT(matrix,utils::Approx(zero));
}
} // namespace

TEST_CASE("phmsd: ranks 1-9 against direct inverse", "[phmsd_regression]") {
  int nex=GENERATE(1,2,3,4,5,6,7,8,9);
  bool unit_determinant=GENERATE(false,true);
  int spin=GENERATE(0,1);
  bool swap_rows=GENERATE(false,true);
  phmsd_regression<HOST_MEMORY>(nex,unit_determinant,spin,swap_rows);
#if defined(ENABLE_DEVICE)
  phmsd_regression<DEVICE_MEMORY>(nex,unit_determinant,spin,swap_rows);
#endif
}
TEST_CASE("phmsd: density with scattered holes and mixed ranks", "[phmsd_regression]") {
  int spin=GENERATE(0,1);
  int batch=GENERATE(1,3,4096);
  phmsd_density_scattered_holes<HOST_MEMORY>(spin,batch);
#if defined(ENABLE_DEVICE)
  phmsd_density_scattered_holes<DEVICE_MEMORY>(spin,batch);
#endif
}
TEST_CASE("phmsd: scaled pivoted host inverses", "[phmsd_regression]") {
  phmsd_host_inverse_scaling<4>();
  phmsd_host_inverse_scaling<5>();
}
} // namespace sfqmc
