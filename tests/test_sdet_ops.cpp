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
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "configuration.hpp"
#include "IO/app_loggers.h" 
#include "utilities/check.hpp"
#include "test_common.hpp"
#include "utilities/mpi_context.h"
#include "numerics/sparse/sparse.hpp"
#include "numerics/operations/split_singular_vals.hpp"
#include "numerics/operations/add_diagonal.hpp"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>

#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/propagate.hpp"
#include "AFQMC/SlaterDeterminantOperations/orthogonalize.hpp"

using std::complex;
using std::cout;
using std::endl;
using std::string;

namespace sfqmc
{
namespace afqmc
{

using namespace afqmc;
template<MEMORY_SPACE MEM>
void sdet_ops_operations()
{

  auto all = nda::range::all;
  using nda::range;

  const int nwalk = 5;
  const int NMO = 4;
  const int NEL = 3;

  using Type        = ComplexType;
  using namespace std::complex_literals;
  using vector       = memory::array<MEM,Type,1>;
  using matrix       = memory::array<MEM,Type,2>;
  using matrix_view  = memory::array_view<MEM,Type,2>;
  using array       = memory::array<MEM,Type,3>;
  using array_view  = memory::array_view<MEM,Type,3>;

  const Type ov_base  = std::log(-7.62332599999999 + 22.20453200000000i);
  const Type ov2_base = std::log(-10.37150000000000 - 7.15750000000000i);
  nda::array<Type,1> ov(nwalk,ov_base);
  nda::array<Type,1> ov2(nwalk,ov2_base);

  // helper function, use with care
  auto copy_to_array = [](auto const& b, auto && A) { 
    for(int i=0; i<A.extent(0); i++) A(i,nda::ellipsis{}) = b();
  };

  // some arbitrary matrices 
  nda::array<Type,2> m_a = { 
                {0.90000 + 0.10000i, 0.40000 + 0.40000i, 1.40000 + 0.20000i, 0.40000 + 0.50000i},
                {2.40000 + 0.20000i, 1.00000 + 0.50000i, 1.60000 + 0.30000i, 0.20000 + 0.10000i},
                {3.00000 + 0.30000i, 1.20000 + 0.10000i, 3.60000 + 0.40000i, 0.10000 + 0.20000i}
                           };
  nda::array<Type,2> m_b = { 
                {1.90000 + 0.60000i, 1.40000 + 0.70000i, 0.40000 + 0.80000i}, 
                {1.40000 + 0.90000i, 0.20000 + 0.50000i, 2.20000 + 0.60000i}, 
                {0.40000 + 0.70000i, 2.60000 + 0.80000i, 0.60000 + 0.90000i}, 
                {1.10000 + 0.50000i, 0.30000 + 0.60000i, 0.90000 + 0.70000i} 
                           };

  matrix A(m_a);
  array B(nwalk,NMO,NEL);
  copy_to_array(m_b,B);

  matrix_view Aref(A());
  array_view Bref(B());

  // csr
  auto Acsr =math::sparse::to_csr<MEM,int,int>(A);

  // views
  auto A_ = A(range(0,2),range(0,3));
  auto B_ = B(all,range(0,3),range(0,2));

  /**** Overlaps ****/
  //SECTION("Overlaps")
  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(A, B, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(Aref, B, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(A, Bref, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(Aref, Bref, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
  }

  //SECTION("range_overlap")
  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(A_, B_, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov2));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(A(range(0,2),range(0,3)), B_, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov2));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(A_, B(all,range(0,3),range(0,2)), ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov2));
  }

  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(Acsr, B, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::Log_Overlap(Acsr, Bref, ovlp);
    CHECK_THAT(ovlp, utils::Approx(ov));
  } 

  /**** Density Matrices *****/
  nda::array<Type,2> v_ref    = {{1.17573619385025996 - 0.01580426445014660i,  -0.25295981756593167 + 0.28594469607401085i,
                  -0.07724502823533341 - 0.09687959052155870i, 0.30512858581808422 - 0.04506898328729603i},
                  {0.17912592806889663 + 0.08374315906672802i,  0.59381451118048767 + 0.13438888951771200i,
                  -0.02021475320201610 - 0.13737561982083193i, 0.32095003919745313 + 0.12832750636154097i},
                  {-0.04919549564646425 + 0.05402065222741825i, -0.00286990878355775 - 0.15806420733175885i,
                  1.05069081188635494 + 0.00793429988912356i,  -0.08048239150997794 + 0.09917405634760490i},
                  {-0.46434598219794548 - 0.09896890422706731i, 0.87746748807670427 - 0.53417787485950319i,
                  0.12162735438647077 + 0.31042401735800573i,  0.17975848308289613 - 0.12651892495668898i}};
  nda::array<Type,2> vc_ref   = {{-0.8721879971495297 + 0.6787593377585239i, 0.3244278932250768 - 1.8083537898275881i,
                   0.6130713546530860 + 0.0399736955598931i,  0.0132562806444336 - 0.2882495766584950i},
                   {0.8853626557603183 + 0.0978868569224204i,  -0.0598704127345155 - 0.0470889603064014i,
                   -0.7392693424168300 - 0.0715317395994149i, 0.3721269544963505 - 0.1797896522886788i},
                   {-0.0567190307022984 - 0.3114847157576828i, -0.1290126440468128 + 0.6815705660308808i,
                   0.3787012855335005 + 0.0039188686237135i,  -0.2005543456941538 + 0.2100886953142371i}};
  nda::array<Type,2> v_ref_2  = {{0.7361983496013835 - 0.0956505507662245i, 0.6467449689807925 + 0.2297471806893873i,
                    0.0189270005620390 - 0.1727975708935829i},
                    {0.2986893588843604 - 0.0099955730815030i, 0.2696068479051557 + 0.0292039710860386i,
                    0.0497734835066391 + 0.1783200397050796i},
                    {0.1020030246826092 + 0.0344707468383766i, -0.2500340021988402 - 0.0826863644855427i,
                    0.9941948024934623 + 0.0664465796801866i}};
  nda::array<Type,2> vc_ref_2 = {{-0.489369975192701 + 0.103038673040713i, -0.858850485405126 - 0.275734238124941i, 1.219791948842170 + 0.118626922447301i},
                     {0.486337033653898 - 0.098631569047656i, 0.595497821653010 + 0.185288949671560i,  -0.455373025165330 - 0.129360996228044i}};

  array g_ref(nwalk,NMO,NMO);    copy_to_array(v_ref,g_ref);
  array gc_ref(nwalk,NEL,NMO);    copy_to_array(vc_ref,gc_ref);
  array g_ref_2(nwalk,3,3);    copy_to_array(v_ref_2,g_ref_2);
  array gc_ref_2(nwalk,2,3);    copy_to_array(vc_ref_2,gc_ref_2);

  //SECTION("density_matrices")
  {
    array G(nwalk, NMO, NMO);
    auto G_ = G(all,range(3),range(3));
    auto Gc_ = G(all,range(2),range(3));
    array Gc(nwalk, NEL, NMO);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));

    det_ops::MixedDensityMatrix(A, B, G, ovlp, false);
    CHECK_THAT(G, utils::Approx(g_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(A(), B(), G(), ovlp(), false);
    CHECK_THAT(G, utils::Approx(g_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));

    det_ops::MixedDensityMatrix(A_, B_, G_, ovlp, false);
    CHECK_THAT(G_, utils::Approx(g_ref_2));

    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(A, B, Gc, ovlp, true);
    CHECK_THAT(Gc, utils::Approx(gc_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(A(), B(), Gc(), ovlp(), true);
    CHECK_THAT(Gc, utils::Approx(gc_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));

    det_ops::MixedDensityMatrix(A_, B_, Gc_, ovlp, true);
    CHECK_THAT(Gc_, utils::Approx(gc_ref_2));
  }

  {
    array G(nwalk, NMO, NMO);
    array Gc(nwalk, NEL, NMO);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));

    det_ops::MixedDensityMatrix(Acsr, B, G, ovlp, false);
    CHECK_THAT(G, utils::Approx(g_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(Acsr, B(), G(), ovlp(), false);
    CHECK_THAT(G, utils::Approx(g_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));

    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(Acsr, B, Gc, ovlp, true);
    CHECK_THAT(Gc, utils::Approx(gc_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));
    ovlp() = Type(0.0);
    det_ops::MixedDensityMatrix(Acsr, B(), Gc(), ovlp(), true);
    CHECK_THAT(Gc, utils::Approx(gc_ref));
    CHECK_THAT(ovlp, utils::Approx(ov));
  }

  // Propagate
  {
    // not testing results, just checking things run. Get specific answers to test
    array SM(nwalk, NMO, NEL);
    matrix P(NMO, NMO);
    array V(nwalk, NMO, NMO);

    SM() = utils::make_random<Type>(nwalk,NMO,NEL);
    P() = utils::make_random<Type>(NMO,NMO);
    auto Vt = utils::make_random<Type>(nwalk,NMO,NMO);
    Vt() = Vt()*0.001;
    V() = Vt(); 
    auto Pcsr = math::sparse::to_csr<MEM,int,int>(P);

    matrix V2(nwalk*NMO, nwalk*NMO);
    V2() = Type(0);
    for(int i=0; i<nwalk; ++i)
      V2(nda::range(i*NMO,(i+1)*NMO),nda::range(i*NMO,(i+1)*NMO)) = utils::make_random<Type>(NMO,NMO); 
    auto Vcsr = math::sparse::to_csr<MEM,int,int>(V2);

    using det_ops::detail::propagate_impl;
    propagate_impl<'N'>(1,SM, P, V, 4);
    propagate_impl<'T'>(1,SM, P, V, 4);
    propagate_impl<'H'>(1,SM, P, V, 4);

    propagate_impl<'N'>(1,SM, Pcsr, V, 4);
    propagate_impl<'T'>(1,SM, Pcsr, V, 4);
    propagate_impl<'H'>(1,SM, Pcsr, V, 4);

    propagate_impl<'N'>(1,SM, P, Vcsr, 4);
    propagate_impl<'T'>(1,SM, P, Vcsr, 4);
    propagate_impl<'H'>(1,SM, P, Vcsr, 4);

    propagate_impl<'N'>(1,SM, Pcsr, Vcsr, 4);
    propagate_impl<'T'>(1,SM, Pcsr, Vcsr, 4);
    propagate_impl<'H'>(1,SM, Pcsr, Vcsr, 4);
  }

  // Propagate noncollinear with diagonal V
  { 
    // not testing results, just checking things run. Get specific answers to test
    array SM(nwalk, 2*NMO, NEL);
    matrix P(2*NMO, 2*NMO); 
    array V(nwalk, 2*NMO, NMO);
    
    SM() = utils::make_random<Type>(nwalk,2*NMO,NEL);
    P() = utils::make_random<Type>(2*NMO,2*NMO);
    auto Vt = utils::make_random<Type>(nwalk,2*NMO,NMO);
    Vt() = Vt()*0.001;
    V() = Vt(); 
    auto Pcsr = math::sparse::to_csr<MEM,int,int>(P);
    
    using det_ops::detail::propagate_impl;
    propagate_impl<'N'>(2, SM, P, V, 4);
    propagate_impl<'T'>(2, SM, P, V, 4);
    propagate_impl<'H'>(2, SM, P, V, 4);
    
    propagate_impl<'N'>(2, SM, Pcsr, V, 4);
    propagate_impl<'T'>(2, SM, Pcsr, V, 4);
    propagate_impl<'H'>(2, SM, Pcsr, V, 4);

    // now with separate spin up/down csr_matrix
  }

  // Orthogonalize
  //SECTION("orthogonalize")
  {
    array Q(B);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    memory::array<MEM,Type,1> oref(nwalk,Type(0.0));
    memory::array<MEM,Type,1> ldet(nwalk,Type(0.0));
    det_ops::orthogonalize(Q,ldet);
    det_ops::Log_Overlap(Q, Q, ovlp);
    CHECK_THAT(oref, utils::Approx(ovlp));
  }

  // Finite temperature functions
  {
    
    // split D matrix test
    {
      nda::array<Type,1> dv = {506.26906413962735, 4.0262905743657393, 0.10523520631662062, 0.0019752342594731467};
      matrix D(nwalk, NMO);
      copy_to_array(dv,D);

      nda::array<Type,1> vmin_ref = {1.0, 1.0, 0.462554865, 0.00868202048};
      nda::array<Type,1> vmax_ref = {0.000449382766, 0.0565057559, 1.0, 1.0};
      matrix dmin_ref(nwalk,NMO);    copy_to_array(vmin_ref,dmin_ref);
      matrix dmax_ref(nwalk,NMO);    copy_to_array(vmax_ref,dmax_ref);

      memory::array<MEM,Type,1> logdet_ref(nwalk,Type(10.5810483195353414));
      memory::array<MEM,Type,1> logdetD(nwalk,Type(0.0));
      memory::array<MEM,Type,1> scl0(nwalk,Type(1.4805672725966263));

      matrix Dmin(nwalk, NMO);
      matrix Dmax(nwalk, NMO);
      math::splitDmatrix(D, Dmin, Dmax, logdetD, scl0);
      CHECK_THAT(Dmin, utils::Approx(dmin_ref));
      CHECK_THAT(Dmax, utils::Approx(dmax_ref));
      CHECK_THAT(logdetD, utils::Approx(logdet_ref));

    }

    // inverse test
    {
      nda::array<Type,2> m_a = {{1.90000 + 0.60000i, 1.40000 + 0.70000i, 0.40000 + 0.80000i,  2.00000 + 0.10000i}, 
                                {1.40000 + 0.90000i, 0.20000 + 0.50000i, 2.20000 + 0.60000i, -0.60000 + 0.80000i}, 
                                {0.40000 + 0.70000i, 2.60000 + 0.80000i, 0.60000 + 0.90000i,  1.40000 - 1.10000i}, 
                                {1.10000 + 0.50000i, 0.30000 + 0.60000i, 0.90000 + 0.70000i,  0.30000 + 0.70000i} 
                              };
      nda::array<Type,2> inv_ref = {{ 0.42120131 - 0.79358205i,  0.24087796 - 0.92608723i, -0.06087011 + 0.14999537i, -0.19921920 + 2.19730690i}, 
                                    {-0.80186316 - 0.02502527i, -0.86324921 - 0.55817475i,  0.62514059 - 0.28389416i,  1.93426356 + 0.72351121i}, 
                                    { 0.05999586 + 0.48146939i,  0.54984296 + 0.82497444i, -0.15735853 - 0.01084387i, -0.69851294 - 1.94260659i}, 
                                    { 0.63360508 + 0.77373987i,  0.17760259 + 1.10656683i, -0.40993694 - 0.05869883i, -1.01426961 - 2.49248591i} 
                                  };
      array ainv_ref(nwalk,NMO,NMO); copy_to_array(inv_ref,ainv_ref);
      array A(nwalk,NMO,NMO); copy_to_array(m_a,A);
      array Ainv(nwalk,NMO,NMO);
      memory::array<MEM,Type,1> logdetA(nwalk,Type(0.0));
      memory::array<MEM,Type,1> logdetA_ref(nwalk,Type(1.3679659506769708-0.727156173959969i));

      det_ops::detail::inverse_logdet(A, logdetA, Ainv);

      CHECK_THAT(Ainv, utils::Approx(ainv_ref));
      CHECK_THAT(logdetA, utils::Approx(logdetA_ref));

    }

    // LU solve test
    {
      nda::array<Type, 2> ma_lu = {{2.1 + 0.2i,  5.0 + 1.1i, -8.0 + 0.0i,  3.4 + 1.0i},
                                  {5.3 + 0.0i, -2.1 + 3.0i,  2.0 + 0.8i,  6.0 - 2.0i},
                                  {7.2 + 0.0i,  5.1 + 0.0i, -4.0 + 2.2i,  6.0 + 0.3i},
                                  {5.0 + 5.0i,  4.2 + 0.0i,  4.2 + 0.0i,  8.0 + 2.0i}
                                  };
      nda::array<Type, 2> mb_lu = {{1.0 + 0.2i, 1.0 + 0.1i, 1.0 - 0.5i,  1.0 + 2.0i},
                                  {1.0 + 0.2i, 1.0 + 0.1i, 1.0 - 0.5i,  1.0 + 2.0i},
                                  {1.0 + 0.2i, 1.0 + 0.1i, 1.0 - 0.5i,  1.0 + 2.0i},
                                  {1.0 + 0.2i, 1.0 + 0.1i, 1.0 - 0.5i,  1.0 + 2.0i}
                                  };

      array A_LU(nwalk,NMO,NMO);  copy_to_array(ma_lu,A_LU);
      array b_LU(nwalk,NMO,NMO);  copy_to_array(mb_lu,b_LU);
      memory::array<MEM,Type,1> ovlp_LU(nwalk,Type(0.0));

      nda::array<Type,2> xv_ref = {{-0.07789269 + 0.03749502i, -0.07278947 + 0.04426364i, -0.04217011 + 0.08487539i, -0.16975077 - 0.08434021i},
                                  { 0.03043269 - 0.06735672i,  0.02337084 - 0.06898762i, -0.01900027 - 0.07877301i,  0.15754602 - 0.03800055i},
                                  {-0.03422502 - 0.01203025i, -0.03472360 - 0.00850803i, -0.03771509 + 0.01262528i, -0.02525056 - 0.07543017i},
                                  { 0.20902585 + 0.03967024i,  0.20882057 + 0.01880871i,  0.20758888 - 0.10636046i,  0.21272093 + 0.41517776i}
                                  };
      
      array x_ref(nwalk,NMO,NMO); copy_to_array(xv_ref,x_ref);
      memory::array<MEM,Type,1> ovlp_ref(nwalk,Type(7.726505629380666+2.2393501663324344i));

      det_ops::detail::LUsolve(A_LU,b_LU,ovlp_LU);

      CHECK_THAT(b_LU, utils::Approx(x_ref));
      CHECK_THAT(ovlp_LU, utils::Approx(ovlp_ref));

    }

    // density matrix test
    {
      
      array G(nwalk, NMO, NMO);
      nda::array<Type,2> v_ref = {{ 0.46345438,  0.28564532,  0.26386935, -0.02614665},
                                  { 0.19396395,  0.65316095, -0.01660395,  0.15844344},
                                  { 0.20418108, -0.00392008,  0.61425088,  0.16850104},
                                  {-0.04220508,  0.33912258,  0.31593366,  0.37148018}
                                };

      //memory::array<MEM,Type,1> pt_ref(nwalk,Type(5.932363985793293+6.283185307179586i));
      memory::array<MEM,Type,1> pt_ref(nwalk,Type(5.932363985793293));

      array g_ref(nwalk, NMO, NMO);  copy_to_array(v_ref,g_ref);
      
      memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));

      {

        memory::array<MEM,Type,1> sclL(1,Type(0.0));
        memory::array<MEM,Type,1> sclR(nwalk,Type(0.22320241149072362));

        nda::array<Type,2> m_ul = {{ 0.50000000,  0.50000000,  0.50000000,  0.50000000},
                                  { 0.50000000, -0.50000000,  0.50000000, -0.50000000},
                                  {-0.50000000, -0.50000000,  0.50000000,  0.50000000},
                                  { 0.50000000, -0.50000000, -0.50000000,  0.50000000}};
        nda::array<Type,2> m_vl = {{ 0.50000000,  0.50000000, -0.50000000,  0.50000000},
                                  { 0.50000000, -0.50000000, -0.50000000, -0.50000000},
                                  { 0.50000000,  0.50000000,  0.50000000, -0.50000000},
                                  { 0.50000000, -0.50000000,  0.50000000,  0.50000000}
                                };
        nda::array<Type,1> m_dl = {35.3041316, 1.0, 1.0, 0.0283252966};
        matrix UL(m_ul);
        vector DL(m_dl);
        matrix VL(m_vl);
      
        nda::array<Type,2> m_ur = {{-0.19527141,  0.12443070,  0.97257064,  0.02219069},
                                  {-0.97012366, -0.08000177, -0.18135273, -0.13989632},
                                  {-0.04090810,  0.98447559, -0.13168960, -0.10859187},
                                  {-0.13804231,  0.09446977, -0.06225259,  0.98394329}
                                  };
        nda::array<Type,1> m_dr = {1.94845449, 1.31603516, 0.75428619, 0.51322728};
        nda::array<Type,2> m_vr = {{-0.18227532, -0.97104555, -0.06830941, -0.13849211},
                                  { 0.24320168,  0.00000000,  0.94671838,  0.21113328},
                                  { 0.99730866,  0.00000000,  0.00000000, -0.07331733},
                                  { 0.00000000,  0.00000000,  0.00000000,  1.00000000}
                                  };    

        array UR(nwalk,NMO,NMO); copy_to_array(m_ur,UR);
        matrix DR(nwalk,NMO); copy_to_array(m_dr,DR);
        array VR(nwalk,NMO,NMO); copy_to_array(m_vr,VR);
      
        // first check computation of only PT = det[G^-1]
        det_ops::Log_Overlap(UL, DL, VL, UR, DR, VR, sclL, sclR, ovlp);

        CHECK_THAT(ovlp, utils::Approx(pt_ref));

        ovlp() = Type(0.0);

        // next check computation of G
        //det_ops::MixedDensityMatrix(UL, DL, VL, UR, DR, VR, G, ovlp, sclL(0), sclR, false, false);
        det_ops::MixedDensityMatrix_v2(UL, DL, VL, UR, DR, VR, G, ovlp, sclL, sclR, false, false);

//        CHECK_THAT(G, utils::Approx(g_ref));
//        CHECK_THAT(ovlp, utils::Approx(pt_ref));
      }

      {
        
        // test with unitaryL = unitaryR = true
        nda::array<Type,2> v_ref = {{ 2.01845628 + 0.61848589i, -0.14007696 + 1.42121428i,  0.85849076 + 0.64936321i,  0.17511422 - 0.13382246i},
                                    { 0.49614863 + 0.55704152i,  0.43209424 + 0.43351669i,  0.61327589 + 0.38363281i, -0.24551059 + 0.88243555i},
                                    {-0.26314996 + 0.11754702i, -0.25208817 - 0.39498037i,  0.31569503 + 0.03437365i, -0.09750865 - 0.12972531i},
                                    {-1.35914169 + 0.15217979i, -0.30619456 - 0.94682906i, -0.74582831 + 0.11717011i,  0.57373942 - 0.39778497i}};

        array g_ref(nwalk, NMO, NMO);  copy_to_array(v_ref,g_ref);

        memory::array<MEM,Type,1> pt_ref(nwalk,Type(2.0386257031701116+2.250415836940481i));

        nda::array<Type,2> m_ul = {{-0.237618 - 0.475639i,  0.557280 + 0.299243i,  0.036048 - 0.330270i,  0.448652 - 0.074379i}, 
                {-0.305195 - 0.383992i, -0.354404 - 0.430958i, -0.097159 + 0.291230i,  0.446308 + 0.393234i}, 
                {-0.446376 - 0.227139i,  0.222971 + 0.224969i,  0.262307 + 0.583585i, -0.485308 + 0.062680i}, 
                {-0.251706 - 0.403126i, -0.418286 - 0.115219i,  0.208887 - 0.585114i, -0.431311 - 0.117788i}};
        nda::array<Type,2> m_vl = {{-0.444569, -0.565290 + 0.079594i, -0.458630 + 0.052340i, -0.512675 + 0.023534i}, 
                { 0.539528,  0.171909 + 0.175095i, -0.493869 - 0.513051i, -0.227021 + 0.300053i}, 
                { 0.584441, -0.085586 - 0.240616i,  0.149100 + 0.503800i, -0.539199 - 0.162564i}, 
                { 0.411941, -0.713791 + 0.197946i, -0.059660 + 0.013674i,  0.509040 - 0.136951i}};
        nda::array<Type,1> m_dl = {2.820064, 1.208732, 0.935833, 0.283179};

        matrix UL(m_ul);
        vector DL(m_dl);
        matrix VL(m_vl);

        nda::array<Type,2> m_ur ={{-0.343131 - 0.185808i, -0.018351 + 0.453505i,  0.531238 - 0.003219i,  0.562208 + 0.208403i}, 
              {-0.415303 - 0.186343i, -0.171196 - 0.128687i, -0.106071 + 0.579018i,  0.117211 - 0.621836i}, 
              {-0.356510 - 0.515866i, -0.372402 + 0.123251i, -0.141028 - 0.558427i, -0.336376 - 0.089615i}, 
              {-0.293124 - 0.401742i,  0.709033 - 0.302533i,  0.165383 + 0.110377i, -0.276500 + 0.206010i}};
        nda::array<Type,1> m_dr ={3.219786, 0.999613, 0.725324, 0.297031};
        nda::array<Type,2> m_vr ={{-0.357210, -0.470761 - 0.008791i, -0.654372 - 0.068256i, -0.466739 + 0.000411i}, 
              { 0.794345, -0.485340 + 0.214702i, -0.113149 + 0.020006i,  0.033013 - 0.270318i}, 
              {-0.309181, -0.594494 + 0.142615i,  0.709217 - 0.004657i, -0.160130 - 0.044656i}, 
              {-0.381869, -0.087885 + 0.339366i, -0.197467 + 0.109234i,  0.634921 - 0.526529i}};

        array UR(nwalk,NMO,NMO); copy_to_array(m_ur,UR);
        matrix DR(nwalk,NMO); copy_to_array(m_dr,DR);
        array VR(nwalk,NMO,NMO); copy_to_array(m_vr,VR);
        
        memory::array<MEM,Type,1> sclL(1,Type(0.128));
        memory::array<MEM,Type,1> sclR(nwalk,Type(0.346));

        ovlp() = Type(0.0);

        det_ops::MixedDensityMatrix(UL, DL, VL, UR, DR, VR, G, ovlp, sclL, sclR, true, true);

        CHECK_THAT(G, utils::Approx(g_ref));
        CHECK_THAT(ovlp, utils::Approx(pt_ref));

      }

    }

    // orthogonalization test
    {

      {
        // for SVD decomp.
        nda::array<Type,2> m_ur_svd = {{ 2.95338068636468 + 8.23020525387664e-01i, 1.89676350413418 + 1.88249259364426i,  2.29295527394499 + 2.54894000652995i,  2.34319847964629e-01 + 5.69443156844357e-01i},
                                      { 1.07343274485525e-01 + 1.07167361317400i, 1.92056258506920 + 2.12568744821291i,  1.05142390313059 + 2.90726943932716i,  2.46015990539850 + 5.04272927692950e-03i},
                                      { 2.04044215935208 + 1.95148570447483i, 1.19694076923315 + 2.81689476834468i,  1.80597460855661 + 1.07259625639558i,  7.51581660566915e-01 + 4.15770164097312e-01i},
                                      { 8.32958929770565e-01 + 1.50468432457451i, 2.47276306737397 + 1.81400088581423i,  2.77946314878267 + 3.64908871831275e-01i,  2.99383312596269 + 7.34679640128662e-01i}
                                      };

        nda::array<Type,1> m_dr_svd = {0.26519788 + 1.51042223i, 1.5752775 + 1.66297619i, 2.87599057 + 2.26262453i, 2.57607341 + 1.70967221i};

        nda::array<Type,2> m_vr_svd = {{ 1.83965103054304 + 5.29357037247383e-01i, 1.68166291717844 +  2.23117260340332i,  3.14025115582430e-01 +  2.23141555986307i,  7.71050134912238e-01 +  1.92197278642536e-01i},
                                      { 1.65538404671351 + 6.70258570197244e-01i, 2.90453815076438 +  7.72131424613451e-01i,  6.99186093684204e-01 +  2.28887778837954i,  2.07529667375284 +  1.75148520153162i},
                                      { 1.50496021591414 + 2.88822990564639e-01i, 1.41407831854736 +  2.32583684589254i,  8.58239250378548e-01 +  1.46127638795796i,  1.78466387814102 +  1.43713954493769e-02i},
                                      { 2.10401127914222 + 2.27165885379993i, 8.33481745909602e-01 +  1.55877970335202i,  1.15031317629187e-01 +  5.94375327177343e-01i,  2.55076604051466 +  3.34673606490562e-01i}
                                      };
        /*
        nda::array<Type,2> m_ur_svd = {{-2.2861125975710705, 0.65274182835413874, -3.43523854892942804E-016,9.31869880197040007E-002},
                                       {-2.2861125975710683, -2.44073701445416273E-015, -0.65274182835413974, -9.31869880197031680E-002},
                                       {-2.2861125975710590, 2.25483453539296768E-016, 0.65274182835413930, -9.31869880197073314E-002},
                                       {-2.2861125975710750, -0.65274182835414007, 2.15580244907982018E-015, 9.31869880197042227E-002}};
        nda::array<Type,1> m_dr_svd = {4.9530324243951060, 1.0000000000000036, 1.0000000000000029, 0.20189651799465577};
        nda::array<Type,2> m_vr_svd = {{-0.50000000000000067,-0.50000000000000011,-0.49999999999999800,-0.50000000000000167},
                                       {0.70710678118654813,-1.04776527778001301E-015,1.49299474187159277E-015,-0.70710678118654713},
                                       {0.0000000000000000,-0.70710678118654735,0.70710678118654779,2.41002114182373110E-015},
                                       {0.49999999999999867,-0.50000000000000022,-0.50000000000000178,0.49999999999999939}};
        */

        array UR(nwalk,NMO,NMO); copy_to_array(m_ur_svd,UR);
        matrix DR(nwalk,NMO); copy_to_array(m_dr_svd,DR);
        array VR(nwalk,NMO,NMO); copy_to_array(m_vr_svd,VR);

        nda::array<Type, 2> ur_ref_svd = {{-3.13074962194799e-01 + 4.05727253574901e-01i, -1.56946908394071e-01 - 5.33243410376989e-01i, -3.01556616691782e-01 - 2.24106472382379e-01i, -1.14318359385931e-01 + 5.23602717393058e-01i},
                                          {-3.84270161991100e-01 + 3.52293726026162e-01i,  5.49207215017357e-01 + 1.56686078437565e-01i, -4.16754112488346e-01 + 2.88016928591331e-01i,  3.05146942521259e-01 - 2.28678995628081e-01i},
                                          {-2.26345495898854e-01 + 3.33101755824866e-01i, -1.14595633095295e-01 - 7.58128418359314e-02i,  4.83154696473021e-01 - 5.08065622180795e-01i,  4.56847574399749e-01 - 3.44459500144956e-01i},
                                          {-1.20674470947688e-01 + 5.37441760354944e-01i, -7.10701673442954e-02 +  5.83874409641062e-01i,  3.18282993820371e-01 +  9.65865063735368e-02i, -4.41147431111152e-01 + 2.13048487493711e-01i}
                                        };
        nda::array<Type, 1> dr_ref_svd = {4.61565437, 1.52510772, 0.9012235,  0.216654};
        
        nda::array<Type, 2> vr_ref_svd = {{ 3.28725423 - 1.07398589i,  4.43006639 + 0.57301881i,  2.34815917 + 1.66192000i,  2.86047582 - 1.48385814i},
                                          { 1.52919972 + 0.70917392i,  0.66189740 - 0.53478447i, -0.15541452 - 0.46046346i,  1.16319431 - 0.49508126i},
                                          {-0.89869699 - 1.13200799i, -1.51036410 - 1.48903646i,  0.28954914 - 2.31601520i, -0.46030020 - 1.90802062i},
                                          { 1.35932660 + 0.22796143i, -0.19359526 + 1.46915652i, -0.03007607 + 0.28876962i,  0.19563680 - 1.39164804i}
                                          };
                
        /*
        nda::array<Type, 2> ur_ref_svd = {{-0.50000000000000044,-0.68378891619938154,-0.18009086063117097,-0.49999999999999861},
                                          {-0.49999999999999994, 0.18009086063117180,-0.68378891619938065, 0.50000000000000022},
                                          {-0.49999999999999795,-0.18009086063117208, 0.68378891619938087, 0.50000000000000189},
                                          {-0.50000000000000144, 0.68378891619937998, 0.18009086063117302,-0.49999999999999928}
                                        };
        nda::array<Type, 1> dr_ref_svd = {22.646379643175244, 0.92311634638663809, 0.92311634638663775, 3.7628256807176116E-002};
        nda::array<Type, 2> vr_ref_svd = {{-0.50000000000000067,-0.50000000000000011,-0.49999999999999800,-0.50000000000000167},
                                          {-0.68378891619938165,0.18009086063117158,-0.18009086063117230,0.68378891619938020},
                                          {-0.18009086063117119,-0.68378891619938031,0.68378891619938131,0.18009086063117258},
                                          {-0.49999999999999867,0.50000000000000067,0.50000000000000133,-0.49999999999999939}
                                        };
        */

        array UR_ref(nwalk,NMO,NMO); copy_to_array(ur_ref_svd,UR_ref);
        matrix DR_ref(nwalk,NMO); copy_to_array(dr_ref_svd,DR_ref);
        array VR_ref(nwalk,NMO,NMO); copy_to_array(vr_ref_svd,VR_ref);

        memory::array<MEM, Type, 1> scl(nwalk,Type(0.0));
        memory::array<MEM, Type, 1> scl_ref(nwalk,Type(1.7667744977546749));

        if(MEM==HOST_MEMORY)
        {
          det_ops::orthogonalize_wSVD(UR,DR,VR,scl);

          CHECK_THAT(UR, utils::Approx(UR_ref));
          CHECK_THAT(DR, utils::Approx(DR_ref));
          CHECK_THAT(VR, utils::Approx(VR_ref));

          CHECK_THAT(scl, utils::Approx(scl_ref));
        }
      }

      {
        // for QR decomp.

        nda::array<Type,2> m_ur_qr = {{-2.06492078,  0.66442708, -0.53408209, -0.67936867},
                                      {-14.87318197, 0.80202448, -3.61794482, -3.45620579},
                                      {-8.43967188,  6.33054898, -3.64130307, -4.36980777},
                                      {-3.08958567,  1.08766034, -1.16092514, -1.03539805}
                                    };
        nda::array<Type,1> m_dr_qr = {11.34491213, 0.89275541, 0.51657435, 0.08814524};
        nda::array<Type,2> m_vr_qr = {{-0.38745756, -0.84147122, -0.22808805, -0.29963088},
                                      { 0.55985516,  0.00000000,  0.76027931,  0.32945040},
                                      { 0.84666756,  0.00000000,  0.00000000, -0.53212220},
                                      { 0.00000000,  0.00000000,  0.00000000,  1.00000000}
                                    };

        array UR(nwalk,NMO,NMO); copy_to_array(m_ur_qr,UR);
        matrix DR(nwalk,NMO); copy_to_array(m_dr_qr,DR);
        array VR(nwalk,NMO,NMO); copy_to_array(m_vr_qr,VR);

        nda::array<Type, 2> ur_ref_qr = {{-1.17995658061036e-01, 3.74066157227406e-02, 6.03730390383380e-01, 7.87519768327402e-01},
                                        {-8.49897443684194e-01, -5.07325053133282e-01,  4.26388554484346e-02, -1.35932165760838e-01},
                                        {-4.82267719617467e-01,  8.57738672631395e-01,  6.75103812037739e-02, -1.64756075812764e-01},
                                        {-1.76548028912200e-01,  7.42051550501439e-02, -7.93179609018852e-01,  5.78092117476014e-01}
                                        };
        nda::array<Type, 1> dr_ref_qr = {2.00237040e+02, 4.67714422, 1.03320492e-01, 4.99408102e-03};
        nda::array<Type, 2> vr_ref_qr = {{-3.86209893710637e-01, -8.41258443758524e-01, -2.41719280228549e-01, -3.10188868329600e-01},
                                        { 4.21185207839637e-01,  0.00000000000000,  7.50451538255373e-01,  3.67947927578920e-01},
                                        { 8.46361103937350e-01,  0.00000000000000,  0.00000000000000, -5.58832858622569e-01},
                                        { 0.00000000000000,  0.00000000000000,  0.00000000000000,  1.00000000000000}
                                        };

        array UR_ref(nwalk,NMO,NMO); copy_to_array(ur_ref_qr,UR_ref);
        matrix DR_ref(nwalk,NMO); copy_to_array(dr_ref_qr,DR_ref);
        array VR_ref(nwalk,NMO,NMO); copy_to_array(vr_ref_qr,VR_ref);

        memory::array<MEM, Type, 1> scl(nwalk,Type(0.0));
        memory::array<MEM, Type, 1> scl_ref(nwalk,Type(-0.008280282078320766));

        det_ops::orthogonalize_wQR(UR,DR,VR,scl);

        CHECK_THAT(UR, utils::Approx(UR_ref));
        CHECK_THAT(DR, utils::Approx(DR_ref));
        CHECK_THAT(VR, utils::Approx(VR_ref));

        CHECK_THAT(scl, utils::Approx(scl_ref));

      }
    }

  }
  
}


TEST_CASE("sdet_ops: operations", "[sdet_ops]")
{
  sdet_ops_operations<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  sdet_ops_operations<DEVICE_MEMORY>();
#endif
}

} // namespace afqmc
} // namespace sfqmc
