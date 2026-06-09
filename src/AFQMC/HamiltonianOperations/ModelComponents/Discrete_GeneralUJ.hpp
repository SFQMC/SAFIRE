/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_DISCRETE_GENERALUJ_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_DISCRETE_GENERALUJ_HPP

#include <vector>
#include <type_traits>
#include <boost/math/tools/roots.hpp>
#if defined(ENABLE_DEVICE)
#include <boost/math/tools/toms748_solve.hpp>
#endif

#include "config.h" // NOLINT(misc-include-cleaner)
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "mpi3/shared_communicator.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/detail/utilities.hpp"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

namespace sfqmc
{
namespace afqmc
{

template<bool MP, bool REAL>
class Discrete_GeneralUJ
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType  >::type;

  // MAM: since the expectation is that the models are very sparse, 
  //      I'm operating under the assumption that all these sparse matrices are
  //      quite small (memory usage). So I'm keeping everything in device memory for now...
  template<class T>
  using csrMat = ma::sparse::csr_matrix<T, int, int, device_allocator<T>>;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;
  using StaticMatrix  = boost::multi::static_array<SPComplexType, 2, device_alloc_type>;

public:

  template<class Vec, class csrM1, class csrM2, class csrM3>
// requires: {Psi(std::move(psi_)) is valid}, {hij(std::move(hij_)) is valid}, ... 
  Discrete_GeneralUJ(afqmc::TaskGroup_& tg_,
                          WALKER_TYPES type,
                          PropagatorTypes ptype,  
                          Vec&& h0_,
                          csrM1&& vn_,
                          csrM2&& vnT_,
                          csrM3&& u_,
                          bool shift_ = false,  
			                    [[maybe_unused]] bool p_shift_ = false,
                          ComplexType e0 = 0
                )
      : TG(tg_),
        walker_type(type),
        propg_type(ptype),
        local_nCV(0),
        shift_one_body_terms(shift_),
        E0(e0),
        h0(std::move(h0_)),
        hMF(h0.extensions(),SPComplexType(0.0)),  // device_allocator is default constructible...
	U(std::move(u_)),
        SpVn(std::move(vn_)),
        SpVnT(std::move(vnT_))
  {
    if(propg_type != DiscreteChargePropagator and 
       propg_type != DiscreteSpinPropagator) 
      APP_ABORT(" Error: Wrong PropagatorTypes argument in Discrete_GeneralUJ. ");      
    local_nCV = SpVn.size(1);
    RUNTIME_CHECK(SpVn.size(0) == SpVnT.size(1), "");
    RUNTIME_CHECK(SpVn.size(1) == SpVnT.size(0), "");
    RUNTIME_CHECK(SpVn.size(0) == h0.size(), "");
    params.reserve(100);
  }

  ~Discrete_GeneralUJ() {}

  Discrete_GeneralUJ(const Discrete_GeneralUJ& other)            = delete;
  Discrete_GeneralUJ& operator=(const Discrete_GeneralUJ& other) = delete;
  Discrete_GeneralUJ(Discrete_GeneralUJ&& other)                 = default;
  Discrete_GeneralUJ& operator=(Discrete_GeneralUJ&& other)      = delete;

  /*
   * n2IJ maps an index in the ordering of the sparse structures to the ordering 
   * of H1 (the generic spin ordering of 1-body operators)
   * n2IJ is expected in host memory. 
   */
  template<class Mat, class map_t>
  void addOneBodyPropagatorMatrix([[maybe_unused]] TaskGroup_& TG_, Mat&& H1, double dt,
                                  [[maybe_unused]] boost::multi::array<ComplexType, 1> const& vMF,
                                  map_t& n2IJ)
  {

    if(not initialized) 
      APP_ABORT(" Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    static_assert(std::decay_t<Mat>::dimensionality == 2,"Incorrect dimensions.");
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO = H1.size(1) / npol;

    RUNTIME_CHECK(H1.size(0) == nspin * npol * NMO, "");
    RUNTIME_CHECK(n2IJ.size() == h0.size(), "");
    RUNTIME_CHECK(n2IJ.size() == hMF.size(), "");

    // if shift_one_body_terms is false, both h0 and hMF are included in vHS
    if(shift_one_body_terms) { 
      SPComplexType Cdt = SPComplexType(SPRealType(dt));
      boost::multi::array_ref<ComplexType, 1> H1D( H1.origin(),
                                                 {H1.num_elements()} );
      Vector<SPComplexType> h0_host(h0.extensions());
      Vector<SPComplexType> hMF_host(hMF.extensions());
      copy_n(h0.origin(),h0.size(),h0_host.origin());
      copy_n(hMF.origin(),hMF.size(),hMF_host.origin());
      for( size_t n=0; n<n2IJ.size(); n++) 
        H1D[ n2IJ[n] ] += static_cast<ComplexType>(Cdt * ( h0_host[n] + hMF_host[n] ));
    }  
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    RUNTIME_CHECK(v.size() == local_nCV, "");
    using std::fill_n;
    fill_n( v.origin(), v.size(), propg_type );
  }

  template<class Vec1, class map_t, class Vec2, class Vec3>
  void update(double dt, Vec1&& nI, Vec2&& n2IJ, map_t& IJ2n, Vec3&& vMF, bool natural_shift)
  {
    // store dt and check?
    initialized = true; 
    setup_Vn_hmf(dt,nI,n2IJ,IJ2n,std::forward<Vec3>(vMF),natural_shift);
  }

  // v(IJ,w) = sum_n Vn(IJ,n) X(n,w)
  template<class MatX,
           class MatV,
           typename = typename std::enable_if_t<(std::decay<MatX>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatV>::type::dimensionality == 2)>,
           typename = void
          >
  void vHS(MatX&& X, MatV&& v, double dt, double a = 1.)
  {
    if(not initialized) 
      APP_ABORT(" Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    RUNTIME_CHECK(X.size(1) == v.size(1), "");
    RUNTIME_CHECK(SpVn.size(1) == X.size(0), "");
    RUNTIME_CHECK(SpVn.size(0) == v.size(0), "");
    RUNTIME_CHECK(v.size(0) == h0.size(0), "");
    RUNTIME_CHECK(v.size(0) == hMF.size(0), "");

    SPComplexType ia = SPComplexType(SPRealType(a));
    ma::product(ia, SpVn, X, SPComplexType(1.0,0.0), v);

    if(shift_one_body_terms) return;
  
    // multiply by '-i' to compensate for factor of 'i' implicit in the propagator
    ia = SPComplexType(0.0, SPRealType(dt*a));

    // v[n][iw] += ia*h0[n];
    ma::elementwise(ma::TOp_PLUS, 0, ia, h0, v);
    ma::elementwise(ma::TOp_PLUS, 0, ia, hMF, v);
  }

  // v(n,w) = sum_IJ VnT(n,IJ) G(w,IJ)
  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vbias(const MatA& G, MatB&& v, [[maybe_unused]] double dt, double a = 1.)
  {
    if(not initialized) 
      APP_ABORT(" Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    SPComplexType ia = SPComplexType(SPRealType(a)); 
    RUNTIME_CHECK(SpVnT.size(0) == v.size(0), "");
    RUNTIME_CHECK(SpVnT.size(1) == G.size(0), "");
    RUNTIME_CHECK(G.size(1) == v.size(1), "");

    ma::product(ia, SpVnT, G, SPComplexType(1.0,0.0), v);
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  int number_of_ke_vectors() const { return local_nCV; }
  int local_number_of_cholesky_vectors() const { return local_nCV; }

private:

  afqmc::TaskGroup_& TG;

  WALKER_TYPES walker_type;

  PropagatorTypes propg_type;

  // this class unfortunately doesn't quite satisfy RAII, 
  // since it needs a timestep and on-site MF occupancies to setup properly.
  bool initialized = false;

  int local_nCV;
  // if shift_one_body_terms=true, h0 is added to the one body propagator.
  // otherwise it is added through vHS
  bool shift_one_body_terms = false;

  ComplexType E0 = 0.0;

  // All sparse matrices have a compact representation of IJ, as defined
  // by n2IJ in ModelHamOps. 

  // constant one-body term associated with the 
  // interacting term.
  Vector<SPComplexType, device_allocator<SPComplexType>> h0;

  // 1-body part of MF substaction. Not given by vMF in discrete case!!!
  Vector<SPComplexType, device_allocator<SPComplexType>> hMF;

  // need to keep a copy of the U matrix. Keeping on host memory.
  ma::sparse::csr_matrix<SPValueType,int,int> U;

  // HS operator 
  csrMat<SPComplexType> SpVn;

  // transposed HS operator 
  csrMat<SPComplexType> SpVnT;

  // save parameters to avoid excesve recalculation
  std::vector<std::tuple<SPRealType,SPRealType,SPComplexType,SPComplexType>> params; 

  // HS operator and 1-body MF substraction terms depend on timestep
  // and on MF onsite occupations.    
  // follows addComponent from Hamiltonians/ModelHamOpsGenerator.icc, 
  // but uses correct prefactors, which are now functions of dt*U and nMFJ = <nI +- nJ>   
  // assumes nMF[i] = <c^{+}_i c_i>_MF is the mean-field site occupations. UHF convenstion for
  // spin ordering (e.g. all up, followed by all down).   
  // assuming onsite densities are real...
  template<class Vec1, class map_t, class Vec2, class Vec3>     
  void setup_Vn_hmf(double dt, Vec1&& nMF, Vec2&& n2IJ, map_t& IJ2n,
		    Vec3&& vMF, bool natural_shift)
  {

#if defined(ENABLE_DEVICE)
    using boost::multi::memory::cuda::fill_n;
# else
    using std::fill_n;
#endif
    static_assert(std::decay_t<Vec1>::dimensionality == 1,"Incorrect dimensions.");
    //int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    //int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO = U.size(0) / 2;

// csrMat need to think how to update matrices on device, might need new access pattern 
#if defined(ENABLE_DEVICE)
    APP_ABORT("Error: Discrete factorization of model hamiltonian not yet available on GPU.");	
#endif
// you also want a way to guarantee also in the cpu that the term already exists, 
// not being dynamically added. Add another version of += that works on gpus and that
// oborts if the term doesn't already exists...

    RUNTIME_CHECK(nMF.size(0) == 2 * NMO, "");
    RUNTIME_CHECK(n2IJ.size() == hMF.size(), "");

    int nFields = 0;
    SPRealType sign = (propg_type == DiscreteChargePropagator) ? SPRealType(1.0) : SPRealType(-1.0);
    size_t M = size_t(NMO);	
    size_t M2 = (walker_type == NONCOLLINEAR) ? 2ul*M : M;
    size_t Madd = (walker_type == NONCOLLINEAR) ? M : 0ul;

    // set to zero
    params.clear();
    ma::fill(hMF,SPComplexType(0.0));      	
    fill_n(vMF.origin(),vMF.num_elements(),SPComplexType(0.0));      	
    fill_n(SpVn.non_zero_values_data(),SpVn.capacity(),SPComplexType(0.0));   
    fill_n(SpVnT.non_zero_values_data(),SpVnT.capacity(),SPComplexType(0.0));   
    
    SPComplexType scl = (propg_type == DiscreteChargePropagator?(1.0):SPComplexType(0.0,-1.0));

    // opposite spin terms
    for(size_t i=0; i<M; ++i) {
      auto vals = U.non_zero_values_data(i);
      auto cols = U.non_zero_indices2_data(i);
      auto nnz = U.num_non_zero_elements(i);
      while( (nnz--) > 0 ) {
        SPValueType Uij = static_cast<SPValueType>(*(vals++));
        size_t j = size_t(*(cols++));
        if( std::abs(Uij) < 1e-6 ) continue;
        if( i > j ) continue;
        // add i up/j dn term 
        size_t I_ = i*M2+i;           // (i up, i up)
        auto nI = IJ2n[I_];
        size_t J_ = (j+M)*M2+j+Madd;  // (j dn, j dn)
        auto nJ = IJ2n[J_];
        auto nMF_ij = SPComplexType(nMF[i]) + sign*SPComplexType(nMF[j+M]); 
        auto [alpha,n_ij,nMF_ij_c] = get_parameters(dt,Uij,nMF_ij,natural_shift);
        vMF[nFields] = alpha * scl * nMF_ij_c; 
        hMF[nI] += sign*Uij*n_ij;
        hMF[nJ] += Uij*n_ij;
        SpVnT[nFields][nI] += alpha * scl; 
        SpVnT[nFields][nJ] += sign * alpha * scl; 
        SpVn[nI][nFields] += alpha * scl; 
        SpVn[nJ][nFields++] += sign * alpha * scl; 
        if( i != j ) {
          // add i dn/j up term 
          I_ = (i+M)*M2+i+Madd;      // (i dn, i dn)
          nI = IJ2n[I_];
          J_ = j*M2+j;               // (j up, j up)
          nJ = IJ2n[J_];
          nMF_ij = SPComplexType(nMF[i+M]) + sign*SPComplexType(nMF[j]); 
          std::tie(alpha,n_ij,nMF_ij_c) = get_parameters(dt,Uij,nMF_ij,natural_shift);
          vMF[nFields] = alpha * scl * nMF_ij_c; 
          hMF[nI] += sign*Uij*n_ij;
          hMF[nJ] += Uij*n_ij;
          SpVnT[nFields][nI] += alpha * scl; 
          SpVnT[nFields][nJ] += sign * alpha * scl; 
          SpVn[nI][nFields] += alpha * scl; 
          SpVn[nJ][nFields++] += sign * alpha * scl; 
        }
      }
    }
    // same spin terms
    for(size_t i=0; i<M; ++i) {
      auto vals = U.non_zero_values_data(i+M);
      auto cols = U.non_zero_indices2_data(i+M);
      auto nnz = U.num_non_zero_elements(i+M);
      while( (nnz--) > 0 ) {
        SPValueType Uij = static_cast<SPValueType>(*(vals++));
        size_t j = size_t(*(cols++));
        if( std::abs(Uij) < 1e-6 ) continue;
        if( i >= j ) continue;
        // add up/up term 
        size_t I_ = i*M2+i;      // (i up, i up)
        auto nI = IJ2n[I_];
        size_t J_ = j*M2+j;      // (j up, j up)
        auto nJ = IJ2n[J_];
        auto nMF_ij = SPComplexType(nMF[i]) + sign*SPComplexType(nMF[j+M]); 
        auto [alpha,n_ij,nMF_ij_c] = get_parameters(dt,Uij,nMF_ij,natural_shift);
        vMF[nFields] = alpha * scl * nMF_ij_c; 
        hMF[nI] += sign*Uij*n_ij; 
        hMF[nJ] += Uij*n_ij; 
        SpVnT[nFields][nI] += alpha * scl; 
        SpVnT[nFields][nJ] += sign * alpha * scl; 
        SpVn[nI][nFields] += alpha * scl; 
        SpVn[nJ][nFields++] += sign * alpha * scl; 
        // add dn/dn term 
        I_ = (i+M)*M2+i+Madd;      // (i dn, i dn)
        nI = IJ2n[I_];
        J_ = (j+M)*M2+j+Madd;      // (j dn, j dn)
        nJ = IJ2n[J_];
        nMF_ij = SPComplexType(nMF[i+M]) + sign*SPComplexType(nMF[j]); 
        std::tie(alpha,n_ij,nMF_ij_c) = get_parameters(dt,Uij,nMF_ij,natural_shift);
        vMF[nFields] = alpha * scl * nMF_ij_c; 
        hMF[nI] += sign*Uij*n_ij;
        hMF[nJ] += Uij*n_ij;
        SpVnT[nFields][nI] += alpha * scl; 
        SpVnT[nFields][nJ] += sign * alpha * scl; 
        SpVn[nI][nFields] += alpha * scl; 
        SpVn[nJ][nFields++] += sign * alpha * scl; 
      }
    }

  }

  // not sure if these equations hold for complex U (actual non-zero complex part...)
  template<typename T>
  std::tuple<SPComplexType,SPComplexType,SPComplexType> get_parameters(double dt, SPValueType U_, T nMF_, bool natural_shift)
  {
    using std::log;
    using std::cos;
    using std::acos;
    using std::cosh;
    using std::abs;
    using std::sqrt;
    using std::get;
    using boost::math::tools::bisect;
    using boost::math::tools::eps_tolerance;
    RUNTIME_CHECK(dt>0.0, "Error in Discrete_GeneralUJ::get_parameters:"
      " dt > 0 is required to determine the parameters of the discrete HS transformation.");
    if(abs(U_) < 1e-8)
      APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: U==0.");
    if(abs(ma::imag(U_)) > 1e-8)
      APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: imag(U) > 0 not yet allowed.");
    if(abs(ma::imag(nMF_)) > 1e-8)
      APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: imag(nMF) > 0. Should not happen.");

    SPRealType nMF = ma::real(nMF_);
    if(propg_type == DiscreteChargePropagator) {
      if(natural_shift) nMF = 1.0;
      if(abs(nMF) < 0.0)
        APP_ABORT("Error in Discrete_GeneralUJ(charge)::get_parameters: nMF<0.");
      if(abs(nMF) > 2.0)
        APP_ABORT("Error in Discrete_GeneralUJ(charge)::get_parameters: nMF>2.");
    } else {
      if(natural_shift) nMF = 0.0;
      if(abs(nMF) > 1.0)
        APP_ABORT("Error in Discrete_GeneralUJ(spin)::get_parameters: abs(nMF)>1.");
    }
    // just keep real part for now... Not sure what to do otherwise
    SPRealType Ud = ma::real(U_);
    // look in table
    for( auto& v : params )
      if( (abs(Ud-get<0>(v)) < 1e-4) and
	  (abs(nMF-get<1>(v)) < 1e-4) ) 
	return std::make_tuple(get<2>(v),get<3>(v),SPComplexType(nMF));

    SPComplexType half(0.5);
    SPComplexType one(1.0);
    SPComplexType two(2.0);
    SPComplexType alpha(0.0),n(0.0);
    if(propg_type == DiscreteChargePropagator) {

      if( abs(nMF-one) < 1e-8 ) {
	// n=1 
	alpha = acos( exp( -SPComplexType(SPRealType(dt))*half*SPComplexType(Ud) ) );   	
      } else if((abs(nMF) < 1e-8) or (abs(nMF-two) < 1e-8)) {
	// n=0/2 
	alpha = acos( sqrt( one / ( two - exp( -SPComplexType(SPRealType(dt))*SPComplexType(Ud) ) ) ) );   	
      } else {	
	// generic case

        double Ud_ = double(Ud);
        ComplexType a0_c = acos( sqrt( 1.0 / ( 2.0 - exp( -dt*ComplexType(Ud_) ) ) ) );
        ComplexType a1_c = acos( exp( -dt*0.5*ComplexType(Ud_) ) );
	double n_ = (nMF < SPRealType(1.0)) ? double(nMF) : 2.0-double(nMF) ; 
        if(Ud < 0.0) {
	  // alpha is imag 
	  // just checking
	  if( (std::abs(ma::real(a0_c)) > 1e-8) or (std::abs(ma::real(a1_c)) > 1e-8) )   	
	    APP_ABORT(" Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = ma::imag(a0_c); 
          double a1 = ma::imag(a1_c); 
          auto f = [&](double a)
            {
              ComplexType c = cos(ComplexType(0.0,a)*(1.0-n_));
              return exp(-1.0*dt*Ud_) - ma::real(cos(ComplexType(0.0,a)*(2.0-n_)) * cos(ComplexType(0.0,a)*n_) / (c*c));
            };
          // [a1,a0] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 )
            APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),eps_tolerance<double>(std::numeric_limits<double>::digits - 1),miter);
          if(miter == 300)
            APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          alpha = SPComplexType(0.0,SPRealType(root.first));	     
        } else {
          // alpha is real
	  if( (std::abs(ma::imag(a0_c)) > 1e-8) or (std::abs(ma::imag(a1_c)) > 1e-8) )   	
	    APP_ABORT(" Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = ma::real(a0_c); 
          double a1 = ma::real(a1_c); 
          auto f = [&](double a)
	    {
	      double c = cos(a*(1.0-n_));
              return exp(-1.0*dt*Ud_) - cos(a*(2.0-n_)) * cos(a*n_) / (c*c);
            }; 
	  // [a0,a1] should bracket the root and f(a) should be monotonically increasing
	  if( f(a0)*f(a1) > 0.0 ) 
	    APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
	  std::uintmax_t miter(300);
	  auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),eps_tolerance<double>(std::numeric_limits<double>::digits - 1),miter);
	  if(miter == 300)
	    APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
          alpha = SPComplexType(SPRealType(root.first));
        }

      }	
      if(  abs(cos(alpha*(one-nMF))) < 1e-8 )
	APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: cosh(a*(1-n) == 0.0)");
      n = one - log( cos(alpha*nMF) / cos(alpha*(two-nMF)) ) / (two*SPComplexType(SPRealType(dt))*SPComplexType(Ud)); 

    } else {

      if( (abs(nMF-one) < 1e-8) or (abs(nMF+one) < 1e-8) ) {
	// n=1/-1 
	alpha = acosh( sqrt( one / ( two - exp( SPComplexType(SPRealType(dt))*SPComplexType(Ud) ) ) ) );   	
      } else if(abs(nMF) < 1e-8) {
	// n=0 
	alpha = acosh( exp( SPComplexType(SPRealType(dt))*half*SPComplexType(Ud) ) );  
      } else {	
	// generic case
        double Ud_ = double(Ud);
        ComplexType a0_c = acosh( sqrt( 1.0 / ( 2.0 - exp( dt*ComplexType(Ud_) ) ) ) );
        ComplexType a1_c = acosh( exp( dt*0.5*ComplexType(Ud_) ) );
	double n_ = double(abs(nMF)); 
        if(Ud < 0.0) {
	  // alpha is imag 
	  // just checking
	  if( (std::abs(ma::real(a0_c)) > 1e-8) or (std::abs(ma::real(a1_c)) > 1e-8) )   	
	    APP_ABORT(" Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = ma::imag(a0_c); 
          double a1 = ma::imag(a1_c); 
          auto f = [&](double a)
            {
              ComplexType c = cosh(ComplexType(0.0,a)*n_);
              return exp(dt*Ud_) - ma::real(cosh(ComplexType(0.0,a)*(1.0-n_)) * cosh(ComplexType(0.0,a)*(1.0+n_)) / (c*c));
            };
          // [a1,a0] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 )
            APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),eps_tolerance<double>(std::numeric_limits<double>::digits - 1),miter);
          if(miter == 300)
            APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          alpha = SPComplexType(0.0,SPRealType(root.first));	     
        } else {
          // alpha is real
	  if( (std::abs(ma::imag(a0_c)) > 1e-8) or (std::abs(ma::imag(a1_c)) > 1e-8) )   	
	    APP_ABORT(" Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = ma::real(a0_c); 
          double a1 = ma::real(a1_c); 
          auto f = [&](double a)
	    {
	      double c = cosh(a*n_);
              return exp(dt*Ud_) - cosh(a*(1.0-n_)) * cosh(a*(1.0+n_)) / (c*c);
            }; 
	  // [a0,a1] should bracket the root and f(a) should be monotonically increasing
	  if( f(a0)*f(a1) > 0.0 ) 
	    APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
	  std::uintmax_t miter(300);
	  auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),eps_tolerance<double>(std::numeric_limits<double>::digits - 1),miter);
	  if(miter == 300)
	    APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
          alpha = SPComplexType(SPRealType(root.first));
        }

      }	
      if(  abs(cosh(alpha*(one-nMF))) < 1e-8 )
	APP_ABORT("Error in Discrete_GeneralUJ::get_parameters: cosh(a*(1-n) == 0.0)");
      n = log(cosh(alpha*(one+nMF)) / cosh(alpha*(one-nMF))) / (two*SPComplexType(SPRealType(dt))*SPComplexType(Ud)); 

    }	
    app_log(1," Discrete Propagator parameter: dt={}, U={}, nMF={}, alpha={}, n={}",dt,double(Ud),
		double(nMF),alpha,n);
    params.emplace_back(std::make_tuple(Ud,nMF,alpha,n));	
    return std::make_tuple(alpha,n,SPComplexType(nMF));
  }

};

} // namespace afqmc

} // namespace sfqmc

#endif
