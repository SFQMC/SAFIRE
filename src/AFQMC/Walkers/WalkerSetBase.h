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

#pragma once

#include <memory>
#include <span>
#include <utility>

#include "configuration.hpp"
#include "config.h"
#include "IO/AppAbort.hpp"
#include "AFQMC/parameters.hpp"
#include "utilities/Random.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"

#include "AFQMC/Walkers/Walkers.hpp"
#include "AFQMC/Walkers/WalkerControl.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/Walkers/WalkerSetInitialGuess.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that contains and handles walkers.
 * Implements communication, load balancing, and I/O operations.   
 * Walkers are always accessed through the handler.
 *
 */

template<MEMORY_SPACE MEM>
class WalkerSetBase
{
public:
  // contiguous_walker = true means that all the data of a walker is continguous in memory
  static const bool contiguous_walker = true;
  // contiguous_storage = true means that the data of all walker is continguous in memory
  static const bool contiguous_storage = true;
  static const bool fixed_population   = true;

  using reference = walker<MEM,ComplexType>;
  using const_reference = walker<MEM,const ComplexType>;

  /// Constructor: build a set of nWalkers walkers of the given dimensions
  /// {rows, naea, naeb}. Walkers are allocated and initialized to valid default
  /// values (unit weight/overlap/phase, zero Slater matrices) but carry no
  /// initial guess; it is for callers that restore walkers from elsewhere, such
  /// as an HDF5 restart file.
  WalkerSetBase(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> r,
                const WalkerSetParameters& params,
                std::array<int, 3> dims,
                int nWalkers,
                bool finite_temperature_
               )
      : mpi(mpi),
        rng(r),
        walker_size(1),
        walker_memory_usage(0),
        bp_walker_size(0),
        bp_walker_memory_usage(0),
        tau_step(0),
        history_pos(0),
        walkerType(params.walker_type),
        finite_temperature(finite_temperature_),
        tot_num_walkers(0),
        walker_buffer(0, 1),
        bp_buffer(0, 0)
  {
    // parse fills load_balance, pop_control, min_weight and max_weight from params
    parse(params);
    setup(dims);
    allocate_walkers(nWalkers);
  }

  /// Constructor: build a set of nWalkers walkers from the trial wavefunction's
  /// initial guess. The dimensions and the finite-temperature flag are derived
  /// from the guess, so no external NMO/nup/ndown is needed. Every walker is
  /// initialized to the guess.
  WalkerSetBase(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> r,
                const WalkerSetParameters& params,
                const WalkerSetInitialGuess& guess,
                int nWalkers
               )
      : WalkerSetBase(mpi, r, params, guess.walker_dims(), nWalkers, guess.isFiniteTemperature())
  {
    utils::check(guess.walker_type == params.walker_type,
                 "WalkerSetBase: the initial guess is {} but the walker set is {}.",
                 walkerTypeToString(guess.walker_type), walkerTypeToString(params.walker_type));
    if(finite_temperature) {
      populate_from_guess_ft(guess.udv());
    } else {
      populate_from_guess(guess.slater());
    }
  }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

  /*
   * Returns the current number of walkers in the set.
   */
  int size() const { return tot_num_walkers; }

  /*
   * Returns the maximum number of walkers in the set that can be stored without reallocation.
   */
  int capacity() const { return int(walker_buffer.extent(0)); }

  /*
   * Returns the maximum number of fields in the set that can be stored without reallocation. 
   */
  int NumBackProp() const { return wlk_desc[3]; }
  /*
   * Returns the maximum number of cholesky vectors in the set that can be stored without reallocation. 
   */
  int NumCholVecs() const { return wlk_desc[4]; }
  /*
   * Returns the length of the history buffers. 
   */
  int HistoryBufferLength() const { return wlk_desc[6]; }

   /*
   * Current imaginary-time slice index. Set to 0 for ground state walker types.
   * Used by FT wavefunction routines to select DL matrix slice.
   */
  int getTauStep() const { return tau_step; }
  void setTauStep(int p) { tau_step = p; }
  void advanceTauStep() { tau_step++; }

  /*
   * Returns and advances the position of the insertion point in the History circular buffers.
   * The FIELDS ring is indexed by history_pos % NumBackProp(), so a single cursor drives
   * both it and the (three times longer) weight history.
   */
  int getHistoryPos() const { return history_pos; }
  void advanceHistoryPos() { history_pos = (history_pos + 1) % wlk_desc[6]; }


  /*
   * Returns a reference to a walker
   */
  auto operator[](int i)
  {
    utils::check(i>=0 and i<tot_num_walkers, "error: index out of bounds.");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return reference(walker_buffer(i,nda::range::all), data_displ, wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  auto operator[](int i) const
  {
    utils::check(i>=0 and i<tot_num_walkers, "error: index out of bounds.");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return const_reference(walker_buffer(i,nda::range::all), data_displ, wlk_desc);
  }

  /*
   * Increases the capacity of the containers to n.
   */
  void reserve(int n);

  /*
   * Adds/removes the number of walkers in the set to match the requested value.
   * Walkers are removed from the end of the set 
   *     and buffer capacity remains unchanged in this case.
   * New walkers are initialized from already existing walkers in a round-robin fashion. 
   * If the set is empty, calling this routine will abort. 
   * Capacity is increased if necessary.
   * Target Populations are set to n.
   */
  void resize(int n);

  /*
   * Finite temperature reset walkers at the beginning of each sweep
  */
  void reset(int n);

  // cleans state of object.
  //   -erases allocated memory
  bool clean();

  /*
   * Resizes back propagation buffers.
   * Must be called before any call to bp-related routines.
   */     
  void resize_bp(int nbp, int nCV, int nref);

  // perform and report tests/timings
  void benchmark(std::string& blist, int maxnW, int delnW, int repeat);

  auto get_target_population() const { return targetN_per_rank; }
  auto get_global_target_population() const { return targetN; }

  /// Dimensions {rows, naea, naeb} of the walkers in this set, in the same layout as
  /// WalkerSetInitialGuess::walker_dims().
  auto walker_dims() const { return std::array<int, 3>{wlk_desc[0], wlk_desc[1], wlk_desc[2]}; }

  auto GlobalPopulation() const
  {
    int res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    res += tot_num_walkers;
    return (mpi->comm += res);
  }

  auto GlobalWeight() const
  {
    RealType res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    nda::array<ComplexType, 1> buff(tot_num_walkers,ComplexType(0.0));
    getProperty(WEIGHT, buff);
    for (int i = 0; i < tot_num_walkers; i++)
      res += std::abs(buff(i));
    return (mpi->comm += res);
  }

  private:
  /*
   * Populates every walker's Slater matrix from the per-spin guess. The set must already be
   * sized; each guess matrix is exactly (rows x naea)/(NMO x naeb).
   */
  void populate_from_guess(WalkerSetInitialGuess::slater_guess guess);

  /*
   * Populates every walker's finite-temperature U/D/V matrices from the rank-4 guess
   * {3, nspin, rows, naea} (D is a full matrix; its diagonal is used). The set must already
   * be sized.
   */
  void populate_from_guess_ft(WalkerSetInitialGuess::udv_guess UDV);

  template<walker_data D>
  auto extract_SM( SpinTypes s ) {
    static_assert(D == SM or D == SMN, "Invalid enum");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,ComplexType,3>(idxm, walker_buffer.data() + i0);     
  } 

  template<walker_data D>
  auto extract_SM( SpinTypes s ) const {
    static_assert(D == SM or D == SMN, "Invalid enum");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,const ComplexType,3>(idxm, walker_buffer.data() + i0); 
  }

  /*
   * extract finite temperature walker matrices
  */
  template<walker_data D>
  auto extract_UM( SpinTypes s ) {
    utils::check(D == UR or D == VR, "Invalid enum");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,ComplexType,3>(idxm, walker_buffer.data() + i0);     
  } 

  template<walker_data D>
  auto extract_UM( SpinTypes s ) const {
    static_assert(D == UR or D == VR, "Invalid enum");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,const ComplexType,3>(idxm, walker_buffer.data() + i0); 
  }

  auto extract_DM( SpinTypes s ) {
    auto i0 = (s==Alpha?data_displ[DR]:data_displ[DR]+wlk_desc[0]);
    std::array<long,2> shape = {tot_num_walkers,wlk_desc[0]};
    std::array<long,2> strides = {walker_buffer.strides()[0],1};
    nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,ComplexType,2>(idxm, walker_buffer.data() + i0);     
  } 

  auto extract_DM( SpinTypes s ) const {
    auto i0 = (s==Alpha?data_displ[DR]:data_displ[DR]+wlk_desc[0]);
    std::array<long,2> shape = {tot_num_walkers,wlk_desc[0]};
    std::array<long,2> strides = {walker_buffer.strides()[0],1};
    nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,const ComplexType,2>(idxm, walker_buffer.data() + i0); 
  }

  public:

  auto SlaterMatrices( SpinTypes s )  
  {
    return extract_SM<SM>(s);
  } 

  auto SlaterMatrices( SpinTypes s ) const
  {
    return extract_SM<SM>(s);
  }

  auto UMatrices( SpinTypes s )  
  {
    return extract_UM<UR>(s);
  } 

  auto DMatrices( SpinTypes s )  
  {
    return extract_DM(s);
  } 

  auto VMatrices( SpinTypes s )  
  {
    return extract_UM<VR>(s);
  } 

  auto UMatrices( SpinTypes s ) const
  {
    return extract_UM<UR>(s);
  } 

  auto DMatrices( SpinTypes s ) const  
  {
    return extract_DM(s);
  } 

  auto VMatrices( SpinTypes s ) const  
  {
    return extract_UM<VR>(s);
  } 

  auto SlaterMatricesN( SpinTypes s )
  {
    utils::check(data_displ[SMN]>=0, "access to uninitialized BP sector. ");
    return extract_SM<SMN>(s);
  }

  auto SlaterMatricesN( SpinTypes s ) const
  {
    utils::check(data_displ[SMN]>=0, "access to uninitialized BP sector. ");
    return extract_SM<SMN>(s);
  }

  /*
   * Rescales every walker weight so that the total weight over the whole population equals
   * the global target population, leaving the mean weight at 1.
   *
   * Collective over the walker set's communicator.
   */
  void rescale_total_weight();

  void popControl();

  // M holds the incoming walkers packed as {walker_buffer row, bp_buffer row}; it comes
  // from an MPI receive buffer, so it is on the host even when the set lives on a device.
  void push_walkers(memory::array_view<HOST_MEMORY, const ComplexType, 2> M);

  void pop_walkers(memory::array_view<HOST_MEMORY, ComplexType, 2> M);

  // given a list of new weights and counts, add/remove walkers and reassign weight accordingly.
  // counts is one {weight, multiplicity} entry per local walker and is reordered in place;
  // walkers beyond the target population are written to M. The new weights are magnitudes, so
  // each walker keeps the phase it had before the branch.
  void branch(std::span<std::pair<double, int>> counts,
              memory::array_view<MEM, ComplexType, 2> M);

  auto get_mpi() const { return mpi; }

  int single_walker_memory_usage() const { return walker_memory_usage; }
  int single_walker_size() const { return walker_size; }
  int single_walker_bp_memory_usage() const { return (wlk_desc[3] > 0) ? bp_walker_memory_usage : 0; }
  int single_walker_bp_size() const { return (wlk_desc[3] > 0) ? bp_walker_size : 0; }

  WALKER_TYPES getWalkerType() const { return walkerType; }

  bool isFiniteTemperature() const { return finite_temperature; }

  std::tuple<BranchingAlgorithm,int,int> population_control_parameters() const 
  { return std::make_tuple(pop_control,min_weight,max_weight); }

  int walkerSizeIO() const
  {
    if (finite_temperature)
      return walker_size; //finite-T walkers include U,D,V matrices
    else if (walkerType == COLLINEAR)
      return wlk_desc[0] * (wlk_desc[1] + wlk_desc[2]) + 10;
    else // since NAEB = 0 in both CLOSED and NONCOLLINEAR cases
      return wlk_desc[0] * wlk_desc[1] + 10;
    return 0;
  }

  // I am going to assume that the relevant data to be copied is continuous,
  // careful not to break this in the future
  void copyToIO(nda::MemoryArrayOfRank<1> auto&& x, int n) const
  {
    using nda::range;
    utils::check(n < tot_num_walkers, "Incorrect argument");
    utils::check(x.size() >= walkerSizeIO(), "Size mismatch");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    x(range(walkerSizeIO())) = walker_buffer(n,range(walkerSizeIO()));
  }

  void copyFromIO(nda::MemoryArrayOfRank<1> auto&& x, int n)
  {
    using nda::range;
    utils::check(n < tot_num_walkers, "Incorrect argument");
    utils::check(x.size() >= walkerSizeIO(), "Size mismatch");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    walker_buffer(n,range(walkerSizeIO())) = x(range(walkerSizeIO()));
  }

  /*
   * Writable view of one scalar property across the population. Properties are columns of
   * walker_buffer, so the view is strided by the walker size and lives in MEM.
   */
  auto getProperty(walker_data id) {
    return walker_buffer(nda::range(tot_num_walkers), data_displ[id]);
  }

  template<typename Arr>
  void getProperty(walker_data id, Arr&& v) const
  //void getProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v) const
  {
    using nda::range;
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    v(range(tot_num_walkers)) = walker_buffer(range(tot_num_walkers),data_displ[id]);
  }

  void setProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v)
  {
    using nda::range;
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    walker_buffer(range(tot_num_walkers),data_displ[id]) = v(range(tot_num_walkers));
  }

  void resetWeights()
  {
    mpi->comm.barrier();
    {
      memory::array<MEM, ComplexType, 1> w_(tot_num_walkers, ComplexType(1.0));
      setProperty(WEIGHT, w_);
    }
    mpi->comm.barrier();
  }

  auto getFields(int ip)
  {
    using nda::range;
    utils::check(ip>=0 and ip<wlk_desc[3], " Error: index out of bounds in getFields. ");
    long i0 = (data_displ[FIELDS] + ip * wlk_desc[4]);
    return bp_buffer(range::all,range(i0,i0+wlk_desc[4]));
  }

  auto getFields()
  {
    using nda::range;
    long i0 = data_displ[FIELDS];
    long nw = bp_buffer.extent(0);
    std::array<long,3> shape = {nw,wlk_desc[3],wlk_desc[4]};
    std::array<long,3> strides = {bp_buffer.strides()[0],wlk_desc[4],1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<MEM,ComplexType,3>(idxm, bp_buffer.data() + i0);
  }

  void storeFields(int ip, nda::MemoryArrayOfRank<2> auto&& V)
  {
    utils::check(ip>=0 and ip<wlk_desc[3], " Error: index out of bounds in getFields. ");
    long nw = bp_buffer.extent(0);
    utils::check(V.shape() == std::array<long,2>{nw,wlk_desc[4]}, 
                 "Shape mismatch");
    auto F = getFields(ip);
    F() = V();
  }

  auto getWeightFactors()
  {
    using nda::range;
    long i0 = data_displ[WEIGHT_FAC];
    return bp_buffer(range::all,range(i0,i0+wlk_desc[6]));
  }

  auto getWeightHistory()
  {
    using nda::range;
    long i0 = data_displ[WEIGHT_HISTORY];
    return bp_buffer(range::all,range(i0,i0+wlk_desc[6]));
  }

  // load balancing algorithm
  void loadBalance(nda::MemoryArrayOfRank<2> auto&& M,  
                   std::vector<int> const& nwalk_counts_old,  
                   std::vector<int> const& nwalk_counts_new)
  {
    if (load_balance == LoadBalanceAlgorithm::simple)
    {
      afqmc::swapWalkersSimple(*this, M, nwalk_counts_old, nwalk_counts_new, mpi->comm);
    }
    else if (load_balance == LoadBalanceAlgorithm::async)
    {
      afqmc::swapWalkersAsync(*this, M, nwalk_counts_old, nwalk_counts_new, mpi->comm);
    }
    mpi->comm.barrier();
  }

  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> getRNG() { return rng; }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng;

  int walker_size, walker_memory_usage;
  int bp_walker_size, bp_walker_memory_usage;
  int tau_step;
  int history_pos;

  // wlk_descriptor: {nmo, naea, naeb, nback_prop, nCV, nRefs, nHist}
  wlk_descriptor wlk_desc;
  wlk_indices data_displ;

  WALKER_TYPES walkerType;

  bool finite_temperature;

  int targetN_per_rank;
  int targetN;
  int tot_num_walkers;

  // Contains main walker data needed for propagation
  memory::array<MEM, ComplexType, 2> walker_buffer;

  // Contains stack of fields and slater matrix references for back propagation
  memory::array<MEM, ComplexType, 2> bp_buffer;

  // performs setup
  void parse(const WalkerSetParameters& params);
  // lay out the walker buffer given {rows, naea, naeb} (= wlk_desc[0..2])
  void setup(std::array<int, 3> dims);
  // reserve capacity for n walkers and initialize them to valid defaults
  void allocate_walkers(int n);

  // the four below are set by parse(); the sentinels only guard against a ctor that forgets to
  // call it

  // load balance algorithm
  LoadBalanceAlgorithm load_balance{LoadBalanceAlgorithm::undefined};

  // branching algorithm
  BranchingAlgorithm pop_control{BranchingAlgorithm::undefined};
  [[maybe_unused]] double min_weight{}, max_weight{};
};

} // namespace afqmc

} // namespace sfqmc

