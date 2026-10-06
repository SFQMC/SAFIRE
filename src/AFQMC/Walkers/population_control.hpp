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

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "config.0.h"
#include "mpi3/request.hpp"
#include "utilities/Random.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"
#include "WalkerSet.hpp"
#include <nda/nda.hpp>

namespace sfqmc::afqmc {

inline nda::vector<int> comb(utils::HostRandomGenerator& rng, nda::vector<RealType> weights, int num_target) {
  memory::buffered_array<HOST_MEMORY, RealType, 1> cumweights(weights.size());
  std::partial_sum(weights.begin(), weights.end(), cumweights.begin());

  RealType total_weight = cumweights[cumweights.size()-1];
  utils::check(total_weight > 1e-6, "The total walker weight collapsed to {}. Something went very wrong.", total_weight);

  memory::buffered_array<HOST_MEMORY, RealType, 1> r(1);
  rng.sampleUniformFields(r);

  // the last tooth can round up to the total weight, and then it has to land on the last walker
  // that has any weight rather than on a zero-weight one behind it; the total weight is positive,
  // so there is one
  int last = weights.size() - 1;
  while(weights[last] <= 0.0) {
    --last;
  }

  nda::vector<int> offspring(weights.size(), 0);
  int iw = 0;
  int ic = 0;
  while(ic < num_target) {
    RealType comb = (ic + r(0))  * total_weight / num_target;
    if(comb < cumweights[iw] || iw == last) {
      offspring[iw] += 1;
      ic++;
    } else {
      iw++;
    }
  }
  return offspring;
}



struct walker_transfer {
  int src;    // sending rank
  int walker; // local index of the walker on src
  int dst;    // receiving rank
  int copies; // copies of the walker that dst ends up with
};

/*
 * Plans the walker exchange that realizes offspring, one copy count per walker of the global
 * population in rank-major order: rank r holds counts[r] walkers, and its walker i is entry
 * counts[0] + ... + counts[r-1] + i. Every rank keeps its count and as many of its own copies as
 * fit. A rank with excess exports the walkers with the most copies first, which covers the excess
 * with the fewest walkers sent, and the exports fill the ranks with a deficit in rank order. A
 * walker is sent once per destination and replicated there, and an export is split only where a
 * deficit runs out.
 *
 * The plan is deterministic, so every rank computes the same one from the same offspring.
 */
inline std::vector<walker_transfer> plan_walker_exchange(nda::vector_const_view<int> offspring,
                                                         std::vector<int> const& counts) {
  long const ntot = std::accumulate(counts.begin(), counts.end(), 0L);
  utils::check(offspring.size() == ntot, "plan_walker_exchange: {} walkers, but the ranks hold {}.",
               offspring.size(), ntot);
  utils::check(std::accumulate(offspring.begin(), offspring.end(), 0L) == ntot,
               "plan_walker_exchange: offspring does not preserve the population.");

  std::vector<walker_transfer> exports;
  // {rank, missing copies}
  std::vector<std::pair<int, int>> deficits;
  std::vector<int> order;
  for(int r = 0, first = 0; r < int(counts.size()); first += counts[r], ++r) {
    int const nw = counts[r];
    auto local = offspring(nda::range(first, first + nw));
    int excess = std::accumulate(local.begin(), local.end(), 0) - nw;
    if(excess < 0) {
      deficits.emplace_back(r, -excess);
    }
    if(excess <= 0) {
      continue;
    }
    order.resize(nw);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return local(a) > local(b); });
    for(int k = 0; excess > 0; ++k) {
      int m = std::min(local(order[k]), excess);
      exports.push_back({r, order[k], -1, m});
      excess -= m;
    }
  }

  std::vector<walker_transfer> plan;
  auto d = deficits.begin();
  for(auto e : exports) {
    while(e.copies > 0) {
      int n = std::min(e.copies, d->second);
      plan.push_back({e.src, e.walker, d->first, n});
      e.copies -= n;
      d->second -= n;
      if(d->second == 0) {
        ++d;
      }
    }
  }
  return plan;
}

/*
 * Replaces every walker of wset by offspring copies of itself, moving walkers between ranks so
 * that each rank ends up with its target population, counts[rank]; see plan_walker_exchange for
 * the layout of offspring and the exchange. Weights are left untouched, so the caller resets them.
 *
 * Collective over mpi.comm.
 */
template<MEMORY_SPACE MEM>
inline void swap_walkers_async(utils::mpi_context_t<>& mpi, WalkerSet<MEM>& wset, nda::vector_const_view<int> offspring,
                               std::vector<int> const& counts) {
  int me = mpi.comm.rank();
  int nw = wset.get_target_population();
  utils::check(int(counts.size()) == mpi.comm.size() && counts[me] == nw,
               "swap_walkers_async: the walker counts do not match {} ranks with {} walkers on rank {}.",
               mpi.comm.size(), nw, me);
  utils::check(wset.size() == nw, "swap_walkers_async: {} walkers, target {}.", wset.size(), nw);
  utils::check_shape(offspring, "offspring", std::accumulate(counts.begin(), counts.end(), 0L));

  auto plan = plan_walker_exchange(offspring, counts);

  // copies of each local walker that stay on this rank
  int const first = std::accumulate(counts.begin(), counts.begin() + me, 0);
  auto local = offspring(nda::range(first, first + nw));
  std::vector<int> keep(local.begin(), local.end());
  for(auto const& t : plan) {
    if(t.src == me) {
      keep[t.walker] -= t.copies;
    }
  }

  // slots of walkers that leave this rank entirely, refilled by arrivals and local copies
  std::vector<int> free_slots;
  for(int i = 0; i < nw; ++i) {
    if(keep[i] == 0) {
      free_slots.push_back(i);
    }
  }
  auto next_slot = free_slots.begin();

  // a rank either sends or receives, and both ends of a rank pair post their messages in plan
  // order, so MPI's non-overtaking rule matches them without per-transfer tags
  std::vector<mpi3::request> sends, recvs;
  // {slot, copies}
  std::vector<std::pair<int, int>> arrivals;
  for(auto const& t : plan) {
    if(t.src == me) {
      wset.isend_walker(t.walker, t.dst, mpi.comm, sends);
    } else if(t.dst == me) {
      int slot = *next_slot++;
      wset.irecv_walker(slot, t.src, mpi.comm, recvs);
      arrivals.emplace_back(slot, t.copies);
    }
  }

  // on a sending rank the free slots can still be in flight
  for(auto& r : sends) {
    r.wait();
  }
  for(int i = 0; i < nw; ++i) {
    for(int c = 1; c < keep[i]; ++c) {
      wset.copy_walker(i, *next_slot++);
    }
  }

  for(auto& r : recvs) {
    r.wait();
  }
  for(auto [slot, copies] : arrivals) {
    for(int c = 1; c < copies; ++c) {
      wset.copy_walker(slot, *next_slot++);
    }
  }
  utils::check(next_slot == free_slots.end(), "swap_walkers_async: {} slots left unfilled.",
               std::distance(next_slot, free_slots.end()));
}


/*
 * Comb population control: resamples the global population by |w| on the root, redistributes
 * the copies, and gives every walker unit weight with the phase it had before.
 *
 * Collective over mpi.comm.
 */
template<MEMORY_SPACE MEM>
inline void population_control(utils::mpi_context_t<>& mpi, utils::HostRandomGenerator& rng, WalkerSet<MEM>& wset) {
  int nw = wset.size();

  // the ranks need not hold the same number of walkers, and each one keeps its own
  std::vector<int> counts = mpi.comm.all_gather_value(nw);
  std::vector<int> displs(counts.size());
  std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
  int ntot = displs.back() + counts.back();

  memory::buffered_array<HOST_MEMORY, ComplexType, 1> complex_weights(nw);
  wset.getProperty(WEIGHT, complex_weights);

  nda::vector<RealType> weights(nw);
  std::ranges::transform(complex_weights, weights.begin(), [](ComplexType w) { return std::abs(w); });

  nda::vector<RealType> all_weights(mpi.comm.root() ? ntot : 0);
  mpi.comm.gatherv_n(weights.data(), nw, all_weights.data(), counts.data(), displs.data(), 0);

  nda::vector<int> offspring(ntot);
  if(mpi.comm.root()) {
    offspring = comb(rng, std::move(all_weights), ntot);
  }
  mpi.broadcast(offspring);
  swap_walkers_async(mpi, wset, offspring, counts);

  // every copy now carries its parent's weight; comb never copies a zero-weight walker
  wset.getProperty(WEIGHT, complex_weights);
  std::ranges::transform(complex_weights, complex_weights.begin(), [](ComplexType w) { return w / std::abs(w); });
  wset.setProperty(WEIGHT, complex_weights);

  // the last completed step's history entry holds the same weight
  if(int nhist = wset.HistoryBufferLength(); nhist > 0) {
    int his_pos = (wset.getHistoryPos() + nhist - 1) % nhist;
    auto history = wset.getWeightHistory();
    history(nda::range(nw), his_pos) = complex_weights;
  }
}

}
