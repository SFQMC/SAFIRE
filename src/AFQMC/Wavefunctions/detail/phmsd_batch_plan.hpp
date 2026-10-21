// Distributed under the Apache License, Version 2.0. See LICENSE.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sfqmc::afqmc {

struct phmsd_batch {
  int walkers = 1, determinants = 4096;
  bool operator==(phmsd_batch const&) const = default;
};

// Conservative planning estimates, not a cap on process/device memory. Counts
// PHMSD live scratch and retained reference/cache arrays; excludes input data,
// caller-owned outputs, BLAS/CUDA contexts, other estimators and allocator slack.
// Double arithmetic avoids integer overflow when inspecting very large inputs.
struct phmsd_batch_model {
  double nmo{}, nelec{}, nact{}, nchol{}, unique_a{}, unique_b{}, rank{}, shell{};
  int cache_walkers = 0;
  bool device = false;

  double bytes(phmsd_batch p, bool energy) const {
    double w=p.walkers, d=std::min(double(p.determinants),shell);
    double u=std::max(unique_a,unique_b), pair=nelec*nact;
    double common=w*(unique_a+unique_b+u + 4*nelec*nmo+4*pair+64);
    // LU buffers, compact/full R assembly, and density output transformation.
    double block=w*d*(4*rank*rank+rank*nact+8*rank+8);
    double elements=common+block+w*nact*nmo;
    if(energy) {
      elements += w*((u+unique_a)*nchol+2*pair*nchol+pair);
      elements += w*d*nchol; // beta KEl
      // Exact cache and GEMM temporary can coexist across spins/shells.
      if(cache_walkers>0) elements += 2*w*(pair*pair+(device ? pair*nchol+pair+nchol : 0));
      elements += std::min(std::max(64.0*1024*1024/16,rank*nelec*nchol),
                           d*rank*nelec*nchol);
    }
    return 1.15*16*elements; // alignment and auxiliary-workspace margin
  }

  phmsd_batch fit(phmsd_batch p, bool energy, double budget) const {
    if(energy && cache_walkers>0) p.walkers=std::min(p.walkers,cache_walkers);
    // Preserve determinant amortization while reducing the dominant walker floor.
    while(bytes(p,energy)>budget && p.walkers>1) p.walkers=std::max(1,p.walkers/2);
    while(bytes(p,energy)>budget && p.determinants>1)
      p.determinants=std::max(1,p.determinants/2);
    return p;
  }

  std::vector<phmsd_batch> candidates(int walkers, int determinants, bool energy,
                                     double budget, bool auto_walkers, bool auto_dets) const {
    std::vector<phmsd_batch> result;
    auto add=[&](phmsd_batch p) {
      p=fit(p,energy,budget);
      if(bytes(p,energy)<=budget && std::find(result.begin(),result.end(),p)==result.end())
        result.push_back(p);
    };
    add({walkers,determinants});
    // Try the larger block early: small-block allocation/launch overhead can
    // otherwise consume the budget before a promising candidate is reached.
    if(auto_dets) add({walkers,8192});
    if(auto_walkers) add({std::max(1,walkers/2),determinants});
    if(auto_dets) add({walkers,1024});
    return result;
  }
};
} // namespace sfqmc::afqmc
