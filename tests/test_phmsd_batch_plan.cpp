// Distributed under the Apache License, Version 2.0. See LICENSE.
#include "catch2/catch_test_macros.hpp"
#include "AFQMC/Wavefunctions/detail/phmsd_batch_plan.hpp"
#include "AFQMC/parameters.hpp"
using namespace sfqmc::afqmc;
TEST_CASE("PHMSD plans respect workspace and explicit upper bounds", "[phmsd_planning]") {
  phmsd_batch_model m{36,27,27,666,28234,28359,5,20000,2,true};
  double budget=2.*1024*1024*1024;
  auto energy=m.fit({8,4096},true,budget);
  CHECK(energy.walkers<=2);
  CHECK(m.bytes(energy,true)<=budget);
  CHECK(m.fit({8,4096},false,budget).walkers==8);
  auto small=m.fit({8,4096},true,0.9*1024*1024*1024);
  CHECK(small.walkers==1);
  CHECK(m.bytes(small,true)<=0.9*1024*1024*1024);
  auto explicit_plans=m.candidates(1,128,true,budget,false,false);
  REQUIRE(explicit_plans.size()==1);
  CHECK(explicit_plans[0].walkers==1);
  CHECK(explicit_plans[0].determinants==128);
  for(auto p:m.candidates(8,4096,true,budget,true,true)) {
    CHECK(m.bytes(p,true)<=budget);
    CHECK(p.walkers<=2);
  }
  // Lowering D cannot remove the all-unique-spin opposite-spin contraction.
  CHECK(m.bytes({1,1},true)>16*(28359.+28234)*666);
  // Oversized dimensions must remain finite, never wrap into a fitting plan.
  m.unique_a=2.e9; m.unique_b=2.e9; m.nchol=2.e9;
  CHECK(std::isfinite(m.bytes({8,4096},true)));
  CHECK(m.bytes(m.fit({8,4096},true,budget),true)>budget);
}
TEST_CASE("PHMSD automatic inputs and explicit overrides round trip", "[phmsd_planning]") {
  WavefunctionParameters p;
  CHECK(p.nwalk_block_size==0);
  CHECK(p.ndet_block_size==0);
  auto j=nlohmann::json(p);
  j["nwalk_block_size"]=3;
  j["ndet_block_size"]=128;
  j["phmsd_workspace_mb"]=512;
  j["phmsd_tuning_seconds"]=0.0;
  auto q=j.get<WavefunctionParameters>();
  CHECK(q.nwalk_block_size==3);
  CHECK(q.ndet_block_size==128);
  CHECK(q.phmsd_workspace_mb==512);
  CHECK(q.phmsd_tuning_seconds==0);
}
