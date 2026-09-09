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

// Regression tests for TwoRDM, DiagonalTwoRDM and SpinCorr: compare the observables against
// reference data produced by the pre-refactor implementations on identical inputs.
//
// The reference file stores the unnormalized numerator sum_w Xw f(G_w), because the old print()
// wrote the denominator as a separate dataset and the dump harness handed it a placeholder of 1
// rather than the walker weight sum. MeasurementOutput instead records the ratio, so the
// comparison divides the reference by sum(Xw) - the denominator MeasurementOutput itself uses.

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "test_common.hpp"
#include "utilities/h5_utils.hpp"

#include <string>
#include <vector>

#include "AFQMC/config.h"
#include "AFQMC/Estimators/Measurements.hpp"
#include "AFQMC/Estimators/Observables/twordm.hpp"
#include "AFQMC/Estimators/Observables/diagonal_twordm.hpp"
#include "AFQMC/Estimators/Observables/spincorr.hpp"
#include "AFQMC/Estimators/Observables/paircorr.hpp"

namespace sfqmc
{
using namespace afqmc;

namespace
{
/// The inputs the pre-refactor observables were fed. The file also carries NAEA/NAEB, and a
/// nave/Wsum pair for the old multi-average interface, none of which the new one has.
struct ReferenceInputs
{
  int NMO, nwalk;
  nda::array<ComplexType, 4> G;
  nda::vector<ComplexType> Xw;
};

std::array<long, 4> g_shape(int nwalk, int NMO, WALKER_TYPES wt)
{
  // CLOSED:       (nwalk, 1, NMO,    NMO)
  // COLLINEAR:    (nwalk, 2, NMO,    NMO)
  // NONCOLLINEAR: (nwalk, 1, 2*NMO,  2*NMO)
  switch (wt)
  {
    case CLOSED:       return {nwalk, 1, NMO,     NMO};
    case COLLINEAR:    return {nwalk, 2, NMO,     NMO};
    case NONCOLLINEAR: return {nwalk, 1, 2 * NMO, 2 * NMO};
    default:
      utils::check(false, "unreachable: unknown walker type");
  }
  return {};
}

/// Length of the i <-> j folded diagonal 2RDM, mirroring DiagonalTwoRDM::compressed_size.
int compressed_diag_size(int NMO, WALKER_TYPES wt)
{
  int dm = NMO * (2 * NMO - 1);
  if (wt == CLOSED) dm -= NMO * (NMO - 1) / 2;
  return dm;
}

ReferenceInputs read_reference_inputs(h5::group const& root,
                                      std::string const& case_group,
                                      WALKER_TYPES wt)
{
  ReferenceInputs r;
  h5::group gcase = root.open_group(case_group);

  h5::group meta = gcase.open_group("Metadata");
  h5::read(meta, "NMO", r.NMO);
  h5::read(meta, "nwalk", r.nwalk);

  auto gshape = g_shape(r.nwalk, r.NMO, wt);
  r.G.resize(gshape);
  r.Xw.resize(r.nwalk);

  h5::group inputs = gcase.open_group("Inputs");
  utils::h5_read(inputs, "G", r.G);
  utils::h5_read(inputs, "Xw", r.Xw);
  return r;
}

std::string reference_file_path()
{
  return std::string(PROJECT_SOURCE_DIR_STR) + "/tests/unit_test_files/rdm_reference.h5";
}

ReferenceInputs load_reference(std::string const& case_group, WALKER_TYPES wt)
{
  const std::string ref_file = reference_file_path();
  utils::check(utils::file_exists(ref_file), "Reference file not found: {}", ref_file);

  h5::file file(ref_file, 'r');
  h5::group root(file);
  return read_reference_inputs(root, case_group, wt);
}

/// Drives an observable over the stored reference Green function. A single reference carries
/// the full overlap, so refCoeff is one and the reference loop runs exactly once - which is
/// what the pre-refactor implementations saw when accumulate() was called directly.
template<MEMORY_SPACE MEM, typename Obs>
void measure_reference(utils::mpi_context_t<boost::mpi3::communicator>& mpi, Obs& obs,
                       ReferenceInputs const& ref, WALKER_TYPES wt, Measurements& meas)
{
  auto G = memory::to_memory_space<MEM>(ref.G);
  memory::buffered_array<MEM, ComplexType, 1> weights(ref.Xw);

  auto referenceLoop = [&G, nwalk = ref.nwalk](auto&& body) {
    memory::buffered_array<MEM, ComplexType, 1> refCoeff(nwalk);
    refCoeff() = 1.0;
    body(refCoeff(), G());
  };

  MeasurementInputs<MEM, decltype(referenceLoop)> inputs{referenceLoop, weights(), wt, ref.NMO};
  MeasurementOutput output{mpi, meas, "", ref.Xw()};
  obs.measure(mpi, output, inputs);
}

/// Writes the measurements into an in-memory h5 file, so that the recorded bins can be read
/// back. Only call this once per Measurements: write() drops the bins it has flushed.
h5::file written(Measurements& meas)
{
  h5::file file{};
  h5::group root(file);
  meas.write(root);
  return file;
}

nda::vector<ComplexType> normalized(nda::vector<ComplexType> const& numerator, ComplexType denominator)
{
  nda::vector<ComplexType> result(numerator.size());
  result() = numerator() / denominator;
  return result;
}

template<MEMORY_SPACE MEM>
void observables_reference_twordm()
{
  auto& mpi = utils::make_unit_test_mpi_context();

  ReferenceInputs ref;
  nda::vector<ComplexType> ref_two_rdm;
  {
    const std::string ref_file = reference_file_path();
    utils::check(utils::file_exists(ref_file), "Reference file not found: {}", ref_file);

    h5::file file(ref_file, 'r');
    h5::group root(file);
    ref = read_reference_inputs(root, "collinear", COLLINEAR);

    ref_two_rdm.resize(3 * ref.NMO * ref.NMO * ref.NMO * ref.NMO);
    h5::group gcase = root.open_group("collinear");
    h5::group avg = gcase.open_group("FullTwoRDM/Average_0");
    utils::h5_read(avg, "two_rdm_000000000", ref_two_rdm);
  }

  const ComplexType denom = nda::sum(ref.Xw);

  TwoRDM<MEM> obs(*mpi, TwoRDMParameters{}, COLLINEAR, ref.NMO);
  Measurements meas;
  measure_reference<MEM>(*mpi, obs, ref, COLLINEAR, meas);

  nda::array<ComplexType, 6> bins;
  {
    h5::file file = written(meas);
    h5::group root(file);
    utils::h5_read(root, "TwoRDM/bins", bins);
  }

  utils::check_shape(bins, "TwoRDM/bins", 1, 3, ref.NMO, ref.NMO, ref.NMO, ref.NMO);
  nda::vector<ComplexType> got(ref_two_rdm.size());
  got() = nda::flatten(bins(0, nda::ellipsis{}));

  CHECK_THAT(got, utils::Approx(normalized(ref_two_rdm, denom)));
}

template<MEMORY_SPACE MEM>
void observables_reference_diagonal_twordm(std::string const& case_group, WALKER_TYPES wt)
{
  auto& mpi = utils::make_unit_test_mpi_context();

  ReferenceInputs ref;
  nda::vector<ComplexType> ref_diag;
  {
    const std::string ref_file = reference_file_path();
    utils::check(utils::file_exists(ref_file), "Reference file not found: {}", ref_file);

    h5::file file(ref_file, 'r');
    h5::group root(file);
    ref = read_reference_inputs(root, case_group, wt);

    ref_diag.resize(compressed_diag_size(ref.NMO, wt));
    h5::group gcase = root.open_group(case_group);
    h5::group avg = gcase.open_group("DiagTwoRDM/Average_0");
    utils::h5_read(avg, "diag_two_rdm_000000000", ref_diag);
  }

  const ComplexType denom = nda::sum(ref.Xw);

  DiagonalTwoRDM<MEM> obs(*mpi, DiagonalTwoRDMParameters{}, wt, ref.NMO);
  Measurements meas;
  measure_reference<MEM>(*mpi, obs, ref, wt, meas);

  nda::array<ComplexType, 2> bins;
  {
    h5::file file = written(meas);
    h5::group root(file);
    utils::h5_read(root, "DiagonalTwoRDM/bins", bins);
  }

  utils::check_shape(bins, "DiagonalTwoRDM/bins", 1, ref_diag.size());
  nda::vector<ComplexType> got(ref_diag.size());
  got() = bins(0, nda::range::all);

  CHECK_THAT(got, utils::Approx(normalized(ref_diag, denom)));
}

template<MEMORY_SPACE MEM>
void observables_reference_spincorr(std::string const& case_group, WALKER_TYPES wt)
{
  auto& mpi = utils::make_unit_test_mpi_context();

  ReferenceInputs ref;
  nda::vector<ComplexType> ref_ss;
  {
    const std::string ref_file = reference_file_path();
    utils::check(utils::file_exists(ref_file), "Reference file not found: {}", ref_file);

    h5::file file(ref_file, 'r');
    h5::group root(file);
    ref = read_reference_inputs(root, case_group, wt);

    // the two channels of the packed upper triangle, laid out end to end
    ref_ss.resize(ref.NMO * (ref.NMO + 1));
    h5::group gcase = root.open_group(case_group);
    h5::group avg = gcase.open_group("SpinSpin/Average_0");
    utils::h5_read(avg, "spinspin_000000000", ref_ss);
  }

  const ComplexType denom = nda::sum(ref.Xw);

  SpinCorr<MEM> obs(*mpi, SpinCorrParameters{}, wt, ref.NMO);
  Measurements meas;
  measure_reference<MEM>(*mpi, obs, ref, wt, meas);

  nda::array<ComplexType, 3> bins;
  {
    h5::file file = written(meas);
    h5::group root(file);
    utils::h5_read(root, "SpinCorr/bins", bins);
  }

  utils::check_shape(bins, "SpinCorr/bins", 1, 2, ref.NMO * (ref.NMO + 1) / 2);
  nda::vector<ComplexType> got(ref_ss.size());
  got() = nda::flatten(bins(0, nda::ellipsis{}));

  CHECK_THAT(got, utils::Approx(normalized(ref_ss, denom)));
}

// PairCorr has no old-implementation reference data. With the identity pairing (ibar == i)
// every term of the correlator collapses onto the same product, so the expected value has a
// closed form that the test can check against:
//   collinear:    P(i,j) = 2 * sum_w Xw * Gup(i,j) * Gdn(i,j)              for j > i, else 0
//   noncollinear: P(i,j) = 2 * sum_w Xw * (G(i,j) G(i',j') - G(i,j') G(i',j)),  i' = i + NMO
// all divided by sum(Xw). This pins down the pair-map plumbing, the normalization, the output
// naming and the fill pattern.
template<MEMORY_SPACE MEM>
void observables_reference_paircorr(std::string const& case_group, WALKER_TYPES wt)
{
  auto& mpi = utils::make_unit_test_mpi_context();

  const ReferenceInputs ref = load_reference(case_group, wt);
  const int NMO = ref.NMO;

  // identity and cyclic-shift pairings, i.e. the "s" and "+x" offsets of a 1d chain. h5 hands
  // the names back in its own order, so the test looks them up rather than assuming one.
  const std::vector<std::string> names{"s", "+x"};
  utils::TemporaryDirectory tmpdir;
  const std::string map_file = (tmpdir / "orbital_map.h5").string();
  if (mpi->comm.root())
  {
    nda::array<int, 2> identity(NMO, 1), shift(NMO, 1);
    for (int i = 0; i < NMO; i++)
    {
      identity(i, 0) = i;
      shift(i, 0)    = (i + 1) % NMO;
    }
    h5::file file(map_file, 'w');
    h5::group orbital_map = h5::group(file).create_group("orbital_map");
    h5::write(orbital_map, names[0], identity);
    h5::write(orbital_map, names[1], shift);
  }
  mpi->comm.barrier();

  PairCorrParameters params;
  params.pairs.filename = map_file;
  params.pairs.group    = "orbital_map";

  PairCorr<MEM> obs(*mpi, params, wt, NMO);
  // this barrier is important to ensure the temp dir is still around. See destructor of TemporaryDirectory.
  mpi->comm.barrier();

  Measurements meas;
  measure_reference<MEM>(*mpi, obs, ref, wt, meas);

  const ComplexType denom = nda::sum(ref.Xw);

  nda::array<ComplexType, 3> identity_block;
  {
    h5::file file = written(meas);
    h5::group root(file);
    h5::group parent = root.open_group("PairCorr");

    // every ordered pair of maps is measured as its own series
    for (auto const& a : names)
    {
      for (auto const& b : names)
      {
        const std::string path = std::format("{}_{}/bins", a, b);
        REQUIRE(parent.has_key(std::format("{}_{}", a, b)));

        nda::array<ComplexType, 3> block;
        utils::h5_read(parent, path, block);
        utils::check_shape(block, path, 1, NMO, NMO);

        if (a == names[0] && b == names[0]) identity_block = block;
      }
    }
  }

  nda::array<ComplexType, 2> expected(NMO, NMO);
  expected() = 0.0;
  for (int iw = 0; iw < ref.nwalk; iw++)
  {
    if (wt == COLLINEAR)
    {
      auto Gup = ref.G(iw, 0, nda::range::all, nda::range::all);
      auto Gdn = ref.G(iw, 1, nda::range::all, nda::range::all);
      for (int i = 0; i < NMO; i++)
        for (int j = i + 1; j < NMO; j++)
          expected(i, j) += 2.0 * ref.Xw(iw) * Gup(i, j) * Gdn(i, j);
    }
    else
    {
      auto G_ = ref.G(iw, 0, nda::range::all, nda::range::all);
      for (int i = 0; i < NMO; i++)
        for (int j = 0; j < NMO; j++)
          expected(i, j) += 2.0 * ref.Xw(iw) *
                            (G_(i, j) * G_(NMO + i, NMO + j) - G_(i, NMO + j) * G_(NMO + i, j));
    }
  }
  expected() = expected() / denom;

  CHECK_THAT(identity_block(0, nda::ellipsis{}), utils::Approx(expected));
  CHECK(nda::max_element(nda::abs(identity_block)) > 1e-12);

}
} // namespace

TEST_CASE("observables_reference: twordm", "[observables_reference][twordm]")
{
  observables_reference_twordm<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  observables_reference_twordm<DEVICE_MEMORY>();
#endif
}

TEST_CASE("observables_reference: diagonal_twordm closed",
          "[observables_reference][diagonal_twordm]")
{
  observables_reference_diagonal_twordm<HOST_MEMORY>("closed", CLOSED);
#if defined(ENABLE_DEVICE)
  observables_reference_diagonal_twordm<DEVICE_MEMORY>("closed", CLOSED);
#endif
}

TEST_CASE("observables_reference: diagonal_twordm collinear",
          "[observables_reference][diagonal_twordm]")
{
  observables_reference_diagonal_twordm<HOST_MEMORY>("collinear", COLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_diagonal_twordm<DEVICE_MEMORY>("collinear", COLLINEAR);
#endif
}

TEST_CASE("observables_reference: diagonal_twordm noncollinear",
          "[observables_reference][diagonal_twordm]")
{
  observables_reference_diagonal_twordm<HOST_MEMORY>("noncollinear", NONCOLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_diagonal_twordm<DEVICE_MEMORY>("noncollinear", NONCOLLINEAR);
#endif
}

TEST_CASE("observables_reference: spincorr closed", "[observables_reference][spincorr]")
{
  observables_reference_spincorr<HOST_MEMORY>("closed", CLOSED);
#if defined(ENABLE_DEVICE)
  observables_reference_spincorr<DEVICE_MEMORY>("closed", CLOSED);
#endif
}

TEST_CASE("observables_reference: spincorr collinear", "[observables_reference][spincorr]")
{
  observables_reference_spincorr<HOST_MEMORY>("collinear", COLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_spincorr<DEVICE_MEMORY>("collinear", COLLINEAR);
#endif
}

TEST_CASE("observables_reference: spincorr noncollinear", "[observables_reference][spincorr]")
{
  observables_reference_spincorr<HOST_MEMORY>("noncollinear", NONCOLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_spincorr<DEVICE_MEMORY>("noncollinear", NONCOLLINEAR);
#endif
}

TEST_CASE("observables_reference: paircorr collinear", "[observables_reference][paircorr]")
{
  observables_reference_paircorr<HOST_MEMORY>("collinear", COLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_paircorr<DEVICE_MEMORY>("collinear", COLLINEAR);
#endif
}

TEST_CASE("observables_reference: paircorr noncollinear", "[observables_reference][paircorr]")
{
  observables_reference_paircorr<HOST_MEMORY>("noncollinear", NONCOLLINEAR);
#if defined(ENABLE_DEVICE)
  observables_reference_paircorr<DEVICE_MEMORY>("noncollinear", NONCOLLINEAR);
#endif
}

} // namespace sfqmc
