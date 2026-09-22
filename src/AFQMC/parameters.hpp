#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <filesystem>

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "utilities/json.hpp"
#include "numerics/shared_array/const_shared_array.hpp"


namespace sfqmc::afqmc {

enum class DriverType {
  afqmc,
  ftafqmc,
};
SAFIRE_DEFINE_ENUM_NAMES(DriverType, afqmc, ftafqmc);

SAFIRE_DEFINE_ENUM(WALKER_TYPES, {
  {UNDEFINED_WALKER_TYPE, "undefined"},
  {CLOSED, "closed"},
  {COLLINEAR, "collinear"},
  {NONCOLLINEAR, "noncollinear"},
});

SAFIRE_DEFINE_ENUM_NAMES(LoadBalanceAlgorithm, undefined, simple, async);
SAFIRE_DEFINE_ENUM_NAMES(BranchingAlgorithm, undefined, pair, comb, serial_comb);

  
enum class PHMSDEnergyAlgorithm {
  reference, // loop over unique configurations, calculate G and evaluate E from scratch
  woodbury, // use ph_reference_energy and ph_excited_energy, which requires compact R matrix
  // fapbq, // not implemented yet
};
SAFIRE_DEFINE_ENUM_NAMES(PHMSDEnergyAlgorithm, reference, woodbury);

struct WalkerSetParameters {
  // an unnamed block cannot be referenced, so it is registered under a generated name
  std::string name{};
  WALKER_TYPES walker_type{COLLINEAR};
  LoadBalanceAlgorithm load_balance_type{LoadBalanceAlgorithm::async};
  BranchingAlgorithm pop_control_type{BranchingAlgorithm::pair};
  double min_weight{0.05};
  double max_weight{4.0};
};
SAFIRE_DEFINE_PARAMETERS(WalkerSetParameters, name, walker_type, load_balance_type, pop_control_type, min_weight,
                         max_weight);


struct WavefunctionParameters {
  std::string name{};
  std::string filename{}; // required

  int ndets_to_read{-1};
  // the two optionals below depend on the hamiltonian type, so resolve_defaults fills them in
  std::optional<PHMSDEnergyAlgorithm> algorithm{};
  std::optional<bool> dense_trial{};
  int nwalk_block_size{8};
  int ndet_block_size{4096};

  // system
};
SAFIRE_DEFINE_PARAMETERS(WavefunctionParameters, name, filename, ndets_to_read, algorithm, dense_trial,
                         nwalk_block_size, ndet_block_size);

struct HamiltonianParameters {
  std::string name{};
  std::string filename{}; // resolve_defaults falls back to the filename of the wavefunction
  int max_memory{2000};   // MiB
  bool shift_1body{};
  int buffer_size{4096};
};
SAFIRE_DEFINE_PARAMETERS(HamiltonianParameters, name, filename, max_memory, shift_1body, buffer_size);

struct PropagatorParameters {
  std::string name{};

  // The optionals below default to 50.0, 10.0, 1.0, true, except for a ModelHamiltonian,
  // where they default to 100.0, 50.0, 50.0, false. resolve_defaults fills them in.
  int taylor_n{6};
  std::optional<double> vbias_bound{};
  std::optional<double> upper_cutoff_scale{};
  std::optional<double> lower_cutoff_scale{};
  // No walker may carry more than max(weight_bound_floor, weight_bound_fraction*N), where N
  // is the global target population, i.e. the total weight the population is rescaled to.
  double weight_bound_floor{100.0};
  double weight_bound_fraction{0.1};
  bool subtractMF{true};
  bool hybrid{true};
  bool printP1eigval{false};
  bool free_projection{false};
  bool denseP1{false};
  std::optional<bool> denseP2{};
  bool debug_verbosity{false};
  bool natural_shift{true};
  bool use_cp_constraint{false};
  bool project_force_bias{false};
};
SAFIRE_DEFINE_PARAMETERS(PropagatorParameters, name, taylor_n, vbias_bound, upper_cutoff_scale,
                         lower_cutoff_scale, weight_bound_floor, weight_bound_fraction,
                         subtractMF, hybrid,
                         printP1eigval, free_projection, denseP1, denseP2, debug_verbosity, natural_shift,
                         use_cp_constraint, project_force_bias);

struct H5PathParameters {
  std::string filename{};
  std::string group{"/"};

  template<MEMORY_SPACE MEM, typename T, int Rank>
  memory::const_shared_array<MEM, T, Rank> load_shared(utils::mpi_context_t<boost::mpi3::communicator>& mpi, std::string const& dataset) const {
    return memory::share_from_root(mpi, [&dataset, this] {
      h5::file f(filename, 'r');
      h5::group root(f);
      h5::group g = root.open_group(group);

      nda::array<T, Rank> result;
      h5::read(g, dataset, result);
      return memory::to_memory_space<MEM>(result);
    });
  }
};
SAFIRE_DEFINE_PARAMETERS(H5PathParameters, filename, group);

struct OneRDMParameters {
  std::optional<H5PathParameters> rotation{};
};
SAFIRE_DEFINE_PARAMETERS(OneRDMParameters, rotation);

struct DiagonalTwoRDMParameters {
};
SAFIRE_DEFINE_EMPTY_PARAMETERS(DiagonalTwoRDMParameters);

struct TwoRDMParameters {
};
SAFIRE_DEFINE_EMPTY_PARAMETERS(TwoRDMParameters);

/// `pairs` points at an h5 group whose datasets each define one orbital pair map. Every
/// dataset in the group becomes a correlator, named after the dataset.
struct PairCorrParameters {
  H5PathParameters pairs{};
};
SAFIRE_DEFINE_PARAMETERS(PairCorrParameters, pairs);

struct SpinCorrParameters {
};
SAFIRE_DEFINE_EMPTY_PARAMETERS(SpinCorrParameters);

struct EnergyEstimatorParameters {
  // the estimator may use a different wavefunction and hamiltonian than the driver
  std::optional<std::string> wavefunction{};
  std::optional<std::string> hamiltonian{};

  // resolve_defaults falls back to the measure_interval_multiplier of the enclosing execute block
  std::optional<int> measure_interval_multiplier{};
};
SAFIRE_DEFINE_PARAMETERS(EnergyEstimatorParameters, wavefunction, hamiltonian, measure_interval_multiplier);

struct MixedEstimatorParameters {
  std::optional<std::string> wavefunction{};
  std::optional<std::string> hamiltonian{};

  // resolve_defaults falls back to the measure_interval_multiplier of the enclosing execute block
  std::optional<int> measure_interval_multiplier{};

  // observables: an observable is measured if and only if its block is present in the input,
  // so one that takes no parameters is requested by an empty block, e.g. "twordm": {}
  std::optional<OneRDMParameters> onerdm{};
  std::optional<DiagonalTwoRDMParameters> diag_twordm{};
  std::optional<TwoRDMParameters> twordm{};
  std::optional<PairCorrParameters> paircorr{};
  std::optional<SpinCorrParameters> spincorr{};
};
SAFIRE_DEFINE_PARAMETERS(MixedEstimatorParameters, wavefunction, hamiltonian, measure_interval_multiplier,
                         onerdm, diag_twordm, twordm, paircorr, spincorr);

struct BackPropEstimatorParameters {
  std::optional<std::string> wavefunction{};
  std::optional<std::string> hamiltonian{};

  // resolve_defaults falls back to the measure_interval_multiplier of the enclosing execute block
  std::optional<std::vector<int>> measure_interval_multiplier{};

  // in units of steps. if not set fall back to the walker_ortho_interval of the
  // enclosing execute block, which is the interval the forward propagation orthogonalizes at
  std::optional<int> walker_ortho_interval{};
  bool path_restoration{true};
  bool extra_path_restoration{false};

  // observables: an observable is measured if and only if its block is present in the input,
  // so one that takes no parameters is requested by an empty block, e.g. "twordm": {}
  std::optional<OneRDMParameters> onerdm{};
  std::optional<DiagonalTwoRDMParameters> diag_twordm{};
  std::optional<TwoRDMParameters> twordm{};
  std::optional<PairCorrParameters> paircorr{};
  std::optional<SpinCorrParameters> spincorr{};
};
SAFIRE_DEFINE_PARAMETERS(BackPropEstimatorParameters, wavefunction, hamiltonian, measure_interval_multiplier,
                         walker_ortho_interval, path_restoration, extra_path_restoration, onerdm, diag_twordm,
                         twordm, paircorr, spincorr);

struct EstimatorParameters {
  // the energy is measured unless the input removes it with "energy": null
  std::optional<EnergyEstimatorParameters> energy{EnergyEstimatorParameters{}};
  std::optional<MixedEstimatorParameters> mixed{};

  // only one bp estimator may be defined at once
  std::optional<BackPropEstimatorParameters> backprop{};
  std::optional<BackPropEstimatorParameters> time_evolved_bp{};
};
SAFIRE_DEFINE_PARAMETERS(EstimatorParameters, energy, mixed, backprop, time_evolved_bp);

/// Reads a parameter whose default resolve_defaults is responsible for filling in.
template<typename T>
const T& resolved(const std::optional<T>& value, std::string_view name) {
  utils::check(value.has_value(), "The parameter '{}' was not resolved. Did resolve_defaults run?", name);
  return *value;
}

/// The measurement intervals of an estimator, in units of the population control interval.
inline const std::vector<int>& measure_interval_multipliers(const BackPropEstimatorParameters& params) {
  const std::vector<int>& multipliers = resolved(params.measure_interval_multiplier, "measure_interval_multiplier");
  utils::check(!multipliers.empty(), "'measure_interval_multiplier' must not be empty.");
  return multipliers;
}


struct ExecuteParameters {
  std::optional<utils::BlockRef<WalkerSetParameters>> walker_set{};
  std::optional<utils::BlockRef<WavefunctionParameters>> wavefunction{}; // required
  std::optional<utils::BlockRef<HamiltonianParameters>> hamiltonian{};
  std::optional<utils::BlockRef<PropagatorParameters>> propagator{};
  EstimatorParameters estimators{};

  int steps{1000};
  int equilibration_steps{100};
  int binsize{1}; // number of measurements averaged into one bin of the results file
  int sweeps{1}; // finite temperature sweeps
  int population_control_interval{DEFAULT_POPULATION_CONTROL_INTERVAL};
  int measure_interval_multiplier{DEFAULT_MEASURE_INTERVAL_MULTIPLIER};
  int walker_ortho_interval{DEFAULT_WALKER_ORTHO_INTERVAL};
  bool print_sweep_step{false}; // ftafqmc only

  double timestep{DEFAULT_TIME_STEP};
  int n_walkers_per_mpi_task{10};

  // fraction of the gap to the average energy that Eshift closes, once per population control
  // interval; defaults to decaying within a tenth of the equilibration phase
  std::optional<double> Eshift_relaxation_factor{};
  std::optional<double> initial_Eshift{};
};
SAFIRE_DEFINE_PARAMETERS(ExecuteParameters, walker_set, wavefunction, hamiltonian, propagator, estimators, steps,
                         equilibration_steps, binsize, sweeps, population_control_interval, measure_interval_multiplier,
                         walker_ortho_interval, print_sweep_step,
                         timestep, n_walkers_per_mpi_task, Eshift_relaxation_factor, initial_Eshift);


struct AFQMCParameters {
  // selects which driver runs the execute blocks
  DriverType driver{DriverType::afqmc};

  // results will be written to `<output_name>.results.h5`. defaults to the name of the input file
  // without its extension, in the current working directory
  std::string output_name{};

  // seeds the random number generators of the whole run, every stage included. Without one
  // resolve_defaults draws one, so a run can always be reproduced from the printed parameters
  std::optional<int> seed{};

  std::vector<ExecuteParameters> execute{};

  // blocks declared outside of an execute block have to be named, so that an execute block can refer to them
  std::vector<WalkerSetParameters> walker_set{};
  std::vector<WavefunctionParameters> wavefunction{};
  std::vector<HamiltonianParameters> hamiltonian{};
  std::vector<PropagatorParameters> propagator{};
};
SAFIRE_DEFINE_PARAMETERS(AFQMCParameters, driver, output_name, seed, execute, walker_set, wavefunction, hamiltonian,
                         propagator);


/// Parses the input json file. An input that does not name its output is named after the input file itself.
AFQMCParameters parse_input_file(const std::filesystem::path &filename);
void print_parameters(const AFQMCParameters& params);

}
