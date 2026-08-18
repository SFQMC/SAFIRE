#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <fstream>
#include <filesystem>

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "utilities/json.hpp"


namespace sfqmc::afqmc {

enum class DriverType {
  afqmc,
  ftafqmc,
  csafqmc,
};
SAFIRE_DEFINE_ENUM_NAMES(DriverType, afqmc, ftafqmc, csafqmc);

SAFIRE_DEFINE_ENUM(WALKER_TYPES, {
  {UNDEFINED_WALKER_TYPE, "undefined"},
  {CLOSED, "closed"},
  {COLLINEAR, "collinear"},
  {NONCOLLINEAR, "noncollinear"},
});

SAFIRE_DEFINE_ENUM_NAMES(LoadBalanceAlgorithm, undefined, simple, async);
SAFIRE_DEFINE_ENUM_NAMES(BranchingAlgorithm, undefined, pair, comb, min_branch, serial_comb);

  
enum class PHMSDEnergyAlgorithm {
  reference, // loop over unique configurations, calculate G and evaluate E from scratch
  woodbury, // use ph_reference_energy and ph_excited_energy, which requires compact R matrix
  // fapbq, // not implemented yet
};
SAFIRE_DEFINE_ENUM_NAMES(PHMSDEnergyAlgorithm, reference, woodbury);

enum class EstimatorType {
  undefined,
  basic,
  mixed,
  energy,
  back_propagation,
  time_evolved_operators,
};
SAFIRE_DEFINE_ENUM_NAMES(EstimatorType, undefined, basic, mixed, energy, back_propagation, time_evolved_operators);


struct ProjectParameters {
  std::string id{"afqmc"};
  int series{};
  int n_groups{1}; // csafqmc only
};
SAFIRE_DEFINE_PARAMETERS(ProjectParameters, id, series, n_groups);

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

  bool rediag{}; // ??
  int ndets_to_read{-1};
  // the two optionals below depend on the hamiltonian type, so resolve_defaults fills them in
  std::optional<PHMSDEnergyAlgorithm> algorithm{};
  std::optional<bool> dense_trial{};
  int nwalk_block_size{8};
  int ndet_block_size{4096};

  // system
};
SAFIRE_DEFINE_PARAMETERS(WavefunctionParameters, name, filename, rediag, ndets_to_read, algorithm, dense_trial,
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

  // The optionals below default to 50.0, 10.0, 1.0, true, true, except for a ModelHamiltonian,
  // where they default to 100.0, 50.0, 50.0, false, false. resolve_defaults fills them in.
  int taylor_n{6};
  std::optional<double> vbias_bound{};
  double external_field_scale{1.0};
  std::optional<double> upper_cutoff_scale{};
  std::optional<double> lower_cutoff_scale{};
  bool apply_constrain{true};
  bool importance_sampling{true};
  bool substractMF{true};
  bool hybrid{true};
  bool printP1eigval{false};
  bool free_projection{false};
  bool denseP1{false};
  std::optional<bool> denseP2{};
  bool debug_verbosity{false};
  bool natural_shift{true};
  std::optional<bool> symmetric_split{};
  bool use_cp_constraint{false};
  bool use_real_vbias{false};
  std::string external_field{""};
  std::string excited{""};
};
SAFIRE_DEFINE_PARAMETERS(PropagatorParameters, name, taylor_n, vbias_bound, external_field_scale, upper_cutoff_scale,
                         lower_cutoff_scale, apply_constrain, importance_sampling, substractMF, hybrid,
                         printP1eigval, free_projection, denseP1, denseP2, debug_verbosity, natural_shift,
                         symmetric_split, use_cp_constraint, use_real_vbias, external_field, excited);

// the name of an observable is a label that the code does not use for anything
struct OneRDMParameters {
  std::string name{};
  std::string rotation{};
  std::string path{"/"};
  bool with_index_list{false};
};
SAFIRE_DEFINE_PARAMETERS(OneRDMParameters, name, rotation, path, with_index_list);

struct DiagTwoRDMParameters {
  std::string name{};
};
SAFIRE_DEFINE_PARAMETERS(DiagTwoRDMParameters, name);

struct TwoRDMParameters {
  std::string name{};
  std::string rotation{};
  std::string path{"/"};
};
SAFIRE_DEFINE_PARAMETERS(TwoRDMParameters, name, rotation, path);

struct PairCorrelatorParameters {
    std::string name{"pair_correlator"};
    std::string filename{""};

    std::vector<std::string> pair_type{}; // required, at least one entry
};
SAFIRE_DEFINE_PARAMETERS(PairCorrelatorParameters, name, filename, pair_type);

struct SpinSpinCorrParameters {
  std::string name{};
};
SAFIRE_DEFINE_PARAMETERS(SpinSpinCorrParameters, name);

struct EstimatorParameters {
  EstimatorType name{};
  bool remove{false};

  // the estimator may use a different wavefunction and hamiltonian than the driver
  std::string wfn{};
  std::string ham{};

  // basic
  bool timers{false};
  int nhist{0};

  // energy
  bool overwrite{false};
  bool print_components{};
  bool print_sign{};

  int equil{};
  int skip{}; // ?

  // mixed
  int equil_multiplier{};

  // bp
  int bp_walker_ortho_interval{10}; // in units of steps
  bool path_restoration{true};
  bool extra_path_restoration{false};

  // resolve_defaults falls back to the measure_interval_multiplier of the enclosing execute block
  std::optional<std::vector<int>> measure_interval_multiplier{};

  // observables
  std::optional<OneRDMParameters> onerdm{};
  std::optional<DiagTwoRDMParameters> diag2rdm{};
  std::optional<TwoRDMParameters> twordm{};
  std::optional<PairCorrelatorParameters> pair_correlators{};
  std::optional<SpinSpinCorrParameters> spinspin{};
};
SAFIRE_DEFINE_PARAMETERS(EstimatorParameters, name, remove, wfn, ham, timers, nhist, overwrite, print_components,
                         print_sign, equil, skip, equil_multiplier, bp_walker_ortho_interval, path_restoration,
                         extra_path_restoration, measure_interval_multiplier, onerdm, diag2rdm, twordm,
                         pair_correlators, spinspin);

/// Reads a parameter whose default resolve_defaults is responsible for filling in.
template<typename T>
const T& resolved(const std::optional<T>& value, std::string_view name) {
  utils::check(value.has_value(), "The parameter '{}' was not resolved. Did resolve_defaults run?", name);
  return *value;
}

/// The measurement intervals of an estimator, in units of the population control interval.
inline const std::vector<int>& measure_interval_multipliers(const EstimatorParameters& params) {
  const std::vector<int>& multipliers = resolved(params.measure_interval_multiplier, "measure_interval_multiplier");
  utils::check(!multipliers.empty(), "'measure_interval_multiplier' must not be empty.");
  return multipliers;
}


struct ExecuteParameters {
  std::optional<utils::BlockRef<WalkerSetParameters>> walker_set{};
  std::optional<utils::BlockRef<WavefunctionParameters>> wavefunction{}; // required
  std::optional<utils::BlockRef<HamiltonianParameters>> hamiltonian{};
  std::optional<utils::BlockRef<PropagatorParameters>> propagator{};
  std::vector<EstimatorParameters> estimator{};


  std::string hdf_read_file{}; // restart from checkpoint
  std::string hdf_write_file{}; // write checkpoint
  int steps{1};
  int sweeps{1}; // finite temperature sweeps
  int population_control_interval{DEFAULT_POPULATION_CONTROL_INTERVAL};
  int measure_interval_multiplier{DEFAULT_MEASURE_INTERVAL_MULTIPLIER};
  int walker_ortho_interval{DEFAULT_WALKER_ORTHO_INTERVAL};
  int checkpoint_interval{-1};
  double weight_reset{0.0}; // in units of time
  double dshift{1.0};
  bool print_sweep_step{false}; // ftafqmc only

  // fix_bias // CSAFQMC only
  // filename
  // ndets_to_read

  double timestep{DEFAULT_TIME_STEP};
  int n_walkers_per_mpi_task{10};
  bool set_nwalker_to_target{};
  double initial_Eshift{}; // make optional
  std::optional<int> seed{};
};
SAFIRE_DEFINE_PARAMETERS(ExecuteParameters, walker_set, wavefunction, hamiltonian, propagator, estimator, hdf_read_file,
                         hdf_write_file, steps, sweeps, population_control_interval, measure_interval_multiplier,
                         walker_ortho_interval, checkpoint_interval, weight_reset, dshift, print_sweep_step,
                         timestep, n_walkers_per_mpi_task, set_nwalker_to_target, initial_Eshift, seed);


struct AFQMCParameters {
  // not a member of the object itself: the driver type is the key the object is stored under
  DriverType driver{DriverType::afqmc};

  ProjectParameters project{};

  std::vector<ExecuteParameters> execute{};

  // blocks declared outside of an execute block have to be named, so that an execute block can refer to them
  std::vector<WalkerSetParameters> walker_set{};
  std::vector<WavefunctionParameters> wavefunction{};
  std::vector<HamiltonianParameters> hamiltonian{};
  std::vector<PropagatorParameters> propagator{};
};
SAFIRE_DEFINE_PARAMETERS(AFQMCParameters, project, execute, walker_set, wavefunction, hamiltonian,
                         propagator);


/// Reads the whole input document. The key of the top level object selects its driver, e.g. {"afqmc": {...}}.
inline AFQMCParameters parse_input_file(const std::filesystem::path& filename) {
  std::ifstream input{filename};
  if(!input) {
    throw std::runtime_error{std::format("Could not open input file '{}'", filename.string())};
  }

  const nlohmann::json raw_parameters = nlohmann::json::parse(input);

  utils::check(raw_parameters.is_object(), "Expected a json object at the top level of the input file, but found {}.",
               raw_parameters.type_name());
  utils::check(raw_parameters.size() == 1, "The input file needs to contain exactly one simulation block.");

  const auto simulation = raw_parameters.cbegin();
  AFQMCParameters params;
  from_json(nlohmann::json(simulation.key()), params.driver);
  simulation.value().get_to(params);
  return params;
}

void print_parameters(const AFQMCParameters& params);

}
