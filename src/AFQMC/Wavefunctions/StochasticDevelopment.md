# StochasticWfn Development Guide

This document tracks the current state of `StochasticWfn` and the methods that must be
overridden for it to function as a full-fledged stochastic trial wavefunction in SAFIRE,
rather than a thin delegate wrapper around `NOMSD`.

---

## Overhaul port status

### Branch layout (`beevus77/SAFIRE`, Jun 2026)

| Branch | Role |
|--------|------|
| **`main`** | Active development — overhaul API + ported stochastic (Phases 1a–3b-var merged Jun 2026). Cut feature branches here and merge back after review. |
| **`stochastic-wfn-phase-3c`** | Phase 3c **complete** (3c-i walker-conditioned sampling + 3c-ii leapfrog, incl. the generalized-`w_p` measurement-energy fix); cut from `main` after 3b-var, pending merge back. |
| **`stochastic-wfn-develop`** | Frozen develop-line reference (Phases 1a–3b complete on the old `boost::variant` stack). Kept for comparison and fixture history; not the integration target. |
| **`upstream/overhaul`** | Upstream architecture base; rebase `main` onto it periodically while upstream settles. |

The stochastic feature was first implemented on the develop-line (`stochastic-wfn-phase-*` branches,
now archived on **`stochastic-wfn-develop`**). It was transplanted onto **`upstream/overhaul`**;
that port is **`main`** (`std::variant`, `memory::const_shared_array`, `Log_Overlap`, …).

| | `stochastic-wfn-develop` | `main` |
|---|---|---|
| Phases 1a–3b implementation | Complete | **Ported** (as of Jun 2026) |
| `safire_lib` build | Verified | **Compiles** |
| `test_afqmc` build + `wfn_factory: sdet` | N/A | **Pass — 10 assertions, Ne_cc-pvdz fixture (Jun 2026)** |
| `[stochastic_wfn]` static suite (1a–3a) | Pass on CPU compute node | **Ported + CPU-verified [overhaul], Ne_cc-pvdz (Jun 2026)** |
| `[stochastic_wfn]` dynamic suite (3b full-G) | Pass on CPU compute node | **Ported + CPU-verified [overhaul], Ne_cc-pvdz (Jun 2026)** |
| `[stochastic_wfn]` 3c-i (`stochastic_conditioned_propagator_step`) | — | **Ported + CPU-verified [overhaul], Ne_cc-pvdz (Jun 2026)** — walker-conditioned inner sampling; finiteness smoke |
| `[stochastic_wfn]` 3c-ii (`stochastic_leapfrog_propagator_step`) | — | **Ported + CPU-verified [overhaul], Ne_cc-pvdz (Jun 2026)** — hybrid leapfrog + importance-reweighted overlap; finiteness smoke (driver-level energy parity is a follow-up) |
| Runtime / driver smoke | Partial (`stochastic_propagator_step`) | **`stochastic_propagator_step` ported + CPU-verified [overhaul] (CLOSED); finiteness only**; **Phase 7 BP integration smokes** (`stochastic_back_propagation_estimator_smoke`, `stochastic_back_propagation_driver_smoke`) at the static delegate limit; **full `DriverFactory` run with VAFQMC-exported Ne cc-pVDZ HDF5 (Stages A/C 3b-var; Stages D/E 3c-i/3c-ii leapfrog, Jun 2026)** — see [Ne cc-pVDZ driver experiments](#ne-cc-pvdz-driver-experiments-jun-2026) |
| GPU build | CPU-only gate in dynamic path | **Not tested** |

**The full `[stochastic_wfn]` tag passes on overhaul (19 cases, 6657 assertions, `mpirun -np 1`,
`Ne_cc-pvdz`, Jun 2026; the 9th case is the Phase 3b-var anchor `stochastic_inner_hamiltonian_same_as_true`; the 10th is the Phase 3c-i smoke `stochastic_conditioned_propagator_step`; the 11th is the Phase 3c-ii leapfrog smoke `stochastic_leapfrog_propagator_step`; the 12th is the Phase 5 observable-DM case `stochastic_mixed_density_matrix_matches_nomsd`; the 13th is the Phase 5 `accumulate_estimators` case `stochastic_accumulate_estimators_matches_nomsd`; the 14th is the Phase 6 mean-field case `stochastic_mean_field_matches_nomsd`; the 15th is the Phase 6 call-order regression `stochastic_mean_field_production_order`; the 16th is the Phase 7 back-prop-reference case `stochastic_back_propagation_matches_nomsd`; the 17th is the Phase 7 call-order regression `stochastic_back_propagation_production_order`; the 18th is the Phase 7 BP estimator integration smoke `stochastic_back_propagation_estimator_smoke`; the 19th is the Phase 7 BP driver integration smoke `stochastic_back_propagation_driver_smoke`).** The Catch2 cases were ported to `tests/test_wfn_factory.cpp` (delegate-limit
parity + Phases 2a/2b/3a static reductions + the 3b full-G dynamic trio + 3c-i conditioned sampling + 3c-ii
leapfrog + the Phase 5 observable mixed DM + accumulate_estimators + the Phase 6 mean field). Porting them surfaced and
fixed three real overhaul-only bugs — see
[Port bugs surfaced by the test port](#port-bugs-surfaced-by-the-test-port-overhaul).

**Remaining gates:** (1) the Phase 1a/1b/1c **infrastructure** tests are still unported (need new
`Wavefunction`-variant accessors — see the *Ported vs. deferred* note below); (2) **GPU**
build/run is untested (the full-G dynamic path is CPU-only gated this phase); (3) multi-rank
(`-np > 1`) is **validated (Jun 2026)** — **the full `[stochastic_wfn]` suite passes at `-np 2`** (all 19
cases) after fixing four bugs (stochastic `Energy` double-count; accumulate-test HDF5;
`full_g::energy_closed` kernel; stochastic `Log_Overlap` cross-rank reduce on distributed walkers). See
[Multi-rank status](#multi-rank--np--1-status-validated-jun-2026-worker6035).

**Ported vs. deferred.** The ported cases test through the **public `Wavefunction` API only**:
`stochastic_overlap_matches_nomsd` (2a), `stochastic_energy_matches_nomsd` (2b),
`stochastic_vbias_matches_nomsd` (3a), and the 3b trio (`stochastic_full_g_matches_compact`,
`stochastic_dynamic_ensemble_smoke`, `stochastic_propagator_step`,
`stochastic_conditioned_propagator_step` [3c-i], `stochastic_leapfrog_propagator_step` [3c-ii]), alongside
the pre-existing `stochastic_wfn_matches_nomsd` / `stochastic_build_smoke`. The develop **infrastructure**
tests
(`stochastic_inner_walkers_init`, `stochastic_inner_outer_infrastructure_independent`,
`stochastic_inner_propagator_construction`, `stochastic_inner_walkers_uninitialized_smoke`) are
**not yet ported**: they reach into the typed `StochasticWfn` (`inner_nomsd()`, `outer_nomsd()`,
`inner_propagator()`, `inner_wset()`) via `boost::apply_visitor`, but the overhaul `Wavefunction`
keeps its `std::variant` **private** and exposes only `is_stochastic_wavefunction()`,
`stochastic_inner_walkers_initialized()`, `initialize_stochastic_inner_walkers()`, `begin_inner_step(wset)`.
Porting them faithfully needs new accessors on the `Wavefunction` variant (at minimum the documented
`stochastic_inner_wset()`, plus typed inner/outer NOMSD + inner-propagator access) — a production-API
decision left open.

### API mapping (develop → overhaul)

| develop | overhaul |
|---------|----------|
| `template<bool MP, …>` | `template<MEMORY_SPACE MEM, …>` |
| `boost::variant` | `std::variant` |
| `boost::multi` / `shared_array` | `nda::array` / `memory::const_shared_array` / `memory::buffered_array` |
| `TaskGroup_` | `std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>` |
| `Overlap` | `Log_Overlap` |
| `test_afqmc_wavefunctions` (`src/AFQMC/Wavefunctions/tests/`) | `test_afqmc` (`tests/`, consolidated binary) |

### Test fixtures on overhaul (the develop fixtures are GONE)

Overhaul **deleted the entire sparse Hamiltonian path** — no `FactorizedSparseHamiltonian` /
`SparseTensor` / `SparseHamiltonian`. `peekHamType` (`Hamiltonians/hdf5_helpers.hpp`, `std` format)
recognizes only `/Hamiltonian/{KPFactorized, DenseFactorized, ModelHamiltonian}`; a legacy
`/Hamiltonian/Factorized` (old sparse cholesky) subgroup now **aborts**
(`"Invalid hdf5 file format in peekHamType()"`).

- **`C_1x1x1_dzvp/ham_chol_sc.h5` no longer exists.** Every `develop` example in this doc that uses it
  is stale — treat as historical. The only cholesky-ish file left in `C_1x1x1_dzvp/` is
  `choldump_phmsd.h5`, which is the old `/Hamiltonian/Factorized` (sparse) format and **aborts in
  `peekHamType`** — do **not** point tests at it.
- The stochastic full-G path (Phase 3b) requires a HamOp that implements `energy_fullG` + the full-G
  `vbias` layout. On overhaul that is **`Real3IndexFactorization` only** (others carry `energy_fullG`
  stubs). The chain is `/Hamiltonian/DenseFactorized` → `RealDenseHamiltonian` →
  `getHamiltonianOperations()` → `Real3IndexFactorization`.

**Canonical overhaul fixture** — dense cholesky + closed-shell RHF, routes through
`Real3IndexFactorization`:

| Role | File |
|------|------|
| Hamiltonian | `tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5` (`/Hamiltonian/DenseFactorized`) |
| Trial (CLOSED/RHF) | `tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5` |

```bash
HAMIL=$PWD/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
WFN=$PWD/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
mpirun -np 1 ./build/tests/bin/test_afqmc --hamil "$HAMIL" --wfn "$WFN" "[stochastic_wfn]"
```

`Ne_cc-pvdz` is **single-determinant RHF only** (no dense-cholesky `wfn_msd.h5`). That suffices for the
minimal port: delegate-limit checks require `ndet == 1` anyway, and `inner_nwalkers`-invariance uses a
static replicated ensemble that is `ndet`-independent. The **delegate-limit** parity test
(`stochastic_wfn_matches_nomsd`, `inner_nsteps = 0`) is in fact **HamOp-agnostic** — at the delegate
limit the stochastic `vbias` uses the *compact* path, so it runs on any cholesky/THC NOMSD fixture
(including the harness's built-in `utils/tests/functional/` files); only the **full-G** tests
(`inner_nsteps > 0`) need the `Ne_cc-pvdz` dense+RHF set. Other dense/KP fixtures for reference:
`He_2x2x2_dzv/ham_chol_uc.h5` = `KPFactorized` (→ `KP3IndexFactorization`, full-G stub — not usable
for the full-G path yet).

---

## Port bugs surfaced by the test port (overhaul)

Porting the `[stochastic_wfn]` cases to `tests/test_wfn_factory.cpp` and running them on CPU exposed
overhaul-only regressions (none present on develop). All are in the stochastic hot path and were
invisible to the pre-existing `stochastic_wfn_matches_nomsd` test, because at `inner_nwalkers = 1`
**every override delegates to `NOMSD`** — the bugs only bite the genuine P>1 reduction and the
dynamic full-G path. This is why the new tests build stochastic trials at `inner_nwalkers = 3`
(invariance) and `inner_nsteps = 1` (full-G), not just the delegate limit.

1. **Log-overlap convention (`StochasticWfn::Log_Overlap`, `StochasticWfn::Energy`) — FIXED.**
   The overhaul renamed the visitor `Overlap` → `Log_Overlap`, and `NOMSD` now stores the **log**
   overlap in the `OVLP` property (`NOMSD::Energy` too: `Ov = log(deno) + log_m`). The ported
   reductions still stored the **linear** effective overlap `(1/P) Σ_p exp(z_p)`, so `OVLP` came out as
   `exp(NOMSD's value)`. Fix: take `std::log` of the reduced linear overlap at the end of both methods.
   The reduction is inherently linear-space (`log((1/P) Σ_p exp z_p)`), so it returns the **principal**
   branch — its phase can differ from `NOMSD`'s unwrapped log-det phase by an integer multiple of 2πi
   while describing the same overlap, so the tests compare `exp(OVLP)`.

2. **Full-G one-body contraction rank mismatch (`full_g::energy_closed`) — FIXED.**
   `nda::tensor::contract(scl, hijf, "ik", Gfull, "wik", …)` was called with `hijf` rank-1 (flat
   `[NMO*NMO]`) and `Gfull` rank-2 (`[nwalk][NMO*NMO]`), but the index labels imply rank-2/rank-3.
   Fix: `nda::reshape` to `h[i][k]` and `G[w][i][k]` before the contract.

3. **Full-G two-body EXX/EJ (`full_g::energy_closed`) — FIXED.**
   With #2 fixed, `E1` matched but EXX/EJ came out **exactly zero**. Two deviations from both the
   develop kernel and the compact `Real3IndexFactorization::energy_impl`: (a) the gemm sliced `Lankf`'s
   **columns** (`Lankf(all, range(i0,iN))`) instead of its **rows** — `Lankf` is `[NMO*local_nCV][NMO]`
   and the `(i,nc)` combined index that FairDivide partitions indexes its rows; (b) the EXX/EJ
   reductions used `nda::blas::dotc` (conjugating) instead of the non-conjugating `nda::blas::dot` that
   `energy_impl` uses (`EXX = Σ T[i][j][nc] T[j][i][nc]`, no conjugation). Both corrected; verified by
   `stochastic_full_g_matches_compact` (full-G energy/force-bias == compact `nd=0` == NOMSD at the anchor).

**Convention reminder for future ports:** the overhaul `OVLP` property is a complex **log** overlap;
any new reduction that builds an overlap in linear space must `log()` it before storing, and any
energy/exchange contraction over the real Cholesky uses the **non-conjugating** `nda::blas::dot`.

---

## Current State

`StochasticWfn` (`StochasticWfn.hpp` / `StochasticWfn.icc`) is a
`template<MEMORY_SPACE MEM, class devPsiT>` class that inherits from `AFQMCInfo` and owns:

```cpp
NOMSD<MEM, devPsiT> nomsd_;                // outer delegate (True Ham; outer-walker-facing)
std::unique_ptr<StochasticInnerStack<…>> inner_stack_;  // inner NOMSD (Ĥ_var) + Propagator + RNG
StochasticInnerEnsemble inner_ensemble_;   // inner WalkerSet + walker-init RNG
int  inner_nwalkers_{1};                   // inner samples P per outer walker
int  inner_nsteps_{0};                     // B̂_T steps from the anchor (0 = static; 1 = canonical)
bool inner_conditioning_{false};           // 3c-i: sample conditioned on each outer walker φ_w
bool inner_leapfrog_{false};               // 3c-ii: shared-ensemble exact Eq. 25 ratio
nda::array<RealType,1> inner_cond_mag_;    // |⟨ψ_q|φ_cond⟩| per inner walker (leapfrog reweight)
// + mechanical members: inner_timestep_, inner_step_pending_ (latch), inner_anchor_ (|φ_T⟩ cache), mpi_
```

**The inner stack (Phases 1a–1c) is complete and ported:** an inner trial ensemble (`inner_wset()`,
sized `P = inner_nwalkers`, two-phase factory/driver init), an independent inner `NOMSD` with its own
`HamiltonianOperations`/`SlaterDetOperations`, and an inner `Propagator` built via `PropagatorFactory`
at construction. The inner `NOMSD` holds the **Variational Hamiltonian `Ĥ_var`** — built from the
`inner_hamiltonian` HDF5 file (Phase 3b-var), or a clone of the True Ham when that key is absent. Only
the dedicated Phase 1a/1b/1c infrastructure unit tests are unported (they need typed accessors the
overhaul `Wavefunction` variant does not expose).

**Every Tier 1 hot-path quantity is stochastic and CPU-verified on overhaul** (`Ne_cc-pvdz`): hybrid
propagation uses the Phase 2a `Log_Overlap` override, local-energy the Phase 2b `Energy` override, and
the force bias the Phase 3a `MixedDensityMatrix_for_vbias` override — each reduces the inner ensemble
(Eq. 27 of arXiv:2505.18519) and reproduces plain `NOMSD` at the single-determinant delegate limit.
`vbias` and `vHS` are trial-independent and **delegate permanently** to `nomsd_`. The Tier 2 observable
mixed DM / `accumulate_estimators` (Phase 5) and the Tier 3 mean field `G_MF` / `vMF` (Phase 6) are
**also stochastic** — the observable mixed DM reduces the ensemble against the outer walkers, while the
mean field reduces it **against itself** (`⟨Ψ_T|Ô|Ψ_T⟩`, no walker). The remaining Tier 2–6 methods
delegate, so outer-facing behavior matches plain `NOMSD` at the delegate limit.

**Inner-ensemble sampling** is static at `inner_nsteps = 0`; at `inner_nsteps > 0` the inner propagator
field-samples `{ψ_p = B̂_T(Y^[p])|φ_T⟩}` from the anchor, in one of three modes (CLOSED/CPU only this
phase): **free projection** (Phase 3b — walker-independent, bare Gaussian fields), **walker-conditioned**
(`inner_conditioning`, Phase 3c-i — `nwalk·P` block-structured, each block importance-sampled toward its
outer walker `φ_w`), and **+ leapfrog** (`inner_leapfrog`, Phase 3c-ii — the shared-ensemble
propagate-then-resample that makes the Eq. 25 overlap-ratio `𝒩(φ)` cancellation exact). Reductions are
always scored against the **True** Hamiltonian (outer `nomsd_`); inner walkers that have left the anchor
use the **un-rotated full-G** contraction (`energy_fullG`; the force bias reaches it via
`Real3IndexFactorization::vbias` G-layout dispatch — no separate seam). Per-phase detail is in
[Implementation phases](#implementation-phases-status-and-plan); the math is in
[Estimator Definitions](#estimator-definitions).

---

## Inner stack architecture (Phase 1c)

Phase 1c refactored the inline `inner_nomsd_` member into a **`StochasticInnerStack`**
abstraction because the inner `Propagator` must hold stable references to a `Wavefunction&`
and a device RNG pointer. `StochasticWfn` is moved into the outer `Wavefunction` variant after
construction; heap allocation keeps those references valid.

### Type layout

```
StochasticWfn
├── nomsd_                          (outer delegate; value member)
├── inner_ensemble_                 (WalkerSet + walker-init RNG; Phase 1a)
└── inner_stack_  ──► StochasticInnerStackImpl  (in WavefunctionFactory.cpp)
                      ├── wfn_      unique_ptr<Wavefunction>  → inner NOMSD
                      ├── prop_     unique_ptr<Propagator>     → bound to wfn_, rng_
                      └── rng_      unique_ptr<DeviceRandomGenerator_t>
```

| Component | Role |
|-----------|------|
| `StochasticInnerStack` | Abstract interface in `StochasticWfn.hpp` (`nomsd()`, `wavefunction()`, `propagator()`). Keeps `Propagator.hpp` out of the header (circular include with `Wavefunction.hpp`). |
| `StochasticInnerStackImpl` | Concrete implementation in `WavefunctionFactory.cpp` (anonymous namespace). |
| `inner_ensemble_.rng` | `RandomGenerator_t` used only when constructing/resizing `inner_wset()`. Separate from the propagator RNG. |
| `stack->rng_` | `DeviceRandomGenerator_t` passed to `PropagatorFactory::buildPropagator`. Rank-decorrelated from `inner_seed`. |

### Why the inner NOMSD is wrapped in `Wavefunction`

`PropagatorFactory::buildPropagator` takes a `Wavefunction&`, not a typed `NOMSD&`. The inner
NOMSD is therefore stored inside a heap-allocated `Wavefunction` variant. `inner_wavefunction()`
exposes that wrapper; `inner_nomsd()` / `inner_wfn()` unwrap to the typed `NOMSD` inside it.
`stochastic_inner_propagator_construction` verifies SDetOp pointer identity between the wrapper
and the typed accessor.

### Factory build path

`buildStochasticNomsdWavefunction()` (in `WavefunctionFactory.h`) now:

1. Builds outer/inner `SlaterDetOperations` and `HamiltonianOperations` via the Phase 1b helpers.
2. Calls `buildStochasticInnerStack()` (defined in `WavefunctionFactory.cpp`) to assemble
   `wfn_`, `rng_`, and `prop_`.
3. Passes the finished `unique_ptr<StochasticInnerStack>` into the `StochasticWfn` constructor.

`InnerPropagatorBuilder` is a thin subclass of `PropagatorFactory` that exposes the protected
`buildPropagator` without registering the result in the factory map — the caller owns the
`Propagator` directly.

**Task groups:** inner `HamOps` and the inner propagator both use the same `(TGprop, TGwfn)`
pair as the outer driver build. Propagator-side Cholesky-vector distribution and `vMF`
reductions assume `TGprop` geometry.

**Destruction order:** in `StochasticInnerStackImpl`, `prop_` is declared after `wfn_` and
`rng_` so the propagator (which references both) is destroyed first.

---

## Dual-Hamiltonian architecture (True vs Variational)

A full `StochasticWfn` carries **two** Hamiltonians, with two distinct jobs. Both are now live: the
outer **True** Ham scores every returned quantity, and the inner **Variational** `Ĥ_var` generates the
stochastic-trial samples (built from `inner_hamiltonian`, or a True-Ham clone when absent — see
[Inner Hamiltonian](#inner-hamiltonian-ĥ_var-vs-clone-fallback) below).

| | **True Hamiltonian `Ĥ`** | **Variational Hamiltonian `Ĥ_var`** |
|---|--------------------------|--------------------------------------|
| **Role** | The physical Hamiltonian — same object `NOMSD` uses. Defines the outer (external) propagator's imaginary-time evolution of the driver walkers, and is the operator in **every quantity `StochasticWfn` returns** (`⟨ψ_p\|Ĥ\|φ_w⟩`, force bias, …). | Pre-optimized, input externally. Its HS propagator `B̂_T(Y) = exp(√Δτ Σ_γ Y_γ L̂_γ^var)` *generates* the stochastic trial: one step on the anchor gives an inner walker `ψ_p = B̂_T(Y^[p])\|φ_T⟩`, a single Slater determinant (Thouless) that the AFQMC machinery already handles. |
| **Lives in** | The **outer `nomsd_`** delegate (the NOMSD internal to `StochasticWfn`). | The **inner stack** — the inner `NOMSD`'s `HamOps` and the inner `Propagator`. **No third `HamOps`**: the inner NOMSD *is* the Variational engine. |
| **Drives** | Outer AFQMC loop; all Tier 1 reductions (`Overlap`/`Energy`/`vbias`) scored against `Ĥ`. | The inner walk that samples `{ψ_p}` (only when `inner_nsteps > 0`). |
| **Provides** | Cholesky for `Energy`/`vbias` contractions (half-rotated against the outer trial). | Cholesky for `B̂_T`'s `vHS`, and — for variance reduction (Phase 3c) — a **force bias**. |
| **Notion of "Energy"** | The AFQMC local energy `⟨Ψ_T\|Ĥ\|φ_w⟩/⟨Ψ_T\|φ_w⟩` (Phase 2b). | **None in the single-step trial.** `B̂_T` needs only `Ĥ_var`'s Cholesky (a `vHS`) plus a force bias; a Variational *local energy* is not required. (It would only appear for a multi-step Variational projection — out of scope — or as an optional diagnostic.) |

So "two Hamiltonians → two propagators → two energies" resolves, for the in-scope **single-step**
trial, to: **one energy (True, outer) + `B̂_T` field-sampling machinery (Variational, inner)**.

### `inner_nsteps` and the single-step goal

`inner_nsteps` counts `B̂_T` (Variational) steps: `0` = static anchor (today, `B̂_T = 𝟙`), `1` =
the canonical single-step stochastic trial (the end goal). `inner_nsteps > 1` (a longer Variational
projection) is **not in scope** and would be the only setting that needs a Variational energy /
population control. Because the end goal is fixed at a single step, `inner_nsteps` is ultimately a
transitional knob that may be retired once the single-step trial is the default.

### Field sampling reuses the propagator; only the bias is new

The inner walk's mechanics — sample `Y`, form `v = √Δτ Σ_γ Y_γ L_γ^var` (a `vHS` with the
Variational Cholesky), exponentiate onto the inner Slater matrices (`apply_propagators`) — are
exactly what `AFQMCBasePropagator` already does on GPU. **Priority: reuse existing propagator
objects/kernels/acceleration; write custom code only for the field-sampling distribution.** The
distribution is the Phase 3b/3c fork:

- **Phase 3b — free / bare sampling.** `Y ~ p_T(Y)` (Gaussian), no force bias. This is precisely the
  inner propagator's **free-projection mode** (`free_projection` zeroes `vbias`, `assemble_X` uses
  bare noise, then `vHS` + `apply_propagators`, then `free_projection_walker_update` — overlap-ratio
  weights, no local energy). So 3b is mostly *configuration*: switch the inner propagator from the
  Phase 1c default (hybrid) to free, and run one step. No bespoke `B̂_T` code, no Variational energy.
- **Phase 3c — `φ_w`-conditioned sampling (Eq. 23).** A custom force bias
  `x̄ ∝ √Δτ · L_γ^var · ⟨φ_T\|c†c\|φ_w⟩/⟨φ_T\|φ_w⟩` that depends on the **outer** walker — which the
  stock inner `vbias` (self-conditioned on `φ_T`) cannot supply. That outer coupling, with the
  propagate-then-resample leapfrog, is the new machinery; it still reuses `vHS`/`apply_propagators`.

### Consequence for the reductions: score with the True Hamiltonian

Because `Ĥ` lives in the **outer `nomsd_`**, every Tier 1 reduction must take its energy/bias
contraction from `nomsd_`, **not** from `inner_nomsd()` (which becomes `Ĥ_var`). The cross **density
matrix** `⟨ψ_p|c†c|φ_w⟩` is Hamiltonian-independent (pure orbital algebra) and can use either
`SDetOp`; only the `HamOp.energy` / `HamOp.vbias` contractions are Ham-specific and must be True.
**Phase 2b `Energy` and Phase 3a `vbias` both follow this** — their contractions run on the outer
`nomsd_` (True Ham), not `inner_nomsd()`.

### Inner Hamiltonian: `Ĥ_var` vs. clone fallback

The inner `NOMSD`'s `HamOps` is the **Variational** `Ĥ_var` when the stochastic wfn block names an
`inner_hamiltonian` HDF5 file (Phase 3b-var); otherwise it is a **clone of the True Ham** (the
pre-3b-var fallback, still valid when no variational file is supplied). At `inner_nsteps = 0` the inner
propagator is dormant (`B̂_T = 𝟙`), so the static Phases 2/3a are insensitive to which one it is. With a
distinct `Ĥ_var`, the inner-vs-outer layout-parity assertions of Phase 1b (equal Cholesky counts) no
longer hold — they were static-phase scaffolding.

---

## Phase 2 overview: `Overlap` (2a) vs `Energy` (2b)

Phase 2 splits the first stochastic reductions on the propagator hot path into two independent
overrides. They share the same static-ensemble limit (`inner_nsteps = 0`, `B̂_T = 𝟙`) and the
same inner/outer walker geometry, but they answer different questions and integrate with
different propagator modes.

| | **Phase 2a — `Log_Overlap`** | **Phase 2b — `Energy`** |
|---|--------------------------|-------------------------|
| **Status** | **Complete**; CPU-verified [overhaul] | **Complete**; CPU-verified [overhaul] |
| **Quantity** | Effective trial overlap `Ov[w]` per outer walker | Local energy `E[w]` (E1, EXX, EJ) and overlap `Ov[w]` per outer walker |
| **Reduction** | Cross overlaps `(1/P) Σ_p ⟨ψ_p\|φ_w⟩` between inner walkers `ψ_p` and outer walkers `φ_w` | `E[w] = Σ_p ⟨ψ_p\|Ĥ\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩` from the same inner/outer cross pairs; `Ov[w]` identical to 2a |
| **Propagator consumer** | **Hybrid** path: `wfn.Log_Overlap(wset, new_overlaps)` in `AFQMCBasePropagator::step` | **Local-energy** path: `wfn.Energy(wset, new_energies, new_overlaps)` |
| **Implementation** | `StochasticWfn.icc` — uses outer `SDetOp`, `FairDivideBoundary` over `nw×P` pairs | `StochasticWfn.icc` — mirrors `NOMSD::Energy_shared`, looping inner walkers as the determinant index (weight `1/P`); the cross DM + `energy_from_G` run on the **outer `nomsd_` (True Ham)** with inner-walker bras |
| **Phase 3 coupling (now done)** | The dynamic ensemble (3b), the `S_p` phase weights (Eq. 26), conditioning (3c-i) and the leapfrog (Eq. 25, 3c-ii) all build on this reduction; `Log_Overlap` is absolute in 3b and importance-reweighted under the leapfrog | Same, plus the off-anchor full-G scoring (3b) and the generalized `w_p` weight (3c-ii) |

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) both
overrides reproduce plain `NOMSD`, verified by `stochastic_overlap_matches_nomsd` (2a) and
`stochastic_energy_matches_nomsd` (2b). `stochastic_wfn_matches_nomsd` gates its energy/overlap
assertions on `ndet == 1` (both overrides diverge from NOMSD for multi-determinant trials by
design).

---

## Stochastic overlap (Phase 2a)

`StochasticWfn::Overlap` is the first Tier 1 method to diverge from the `nomsd_` delegate. It
computes the **effective trial overlap** of each outer driver walker against the inner ensemble,
following Eq. 24 of [arXiv:2505.18519](https://arxiv.org/abs/2505.18519) ("Implementing advanced
trial wave functions in fermion quantum Monte Carlo via stochastic sampling", Xiao *et al.*,
J. Chem. Phys. **163**, 164109 (2025)).

### Paper formulation

The stochastic trial is an integral over auxiliary fields,
`|Ψ_T⟩ = ∫ dY p_T(Y) B̂_T(Y) |φ_T⟩` (Eq. 21), with `|φ_T⟩` a **single** Slater determinant and
`B̂_T(Y)` an exponential one-body propagator. Each sampled field path `Y^[p]` gives a single
determinant `B̂_T(Y^[p])|φ_T⟩`. The effective bra (Eq. 24) is a phase-weighted, importance-sampled
reduction over `P` paths; the quantity that drives the walk is the **overlap ratio** (Eq. 25-26),
in which the per-walker normalization `𝒩(φ)` cancels.

### What Phase 2a implements (static-ensemble limit)

At `inner_nsteps = 0` there is no field sampling: `B̂_T = 𝟙`, so the phase factor `S(Y^[p])` and
importance reweighting are degenerate and Eq. 24 collapses to a plain average. The implemented
reduction over the `P = inner_nwalkers` inner walkers `ψ_p` (`inner_wset()`) is

```
Ov[w] = (1/P) Σ_p ⟨ψ_p | φ_w⟩
```

where `⟨ψ_p|φ_w⟩` is a single-determinant overlap between an **inner**-walker and an **outer**-walker
Slater matrix. This is *not* `inner_nomsd().Overlap(inner_wset())` (inner-trial-vs-inner-walkers, the
Phase 1b sanity check) — it is a **cross** overlap between the two walker sets.

Implementation notes (`StochasticWfn.icc`):

- Each `⟨ψ_p|φ_w⟩` uses `SlaterDetOperations::Overlap(ψ_p, φ_w, LogOverlapFactor, herm=false)`, which
  forms `det(ψ_p^† φ_w)` — the same convention and `LogOverlapFactor` as `NOMSD::Overlap` (which passes
  the pre-transposed trial with `herm=true`). CLOSED squares the alpha overlap; COLLINEAR multiplies
  alpha and beta; NONCOLLINEAR uses the single packed matrix — mirroring the three
  `NOMSD::Overlap_shared` branches, with the CI weight `conj(ci[nd])` replaced by `1/P`.
- The `nw*P` (outer, inner) pairs are split across the local task group via `FairDivideBoundary`, then
  summed with `all_reduce` over `TG_local`. A new `DeviceBufferManager buffer_manager` member backs the
  transient overlap vector in the `Log_Overlap(wset)` overload (mirrors NOMSD). The outer `nomsd_`'s
  `SlaterDetOperations` and `TG_` (= outer `TGwfn`) are reused.

### Why the visitor `Log_Overlap` is overridden directly

The hybrid propagation path (`AFQMCBasePropagator::step`) consumes the overlap **only as a step-to-step
ratio** `ratioOverlaps = new/old` in `hybrid_walker_update`: its argument drives the cosine (phase)
constraint and its log feeds the hybrid energy, and the stored `OVLP` property is overwritten each step.
This is exactly Eq. 25's structure, so overriding the visitor `Log_Overlap` is the correct integration
point. The exact `𝒩(φ)` cancellation of Eq. 25 requires numerator and denominator to share the inner
field samples conditioned on the old walker — the paper's propagate-then-resample *leapfrog*, which
lives in the propagator hot path. Phase 2a's `Log_Overlap` produces just the absolute reduction above
(exact at `inner_nsteps = 0`); the phase weights `S_p`, walker-conditioning, and the leapfrog are added
in [Phase 3b/3c](#phase-3--propagator-hot-path-completion-and-the-dynamic-ensemble).

### Delegate limit and the multi-determinant caveat

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`, inner walker = the
trial determinant) the reduction is exactly `⟨φ_T|φ_w⟩` = the NOMSD overlap. **This holds only for
single-determinant trials** (RHF/UHF files): the paper's anchor `|φ_T⟩` is a single determinant, and the
inner walkers are initialized from `getInitialGuess` (`Psi0`, the dominant determinant). For a
multi-determinant trial (e.g. `wfn_msd.h5`) the single-determinant inner ensemble cannot reproduce the
CI-weighted NOMSD overlap, so the overridden `Log_Overlap` diverges from NOMSD there by design. The dedicated
`stochastic_overlap_matches_nomsd` test exercises `Log_Overlap` in isolation and gates the NOMSD-equality
assertion on `ndet == 1`; the `stochastic_wfn_matches_nomsd` anchor likewise gates its overlap/energy
assertions on `ndet == 1`.

**This caveat applies identically to every stochastic override** — the Phase 2b `Energy` and Phase 3a
`MixedDensityMatrix_for_vbias`/`vbias` below each reproduce NOMSD only at the single-determinant
delegate limit and diverge for multi-determinant trials by design, so their dedicated tests
(`stochastic_energy_matches_nomsd`, `stochastic_vbias_matches_nomsd`) gate NOMSD-equality on `ndet == 1`.

---

## Estimator Definitions

Every override in Phases 2–6 computes the same kind of object: a **mixed expectation value** of an
operator `Ô` between the stochastic trial `⟨Ψ_T|` and an outer walker `|φ_w⟩`. This section fixes
notation and writes each estimator we will need — overlap, local energy, mixed density matrix, force
bias — as one specialization of Eq. 27 of [arXiv:2505.18519](https://arxiv.org/abs/2505.18519), so the
later phases differ only in the choice of `Ô`. Equation numbers below refer to that paper.

### The one ansatz

```
|Ψ_T⟩ = ∫ dY  p_T(Y)  B̂_T(Y) |φ_T⟩                         (Eq. 21)
```

`|φ_T⟩` is a single Slater determinant (the *anchor*; `|Ψ_T⁰⟩` in the paper, taken as one determinant),
`B̂_T(Y) = exp(one-body)` is an exponential one-body operator built from auxiliary fields `Y`, and
`p_T(Y)` is the Hubbard–Stratonovich Gaussian weight. By Thouless' theorem each field path `Y^[p]`
(`p = 1 … P`) generates a single inner-walker determinant, written here through its bra:

```
⟨ψ_p|  ≡  ⟨φ_T| B̂_T(Y^[p])                                 (one Slater determinant)
```

In code `ψ_p` is `inner_wset()[p]` and `φ_w` is the outer driver walker `wset[w]`. Define the per-sample
overlap and its phase (Eq. 26):

```
O_p(φ_w)  ≡  ⟨ψ_p | φ_w⟩
S_p(φ_w)  ≡  O_p(φ_w) / |O_p(φ_w)|                          (Eq. 26)
```

### The master estimator (Eq. 27)

For any operator `Ô`, the stochastic-trial mixed estimator on outer walker `w` is the phase-weighted
average over the inner ensemble of an ordinary **per-pair local estimate**:

```
                  ⟨Ψ_T|Ô|φ_w⟩       Σ_p  ⟨Ô⟩_{p,w} · S_p(φ_w)
   ⟨Ô⟩_w   ≡   ───────────────  =  ─────────────────────────         (Eq. 27)
                  ⟨Ψ_T|φ_w⟩              Σ_p  S_p(φ_w)

   ⟨Ô⟩_{p,w}  ≡  ⟨ψ_p|Ô|φ_w⟩ / ⟨ψ_p|φ_w⟩
```

`⟨Ô⟩_{p,w}` is exactly the single-determinant (Wick / Slater–Condon) local estimate of `Ô` between the
inner determinant `ψ_p` and the outer walker `φ_w` — the same per-determinant-pair quantity `NOMSD`
already forms against a fixed trial. The stochastic trial only adds the phase-weighted average
`Σ_p (·) S_p / Σ_p S_p`. **Each estimator below is Eq. 27 with a different `Ô`** (the overlap is the
lone exception — it is the normalization, not a ratio of this form).

### The four estimators

| Estimator | `Ô` | Per-pair local estimate `⟨Ô⟩_{p,w}` | SAFIRE method | Phase |
|-----------|-----|--------------------------------------|---------------|-------|
| Overlap | `𝟙` | (absolute normalization; see below) | `Overlap` | 2a ✓ |
| Local energy | `Ĥ` | `E_{p,w} = ⟨ψ_p\|Ĥ\|φ_w⟩ / ⟨ψ_p\|φ_w⟩` | `Energy` | 2b ✓ |
| Mixed density matrix | `c†_i c_j` | `(G_{p,w})_{ij} = ⟨ψ_p\|c†_i c_j\|φ_w⟩ / ⟨ψ_p\|φ_w⟩` | `MixedDensityMatrix[_for_vbias]` | 3a ✓ |
| Force bias | `L̂_γ` | `Σ_{ij} (L_γ)_{ij} (G_{p,w})_{ij}` | `vbias` | 3a ✓ |

**1 — Overlap (`Ô = 𝟙`).** Not a ratio of the Eq. 27 form; it is the *absolute* bra normalization
(Eq. 24):

```
⟨Ψ_T|φ_w⟩ = (𝒩(φ_w)/P) Σ_p S_p(φ_w),     𝒩(φ_w) = ∫ dY |⟨φ_T| p_T(Y) B̂_T(Y) |φ_w⟩|   (Eq. 24)
```

What the random walk consumes is never this absolute value but the step-to-step **ratio**
`⟨Ψ_T|φ'⟩/⟨Ψ_T|φ⟩` (Eq. 25), in which `𝒩(φ_w)` cancels — see
[Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a). Phase 2a stores the static-limit
absolute overlap; the phase weights `S_p` are added in Phase 3b and the exact `𝒩` cancellation (the
leapfrog) in Phase 3c-ii.

**2 — Local energy (`Ô = Ĥ`).** Eq. 27 with the Hamiltonian; this is the per-walker term of the AFQMC
mixed-energy estimator (Eq. 16):

```
E_L[w] = Σ_p E_{p,w} S_p(φ_w) / Σ_p S_p(φ_w),     E_{p,w} = ⟨ψ_p|Ĥ|φ_w⟩ / ⟨ψ_p|φ_w⟩
```

`E_{p,w}` decomposes into the one-body (`E1`) and two-body Coulomb / exchange (`EJ` / `EXX`) pieces
exactly as `NOMSD::Energy` does today, each built from the per-pair Green's function `G_{p,w}` of
estimator 3. This is Phase 2b.

**3 — Mixed density matrix / one-particle Green's function (`Ô = c†_i c_j`).**

```
G_{ij}[w] = Σ_p (G_{p,w})_{ij} S_p(φ_w) / Σ_p S_p(φ_w),
(G_{p,w})_{ij} = ⟨ψ_p|c†_i c_j|φ_w⟩ / ⟨ψ_p|φ_w⟩
```

`G_{p,w}` is the standard single-pair mixed Green's function (`SDetOp.MixedDensityMatrix` between `ψ_p`
and `φ_w`). It is the shared ingredient of both the local energy (2, Phase 2b ✓) and the force bias
(4, Phase 3a).

**4 — Force bias (`Ô = L̂_γ`).** The `L̂_γ = Σ_{ij} (L_γ)_{ij} c†_i c_j` are the one-body Cholesky / HS
operators (`V̂ = -½ Σ_γ L̂_γ²`). Being one-body, the force bias is just the Cholesky contraction of the
mixed DM; the auxiliary-field shift is (Eq. 15)

```
x̄_γ[w] = -√Δτ · ⟨Ψ_T|L̂_γ|φ_w⟩ / ⟨Ψ_T|φ_w⟩ = -√Δτ · Σ_{ij} (L_γ)_{ij} G_{ij}[w]        (Eq. 15)
```

so SAFIRE's `vbias` computes the determinant-dependent factor `⟨Ψ_T|L̂_γ|φ_w⟩/⟨Ψ_T|φ_w⟩ = L · G[w]`
exactly as in deterministic AFQMC, but with the stochastic `G[w]` of estimator 3 (the propagator
applies the `√Δτ` / timestep prefactor). Phase 3a ✓ (`MixedDensityMatrix_for_vbias` → `vbias`).

### Static-ensemble limit (`inner_nsteps = 0`, `B̂_T = 𝟙`)

All `ψ_p` collapse to the anchor `φ_T`, the phase weights `S_p` become identical, and Eq. 27 reduces to
the single deterministic estimate against `φ_T`:

| Estimator | Static-limit value |
|-----------|--------------------|
| Overlap | `⟨φ_T\|φ_w⟩` |
| Local energy | `⟨φ_T\|Ĥ\|φ_w⟩ / ⟨φ_T\|φ_w⟩` |
| Mixed DM | `⟨φ_T\|c†_i c_j\|φ_w⟩ / ⟨φ_T\|φ_w⟩` |
| Force bias | `L · G` |

For a **single-determinant** anchor (`φ_T = Psi0`) these are precisely the plain-`NOMSD` estimates — the
delegate-limit regression each phase must reproduce. A multi-determinant `NOMSD` trial is *not* recovered
by a single-determinant anchor (see
[Delegate limit and the multi-determinant caveat](#delegate-limit-and-the-multi-determinant-caveat)).

### Walker-conditioned sampling (the Phase 3c coupling)

Away from the static limit the inner field paths are importance-sampled **conditioned on the outer
walker** (Eq. 23):

```
𝒫(Y; φ_w) = |⟨φ_T| p_T(Y) B̂_T(Y) |φ_w⟩| / 𝒩(φ_w)
```

so both the inner ensemble `{ψ_p}` and the phase weights `S_p(φ_w)` depend on `φ_w`. This walker coupling
(Phase 3c-i), together with the propagate-then-resample *leapfrog* that makes the `𝒩(φ_w)` cancellation
in the Eq. 25 ratio exact (Phase 3c-ii), is what Phase 3c adds on top of the dynamic, walker-independent
ensemble of Phase 3b. Phases 2a/2b/3a implement the `B̂_T = 𝟙` static limit, where the ensemble is
walker-independent, `S_p` is degenerate, and Eq. 27 collapses to the plain `(1/P) Σ_p` average.

---

## Stochastic local energy (Phase 2b)

`StochasticWfn::Energy` is the second Tier 1 method to diverge from the `nomsd_` delegate. It
overrides `Energy(wset)` / `Energy(wset, E, Ov)` to return a **stochastic local energy** (E1, EXX,
EJ) and a **consistent overlap** for each outer walker, reduced from the inner ensemble in the
same static limit as Phase 2a, following Eq. 27 of
[arXiv:2505.18519](https://arxiv.org/abs/2505.18519).

### What Phase 2b implements (static-ensemble limit)

At `inner_nsteps = 0` the phase factor `S_p` and importance reweighting are degenerate (`B̂_T = 𝟙`),
so Eq. 27 collapses to the overlap-weighted average over the `P = inner_nwalkers` inner walkers
`ψ_p` (`inner_wset()`):

```
E[w]  = Σ_p ⟨ψ_p|Ĥ|φ_w⟩ / Σ_p ⟨ψ_p|φ_w⟩,     Ov[w] = (1/P) Σ_p ⟨ψ_p|φ_w⟩
```

where `⟨ψ_p|φ_w⟩` is the same inner-vs-outer **cross** overlap as Phase 2a (not
`inner_nomsd().Energy(inner_wset())`, which is inner-trial-vs-inner-walkers). `Ov[w]` is the same
reduction as the Phase 2a `Overlap` — identical per-pair terms `(1/P)⟨ψ_p|φ_w⟩`, hence equal up to
floating-point summation order (the `1/P` cancels in `E` but is kept in `Ov`) — so hybrid and
local-energy modes agree in the delegate limit. (Phase 2a sums the `nw×P` pairs via `FairDivide` +
`all_reduce`; here we sum per inner walker — `stochastic_energy_matches_nomsd` checkpoint (3)
asserts the two agree within `Approx`.)

Implementation notes (`StochasticWfn.icc`):

- The reduction **mirrors `NOMSD::Energy_shared`** with the trial-determinant loop replaced by the
  inner-walker loop and the CI weight `conj(ci[nd])` replaced by `1/P`. For each inner walker `ψ_p`
  it forms the cross mixed density matrix against every outer walker via
  `nomsd_.DensityMatrix(wset, ψ_p, …, herm=false, …)`, accumulates `(1/P)⟨ψ_p|φ_w⟩` into `Ov`
  and `(1/P)⟨ψ_p|φ_w⟩ E_{p,w}` into `E`, then divides `E` by `Ov`. The energy of each cross DM is
  evaluated by `nomsd_.energy_from_G(eloc2, G, nd=0, addH1=root)`, a **protected** named seam
  over `HamOp.energy`. Both the DM and the energy run on the **outer `nomsd_` (True Hamiltonian)** —
  the bras `ψ_p` are the inner walkers, but the operator is `Ĥ` (see the dual-Hamiltonian note
  below). **NOMSD's public interface is unchanged:** `StochasticWfn` is declared a `friend` of
  `NOMSD`, so it reaches the protected `energy_from_G` and `dm_size(false)` (the compact `G` size)
  directly — no public helpers were added.
- `herm = false`: the inner-walker Slater matrix is `[NMO, NEL]` (the same convention Phase 2a
  `Overlap` uses), not the pre-transposed trial NOMSD passes with `herm = true`. The two produce
  the same `G`.
- `nd = 0`: `HamOp.energy`'s `nd` selects the **half-rotated integrals of trial determinant `nd`**,
  which are tied to that determinant's bra orbitals. In the static-ensemble limit every inner walker
  equals the anchor (`getInitialGuess == OrbMats[0]`), so `nd = 0` is the correct half-rotation. Once
  `inner_nsteps > 0` moves inner walkers off the anchor, **Phase 3b** drops the half-rotation entirely
  for those walkers and scores them with the un-rotated full-G contraction (`energy_from_fullG`).
- The reduction runs through the **outer `nomsd_`** (the True Hamiltonian), which shares `TG_` so the
  `DensityMatrix` / `energy_from_G` barriers stay consistent. A `LocalTGBufferManager
  shm_buffer_manager` member backs the shared `G` buffer; per-core energies are reduced with
  `all_reduce` over `TG_local`, then `E /= Ov`.
- **✓ Dual-Hamiltonian sourcing.** The contraction is scored against the **True** Hamiltonian via the
  outer `nomsd_`, *not* `inner_nomsd()` — see
  [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational).
  This is a correctness requirement now that the inner `HamOps` can be the Variational `Ĥ_var`
  (Phase 3b-var): `inner_nomsd().energy_from_G` would return the wrong operator. The cross **density
  matrix** is Hamiltonian-independent, so running it on `nomsd_` too just keeps Overlap (2a) and Energy
  (2b) on the same `SDetOp`. The Phase 3a `vbias` contraction follows the same rule.
- **Distributed Cholesky (`NGroupsPerTG > 1`) is guarded with `APP_ABORT`**: a single `energy_from_G`
  sees only one group's Cholesky vectors. The ring reduction (`NOMSD::Energy_distributed`) is a
  later phase; Phase 2b targets the shared limit (and the CPU test runs single-group).

### Delegate limit and the multi-determinant caveat

Same as Phase 2a (see *Delegate limit and the multi-determinant caveat* under Stochastic overlap):
at the single-determinant delegate limit `Energy` reproduces NOMSD exactly (`E = ⟨φ_T|Ĥ|φ_w⟩/⟨φ_T|φ_w⟩`,
`Ov = ⟨φ_T|φ_w⟩`); for a multi-determinant trial it diverges by design.
`stochastic_energy_matches_nomsd` gates its NOMSD-equality assertions on `ndet == 1`.

### Why 2b is separate from 2a

- **Different propagator entry points:** hybrid mode never calls `Energy` for the step update;
  local-energy mode calls `Energy(wset, E, Ov)` and does not use the Phase 2a `Overlap` result
  for weights.
- **Different outputs:** `Overlap` returns a single complex vector; `Energy` fills energy
  components and may also write overlaps used by `local_energy_walker_update`.
- **Independent testability:** Phase 2a is verified via `stochastic_overlap_matches_nomsd` (no
  following `Energy` call); Phase 2b via `stochastic_energy_matches_nomsd` (Energy in isolation).
  `stochastic_wfn_matches_nomsd` exercises both overrides with its energy/overlap assertions gated
  on `ndet == 1`.

### Dynamic-ensemble extensions (Phase 3b/3c, done)

Phase 2b reduces the static ensemble only. The `S_p` phase weights (Eq. 26), the off-anchor full-G
scoring, walker-conditioning, the leapfrog (Eq. 25), and the generalized `w_p` weight are layered on by
**Phases 3b/3c** — see [Phase 3](#phase-3--propagator-hot-path-completion-and-the-dynamic-ensemble).

---

## Stochastic mixed DM + force bias (Phase 3a)

`StochasticWfn::MixedDensityMatrix_for_vbias` is the third (and last static) Tier 1 method to diverge
from the `nomsd_` delegate. It overrides `MixedDensityMatrix_for_vbias(wset, G)` to return the
**stochastic mixed density matrix** each outer walker's force bias contracts against, reduced from the
inner ensemble in the same static limit as Phases 2a/2b, following estimators 3/4 of
[arXiv:2505.18519](https://arxiv.org/abs/2505.18519). `vbias` is unchanged — it stays a permanent
delegate to `nomsd_.vbias`, now fed the stochastic `G`.

### What Phase 3a implements (static-ensemble limit)

At `inner_nsteps = 0` the phase factor `S_p` and importance reweighting are degenerate (`B̂_T = 𝟙`),
so Eq. 27 collapses to the overlap-weighted average over the `P = inner_nwalkers` inner walkers `ψ_p`:

```
G[w]   = Σ_p ⟨ψ_p|c†c|φ_w⟩ / Σ_p ⟨ψ_p|φ_w⟩            (estimator 3)
x̄_γ[w] = L_γ · G[w]                                    (estimator 4; the √Δτ prefactor stays in the propagator)
```

`⟨ψ_p|φ_w⟩` is the same inner-vs-outer **cross** overlap as Phases 2a/2b. The `1/P` cancels between
numerator and denominator, so `G[w]`'s overlap normalization is the same reduction as the Phase 2a/2b
`Ov`.

Implementation notes (`StochasticWfn.icc`):

- The reduction **mirrors `NOMSD::MixedDensityMatrix_shared`'s accumulate-then-normalize** with the
  trial-determinant loop replaced by the inner-walker loop and the CI weight `conj(ci[nd])` replaced by
  `1/P`. For each inner walker `ψ_p` it forms the normalized cross mixed DM and its overlap against
  every outer walker via `nomsd_.DensityMatrix(wset, ψ_p, …, herm = false, compact_G_for_vbias,
  transposed_G_for_vbias_)` (reusing the Phase 2b machinery), accumulates the overlap-weighted numerator
  `(1/P)⟨ψ_p|φ_w⟩ G_{p,w}` and the denominator `(1/P)⟨ψ_p|φ_w⟩`, then divides. `herm = false` because the
  inner-walker Slater matrix is `[NMO, NEL]` (the Phase 2a/2b convention), not the pre-transposed trial
  NOMSD passes with `herm = true`.
- **Output layout is the outer `nomsd_` `vbias` layout** (`compact_G_for_vbias` / `transposed_G_for_vbias_`,
  reached via the existing `friend` access — no new NOMSD public API), so the unchanged `vbias` delegate
  contracts it exactly as in deterministic AFQMC.
- **✓ Dual-Hamiltonian sourcing.** Like the Phase 2b energy, the cross DM and the `vbias` contraction
  run on the **outer `nomsd_` (True Ham)**, not `inner_nomsd()` — a correctness requirement now that the
  inner `HamOps` can be the Variational `Ĥ_var` (Phase 3b-var). At `inner_nsteps = 0` every inner walker
  equals the anchor, so the compact DM's bra basis matches `nomsd_`'s half-rotated Cholesky (`vbias`
  carries no per-determinant `nd`); once inner walkers move off the anchor, **Phase 3b** switches the
  `vbias` contraction to the un-rotated full-G layout (via `Real3IndexFactorization::vbias` dispatch).
- The shared output `G` is partitioned by its `Gsize` dimension across `TG_local` (all walkers stay in
  each band), exactly as `MixedDensityMatrix_shared` partitions its normalization, so each band is written
  by one core — no mutex, no double counting. The weighted numerator is accumulated in a **full-precision
  shared scratch** and `copy_n_cast` into the (possibly single-precision) caller `G` once at the end, so
  the reduction never mixes precisions in `ma::add` / `ma::elementwise` (mirrors the propagator's own SP
  handling).
- **Distributed Cholesky (`NGroupsPerTG > 1`) is guarded with `APP_ABORT`**, as in Phase 2b: the
  distributed propagators pass cross-group `G` buffers this shared-limit reduction does not handle.

### The milestone

With `vHS` delegating (Phase 4, trial-independent) and `Overlap`/`Energy` stochastic (Phases 2a/2b),
overriding `MixedDensityMatrix_for_vbias` makes **every Tier 1 hot-path quantity stochastic** — so the
static-limit propagator is feature-complete at the method level, the groundwork for the first complete
stochastic-trial AFQMC step. Each override is unit-verified against NOMSD at the delegate limit; an
actual end-to-end propagator run was an integration follow-up at 3a completion (now covered for the
**dynamic** case by `stochastic_propagator_step` in Phase 3b — see
[Phase 3b-specific tests](#phase-3b-specific-tests-implemented)).

### Delegate limit and the multi-determinant caveat

Same as Phases 2a/2b (see *Delegate limit and the multi-determinant caveat* under Stochastic overlap):
at the single-determinant delegate limit the stochastic mixed DM and force bias equal NOMSD's; for a
multi-determinant trial they diverge by design. `stochastic_vbias_matches_nomsd` gates its
NOMSD-equality assertions on `ndet == 1`.

### Dynamic-ensemble extensions (Phase 3b/3c, done)

Phase 3a reduces the static ensemble only; the `S_p` phase weights, off-anchor full-G force bias,
walker-conditioning, and the leapfrog are added by **Phases 3b/3c** — see
[Phase 3](#phase-3--propagator-hot-path-completion-and-the-dynamic-ensemble).

---

## Input keys

Stochastic-specific keys are stripped from plain `NOMSD` input via `strip_stochastic_input_keys()`
(single source of truth in `StochasticWfn.hpp`; `WavefunctionFactory::strip_stochastic_factory_keys`
delegates to it). All stochastic keys require `stochastic: true` in `WavefunctionFactory::interpret_inputs`.

| Key | Default | Phase | Description |
|-----|---------|-------|-------------|
| `stochastic` | `false` | — | Selects `StochasticWfn` instead of plain `NOMSD` on the HDF5 NOMSD path. |
| `inner_nwalkers` | `1` | 1a | Size of the owned inner `WalkerSet`. Must be ≥ 1. |
| `inner_nsteps` | `0` | 1c/3b | Number of free-projection `B̂_T` steps applied (from the anchor) per outer step. `0` = static anchor (Phases 1c–3a). **`> 0` (Phase 3b)** drives the inner free-projection propagator; forces free-projection mode in `buildStochasticInnerStack` and the un-rotated full-G True-Ham scoring (CLOSED trials only this phase). |
| `inner_conditioning` | `false` | 3c-i ✓ | When `true`, importance-sample the inner fields **conditioned on each outer walker** `φ_w` (Eq. 23) instead of the walker-independent free projection of 3b. The inner ensemble becomes block-structured `nwalk·P` (block `w` conditioned on `φ_w` via the custom force bias `x̄(φ_w)=√Δτ·L^var·⟨φ_T\|c†c\|φ_w⟩/⟨φ_T\|φ_w⟩`), and `buildStochasticInnerStack` builds the inner propagator in **importance-sampling** mode so `assemble_X` applies that bias. Requires `inner_nsteps > 0` (CLOSED/CPU, as 3b). See [Phase 3c-i](#phase-3c-i--walker-conditioned-inner-sampling-complete-cpu-verified-overhaul). |
| `inner_leapfrog` | `false` | 3c-ii ✓ | When `true`, the propagate-then-resample **leapfrog**: each outer step resamples the inner ensemble conditioned on the **old** walker and uses it for both the old and new effective overlap, so the hybrid step ratio `new/old` is exactly Eq. 25 and `𝒩(φ)` cancels. Switches `Log_Overlap` to the **importance-reweighted** `Σ_p⟨ψ_p\|χ⟩/\|⟨ψ_p\|φ_cond⟩\|` and refreshes the stored old overlap in `begin_inner_step(wset)`. Requires `inner_conditioning: true`. Hybrid propagation (CLOSED/CPU). See [Phase 3c-ii](#phase-3c-ii--propagate-then-resample-leapfrog-complete-cpu-verified-overhaul). |
| `inner_seed` | `777` | 1c | Seed for the inner propagator device RNG. `0` selects a time-based seed (same convention as the driver `seed`). Rank-decorrelated via `split_seed`. |
| `inner_propagator` | *(optional subtree)* | 1c/3b | Propagator input block for `PropagatorFactory`. If omitted, factory defaults apply. `system` and `name` are injected when missing (`name` suffix `_inner_propagator`). At `inner_nsteps > 0` the free-projection mode flags are forced; `timestep` (default `0.01`) sets the `B̂_T` step `dt`. |
| `inner_hamiltonian` | *(optional sub-block; `filename` required, `system` inherited)* | 3b-var ✓ | Sub-block of the stochastic wfn block naming the pre-optimized **Variational** Cholesky Hamiltonian `Ĥ_var` that defines `B̂_T` (same basis/NMO as the True Ham). **Parsed and built (3b-var):** `WavefunctionFactory` builds it on demand through the `HamiltonianFactory` it holds; absent ⇒ the inner stack clones the True Ham. A distinct variational file (VAFQMC-exported `ham_var.h5`) is exercised by the Ne driver Stages C/E; there is no in-repo unit-test fixture yet (the `stochastic_inner_hamiltonian_same_as_true` anchor reuses the True-Ham file). See [Phase 3b-var](#phase-3b-var--variational-hamiltonian-factory-plumbing-complete) and [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational). |

Example wavefunction block fragment:

```json
{
  "name": "wfn0",
  "type": "nomsd",
  "stochastic": true,
  "inner_nwalkers": 3,
  "inner_nsteps": 0,
  "inner_seed": 777,
  "filename": "wfn.h5",
  "system": "default"
}
```

Optional explicit inner propagator settings:

```json
"inner_propagator": {
  "type": "afqmc",
  "timestep": 0.01
}
```

---

## Factory integration

- Selected via `stochastic: true` inside the existing `type == "nomsd"` path in
  `WavefunctionFactory::fromHDF5`.
- Reads the same NOMSD HDF5 data (`Wavefunction/NOMSD`, `PsiT_*`, CI coefficients, etc.).
- When `stochastic: true`, `fromHDF5` builds a `NomsdSdetPair` upfront and routes through
  `buildStochasticNomsdWavefunctionWithPrecision()` (separate from the plain `NOMSD` path).
- Plain `NOMSD` builds use `buildNomsdWavefunctionWithPrecision()`; the `stochastic` flag is
  not threaded through those templates.
- Registered as explicit instantiations in the `Wavefunction` `std::variant` **[overhaul]** (was `boost::variant` on develop).
- **Two-phase inner-walker init:** (1) construct `StochasticWfn` (inner propagator built here);
  (2) call `maybe_initialize_stochastic_inner_walkers()` from `WavefunctionFactory` (drivers
  and tests) once `getInitialGuess()` and the outer-walker input block are available.

**Dual-bundle and inner-stack factory helpers** (`WavefunctionFactory.h` / `.cpp`):

| Helper | Role |
|--------|------|
| `NomsdSdetPair` / `makeOuterInnerSlaterDetOperations()` | Two independent `SlaterDetOperations` instances (same layout flags). |
| `NomsdHamOpsPair` / `makeOuterInnerHamOps()` | Two independent `getHamOps()` builds (in-place brace init; `HamOps` is move-only). The outer build always uses the True Ham `h`; **at Phase 3b-var the inner build takes the Variational Ham** that `WavefunctionFactory` builds from the `inner_hamiltonian` block, else clones the True Ham. |
| `clone_orbitals()` | Deep copy of CI/orbital vectors for the inner `NOMSD`. |
| `buildStochasticInnerStack()` | Assembles inner `Wavefunction` + device RNG + `Propagator` (`.cpp` only; avoids `Propagator.hpp` in header). |
| `buildStochasticNomsdWavefunction*()` | Assembles `StochasticWfn` with outer infrastructure + pre-built inner stack. |
| `buildNomsdWavefunction*()` | Plain `NOMSD` only (no stochastic branching). |

This is intentionally **not** first-class like `PHMSD` (no separate HDF5 type, no dedicated
factory branch). That remains deferred to Phase 8.

### Accessors

| Accessor | Role |
|----------|------|
| `inner_wset()` | Owned inner `WalkerSet` (stochastic trial ensemble) |
| `inner_nwalkers()` / `inner_nsteps()` | Parsed ensemble size and step count |
| `inner_wfn()` / `inner_nomsd()` | Inner `NOMSD` inside `inner_stack_` |
| `inner_wavefunction()` | Inner `NOMSD` as `Wavefunction&` (object the propagator binds to) |
| `inner_propagator_built()` / `inner_propagator()` | Inner `Propagator` (always built at construction; dormant when `inner_nsteps = 0`) |
| `outer_nomsd()` | Outer delegate `NOMSD` (same object as `nomsd_`) |
| `inner_walkers_initialized()` | Whether phase-2 walker init has run |

`Wavefunction` variant wrappers: `is_stochastic_wavefunction()`,
`stochastic_inner_walkers_initialized()`, `stochastic_inner_wset()`, and
`initialize_stochastic_inner_walkers()`.

**Outer-facing infrastructure:** `getSlaterDetOperations()` and the Tier 4–5 layout/Hamiltonian
queries use `nomsd_` (outer) — and under the dual-Hamiltonian design they **stay** there, because
they describe the **True** Ham the outer driver sees (see [Tier 4](#tier-4--layout-and-metadata-queries)
/ [Tier 5](#tier-5--hamiltonian-and-slater-infrastructure)). The inner `HamOps`/`SDetOp` live inside
`inner_stack_->nomsd()` and hold the **Variational** Ham (from Phase 3b); they are internal to the
inner walk and are deliberately *not* surfaced through these outer-facing accessors.

---

## Target Architecture

> `StochasticWfn` owns **two** engines (see
> [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational)):
>
> - **Outer `nomsd_` (True Hamiltonian)** — scores every returned quantity and faces the outer driver.
> - **Inner AFQMC stack (Variational Hamiltonian)** — generates the stochastic-trial samples:
>   - **Walker set** — stochastic ensemble defining the trial wavefunction *(Phase 1a ✓)*
>   - **NOMSD** — single-determinant anchor + its own `HamOps`/`SDetOp`, holding `Ĥ_var` (from
>     `inner_hamiltonian`) or a True-Ham clone fallback *(Phase 1b ✓ / 3b-var ✓)*.
>   - **Propagator** — built and bound at construction; applies `B̂_T` under `Ĥ_var` when
>     `inner_nsteps > 0` — free projection (3b), walker-conditioned (3c-i), or leapfrog (3c-ii)
>     *(Phase 1c ✓ infrastructure; driven from 3b)*.
> - **Effective outer quantities** — overlap *(2a ✓)*, local energy *(2b ✓)*, mixed DM / force bias
>   *(3a ✓)*, all reduced from the inner ensemble (static 3a → dynamic 3b → conditioned/leapfrog 3c)
>   and scored against the True Ham.
>
> The outer AFQMC driver calls `StochasticWfn` methods as it does `NOMSD`/`PHMSD`; `StochasticWfn`
> orchestrates the inner (Variational) sampling and returns the True-Ham effective quantities
> (overlap, local energy, mixed density matrix, bias potentials) to the outer propagator.

---

## Methods to Override

The `Wavefunction` visitor dispatches all calls below. Any method not listed still delegates to
`nomsd_` today and may need overriding once stochastic outputs diverge from the delegate limit.

Methods are grouped by priority. **Tier 1** methods are on the propagator hot path (every
outer timestep). These are the hardest and most important to replace.

---

### Tier 1 — Propagator hot path (highest priority)

Called every outer propagation step from `AFQMCBasePropagator`, `AFQMCModelPropagator`, and
distributed propagator variants.

**All Tier 1 methods are complete through Phase 3c** (the column below records the status; the dynamic
detail is in [Phase 3](#phase-3--propagator-hot-path-completion-and-the-dynamic-ensemble)).

| Method | Status |
|--------|--------|
| `MixedDensityMatrix_for_vbias(wset, G)` | **✓ stochastic.** Reduces per-pair cross mixed DMs over the inner ensemble, `G[w] = Σ_p ⟨ψ_p\|c†c\|φ_w⟩ · w_p / Σ_p w_p` (estimator 3 of arXiv:2505.18519), in the outer `nomsd_` `vbias` layout; cross DM built via `nomsd_.DensityMatrix(…, herm=false)` on the **True Ham**. Weight `w_p` is `1/P` at the static limit (3a), the phase `S_p` once dynamic (3b), and the generalized importance weight under conditioning/leapfrog (3c). Matches NOMSD at the single-determinant delegate limit. See [Phase 3a](#stochastic-mixed-dm--force-bias-phase-3a). |
| `vbias(G, v, dt)` | **✓ permanent delegate to `nomsd_.vbias`**, fed the stochastic `G[w]`. The `L·G` contraction (estimator 4, `x̄_γ[w] = L_γ · G[w]`) is trial-independent given `G`, scored against the **True Ham**; off the anchor it reaches the un-rotated full-G layout via `Real3IndexFactorization::vbias` dispatch. `√Δτ` prefactor stays in the propagator. |
| `vHS(X, v, dt)` | **✓ permanent delegate** (no override). Trial-independent: `v = √Δτ · Σ_γ X_γ L_γ` uses only the fields `X` and the Cholesky `L` — no half-rotation, no inner-ensemble quantity. |
| `Log_Overlap(wset)` / `Log_Overlap(wset, Ov)` | **✓ stochastic.** `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩` (Eq. 24, static/3b); under the leapfrog (3c-ii) the importance-reweighted `Σ_p ⟨ψ_p\|φ_w⟩/\|⟨ψ_p\|φ_cond⟩\|`, so the hybrid step ratio `new/old` is exactly Eq. 25. Consumed by **hybrid** propagation. Matches NOMSD at the delegate limit. |
| `Energy(wset)` / `Energy(wset, E, Ov)` | **✓ stochastic.** `E[w] = Σ_p ⟨ψ_p\|Ĥ\|φ_w⟩ · w_p / Σ_p w_p` (E1, EXX, EJ) with the same weight progression as the mixed DM (Eq. 27); off-anchor inner walkers scored with the un-rotated full-G kernel (`energy_from_fullG`). `Ov` consistent with `Log_Overlap`. Consumed by **local-energy** propagation; matches NOMSD at the delegate limit. |

---

### Tier 2 — Mixed estimators and observable handlers

Called from `MixedObsHandler`, `FullObsHandler`, force estimators, and related code paths.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `MixedDensityMatrix(wset, G, ...)` | **✓ stochastic (Phase 5).** Inner-ensemble reduction `G[w] = Σ_p w_p G_{p,w}/Σ_p w_p` (estimator 3) in the caller's observable layout — the observable analogue of `MixedDensityMatrix_for_vbias`. Honors `compact` at the static limit; rejects `compact && inner_nsteps>0` (off-anchor compact DMs are bra-basis-dependent, only the full layout averages). Delegates to `nomsd_` at the delegate limit. |
| `MixedDensityMatrix(wset, G, Ov, ...)` | **✓ stochastic (Phase 5).** As above; the 3-arg workhorse, also returns the per-walker **log** effective overlap (linear `(1/P) Σ_p ⟨ψ_p\|φ_w⟩` reduced then `log`-converted, matching NOMSD's observable-DM overlap convention). |
| `DensityMatrix(wset, Ref, G, Ov, ...)` | **✓ delegate is exact (Phase 5).** The bra is the externally supplied `Ref`, not the stochastic trial — pure orbital algebra, trial- and Hamiltonian-independent (NOMSD never touches `OrbMats`). Delegating to `nomsd_` is exact; no stochastic reduction applies. (Which references a stochastic trial *exposes* was the Tier 6 question, resolved in Phase 7: the outer-NOMSD set, inner-ensemble-agnostic.) |
| `accumulate_estimators(...)` | **✓ stochastic (Phase 5).** Feeds the inner-ensemble-reduced full mixed Green's function (the Phase 5 `MixedDensityMatrix`, full layout) to the observables via NOMSD's identical accumulate loop. Delegates at the delegate limit. Time-evolved / back-propagated (X/Yc/M) observables are supported too: the `M + T(X)·G_full·Yc` transform is linear in the DM, so the inner-ensemble average commutes with it (X/Yc/M are trial-independent operator state). |

---

### Tier 3 — Mean-field subtraction

Called during propagator setup and mean-field initialization.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `G_MF()` | **✓ stochastic (Phase 6).** Builds the trial's OWN mean-field Green's function `⟨Ψ_T\|c†c\|Ψ_T⟩/⟨Ψ_T\|Ψ_T⟩` by reducing the inner ensemble **against itself** — a double sum over inner-walker pairs `[Σ_{p,q}⟨ψ_p\|c†c\|ψ_q⟩]/[Σ_{p,q}⟨ψ_p\|ψ_q⟩]` (uniform weight `1/P`), the inner-ensemble analogue of NOMSD's multi-determinant `G_MF`. No outer walker. Collapses to the anchor density (= NOMSD's `ndet==1` result) at the static replicated limit. **Delegates to `nomsd_` (the anchor mean field)** unless the inner ensemble is in its walker-independent `P`-sample form (`inner.size() == inner_nwalkers_ > 1`) — i.e. delegates at `inner_nwalkers == 1` and whenever a conditioned/leapfrog resample has expanded it to `nwalk·P`. Used by the **discrete/model** propagator setup only. |
| `vMF(v, dt)` | **✓ stochastic (Phase 6).** Mean-field bias `v_n = L_n·G_MF` (estimator 4 on the trial's own mean-field DM), scored against the **True** Ham. Builds the stochastic `G_MF` (full layout) and contracts it via `nomsd_.vbias_from_G` — the same full-G contraction the off-anchor force bias uses, equal to NOMSD's compact-G `vMF` at the static limit (full-G == compact at the anchor). **Same `P`-sample gate as `G_MF`** (delegates at `inner_nwalkers == 1` and for a conditioned/expanded ensemble). Computed once per timestep in `generateP1`, which runs *after* `begin_inner_step` in `Propagate`. |

---

### Tier 4 — Layout and metadata queries

**✓ Verified (Phase 7): all stay on `nomsd_` (True Ham)** — correct by construction (the propagator and
the Tier 1–3 methods all run against them).

Currently forwarded from **outer** `nomsd_` / its `HamOp`. **Dual-Hamiltonian rule:** these describe
what the **outer** driver/propagator sees — the auxiliary-field count it samples, the `vHS`/`G`
layouts it consumes — so they are all about the **True** Ham and **stay on `nomsd_`**. The inner
(Variational) Ham has its *own*, generally different, Cholesky count and layout, used only inside the
inner propagator; it must **not** drive these outer-facing queries. (The Phase 1b/1c claim that inner
== outer layout holds only while the inner Ham is a True clone — it lapses at Phase 3b.)

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `size_of_G_for_vbias()` | Returns `nomsd_` DM dimension for `vbias` layout. | Stay on `nomsd_` (True Ham) — the stochastic `MixedDensityMatrix_for_vbias` reduces `G[w]` into exactly this layout. |
| `transposed_G_for_vbias()` | From outer `HamOp` flags. | Stay on `nomsd_` (True Ham) — the layout of the stochastic `G` passed to `vbias`. |
| `transposed_G_for_E()` | From outer `HamOp` flags. | Stay on `nomsd_` (True Ham) — the layout of the stochastic `G` used in energy evaluation. |
| `transposed_vHS()` | From outer `HamOp` flags. | Stay on `nomsd_` (True Ham) — `vHS` is a permanent True-Ham delegate (Phase 4). |
| `getWalkerType()` | Returns outer `NOMSD` walker type. | Stay on `nomsd_` — unchanged (the inner ensemble uses the same walker type). |
| `local_number_of_cholesky_vectors()` | From outer `HamOp`. | **Stay on `nomsd_` (True Ham)** — this counts the auxiliary fields the *outer* propagator samples for the True-Ham `vHS`. The inner Variational CV count is separate and internal. |
| `global_number_of_cholesky_vectors()` | From outer `HamOp`. | Stay on `nomsd_` (True Ham) — global True-Ham CV count for the outer MPI distribution. |
| `global_origin_cholesky_vector()` | From outer `HamOp`. | Stay on `nomsd_` (True Ham) — outer-loop CV offset for this task group. |
| `distribution_over_cholesky_vectors()` | From outer `HamOp`. | Stay on `nomsd_` (True Ham) — outer-loop CV distribution flag. |
| `spin_dependent_vHS()` | From outer `HamOp`. | Stay on `nomsd_` (True Ham) — whether the outer `vHS` expects spin-resolved buffers. |

---

### Tier 5 — Hamiltonian and Slater infrastructure

**✓ Verified (Phase 7): all stay on `nomsd_` (True Ham)** — `generalizedFockMatrix` resolved as a
True-Ham delegate in Phase 5; the rest are outer-facing queries correct by construction.

**Dual-Hamiltonian rule:** every **outer-facing** Hamiltonian query is about the **True** Ham, so it
**stays on the outer `nomsd_`** — it must *not* be routed to `inner_nomsd()`, whose `HamOps` becomes
the **Variational** Ham (Phase 3b). The inner (Variational) Ham metadata is consumed only
*internally* by the inner `PropagatorFactory` when it builds `B̂_T`. So these accessors are largely
already correct; the earlier "route to inner" notes were written before the dual-Ham split and are
superseded.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `getHamType()` | Returns `nomsd_.HamOp.getHamType()`. | **Keep on `nomsd_` (True Ham)** — the outer-facing Hamiltonian type. (The inner Variational `HamOps` has its own type, used only by the inner propagator.) |
| `getFieldTypes(...)` | Delegates to outer `HamOp`. | **Keep on `nomsd_` (True Ham)** for the outer driver's field layout. The inner Variational field layout is consumed separately by the inner `PropagatorFactory` (already handled inside the inner stack). |
| `update_potentials(...)` | Delegates to outer `HamOp`. | Update the True-Ham potentials (`nomsd_`) for the outer loop; the inner stack updates `Ĥ_var` independently if grids/ions move. |
| `generalizedFockMatrix(...)` | **✓ True-Ham delegate (Phase 5).** Now declared on `StochasticWfn` as a delegate to `nomsd_` (True Ham), mirroring NOMSD/PHMSD. Contracts a *supplied* DM against the Ham (no trial reduction); the supplied DM comes from the stochastic `MixedDensityMatrix`/`accumulate_estimators`. Reached only via the `generalizedFockMatrix` observable (commented out in all handlers today). |
| `getOneBodyPropagatorMatrix(...)` | Delegates to outer `HamOp`. | **Keep on `nomsd_` (True Ham)** — the outer propagator's one-body matrix. (`B̂_T`'s one-body matrix comes from `Ĥ_var` inside the inner stack.) |
| `vHS_sparse(...)` | Delegates to outer `HamOp`. | Sparse `vHS` of the **True** Ham for the outer propagator (`vHS` itself is a permanent delegate — Phase 4). |
| `getSlaterDetOperations()` | Returns `nomsd_` SlaterDetOperations pointer. | **Keep returning the outer `nomsd_` SDetOp** — it is Hamiltonian-independent (orbital algebra), so the reductions can use it for the cross DM regardless of which Ham scores the contraction. |

---

### Tier 6 — Back propagation

Called from `BackPropagatedEstimator` and related reference-tracking code.

**Resolution (Phase 7): outer-NOMSD delegate, inner-ensemble-agnostic** — the stochastic trial exposes
exactly the outer `nomsd_`'s reference set (the True-Ham trial's weighted determinant expansion) and
ignores the inner ensemble entirely; all three methods stay `nomsd_` delegates. For the intended
single-determinant anchor (Eq. 21) that set is `{φ_T} = OrbMats(0)` with weight 1; for a
multi-determinant outer trial it is the full CI expansion (`nrefs = ndet`, weights `ci[i]`) — identical
to plain NOMSD in both cases. The `BackPropagatedEstimator`/`FullObsHandler` fill references for one
walker and **broadcast them to all walkers** (so references must be walker-INDEPENDENT) and capture them
once per BP block over a fixed window (so they must be time-stable) — the outer trial's references
satisfy both (frozen while the inner ensemble resamples). Exact vs plain NOMSD by construction; the
scientific caveat for a genuine `inner_nsteps>0` trial is that back-propagation is scored against the
**outer** trial, not the field-sampled spread `{ψ_p}`, consistent with how `vMF`/`G_MF` collapse to the
anchor (Phase 6). The faithful inner-ensemble reference set is deferred — see the
[Phase 7](#phase-7--layout-queries-and-back-prop-references) follow-up spec.

| Method (overhaul name) | Status |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `total_number_of_references()` | **✓ outer-NOMSD delegate (Phase 7).** Returns `nomsd_.total_number_of_references()` (independent of `inner_nwalkers`; equals `ndet` for multi-det outer trials). |
| `getReferenceWeight(i)` | **✓ outer-NOMSD delegate (Phase 7).** Returns `nomsd_.getReferenceWeight(i)` (= `ci[i]`). |
| `getReferences(n, Refs)` | **✓ outer-NOMSD delegate (Phase 7).** Fills the outer trial's reference Slater matrices from `nomsd_` (`OrbMats`, H-conjugated) — walker-independent and time-stable, as the BP estimator requires. |

---

### Tier 7 — Construction and factory (infrastructure)

Not wavefunction visitor methods, but required for a full-fledged type.

| Component | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `StochasticWfn` constructor | Builds outer `nomsd_`; accepts pre-built `StochasticInnerStack` (inner NOMSD + propagator + RNG). Parses `inner_nwalkers`, `inner_nsteps`, `inner_seed`, `inner_propagator`. Defers inner `WalkerSet` resize. | **Builds two `HamOps`** (3b-var ✓): the True Ham for `nomsd_` (from `h`) and the **Variational** Ham for the inner stack — `WavefunctionFactory` builds the second Cholesky Hamiltonian from `inner_hamiltonian` through the `HamiltonianFactory` it holds (absent ⇒ clones the True Ham). Remaining: population control, first-class HDF5 type (Phase 8). |
| `interpret_inputs(pt)` | Validates NOMSD keys plus all stochastic keys listed above. Rejects `inner_nsteps > 0`. | Relax the `inner_nsteps` guard when Phase 3b invokes `inner_propagator()`. |
| `WavefunctionFactory` | `stochastic: true` on NOMSD HDF5 path; `buildStochasticInnerStack()` + `buildStochasticNomsdWavefunction*`; `maybe_initialize_stochastic_inner_walkers()` after build. | First-class `stochasticwfn` type with its own `fromHDF5` branch (Phase 8). |
| `getWavefunctionType()` | Not aware of `StochasticWfn`. | Detect stochastic trial wavefunction files on disk. |

---

## Implementation phases (status and plan)

The per-phase reference for the whole feature: what each phase delivers, its key members/factory
wiring/reductions, its tests, and what it defers. **Phases 1a–3c (3c-i + 3c-ii), Phase 4, Phase 5, Phase 6, and Phase 7 are
complete and CPU-verified on overhaul** (`Ne_cc-pvdz`); Phase 8 is the remaining plan. (Phases 1a–1c decompose the
original "own the full inner stack" step — walker set, `NOMSD`, `HamOps`, and propagator — which
should *not* be implemented as a single monolith.)

### Phase 1a — Inner walker ensemble skeleton (**complete**)

**Goal:** Introduce the two-walker-set model (outer driver walkers vs. inner trial ensemble)
with no change to outer-facing behavior.

| Item | Status |
|------|--------|
| Members | `StochasticInnerEnsemble` with `unique_ptr<WalkerSet>`, RNG, and init flag. |
| Inputs | `inner_nwalkers` (default `1`) in `StochasticWfn` and `WavefunctionFactory` input. |
| Initialization | Two-phase init via `initialize_inner_walkers()` / `maybe_initialize_stochastic_inner_walkers()` in factory and all AFQMC drivers. |
| Public API | Tier 1–6 methods still delegate to `nomsd_` on outer walkers. Accessors `inner_wset()`, `inner_wfn()` / `inner_nomsd()`, `outer_nomsd()`. |
| Task group | Same `TGwfn` as inner `NOMSD`. |
| Tests | `stochastic_wfn_matches_nomsd`, `stochastic_inner_walkers_init`, `stochastic_inner_walkers_uninitialized_smoke` in `test_wfn_factory.cpp`. |

### Phase 1b — Independent inner `HamOps` and `SDetOp` (**complete**)

**Goal:** Duplicate the factory's `getHamOps()` + `SlaterDetOperations` build for a second inner
`NOMSD` so the inner stack is truly independent of the outer delegate. `HamOps` still lives
inside each `NOMSD`; Phase 1b adds a second `NOMSD` with its own moved-in infrastructure rather
than splitting `HamOps` out of `NOMSD` globally.

> **Placeholder:** the inner `HamOps` is currently a **clone of the True Hamiltonian**. Under the
> [dual-Hamiltonian end goal](#dual-hamiltonian-architecture-true-vs-variational) it
> becomes the **Variational** Hamiltonian at Phase 3b. The independence Phase 1b establishes is what
> makes that swap a localized change; the inner-vs-outer layout-parity assertions below hold only
> while the inner Ham is a True clone.

| Item | Status |
|------|--------|
| Members | `nomsd_` (outer delegate) + inner `NOMSD` (now inside `inner_stack_`). |
| Constructor | Outer SDetOp/HamOps moved into `nomsd_`; inner bundles assembled in `buildStochasticInnerStack()`. |
| Accessors | `inner_nomsd()`, `outer_nomsd()`; `inner_wfn()` aliases inner. |
| Factory | `makeOuterInnerSlaterDetOperations`, `makeOuterInnerHamOps`, `buildStochasticNomsdWavefunction*`; plain path unchanged. |
| Outer behavior | Tier 1–6 still delegate to `nomsd_`; `getSlaterDetOperations()` returns outer SDetOp. |
| Tests | `stochastic_inner_outer_infrastructure_independent` in `test_wfn_factory.cpp`. |

**Verified:** distinct outer/inner `NOMSD` and `SlaterDetOperations` pointers; matching layout
metadata (Cholesky counts, transpose flags, `size_of_G_for_vbias`, walker/Ham types); inner
overlap/energy on `inner_wset()` matches plain `NOMSD` at delegate limit.

### Phase 1c — Inner propagator (static default) (**complete**)

**Goal:** Own an inner `Propagator` wired through `PropagatorFactory` at wavefunction build
time. Default `inner_nsteps = 0` (no propagation; static ensemble). The propagator exists and
is testable but is not called from `StochasticWfn` methods until Phase 3b.

| Item | Status |
|------|--------|
| Architecture | `StochasticInnerStack` / `StochasticInnerStackImpl`; heap `Wavefunction` + `Propagator` + device RNG. |
| Inputs | `inner_nsteps` (default `0`, `> 0` rejected), `inner_seed` (default `777`), optional `inner_propagator` subtree. |
| Key stripping | `strip_stochastic_input_keys()` centralizes stochastic key removal. |
| Factory | `buildStochasticInnerStack()`, `InnerPropagatorBuilder`; propagator built with `TGprop` + inner `wavefunction()` + `rng_`. |
| Accessors | `inner_wavefunction()`, `inner_propagator_built()`, `inner_propagator()`, `inner_nsteps()`. |
| Outer behavior | Unchanged; delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) preserved. |
| Tests | `stochastic_inner_propagator_construction` in `test_wfn_factory.cpp`. |

**Verified:** factory constructs without error; inner propagator bound to inner `Wavefunction` with
matching Cholesky layout; default hybrid propagation. Exercised by the ported overhaul suite; the
dedicated `stochastic_inner_propagator_construction` test is develop-only (not yet ported).

**Still deferred (post-1c):** invoking `inner_propagator()` from overrides and enabling
`inner_nsteps > 0` (Phase 3b). *(Note: under the dual-Hamiltonian design, outer-facing layout/SDetOp
queries deliberately stay on `nomsd_` (True Ham) — they are not rerouted to the inner stack; the
old "Phase 7 reroute" idea is superseded. See [Tier 4](#tier-4--layout-and-metadata-queries).)*

### Phase 2a — Stochastic `Log_Overlap` (**complete**; CPU-verified [overhaul])

**Goal:** Override `Log_Overlap` to reduce the inner ensemble into an effective stochastic-trial
overlap per outer walker. First Tier 1 method to diverge from the `nomsd_` delegate.

| Item | Status |
|------|--------|
| Method | `Log_Overlap(wset)` / `Log_Overlap(wset, Ov)` in `StochasticWfn.icc` |
| Reduction | `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩` (Eq. 24, static limit); see [Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a) |
| Infrastructure | `DeviceBufferManager buffer_manager`; outer `SDetOp` + `TG_`; `FairDivideBoundary` over `nw×P` |
| Propagator | **Hybrid** mode consumes this override directly |
| Tests | `stochastic_overlap_matches_nomsd` |

**Verified (CPU, develop + [overhaul]):** `stochastic_overlap_matches_nomsd` — `inner_nwalkers`
invariance (replicated ensemble) + delegate-limit overlap == NOMSD at `ndet == 1`.

### Phase 2b — Stochastic `Energy` (**complete**; CPU-verified [overhaul])

**Goal:** Override `Energy` to reduce inner-ensemble cross local energies to per-outer-walker `E`
and `Ov`, consistent with the Phase 2a overlap reduction. Unblocks correct **local-energy**
propagation; hybrid mode already uses Phase 2a for overlaps.

| Item | Status |
|------|--------|
| Method | `Energy(wset)` / `Energy(wset, E, Ov)` in `StochasticWfn.icc` (override) |
| Reduction | `E[w] = Σ_p ⟨ψ_p\|Ĥ\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩`, `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩`; mirrors `NOMSD::Energy_shared` (inner-walker loop, weight `1/P`); see [Stochastic local energy (Phase 2b)](#stochastic-local-energy-phase-2b) |
| Infrastructure | `nomsd_.DensityMatrix` (cross DM, `herm=false`) + `nomsd_.energy_from_G` on the **outer (True Ham)** delegate; `StochasticWfn` is a `friend` of `NOMSD` and reaches the **protected** `energy_from_G` seam + `dm_size(false)` (no public NOMSD API added); `LocalTGBufferManager shm_buffer_manager` member; `nd = 0` static-limit anchor half-rotation |
| Propagator | **Local-energy** mode: `wfn.Energy(wset, new_energies, new_overlaps)` in `AFQMCBasePropagator::step` |
| Tests | `stochastic_energy_matches_nomsd` (new, mirrors the 2a overlap test): invariance, delegate-limit parity (`ndet == 1`), Overlap/Energy `Ov` consistency, and the direct 3-arg propagator entry point; `stochastic_wfn_matches_nomsd` energy/overlap assertions gated on `ndet == 1` |
| Guard | `inner_nsteps > 0` parse abort stays until Phase 3b; `NGroupsPerTG > 1` aborts in `Energy` (distributed-Cholesky reduction deferred) |

**Verified (CPU, develop + [overhaul]):** `stochastic_energy_matches_nomsd` — `inner_nwalkers`
invariance, delegate-limit E1/EXX/EJ/overlap == NOMSD at `ndet == 1`, and Overlap/Energy `Ov`
consistency across the two code paths.

**Still deferred (post-2b):** the static force bias is now done (Phase 3a ✓); remaining — invoking
`inner_propagator()`, enabling `inner_nsteps > 0`, the per-walker `nd > 0` half-rotation, and the
phase factor `S_p` (Phase 3b); walker-conditioned importance sampling + the leapfrog (Phase 3c);
distributed-Cholesky `Energy` / force bias (later).

### Phase 3 — Propagator hot path completion and the dynamic ensemble

Phase 3 finishes the force bias (the last hot-path quantity) and then turns the **static** inner
ensemble into the **dynamic, field-sampled** ensemble the paper actually defines. The original
one-line plan ("`MixedDensityMatrix_for_vbias` and `vbias`") quietly carried all of this — force
bias **plus** inner propagation, phase factors, importance sampling, and the leapfrog — in a single
step. That is too much at once, so Phase 3 is split into three subphases of escalating coupling.
**3a is a static reduction that mirrors 2a/2b — well-scoped, and the large milestone; 3b and 3c are
the genuinely new dynamic machinery and carry the real risk.**

Hot-path call order per outer step (`AFQMCBasePropagator::step`): `MixedDensityMatrix_for_vbias` →
`vbias` → `vHS` → apply propagators → `Overlap` (hybrid) / `Energy` (local energy) → walker update.
After Phase 2, `Overlap`/`Energy` are stochastic and `vHS` delegates permanently (trial-independent,
Phase 4); **with Phase 3a complete (`MixedDensityMatrix_for_vbias` stochastic, `vbias` fed the
stochastic `G`), every hot-path quantity now has a stochastic implementation** — so the static-limit
propagator is feature-complete at the method level (each override unit-verified against NOMSD at the
delegate limit; an end-to-end `step()` run was an integration follow-up until Phase 3b's
`stochastic_propagator_step`). Phases 3b/3c then turned the static ensemble dynamic (free projection →
walker-conditioning → leapfrog), all complete.

#### Phase 3a — Static stochastic mixed DM + force bias (**complete**, CPU-verified [overhaul])

**Goal:** override `MixedDensityMatrix_for_vbias` so the force-bias step uses the stochastic trial
(`vbias` stays a permanent delegate, now fed the stochastic `G`). Static-ensemble limit
(`inner_nsteps = 0`), mirroring 2a/2b. See [Stochastic mixed DM + force bias
(Phase 3a)](#stochastic-mixed-dm--force-bias-phase-3a).

| Item | Plan |
|------|------|
| Methods | `MixedDensityMatrix_for_vbias(wset, G)`, `vbias(G, v, dt, a)` |
| Reduction | Estimators 3/4 of [the four estimators](#the-four-estimators), static limit: `G[w] = Σ_p ⟨ψ_p\|c†c\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩`, then `x̄_γ[w] = L_γ · G[w]`. The `1/P` cancels (as in 2b), so `G[w]`'s overlap normalization matches the Phase 2a/2b `Ov`. |
| Infrastructure | Per-inner-walker cross mixed DM via `DensityMatrix(wset, ψ_p, …, herm=false)` (reuses the 2b machinery); reduce `G[w]` in the `vbias` layout (`compact_G_for_vbias`, `transposed_G_for_vbias`); the contraction delegates to **`nomsd_.HamOp.vbias` (True Ham)** on the reduced `G[w]` with the `nd = 0` static-anchor half-rotation. **Dual-Ham:** like the Phase 2b energy, the `vbias` contraction is scored against the **True** Hamiltonian (outer `nomsd_`), not `inner_nomsd()` — fold in the same sourcing revisit. |
| Milestone | With `vHS` delegating, **every Tier 1 hot-path quantity is now stochastic** — the static-limit propagator is feature-complete at the method level. An end-to-end `step()` run was an integration follow-up at 3a (covered for the dynamic case in Phase 3b by `stochastic_propagator_step`). |
| Tests | ✓ `stochastic_vbias_matches_nomsd` (mirrors `stochastic_energy_matches_nomsd`): `inner_nwalkers` invariance + delegate-limit `G` / `vbias` (`x̄`) equality vs NOMSD at `ndet == 1`. `stochastic_wfn_matches_nomsd` gates its `MixedDensityMatrix_for_vbias` / `vbias` assertions on `ndet == 1`. |
| Guards | `inner_nsteps > 0` stays rejected (still static); `NGroupsPerTG > 1` aborts in `MixedDensityMatrix_for_vbias`, deferred as in 2b. |

#### Phase 3b — Dynamic ensemble + phase weights (**complete**, CPU-verified [overhaul]; `Ĥ_var` added in 3b-var)

**Goal:** make the inner ensemble genuinely stochastic by turning on `B̂_T`: the inner propagator
field-samples `{ψ_p = B̂_T(Y^[p])\|φ_T⟩}` (free projection), every reduction upgrades from `(1/P) Σ_p`
to the phase-weighted average of Eq. 27, and the True-Ham scoring is made correct for inner walkers
that have left the anchor. Reductions stay **scored against the True Hamiltonian** (outer `nomsd_`).

**Scope landed this phase (CPU-verified on develop and overhaul):** dynamic free-projection sampling, the
`S_p` phase weights, the `reduce_inner_cross_dm` refactor, the un-rotated full-G True-Ham scoring, and
end-to-end outer propagator integration (`stochastic_propagator_step`). **Deferred at 3b:** the
**Variational Hamiltonian `Ĥ_var`** — the inner stack stayed a True-Ham clone, so `B̂_T` was generated
by the cloned True Cholesky; the machinery is fully exercised, only the generator differed from the
paper's intent. The **factory plumbing for a *second* `Hamiltonian`** is now done in
[Phase 3b-var](#phase-3b-var--variational-hamiltonian-factory-plumbing-complete)
(input key **`inner_hamiltonian`**, built on demand by `WavefunctionFactory` through the
`HamiltonianFactory` it now holds). A distinct **variational HDF5 fixture** is now generated from
VAFQMC (see [Ne cc-pVDZ driver experiments](#ne-cc-pvdz-driver-experiments-jun-2026)); the
same-file unit-test anchor remains in `stochastic_inner_hamiltonian_same_as_true`.

| Item | Status |
|------|--------|
| Sampling | ✓ Inner propagator forced to **free-projection** mode (`buildStochasticInnerStack`) when `inner_nsteps > 0` and `inner_conditioning = false`; conditioned trials use importance sampling (3c-i). Each outer step the ensemble is **reset to the anchor `\|φ_T⟩` then advanced `inner_nsteps` `B̂_T` steps** (`maybe_advance_inner_ensemble` or `advance_inner_ensemble_conditioned`), driven by a per-step latch `begin_inner_step(wset)` at the top of `AFQMCBasePropagator::Propagate` (leapfrog mode also refreshes the stored `OVLP` there). Reset-then-resample (not a continuous free walk): the single-step trial is `B̂_T` applied to the anchor; recovers 3a exactly at `inner_nsteps = 0`. The `B̂_T` `dt` is `inner_propagator.timestep` (default 0.01); the inner propagator rebuilds its one-body/CV scaling for this `dt` on first `Propagate` (so it is the single effective source — no build-vs-runtime drift). |
| Fail-fast / limits | ✓ `inner_nsteps > 0` is rejected **at `StochasticWfn` construction** unless the trial is **CLOSED (RHF)** and the build is **CPU** (`!ENABLE_DEVICE`) — the un-rotated full-G kernels are CLOSED/CPU-only this phase, so misuse fails immediately, not deep in the first reduction's HamOp dispatch. |
| Phase weight | ✓ `⟨Ô⟩_w = Σ_p ⟨Ô⟩_{p,w} S_p / Σ_p S_p` with `S_p[w] = ⟨ψ_p\|φ_w⟩/\|⟨ψ_p\|φ_w⟩\|` (Eq. 26) in `Energy`/`MixedDensityMatrix_for_vbias`; degenerate `S_p` recovers 3a. **`Log_Overlap` stays the absolute `(1/P) Σ_p ⟨ψ_p\|φ_w⟩`** (Eq. 24's normalization) — dropping the magnitude would break `Log_Overlap == NOMSD` at the delegate limit, and the exact `𝒩(φ)` cancellation is the Phase 3c leapfrog. (The earlier Tier-1 "Overlap: replace 1/P with `S_p`" note was imprecise.) |
| Refactor | ✓ The shared per-inner-walker cross-DM loop is now a private `reduce_inner_cross_dm(wset, compact, transposed, Gsize, D, Ov, accumulate)` (the callback folds `G_{p,w}` into each estimator's `S_p`-weighted numerator); it also owns the `maybe_advance_inner_ensemble()` resample. `Log_Overlap` keeps its own `nw×P` FairDivide loop. |
| Half-rotation | ✓ Resolved via the **un-rotated full-G contraction** (not per-walker re-rotation): once `ψ_p ≠` anchor the reductions form the **full** NMO×NMO cross `G` and contract it with the full (un-rotated) Cholesky/bare `hij`. The EXX/EJ/E1 kernel is a single shared `full_g::energy_closed` (`HamiltonianOperations/full_g_estimators.hpp`, reusing the proven `energy_impl` structure with an identity-rotated Cholesky), written HamOp-agnostically (G requested as `[nwalk][NMO*NMO]` transposed). On overhaul only the dense `Real3IndexFactorization` route is live (`THCOps`, `ModelHamOps`, `KPTHCOps`, `KP3IndexFactorization` carry `energy_fullG` stubs pending coverage); the sparse `SparseTensor` route existed only on develop (`Likn` densified via `Matrix2MA`, exercised there by `ham_chol_sc.h5`). `ma_rotate::getLank` could not be reused (it reinterprets a real `Likn` as complex). **The full-G force bias has no separate `vbias_fullG` seam** — `Real3IndexFactorization::vbias` dispatches on the G layout (compact vs full NMO×NMO), so the un-rotated contraction is reached through the ordinary `vbias`/`vbias_from_G` path; only `energy_fullG` remains a dedicated seam (`energy_from_fullG` on NOMSD). **CLOSED (RHF) trials only this phase** (rejected at `StochasticWfn` construction otherwise; also CPU-only); COLLINEAR/NONCOLLINEAR full-G are a follow-up. |
| Tests | ✓ **`stochastic_full_g_matches_compact`** — full-G at the anchor == compact `nd = 0` == NOMSD (new-kernel validation). ✓ **`stochastic_dynamic_ensemble_smoke`** — resample + **all four** stochastic overrides (`MixedDensityMatrix_for_vbias`/`vbias`, `Energy`/`energy_fullG`, `Log_Overlap`) on the moved ensemble; asserts finite. ✓ **`stochastic_propagator_step`** — real OUTER `AFQMCBasePropagator::Propagate()` on a dynamic trial (`inner_nsteps = 1`, `inner_nwalkers = 4`); full hot path through the propagator (`begin_inner_step` → resample → `MixedDensityMatrix_for_vbias`/`vbias` (full-G via layout dispatch) → `vHS` → apply → `Log_Overlap`), including G-buffer sizing from `size_of_G_for_vbias()`; asserts finite weights/energies/overlaps over 3 steps (smoke, not NOMSD parity). All three gated on **CLOSED (RHF)**. **[develop]** CPU-verified with `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5` (also passes with `wfn_msd.h5`). **[overhaul] Ported + CPU-verified** on `Ne_cc-pvdz` (`ham_chol_dense.h5` + `wfn_rhf.h5`), after fixing the full-G kernel bugs (see [Port bugs](#port-bugs-surfaced-by-the-test-port-overhaul)). Remaining (research-level): `P → ∞` convergence and `inner_seed` stability. |

#### Phase 3b-var — Variational Hamiltonian factory plumbing (**complete**)

**Goal:** let the inner (Variational) stack be built from a **separate** Cholesky Hamiltonian `Ĥ_var`
read from its own HDF5 file, instead of cloning the True Ham. This is the factory-plumbing follow-up
that Phase 3b explicitly deferred (the inner stack was a True-Ham clone, so `B̂_T` was generated by the
cloned True Cholesky). The dual-Hamiltonian end goal becomes real here: the outer `nomsd_` scores
against the **True** Ham (unchanged), and `B̂_T` is generated by the **Variational** Cholesky.

**Design decision (where the second Hamiltonian gets built):** `WavefunctionFactory` now **holds a
`HamiltonianFactory&`** (added to its constructor) and builds `Ĥ_var` on demand inside `fromHDF5` when
a stochastic trial names one via `inner_hamiltonian`. It remains a Hamiltonian *consumer* — it does
not own Hamiltonians or duplicate any build logic; it asks the `HamiltonianFactory` (a deliberately
small class) to build the second one, exactly as the driver does for the True Ham. **`DriverFactory`
is untouched** — the alternative of building `Ĥ_var` in the driver and threading a second `Hamiltonian*`
through `getWavefunction` was prototyped and rejected (it spread the change across the driver's call
sites); concentrating it in `WavefunctionFactory` keeps the feature in one layer. The
`inner_hamiltonian` key is consumed entirely at the factory level (read from the raw input in
`fromHDF5`) and is **never forwarded into the wavefunction's own ptree**, so `StochasticWfn` /
`strip_stochastic_input_keys` are unchanged.

| Item | Status |
|------|--------|
| Input key | ✓ **`inner_hamiltonian`** — an optional sub-block of the stochastic wfn block naming the Variational Ham (`filename` required; `system` inherited from the wfn block if absent). `WavefunctionFactory::interpret_inputs` rejects it unless `stochastic: true` and lists it as a known pass-through key; it is **not** copied into the cleaned ptree, so it stays out of the wavefunction. |
| Factory wiring | ✓ The `WavefunctionFactory` constructor is **overloaded**: the original `WavefunctionFactory(InfoMap)` is preserved (member `HamFac_` is a `HamiltonianFactory*` defaulting to `nullptr`), and a new `WavefunctionFactory(InfoMap, HamiltonianFactory&)` sets `HamFac_`. So every existing call site (all in tests) compiles **unchanged**; only callers that need an inner Hamiltonian use the two-arg form. `fromHDF5` resolves `Hamiltonian& inner_ham`: if the wfn block has `inner_hamiltonian`, it builds/fetches that Ham via `HamFac_` (registered under the namespaced ID `<wfn>__inner_hamiltonian__`, idempotent via the new `HamiltonianFactory::has_input`) — aborting with a clear message if `HamFac_` is null (single-arg-constructed factory); otherwise `inner_ham = h` (the True Ham, clone path). It passes `inner_ham` to `buildStochasticNomsdWavefunction`, whose inner HamOps now come from `inner_ham.getHamiltonianOperations(...)` (outer `nomsd_` HamOps still from `h`). Both half-rotated with the **same** trial orbitals; only the integrals differ. |
| Driver | ✓ **`DriverFactory` call sites unchanged.** `AFQMCFactory` constructs `WfnFac(InfoMap, HamFac)` so production input decks can name `inner_hamiltonian`; `DriverFactory::getWavefunction` is untouched. |
| Layout queries | ✓ No change. All outer-facing Tier-4/5 metadata (Cholesky counts, transpose flags, `size_of_G_for_vbias`, Ham type) stay on `nomsd_` (True Ham), exactly as the dual-Hamiltonian rule requires. `Ĥ_var`'s (generally different) Cholesky count is internal to the inner propagator. |
| Tests | ✓ **`stochastic_inner_hamiltonian_same_as_true`** (`tests/test_wfn_factory.cpp`, `[stochastic_wfn]`) — one factory builds two trials: `wfn_clone` (no `inner_hamiltonian` → inner clones the True Ham) and `wfn_hvar` (`inner_hamiltonian` = the **same** integral file → factory builds a second Ham and uses it for the inner stack). On the dynamic path (`inner_nsteps = 1`, so the inner Ham drives `B̂_T`) the two must agree to `1e-9` on `Energy`/`Log_Overlap`/`vbias` over 3 resampled steps. A `HamFac.has_input("wfn_hvar__inner_hamiltonian__")` check proves the factory actually traversed the `inner_hamiltonian` path (built + registered the Ham) rather than silently ignoring the key. Gated on CLOSED (RHF) + CPU. **CPU-verified** on a compute node (`Ne_cc-pvdz`, `mpirun -np 1`): passes in isolation (2675 assertions) and as part of the full `[stochastic_wfn]` tag (11 cases, 5831 assertions). The build also confirmed the constructor overload is non-breaking — the untouched estimator/propagator/phmsd test TUs recompile and link against the new header unchanged. |
| Driver validation | ✓ **Ne cc-pVDZ end-to-end driver runs** (Jun 2026, compute node, `mpirun -np 1`) with VAFQMC-exported `ham.h5` / `wfn.h5` / `ham_var.h5` — see [Ne cc-pVDZ driver experiments](#ne-cc-pvdz-driver-experiments-jun-2026). Stage C confirms a *different* `Ĥ_var` is loaded for the inner stack (`enuc` shift, smaller inner mean-field subtraction, `H1 is not hermitian` warning on the inner propagator only) while outer energies stay on the True Ham. |
| Follow-ups | COLLINEAR/NONCOLLINEAR, GPU, and the non-`Real3IndexFactorization` HamOps stubs remain as in Phase 3b. Stage B (dynamic path, True Ham inner clone) showed **walker population collapse** at `P = 4` on the smoke trial — retry with larger `inner_nwalkers` / outer walker count before treating as a regression. Longer VAFQMC training (`train_ne_smoke.py --iterations 5000+`) for production-quality trials. |

##### Ne cc-pVDZ driver experiments (Jun 2026)

First **production `DriverFactory` / `safire` executable** runs with a stochastic trial built from
**VAFQMC-optimized** HDF5 (not the in-repo `Ne_cc-pvdz` RHF fixture). Validates the full
dual-Hamiltonian input path merged to `main` with Phase 3b-var.

**VAFQMC side** (branch `safire/ne-cc-pvdz-smoke` on `beevus77/vafqmc`; tools live under `tools/`):

```bash
# From the vafqmc repo root (Python 3.11 venv; see that branch's uv.lock / README)
python tools/build_ne_hamiltonian.py
# → hamiltonian.pkl  (PySCF Ne cc-pVDZ RHF, chol_cut=1e-6)

python tools/train_ne_smoke.py
# → checkpoints/checkpoint.pkl  (500 iterations by default; spin_mixing=false, init_tsteps=[0.01])

python tools/export_safire.py \
    --hamiltonian hamiltonian.pkl \
    --checkpoint checkpoints/checkpoint.pkl \
    --out-dir safire_export \
    --variational
# → safire_export/ham.h5      (True Ham: physical hcore/ceri/enuc)
# → safire_export/wfn.h5      (NOMSD anchor: optimized orbitals, ci_coeffs=[1])
# → safire_export/ham_var.h5  (Variational Ham: optimized hmf/vhs + projection-shifted enuc)
```

| Script | Role |
|--------|------|
| `tools/build_ne_hamiltonian.py` | PySCF RHF Ne cc-pVDZ → `hamiltonian.pkl` |
| `tools/train_ne_smoke.py` | Short VAFQMC optimization (`cfg.seed = 1`, `spin_mixing: false`, `mf_subtract: false`) |
| `tools/export_safire.py` | Writes SAFIRE `/Hamiltonian/DenseFactorized` + `/Wavefunction/NOMSD` HDF5 (`--variational` adds `ham_var.h5`) |

**SAFIRE side** — create a run directory, copy the three HDF5 files, and use three staged input decks
(`project.series` 0/1/2 → `qmc.s000` / `qmc.s001` / `qmc.s002` scalar output):

```bash
RUN=~/development/run_ne_stochastic
mkdir -p "$RUN" && cd "$RUN"
cp /path/to/vafqmc/safire_export/{ham.h5,wfn.h5,ham_var.h5} .

module load openmpi   # Flatiron: match the OpenMPI used at SAFIRE configure time
SAFIRE=~/development/SAFIRE/build/bin/safire

# Stage A — static anchor (inner_nsteps = 0)
mpirun -np 1 "$SAFIRE" afqmc_static.json 2>&1 | tee afqmc_static.out

# Stage B — dynamic inner loop, True Ham clone (inner_nsteps = 1, no inner_hamiltonian)
mpirun -np 1 "$SAFIRE" afqmc_dynamic_clone.json 2>&1 | tee afqmc_dynamic_clone.out

# Stage C — dynamic inner loop + Variational Ham (inner_hamiltonian → ham_var.h5)
mpirun -np 1 "$SAFIRE" afqmc_var.json 2>&1 | tee afqmc_var.out

# Stage D — Phase 3c-i: walker-conditioned sampling on the Variational Ham (inner_conditioning)
mpirun -np 1 "$SAFIRE" afqmc_cond.json 2>&1 | tee afqmc_cond.out

# Stage E — Phase 3c-ii: + propagate-then-resample leapfrog (inner_leapfrog)
mpirun -np 1 "$SAFIRE" afqmc_leapfrog.json 2>&1 | tee afqmc_leapfrog.out
```

**Input deck skeleton** (all stages: `walker_type: CLOSED`, outer `hamiltonian: ham.h5`,
`wavefunction: wfn.h5`, `stochastic: true`, `timestep: 0.01`, `steps: 500`,
`n_walkers_per_mpi_task: 4`, `seed: 12345`). Stage-specific wavefunction keys:

| Stage | `project.series` | `inner_nsteps` | `inner_hamiltonian` | extra wfn keys | Purpose |
|-------|------------------|----------------|---------------------|----------------|---------|
| **A** | 0 | `0` | *(absent)* | — | Static stochastic anchor; no inner `B̂_T` |
| **B** | 1 | `1` | *(absent)* | — | Dynamic `B̂_T` with inner stack cloning True Ham |
| **C** | 2 | `1` | `{ "filename": "ham_var.h5" }` | — | Dynamic `B̂_T` (3b) from VAFQMC variational Cholesky |
| **D** | 3 | `1` | `{ "filename": "ham_var.h5" }` | `inner_conditioning: true` | Phase 3c-i: walker-conditioned sampling (no leapfrog) |
| **E** | 4 | `1` | `{ "filename": "ham_var.h5" }` | `inner_conditioning: true`, `inner_leapfrog: true` | Phase 3c-ii: + propagate-then-resample leapfrog |

Common stochastic block (example Stage C):

```json
"wavefunction": {
  "filename": "wfn.h5",
  "stochastic": true,
  "inner_nwalkers": 4,
  "inner_nsteps": 1,
  "inner_seed": 777,
  "inner_hamiltonian": { "filename": "ham_var.h5" },
  "inner_propagator": { "timestep": 0.01 }
}
```

(`inner_propagator.timestep` is read by `StochasticWfn` for the inner `B̂_T` step; the propagator
factory warns `Unknown key: timestep` — harmless.)

**Results (Jun 2026, `stochastic-wfn-phase-3b-var` → merged to `main`, commit `9a85ee6`, compute node
`worker7332`):**

| Stage | Driver | Scalar file | Outcome |
|-------|--------|-------------|---------|
| **A** | Completed 500 steps | `qmc.s000.scalar.dat` | **Healthy.** `StochasticWfn`, CLOSED, local energy at init ≈ **−128.48 Ha**. Weight ≈ 4 throughout; mean `EnergyEstim__nume_real` ≈ **−128.66 Ha** over 50 measurement blocks. |
| **B** | Completed (no crash) | `qmc.s001.scalar.dat` | **Population collapse.** Weight → 0 after block 2; energies / `Eshift` → NaN for the remainder. First block energy ≈ −128.63 Ha; unstable dynamic path with True Ham inner at `P = 4` on the smoke trial. |
| **C** | Completed 500 steps | `qmc.s002.scalar.dat` | **Healthy — main integration result.** Log shows **two** Hamiltonian inits: outer `ham.h5` (`enuc = 0`) and inner `ham_var.h5` (`enuc ≈ −65.94`). Inner mean-field subtraction **0.02** vs **0.95** in Stage B (confirms distinct `Ĥ_var`). Warning `H1 is not hermitian!` on inner propagator only. Weight ≈ 2.8–4.1; mean energy ≈ **−128.74 Ha**. |

**Interpretation:** Stages A and C validate the VAFQMC → SAFIRE export and the `inner_hamiltonian`
factory path in a real driver run. Stage C is the first demonstration that a **different** variational
Hamiltonian drives inner `B̂_T` while outer scoring stays on the True Ham. Stage B's collapse is likely
a **small-ensemble / short-smoke-trial** instability (not a driver init failure); re-run with larger
`inner_nwalkers` and `n_walkers_per_mpi_task` before filing a code bug.

##### Phase 3c comparison — conditioning + leapfrog (Jun 2026, leapfrog validation)

Stages **C/D/E** re-run on the 3c-ii binary (compute node `worker7014`, current `main` + Phase 3c
commits, `mpirun -np 1`) against the **same** `ham_var.h5` trial — isolating the effect of the sampling
sophistication (3b → 3c-i → 3c-ii) at fixed `P = inner_nwalkers = 4`, `n_walkers_per_mpi_task = 4`,
500 steps. `E` = per-block `EnergyEstim__nume_real / __deno_real`, mean ± block std over 50 measurement
blocks (10-block equilibration skipped); weight = `min/mean/max` over the same blocks. Stage A is the
static (deterministic NOMSD-trial) anchor.

| Stage | Method | `E` mean (Ha) | `E` block-std | weight min/mean/max | blocks / NaN |
|-------|--------|---------------|---------------|---------------------|--------------|
| **A** | static anchor (analytic trial) | **−128.671** | 0.081 | 3.55 / 4.08 / 5.03 | 50 / 0 |
| **C** | 3b free-projection | **−128.733** | 0.161 | 1.75 / 4.29 / 8.36 | 50 / 0 |
| **D** | 3c-i conditioning (no leapfrog) | **−128.774** | 0.151 | 1.80 / 4.48 / **11.58** | 50 / 0 |
| **E** | **3c-ii leapfrog** (measurement-`Energy` fix) | **−128.785** | 0.157 | **3.50 / 4.07 / 5.04** | 50 / 0 |

(Stage E numbers are **after** the generalized-`w_p` measurement-`Energy` fix; see [Phase 3c-ii](#phase-3c-ii--propagate-then-resample-leapfrog-complete-cpu-verified-overhaul). Pre-fix Stage E mis-scored the
observable as `−128.738 / std 0.118` — biased low with artificially small variance because the
old-conditioned ensemble was weighted with the scored-walker phase `S_p` instead of `w_p`.)

**Findings (3c-ii acceptance met at smoke scale):**
- **Weight stability is the leapfrog payoff.** The leapfrog (E) holds the weight band at **3.50–5.04 —
  matching the deterministic anchor's 3.55–5.03** — because the exact `𝒩(φ)` cancellation in the Eq. 25
  ratio makes the step-to-step overlap ratio as well-behaved as a deterministic trial. This is the clear,
  robust signal at smoke scale, and it is *unaffected* by the measurement-energy fix (the propagation
  ratio is a separate code path).
- **Why the leapfrog (not just conditioning) matters.** 3c-i (conditioning **without** the leapfrog, D)
  has the **widest** weight band (1.80–11.58) — conditioning the sampling while the step ratio still
  divides overlaps from two *differently* conditioned ensembles (old from the previous step) is noisier
  than even 3b. The leapfrog's shared-ensemble ratio is what delivers the stability.
- **Energy consistency (unbiased).** The conditioned stages now agree: D (directly conditioned on the
  scored walker) **−128.774** and E (leapfrog, reweighted to the scored walker via `w_p`) **−128.785**
  — both ≈ 0.04–0.05 Ha below 3b's −128.733, and ≈ 0.1 below the static anchor (the physical `B̂_T`
  trial improvement). The measurement-`Energy` fix bringing E into agreement with the
  directly-conditioned D is the validation that the generalized estimator is unbiased.
- **Energy block-variance is comparable across the conditioned stages** (D 0.151, E 0.157) at this scale;
  the leapfrog does **not** visibly reduce the *energy* block-variance with only `P = 4` / 4 walkers on a
  500-iter trial — its demonstrated win here is weight stability. A clean energy-variance-reduction claim
  needs more walkers + a longer trial.
- All stages complete 500 steps with **no NaN and no population collapse** (unlike Stage B's True-Ham
  clone); Stage E ran cleanly with `inner_conditioning: true` + `inner_leapfrog: true` echoed in the
  `StochasticWfn input` log.

**Caveats (smoke scale).** Short VAFQMC trial (500 iters) and low statistics (`P = 4`, 4 outer walkers).
The measurement-`Energy` conditioning is now **exact** (generalized `w_p`); a production claim still needs
more walkers + a longer trial. Files: `qmc.s002/s003/s004.scalar.dat`,
`afqmc_var.json` / `afqmc_cond.json` / `afqmc_leapfrog.json` in `~/development/run_ne_stochastic`.

**Harmless log noise:** OpenMPI `oob_tcp_if_exclude` subnet message; spdlog `string pointer is null` at
startup; HDF5 `type mismatch long != int` when reading exported `wfn.h5` dims.

**Executable note:** the driver binary is `${BUILD_DIR}/bin/safire` (not `build/src/safire`). Build on a
compute node with the same toolchain/MPI as `test_afqmc`; if linking `safire` fails on `main.cpp` with
system Boost headers, add `-isystem` for the nix Boost 1.87 include path at configure time or link
`Boost::headers` on the `safire` target.

#### Phase 3c — Walker-conditioned sampling + propagate-then-resample leapfrog

**Status:** branch **`stochastic-wfn-phase-3c`** (cut from `main` after 3b-var merge). Decomposed into
**3c-i** (walker-conditioned sampling — *complete*, the custom force bias + `nw·P` block-structured
ensemble) and **3c-ii** (the propagate-then-resample leapfrog — *complete*, exact Eq. 25 ratio via the
importance-reweighted overlap + top-of-step refresh). The two pieces 3c bundles are: **(A)**
importance-sample the inner fields conditioned on each outer walker (Eq. 23), and **(B)** the leapfrog
so the step-to-step overlap-ratio `𝒩(φ)` cancellation (Eq. 25) is exact. 3c-i lands (A); 3c-ii lands (B)
— exact for both hybrid and local-energy propagation (the generalized `w_p` weight also makes the
measurement energy unbiased); only the `inner_nsteps > 1` moving-bra bias remains a follow-up.

##### Phase 3c-i — Walker-conditioned inner sampling (**complete**, CPU-verified [overhaul])

**Goal:** replace 3b's walker-independent free-projection ensemble with one that is importance-sampled
**conditioned on each outer walker** `φ_w` (Eq. 23 of arXiv:2505.18519), using the **force-biased
Gaussian** route SAFIRE chose (reuse `assemble_X → vHS → apply_propagators`; *not* the paper's
Metropolis MCMC). Opt-in via the **`inner_conditioning`** key (default `false` ⇒ exact Phase 3b
fallback). With `inner_leapfrog = false` (default), the within-step resample stays at the first
reduction latch (as in 3b); hybrid overlap ratios are **not** Eq.-25-exact until 3c-ii is enabled
(see [Phase 3c-ii](#phase-3c-ii--propagate-then-resample-leapfrog-complete-cpu-verified-overhaul)).

| Item | Status |
|------|--------|
| Conditioning bias | ✓ `x̄(φ_w) = √Δτ · L^var · ⟨φ_T\|c†c\|φ_w⟩/⟨φ_T\|φ_w⟩` (Eq. 23). The inner trial **is** the anchor `φ_T`, so this **reuses `inner_nomsd()`'s own `MixedDensityMatrix` + `vbias_from_G` on the OUTER walker set** — no bespoke contraction. The `√Δτ·[2 for CLOSED]` prefactor and overlap normalization come from the standard `HamOp.vbias`/`MixedDensityMatrix`, matching the field-shift convention `assemble_X` expects. Computed const-correctly in `StochasticWfn::conditioned_resample` (the outer `wset` is const there). |
| `nw·P` ensemble | ✓ `advance_inner_ensemble_conditioned(X_bias, nw)` grows the inner `WalkerSet` to `nwalk·P` (resize-on-mismatch — outer count fluctuates with population control), resets every walker to the anchor, then advances `inner_nsteps` conditioned steps. **Slot-major layout** `q = ip·nwalk + w` (block `w` conditioned on `φ_w`) so each per-slot batch of `nwalk` inner walkers is contiguous. `X_bias(w,·)` is broadcast to its `P` samples. |
| Propagator seam | ✓ `AFQMCBasePropagator::Propagate_conditioned(wset, Xbias, dt)` seeds the fields with the external per-walker bias, runs `assemble_X → vHS → apply_propagators`, and **skips** the inner energy/overlap/walker-weight update (the inner determinants are field samples; their weights are not consumed — the reductions weight by `1/P` and the phase `S_p`). Exposed through the `Propagator<MEM>` variant. **No `assemble_X` signature change:** the conditioned inner propagator is built `free_projection = false` (importance sampling) so `assemble_X` applies the bias; `Propagate_conditioned` asserts `not free_projection`. |
| Block-structured reductions | ✓ `reduce_inner_cross_dm` and `Log_Overlap` gain a conditioned branch: loop the `P` per-walker samples; for slot `ip` pair its `nwalk` inner walkers **diagonally** with the `nwalk` outer walkers via the **batched-reference** `det_ops::MixedDensityMatrix(A_rank3, B_rank3, …)` / `Log_Overlap` (one inner reference per outer walker). Same `det(A†B)` convention as the existing `herm=false` single-reference path. Phase `S_p(w)`, denominator `D(w)=Σ_p S_p`, and the `accumulate` callback are unchanged; `Energy`/`MixedDensityMatrix_for_vbias`/`vbias` above the reduction are untouched. |
| Factory | ✓ `buildStochasticInnerStack` builds the inner propagator in **importance-sampling** mode (`free_projection=false`, `importance_sampling=true`) when `inner_conditioning`, else the 3b free-projection mode. `inner_conditioning` threaded through `WavefunctionFactory::interpret_inputs` (+ `StochasticWfn::interpret_inputs` / `strip_stochastic_input_keys`); rejected unless `stochastic: true`; construction aborts if `inner_conditioning && inner_nsteps == 0`. |
| Tests | ✓ **`stochastic_conditioned_propagator_step`** (`tests/test_wfn_factory.cpp`, `[stochastic_wfn]`) — a real OUTER `AFQMCBasePropagator::Propagate()` over the conditioned dynamic trial (`inner_conditioning=true`, `inner_nsteps=1`, `inner_nwalkers=4`, `nwalk=11` ⇒ `44` inner walkers); asserts finite weights/energies/overlaps over 3 steps. The internal `inner.size() == nwalk·P` checks in the reductions validate the resize. CLOSED+CPU; `inner_conditioning=false` leaves the 3b cases bit-identical. **[overhaul] CPU-verified** Jun 2026: passes in the full tag (**11 cases / 5831 assertions**, `mpirun -np 1`, `Ne_cc-pvdz`). |

**Leapfrog (3c-ii):** landed in [Phase 3c-ii](#phase-3c-ii--propagate-then-resample-leapfrog-complete-cpu-verified-overhaul)
via top-of-step `begin_inner_step(wset)` refresh (not the original end-of-step hook), with the
generalized-`w_p` weight making the measurement / local-energy estimators exact. Driver-validated at
smoke scale (Ne Stage E: unbiased energy, weight stability — see
[Phase 3c comparison](#phase-3c-comparison--conditioning--leapfrog-jun-2026-leapfrog-validation)).
**Still open:** a production-scale energy-variance-reduction-vs-3b claim (more walkers + longer trial),
and the `inner_nsteps > 1` moving-bra bias (canonical trial is `inner_nsteps = 1`).

##### Phase 3c-ii — Propagate-then-resample leapfrog (**complete**, CPU-verified [overhaul])

**Goal:** make the `𝒩(φ)` cancellation in the Eq. 25 overlap ratio exact — the variance reduction /
exactness the paper relies on. Opt-in via **`inner_leapfrog`** (default `false`; requires
`inner_conditioning`, hence `inner_nsteps > 0`, CLOSED/CPU). Off ⇒ exact 3c-i behavior.

**Key correctness point (refines the original "re-store OVLP" plan):** with a shared conditioned
ensemble, the *absolute* `(1/P)Σ_p⟨ψ_p|χ⟩` reduction does **not** reproduce Eq. 25 — `Σ_p⟨ψ_p|φ'⟩ /
Σ_p⟨ψ_p|φ⟩` differs from `Σ_p(⟨ψ_p|φ'⟩/⟨ψ_p|φ⟩)S_p / Σ_p S_p` unless every `|⟨ψ_p|φ⟩|` is equal. The
leapfrog overlap must be **importance-reweighted**, `Õv(χ) = Σ_p ⟨ψ_p|χ⟩/|⟨ψ_p|φ_cond⟩|`, so the
old overlap (`χ = φ_cond`) is `Σ_p S_p` and `new/old` is exactly Eq. 25 (`𝒩(φ)` cancels via Eq. 24
`⟨Ψ̄_T|φ⟩ = (𝒩(φ)/P)Σ_p S_p`).

| Item | Status |
|------|--------|
| Conditioning / force bias | ✓ done in 3c-i (`𝒫(Y;φ_w)`, `x̄ ∝ √Δτ·L^var·⟨φ_T\|c†c\|φ_w⟩/⟨φ_T\|φ_w⟩`). |
| Reweighted overlap | ✓ `Log_Overlap` leapfrog branch accumulates `Σ_p ⟨ψ_p\|χ⟩/\|⟨ψ_p\|φ_w^cond⟩\|` (per-sample reweight by the conditioning-walker magnitude `inner_cond_mag_`, computed by `compute_inner_cond_mag` right after each conditioned resample via the batched `det_ops::Log_Overlap`, slot-major `q = ip·nwalk + w`). |
| Generalized weight (Energy / DM) | ✓ `reduce_inner_cross_dm` (used by `Energy` and `MixedDensityMatrix_for_vbias`) weights each sample by the **generalized importance weight** `w_p = ⟨ψ_p\|χ⟩/\|⟨ψ_p\|φ_cond⟩\|` in leapfrog mode (vs the scored-walker phase `S_p`). This is the exact Eq. 27 estimator when the scored walker `χ ≠ φ_cond` (measurement `Energy` after propagation; local-energy mode step-5). At the top of the step `χ = φ_cond` so `w_p = S_p` and the force-bias DM / 3c-i are unchanged. Resolves the deferred **measurement-`Energy`** and **local-energy-mode** conditioning. |
| Leapfrog (top-of-step refresh) | ✓ Instead of the doc's original end-of-step hook, `begin_inner_step(wset)` (now taking the walker set, dispatched through the `Wavefunction` variant; the single call site is the top of `AFQMCBasePropagator::Propagate`) **resamples conditioned on the current (old) walker and refreshes `OVLP = Õv(φ)` against that ensemble**. The post-propagation `Log_Overlap` reuses the SAME ensemble (latch consumed), so `ratioOverlaps = exp(new_ovlp − old_ovlp)` (`hybrid_walker_update`) is exactly Eq. 25. Mathematically identical to end-of-step (ensemble conditioned on the old walker either way) but robust to between-step orthogonalization — everything is recomputed fresh against the current walker, so the LogOverlapFactor never drifts. |
| Scope | **Hybrid and local-energy** propagation are both exact (the overlap ratio via the reweighted `Log_Overlap`; the local energy / measurement energy via the generalized `w_p`). The only remaining limitation: `inner_nsteps > 1` reuses the anchor-based bias each step (canonical trial is `inner_nsteps = 1`; a moving-bra bias is a follow-up). |
| Tests | ✓ **`stochastic_leapfrog_propagator_step`** (`tests/test_wfn_factory.cpp`, `[stochastic_wfn]`) — a real OUTER hybrid `AFQMCBasePropagator::Propagate()` over the leapfrog trial (`inner_conditioning = inner_leapfrog = true`, `inner_nsteps = 1`, `inner_nwalkers = 4`); calls `Energy(wset)` after each step (exercising the `χ ≠ φ_cond` measurement path); asserts finite weights/energies/overlaps over 3 steps. CLOSED+CPU; `inner_leapfrog = false` leaves 3c-i/3b bit-identical. **[overhaul] CPU-verified** Jun 2026: passes in the full tag (**11 cases / 5831 assertions**, `mpirun -np 1`, `Ne_cc-pvdz`). |
| Driver validation | ✓ **Ne cc-pVDZ Stage E** (Jun 2026, `worker7014`, `mpirun -np 1`) — leapfrog on the VAFQMC `ham_var.h5` trial. The leapfrog's robust win is **weight stability**: the band holds at **3.50–5.04**, matching the deterministic anchor's 3.55–5.03 (vs 3b 1.75–8.36 and 3c-i 1.80–11.58) — the exact `𝒩(φ)` cancellation in the ratio. With the generalized-`w_p` fix the **measurement energy is unbiased** (**−128.785**, agreeing with the directly-conditioned 3c-i −128.774; pre-fix `S_p` mis-scored it as −128.738). No NaN / collapse. Energy *block-variance* is comparable across conditioned stages at smoke scale (not a visible reduction with `P = 4`). See [Phase 3c comparison](#phase-3c-comparison--conditioning--leapfrog-jun-2026-leapfrog-validation). |

**Sequencing note:** 3a alone delivers a working (static) stochastic-trial propagator, delegate-limit
verified. 3b/3c can follow once 3a is stable; the observable/mean-field phases below reuse the same
reductions and inherit whichever depth (static after 3a, dynamic after 3b/3c) is in force.

### Phase 4 — `vHS` (resolved: no override needed)

`vHS` is **trial-independent** — `v = √Δτ · Σ_γ X_γ L_γ` uses only the auxiliary fields and the
(trial-independent) Cholesky tensor `L`, with no half-rotation or inner-ensemble quantity (verified
in `Real3IndexFactorization::vHS`). `StochasticWfn` delegates to `nomsd_` permanently; there is no
stochastic `vHS` work. Recorded here so the hot path is fully accounted for (see Phase 3a).

### Phase 5 — `MixedDensityMatrix` / `DensityMatrix` / `accumulate_estimators` (**complete**)

Mixed and back-propagated observables. The observable `MixedDensityMatrix` reuses the Phase 3a
mixed-DM reduction (estimator 3) with the observable layout/options; `accumulate_estimators` feeds that
stochastic full mixed DM to the observable handlers — including the **time-evolved** (`TimeEvolvedObsHandler`)
path, where the X/Yc/M operator transform is linear in the DM and so commutes with the inner-ensemble
average; `DensityMatrix` (reference-parameterized) and `generalizedFockMatrix` (DM-in contraction)
resolve to exact True-Ham delegates. The only Phase-5-adjacent item left is the back-propagation
*reference* semantics for a stochastic trial (which references it exposes), which is Tier 6. Status
detail below.

#### Phase 5 — observable `MixedDensityMatrix` (**complete**, CPU-verified [overhaul])

The two observable `MixedDensityMatrix` overloads (`(wset, G, compact)` and `(wset, G, Ov, compact)`)
were the pre-Phase-5 pure delegates to `nomsd_`; they are now the **observable analogue of
`MixedDensityMatrix_for_vbias`**. Both route through one 3-arg workhorse
(`StochasticWfn.icc`) that, away from the delegate limit, calls the shared `reduce_inner_cross_dm`
to form `G[w] = Σ_p w_p G_{p,w} / Σ_p w_p` (estimator 3 of arXiv:2505.18519) — the **same** inner-
ensemble reduction and weight progression (`1/P` static → `S_p` dynamic → generalized under
conditioning/leapfrog) as the force-bias DM — but returned in the **caller's observable layout** and
with the **log** overlap convention NOMSD's observable DM reports (the linear effective overlap
`(1/P) Σ_p ⟨ψ_p|φ_w⟩` is `std::log`-converted at the end, exactly as `Energy`/`Log_Overlap` do). The
2-arg overload mirrors NOMSD: it builds a scratch overlap vector and calls the 3-arg form. Delegates
to `nomsd_` at the single-determinant delegate limit; matches NOMSD there.

**Compact vs. full off the anchor.** The per-pair *compact* mixed DM (`[nel × NMO]`) has its `nel`
rows in the **bra `ψ_p`'s own occupied basis** (`det_ops::MixedDensityMatrix` compact layout), so
once the inner walkers leave the anchor (`inner_nsteps > 0`) each `ψ_p` lives in a different basis and
the compact DMs cannot be averaged across the ensemble — only the basis-independent full `NMO × NMO`
layout can. At the static limit every `ψ_p ≡` anchor, so compact is well-defined (anchor basis) and
equals NOMSD. The override therefore reduces in the caller's layout at the static limit and **rejects
`compact && inner_nsteps > 0`** with a `utils::check` (request the full layout for a dynamic ensemble).
This is the observable counterpart of `MixedDensityMatrix_for_vbias` *forcing* the full layout off the
anchor. `accumulate_estimators` always builds the full mixed DM (`compact = false`) through its own
override, so it never requests a compact observable layout off the anchor; at the delegate limit both
methods delegate to `nomsd_`.

Test: `stochastic_mixed_density_matrix_matches_nomsd` (`tests/test_wfn_factory.cpp`, `[stochastic_wfn]`).
Unlike `vbias`, the observable mixed DM **is** exposed on the `Wavefunction` variant, so it compares
`G` directly in **both** the compact and full layouts (plus `exp(Ov)`): (1) `inner_nwalkers` invariance
via a static replicated ensemble (1 vs 3), (2) delegate-limit equality vs NOMSD gated on `ndet == 1`.
Built + CPU-verified on `worker6049` (`Ne_cc-pvdz`, `mpirun -np 1`): the new case passes (8 assertions)
and at Phase 5 completion the full `[stochastic_wfn]` tag stood at **13 cases / 5843 assertions** (from
11 / 5831; the current suite is **19 / 6657** after Phase 7).

#### Phase 5 — `accumulate_estimators` (**complete**, CPU-verified [overhaul])

`accumulate_estimators` is the real observable-handler entry point — `MixedObsHandler` calls the 5-arg
form, `TimeEvolvedObsHandler` the 10-arg (X/Yc/M) form. NOMSD's `ndet == 1` path builds the **full**
mixed Green's function of its trial (compact `DensityMatrix` against `OrbMats(0)`, un-compacted via
`T(OrbMats)·Gc`) and feeds it to the observables. The stochastic override builds that full mixed DM
from the **inner ensemble instead** — it is exactly the Phase 5 `MixedDensityMatrix(compact = false)`
(estimator 3, full `[nw][nspin][npol·NMO][npol·NMO]` layout) — then runs NOMSD's identical observable
loop (`v.accumulate(iav, Gfull, Gfull_h, wgt, importanceSampling)` over both property lists). Delegates
to `nomsd_` at the delegate limit.

**Time-evolved / back-propagated observables (X/Yc/M, `TimeEvolvedObsHandler`).** NOMSD's `ndet == 1`
transform is `Gfull[is] ← M[is] + T(X[is])·T(OrbMats)·Gc[is]·Yc[is]`. Grouping it as
`M + T(X)·(T(OrbMats)·Gc)·Yc = M + T(X)·G_full·Yc`, the transform is **linear in the mixed DM `G_full`**
and `X`/`Yc`/`M` are trial-independent operator/walker quantities (from `PropagateOperators`). So the
inner-ensemble average (estimator 3) **commutes** with the transform: feeding the *stochastic* full
mixed DM and applying the **same** `M + T(X)·G_full·Yc` per spin (`M` added once) equals averaging the
per-inner-walker transforms `Σ_p w_p[M + T(X)·G_{full,p,w}·Yc]/Σ_p w_p`. The override therefore builds
the stochastic full mixed DM, then applies NOMSD's identical X/Yc/M gemms — no anchor-specific
back-propagation reference is needed for this *mixed* (importance-sampled) estimator. (Full
back-propagation *reference* semantics for a stochastic trial — which reference set it exposes — remain
the open [Tier 6](#tier-6--back-propagation) question; they do not affect this transform.)

Test: `stochastic_accumulate_estimators_matches_nomsd` (`tests/test_wfn_factory.cpp`, `[stochastic_wfn]`).
A real `full1rdm` (one-body RDM, no rotation) observable is driven through `accumulate_estimators` — the
identical code path the `Observable` variant takes, since `accumulate_estimators` is templated on the
observable type and only calls `v.accumulate(...)`, so a `std::vector<full1rdm>` exercises it directly.
The accumulated 1-RDM is read back via the observable's HDF5 `print()` (its `DMAverage` is private) and
compared against NOMSD, **both for the mixed path and the time-evolved path** (with deterministic
non-trivial X/Yc/M operators): `inner_nwalkers` invariance (static replicated 1 vs 3) and delegate-limit
equality at `ndet == 1`. CPU-verified on `worker6049` (`Ne_cc-pvdz`, `mpirun -np 1`).

#### Phase 5 — `DensityMatrix` and `generalizedFockMatrix` (resolved as True-Ham delegates)

Neither needs an inner-ensemble reduction:

- `DensityMatrix(wset, Ref, G, Ov, compact, herm)` — the bra is an **externally supplied reference**
  `Ref`, the ket the walker; it is pure orbital algebra **independent of both the trial and the
  Hamiltonian** (NOMSD's implementation never touches its own `OrbMats`). The stochastic trial plays no
  role, so the delegate to `nomsd_` is **exact** (documented inline). Which reference set a stochastic
  trial *exposes* for back-propagation was the separate Tier 6 question, **resolved in Phase 7** (the
  outer-NOMSD reference set, inner-ensemble-agnostic) — it does not change the meaning of this
  `Ref`-parameterized method. (No production caller today; reached only via Tier 6.)
- `generalizedFockMatrix(...)` — now declared on `StochasticWfn` as a delegate to `nomsd_` (the True
  Ham), mirroring NOMSD/PHMSD. It contracts a **supplied** DM against the Ham (no trial reduction), and
  the True Ham is the correct operator. Reached only via the `generalizedFockMatrix` observable, which is
  currently commented out in all handlers; the delegate completes the API so it compiles if re-enabled.

### Phase 6 — `G_MF` / `vMF` (**complete**, CPU-verified [overhaul])

Mean-field subtraction consistency: the stochastic mean-field Green's function / Cholesky contribution,
built from the inner ensemble. **Done static-limit-first, like 3a.**

**The key structural difference from every prior phase:** `G_MF` / `vMF` are **trial-against-itself**
quantities — the trial's own expectation `⟨Ψ_T|Ô|Ψ_T⟩/⟨Ψ_T|Ψ_T⟩` with **no outer walker**. The Tier 1/2
estimators (Phases 2–5) are *mixed* (`⟨Ψ_T|Ô|φ_w⟩`, inner ensemble against the OUTER walkers); the
mean-field reduces the inner ensemble **against itself**. So this is the inner-ensemble analogue of
**NOMSD's *multi-determinant* `G_MF`/`vMF`** (a double sum over the determinant expansion), with the
inner ensemble `{ψ_p}` playing the role of the determinants and a uniform weight `1/P`:

```
G_MF = ⟨Ψ_T|c†c|Ψ_T⟩/⟨Ψ_T|Ψ_T⟩ = [Σ_{p,q} ⟨ψ_p|c†c|ψ_q⟩] / [Σ_{p,q} ⟨ψ_p|ψ_q⟩]
```

This is the *exact* mean field of the sampled trial state `|Ψ_T⟩ ≈ (1/P) Σ_p |ψ_p⟩` (the double sum,
**not** a diagonal `(1/P) Σ_p ⟨ψ_p|·|ψ_p⟩` average — that would drop the interference between samples
and is a different object), and it mirrors NOMSD's existing multi-determinant code structure (cross-pair
double sum, `det_ops::MixedDensityMatrix` per pair already normalized by its pair overlap, the linear
pair overlaps `exp(ov_pq)` supplying the weights, CLOSED doubling the alpha log-overlap). Implementation:
a private `reduce_inner_mean_field_dm` (`StochasticWfn.icc`) reduces the **`P = inner_nwalkers_`
walker-independent samples** into the normalized full `[1, nspin·npol·NMO·npol·NMO]` mean-field DM (bras
FairDivide'd across the comm with a full ket batch per bra, then `Gsum`/`Osum` all-reduced); `G_MF`
reshapes/`share_from_root`s it (same return type/shape as `NOMSD::G_MF`), and `vMF` contracts it through
`nomsd_.vbias_from_G` (True Ham, full-G layout). Both **delegate to `nomsd_` (the anchor mean field)**
unless `mean_field_uses_inner_ensemble()` is true — i.e. when `inner_nwalkers_ > 1` **and**
`inner.size() == inner_nwalkers_` (walker-independent `P`-sample form). They delegate when
`inner_nwalkers_ == 1` (a single sample is degenerate; subsumes the usual delegate limit) **or** whenever
a conditioned/leapfrog resample has expanded the ensemble to `nwalk·P` (see the call-order note below).

**Call order in `Propagate` (the reason for the `P`-sample gate).** `AFQMCBasePropagator::Propagate`
calls `begin_inner_step(wset)` **first**, then `generateP1(dt)` (→ `vMF`, and `G_MF` for discrete Hams)
only if `dt` changed. `begin_inner_step` differs by mode: in the **non-leapfrog** paths (static, 3b free
projection, 3c-i) it only *arms a latch* — the actual resample happens later inside the force-bias
reduction — so the inner ensemble is still in its `P`-sample form when `vMF` runs (at the static limit it
is the anchor; `vMF` reduces it to the anchor mean field). In the **leapfrog** path (3c-ii)
`begin_inner_step` *eagerly* resamples (to refresh the importance-reweighted old overlap), **expanding the
ensemble to `nwalk·P` walker-conditioned samples before `vMF` runs**. The trial mean field has no
outer-walker dependence, so that conditioned ensemble must **not** be summed (a `1/(nwalk·P)²`-weighted
double sum over walker-conditioned samples is the wrong object); `vMF`/`G_MF` detect the non-`P`-sample
size and delegate to the anchor mean field instead. Regression-tested by
`stochastic_mean_field_production_order` (reproduces the `begin_inner_step`-before-`vMF` order under
leapfrog and asserts `vMF`/`G_MF` still equal NOMSD). Because `vMF` is recomputed only when `dt` changes,
getting the **first** value right is what matters.

**Static-limit-first / what's deferred.** At the static replicated limit every `ψ_p ≡` anchor, so the
reduction's double sum collapses to the anchor density `⟨φ_T|c†c|φ_T⟩` — exactly NOMSD's `ndet==1`
`G_MF`/`vMF` — the verified regression (the `inner_nwalkers = 3` static case runs the full reduction;
`= 1` delegates). **Note that in the current call structure the reduction in practice always sees the
anchor:** `vMF` is computed once, at the first `generateP1` (when `dt` first changes), which runs *before*
the force-bias reduction that fires the non-conditioned (3b) free-projection resample — so the `P`-sample
ensemble has not yet left the anchor, and the conditioned (3c) ensemble is delegated. The reduction
machinery is therefore in place and statically verified, but a **genuine dynamic mean field** (recomputing
`vMF` on a `B̂_T`-advanced, non-anchor `P`-sample ensemble) is **not reached by the present plumbing and is
intentionally deferred** — off the anchor the inner samples are single determinants whose double sum is a
*different object* from NOMSD's analytic trial mean field, and `vMF` is variance-reduction-only so it need
not be exact. That dynamic mean field, the conditioned/leapfrog case (delegated to the anchor today), and
multi-rank (`-np > 1`) consistency of the per-rank-decorrelated ensemble are **deferred** (the same
`-np > 1` gate open for every stochastic reduction). **Phase 6 is complete at the unit-test bar —
static-limit parity plus delegate robustness across the production call order; production/driver validation
of a dynamic mean field is a follow-up.** (The reduction sums the **full** `P²` pair grid — the exact
`⟨Ψ_T|Ô|Ψ_T⟩/⟨Ψ_T|Ψ_T⟩` for the sampled state — rather than NOMSD's multi-det upper-triangle convention;
both agree at the static limit, and the full grid is unambiguous.)

### Phase 7 — Layout queries and back-prop references (**complete**, CPU-verified [overhaul])

**Tier 4–5 (layout/metadata queries) — verified.** Under the dual-Hamiltonian design every outer-facing
query (`number_of_cholesky_vectors`, `getHamType`, `getFieldTypes`, `getOneBodyPropagatorMatrix`, the
`size_of_G`/`transposed_*`/`spin_dependent_vHS` flags, the Cholesky-count/distribution queries,
`getSlaterDetOperations`, …) **stays on the outer `nomsd_` (True Ham)** — it describes what the outer
driver/propagator sees, not the inner Variational stack. These were correct by construction (the
propagator, the Tier 1 reductions, and the Tier 2/3 observables all run against them), so Phase 7's Tier
4–5 work is verification, not a reroute (see [Tier 4](#tier-4--layout-and-metadata-queries) /
[Tier 5](#tier-5--hamiltonian-and-slater-infrastructure)).

**Tier 6 (back-propagation references) — outer-NOMSD delegate, the substantive decision.** The
`BackPropagatedEstimator`/`FullObsHandler` consume a trial as a weighted determinant expansion
`Ψ_T = Σ_iref ci_iref |Ref_iref⟩` via `total_number_of_references()` (buffer/BP-history sizing),
`getReferences(n, Refs)` (the reference Slater matrices — **filled for one walker then broadcast to ALL
walkers**, so they must be walker-INDEPENDENT, and captured once per BP block over a fixed window, so
time-stable), and `getReferenceWeight(i)`. The chosen semantics for a stochastic trial is to expose
exactly the **outer** `nomsd_`'s reference set and ignore the inner ensemble — i.e. keep all three as
`nomsd_` delegates, made explicit + documented in `StochasticWfn.hpp`. For the single-determinant anchor
that set is `{φ_T}` with weight 1; for multi-det outer trials it is the full CI expansion — identical to
plain NOMSD in both cases. Rationale: the outer trial's references are walker-independent and
time-stable (frozen while the inner ensemble resamples); exact vs plain NOMSD by construction, with the
scientific caveat that for `inner_nsteps>0` back-propagation is scored against the outer trial, not the
field-sampled spread `{ψ_p}`, consistent with `vMF`/`G_MF` (Phase 6). Tests:
`stochastic_back_propagation_matches_nomsd` verifies reference COUNT, per-reference WEIGHT, reference
Slater matrices, and Tier 4–5 layout parity all equal plain NOMSD's **unconditionally** and **independent
of `inner_nwalkers`**; `stochastic_back_propagation_production_order` pins the dynamic case — after
leapfrog `begin_inner_step` expands the inner ensemble to `nwalk·P`, the references stay frozen at the
outer trial; `stochastic_back_propagation_estimator_smoke` and
`stochastic_back_propagation_driver_smoke` exercise `BackPropagatedEstimator` end to end at the
**static delegate limit** (`inner_nsteps = 0`) through `EstimatorHandler` and `DriverFactory`
respectively (finiteness of the accumulated back-propagated one-RDM only). A dynamic
(`inner_nsteps > 0`) BP integration smoke currently accumulates **NaN** — numerical stability off the
static limit is a follow-up.

**Deferred (Option B — faithful inner-ensemble references).** The faithful expansion would set
`nrefs = P`, references = the inner samples `{ψ_p}`, weights `1/P` (→ `S_p`). It is **deferred research**,
blocked by: (1) **walker-dependence** — only the non-conditioned `P`-sample ensemble is shared across
outer walkers; the conditioned/leapfrog (Phase 3c) `nwalk·P` ensemble is walker-specific and incompatible
with the estimator's broadcast-to-all-walkers structure; (2) **fixed-window vs. resample** — BP captures
references once per block and back-propagates over a window, but the inner ensemble resamples every step,
so which snapshot defines the reference is unsettled; (3) **cost** — `P×` the back-propagation work
(propagating `P` references backward instead of 1). Unlike `vMF`, BP would capture references at
*measurement* time, so Option B would genuinely use field-sampled `ψ_p` — making it the natural home for a
future dynamic back-propagated estimator once those three are resolved.

### Phase 8 — Factory / HDF5 first-class treatment

Remove the `stochastic` boolean flag hack; dedicated `fromHDF5` branch like `PHMSD`.

### What to avoid as an early step

- **Overriding `Overlap`/`Energy` before 1a** — without an owned inner ensemble, this only
  re-wraps `nomsd_` with extra indirection.
- **Building the inner propagator before inner walkers exist** — highest coupling, hardest to
  test, and useless until `Overlap`/`Energy` consume the ensemble. *(Phase 1c builds the
  propagator at wavefunction construction time, but does not invoke it; walker init remains
  the separate two-phase step from 1a.)*
- **Splitting `HamOps` out of `NOMSD` globally** — a `NOMSD` refactor, not a `StochasticWfn`
  feature; Phase 1b instead duplicates full `NOMSD` instances with separate moved-in `HamOps`.

---

## Validation Checkpoints

The checklist below defines the **regression targets** for stochastic work. The static (1a–3a) and
dynamic (3b) suites are **ported and CPU-verified on overhaul** (`Ne_cc-pvdz`, `-np 1`); the Phase
1a/1b/1c infrastructure tests remain develop-only. Per-test checkpoints are in the per-phase tables
below; the per-phase fixtures/commands are the canonical ones from the
[port-status section](#test-fixtures-on-overhaul-the-develop-fixtures-are-gone).

### Delegate limit (primary regression anchor)

The core anchor is the limit where the inner ensemble collapses to a single deterministic state
(`inner_nwalkers = 1`, `inner_nsteps = 0`), where every override must reproduce plain `NOMSD`:

- `stochastic_wfn_matches_nomsd` — all four overrides (`Log_Overlap`, `Energy`, `MixedDensityMatrix_for_vbias`, `vbias`) at the delegate limit, NOMSD-equality gated on `ndet == 1` (each diverges for multi-determinant trials by design).
- `stochastic_overlap_matches_nomsd` / `stochastic_energy_matches_nomsd` / `stochastic_vbias_matches_nomsd` — per-method `inner_nwalkers` invariance + delegate-limit equality (details in the per-phase tables).
- `stochastic_propagator_step` — outer `Propagate()` on a dynamic trial completes (finiteness only, not parity).
- `stochastic_inner_outer_infrastructure_independent` / `stochastic_inner_propagator_construction` — dual-infrastructure + inner-propagator wiring (**develop-only; not yet ported**).

Still open (integration/research): dynamic-trial energies/overlaps vs analytic `NOMSD` within
stochastic error bars; `MixedObsHandler` forces/densities; GPU build + memory/task-group load.

### Phase 1a-specific tests (implemented)

*Phase 1a **infrastructure** tests remain **develop-only** (see *Ported vs. deferred* in the port-status section). The keystone delegate-limit anchor `stochastic_wfn_matches_nomsd` is ported and CPU-verified on overhaul — see [Delegate limit](#delegate-limit-primary-regression-anchor).*

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_walkers_init` | Inner walkers initialize to `inner_nwalkers`; Slater matrices match `getInitialGuess()`; inner `NOMSD` overlap/energy on `inner_wset()` matches reference; replicated ensemble gives identical observables. |
| `stochastic_inner_walkers_uninitialized_smoke` | Stochastic wavefunction built without `walker_pt` stays uninitialized until `maybe_initialize_stochastic_inner_walkers()`; then `stochastic_inner_wset()` is usable. (`inner_wset()` before init calls `APP_ABORT` — not exercised in-process.) |

### Phase 1b-specific tests (implemented)

*Develop only — **not yet ported to overhaul** (infrastructure test; reaches into typed
`StochasticWfn` internals the overhaul `Wavefunction` variant does not expose — see the
*Ported vs. deferred* note in the port-status section).*

Requires a **NOMSD** HDF5 input (`getWavefunctionType() == "NOMSD"`); PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_outer_infrastructure_independent` | `outer_nomsd()` and `inner_nomsd()` are distinct objects with distinct `getSlaterDetOperations()` pointers; `StochasticWfn::getSlaterDetOperations()` returns outer; inner/outer layout metadata match (Cholesky distribution, transpose flags, `size_of_G_for_vbias`, walker/Ham types); inner overlap/energy on `inner_wset()` match plain `NOMSD` reference. |

### Phase 1c-specific tests (implemented)

*Develop only — **not yet ported to overhaul** (infrastructure test; needs typed
`inner_wavefunction()`/`inner_propagator()` access not exposed on the overhaul `Wavefunction` variant
— see the *Ported vs. deferred* note in the port-status section).*

Requires a **NOMSD** HDF5 input; PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_propagator_construction` | `inner_nsteps() == 0`; `inner_propagator_built()`; `inner_wavefunction()` shares SDetOp with `inner_nomsd()` but not outer; layout parity between inner NOMSD and wrapped `Wavefunction`; propagator Cholesky count matches inner NOMSD; default hybrid (not free) propagation. |

### Phase 2a-specific tests (implemented)

***[overhaul] Ported + CPU-verified*** in `tests/test_wfn_factory.cpp` on the canonical `Ne_cc-pvdz`
fixture (`ham_chol_dense.h5` + `wfn_rhf.h5`). Both checkpoints run on the single-determinant RHF trial:
(1) invariance via a replicated static ensemble (`inner_nwalkers` 1 vs 3), (2) delegate limit at
`ndet == 1`. Overlaps are compared as `exp(OVLP)` (the property is a *log* overlap on overhaul — see
[Port bugs](#port-bugs-surfaced-by-the-test-port-overhaul)).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_overlap_matches_nomsd` | `Log_Overlap` in isolation (no following `Energy`): (1) **inner_nwalkers invariance**; (2) **delegate limit** — stochastic overlap equals NOMSD at `ndet == 1`. |

### Phase 2b-specific tests (implemented)

***[overhaul] Ported + CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz`
(`ham_chol_dense.h5` + `wfn_rhf.h5`); all four checkpoints run on the single-determinant RHF trial
(invariance via the replicated ensemble, delegate limit at `ndet == 1`, Log_Overlap/Energy `Ov`
consistency, and the 3-arg `Energy(wset, E, Ov)` entry point).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_energy_matches_nomsd` | `Energy` in isolation: (1) **inner_nwalkers invariance** — a static replicated ensemble gives E1/EXX/EJ/energy/overlap independent of `inner_nwalkers`; (2) **delegate limit** — stochastic energy and overlap equal NOMSD at `ndet == 1`; (3) **Log_Overlap/Energy consistency** — `Log_Overlap(wset)` equals the overlap from `Energy(wset)` (locks the shared-reduction invariant across the two differently-computed code paths); (4) **propagator entry point** — the direct 3-arg `Energy(wset, E, Ov)` agrees with the property-setter form. Mirrors `stochastic_overlap_matches_nomsd`. |

`stochastic_wfn_matches_nomsd` now also exercises the `Energy` override (its energy/overlap
assertions are gated on `ndet == 1`; since Phase 3a the `MixedDensityMatrix_for_vbias` / `vbias`
assertions are gated on `ndet == 1` too — see [Phase 3a-specific tests](#phase-3a-specific-tests-implemented)).

### Phase 3a-specific tests (implemented)

***[overhaul] Ported + CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz`
(`ham_chol_dense.h5` + `wfn_rhf.h5`). The overhaul `vbias(wset, X, dt)` drives
`MixedDensityMatrix_for_vbias` internally, so the test compares the **force bias `X` (= L·G)** for
invariance and the delegate limit; the intermediate `G` is not surfaced on the variant.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_vbias_matches_nomsd` | `MixedDensityMatrix_for_vbias` + `vbias`: (1) **inner_nwalkers invariance** — a static replicated ensemble gives a mixed DM `G` and force bias `x̄` independent of `inner_nwalkers`; (2) **delegate limit** — stochastic `G` and `x̄` equal NOMSD at `ndet == 1`. Mirrors `stochastic_energy_matches_nomsd`. |

`stochastic_wfn_matches_nomsd` also exercises both Phase 3a methods, with its
`MixedDensityMatrix_for_vbias` / `vbias` assertions gated on `ndet == 1`.

### Phase 3b-specific tests (implemented)

***[overhaul] Ported + CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz`
(`ham_chol_dense.h5` + `wfn_rhf.h5`), CPU/CLOSED only. Porting these caught and fixed the full-G
kernel bugs (one-body rank mismatch; EXX/EJ wrong `Lankf` slice axis + `dotc`→`dot`) — see
[Port bugs](#port-bugs-surfaced-by-the-test-port-overhaul).

Requires a **NOMSD** HDF5 input and a **CLOSED (RHF)** trial (`wfn_rhf.h5`); UHF/MSD/PHMSD inputs skip
the dynamic/full-G checks silently. On overhaul the tests also `return` for `DEVICE_MEMORY`
(full-G kernels are CPU-only this phase).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_full_g_matches_compact` | Un-rotated full-G energy and force bias at the anchor equal the compact `nd = 0` path and NOMSD (single-determinant delegate limit). |
| `stochastic_dynamic_ensemble_smoke` | With `inner_nsteps > 0`, inner resample + all four stochastic overrides on moved walkers; asserts finite energies/overlaps (and vbias path via MixedDM + `vbias`, full-G via layout dispatch). |
| `stochastic_propagator_step` | Real outer `AFQMCBasePropagator::Propagate()` on a dynamic trial (`inner_nsteps = 1`, `inner_nwalkers = 4`); full propagator hot path; asserts finite weights/energies/overlaps over 3 steps. |

### Phase 3b-var-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`), CPU/CLOSED only (passes in isolation, 2675 assertions, and in the full
`[stochastic_wfn]` tag, 11 cases / 5831 assertions). Requires a **NOMSD** + **CLOSED (RHF)** input;
other inputs skip silently.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_hamiltonian_same_as_true` | Inner stack built from a separate Hamiltonian via the `inner_hamiltonian` key (same integral file) reproduces the clone path: `wfn_clone` (no `inner_hamiltonian`) and `wfn_hvar` (`inner_hamiltonian` = same file) agree to `1e-9` on `Energy`/`Log_Overlap`/`vbias` over 3 dynamic (`inner_nsteps = 1`) steps. `HamFac.has_input("wfn_hvar__inner_hamiltonian__")` (+ its `REQUIRE_FALSE` for `wfn_clone`) confirms the factory actually built/registered the second Ham. |

### Phase 3c-i-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`), CPU/CLOSED only. Requires a **NOMSD** + **CLOSED (RHF)** input; other inputs skip
silently. `inner_conditioning = false` leaves all Phase 3b cases bit-identical (exact fallback). Passes
in the full `[stochastic_wfn]` tag (**11 cases / 5831 assertions**, `mpirun -np 1`, Jun 2026).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_conditioned_propagator_step` | Real outer `AFQMCBasePropagator::Propagate()` over the **walker-conditioned** dynamic trial (`inner_conditioning = true`, `inner_nsteps = 1`, `inner_nwalkers = 4`, `nwalk = 11` ⇒ `nwalk·P = 44` inner walkers); the inner ensemble resizes to `nwalk·P` and is sampled with the per-walker conditioning bias; asserts finite weights/energies/overlaps over 3 steps. The internal `inner.size() == nwalk·P` checks in `reduce_inner_cross_dm` / `Log_Overlap` validate the block-structured resize. Finiteness smoke; hybrid Eq.-25 overlap exactness requires `inner_leapfrog` (3c-ii). |

### Phase 3c-ii-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`), CPU/CLOSED only, hybrid propagation. Requires a **NOMSD** + **CLOSED (RHF)** input;
other inputs skip silently. `inner_leapfrog = false` leaves 3c-i/3b bit-identical (exact fallback).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_leapfrog_propagator_step` | Real outer **hybrid** `AFQMCBasePropagator::Propagate()` over the leapfrog trial (`inner_conditioning = inner_leapfrog = true`, `inner_nsteps = 1`, `inner_nwalkers = 4`); `begin_inner_step(wset)` resamples conditioned on the old walker + refreshes the importance-reweighted old overlap, so the step ratio `new/old` is exactly Eq. 25 (`𝒩(φ)` cancels). Asserts finite weights/energies/overlaps over 3 steps (finiteness smoke; energy-vs-analytic-AFQMC + variance reduction are the driver-level research validation). |

### Phase 5-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`). Requires a **NOMSD** input; other inputs skip silently. Covers the observable
`MixedDensityMatrix` and `accumulate_estimators` (`DensityMatrix` / `generalizedFockMatrix` are exact
True-Ham delegates needing no test — see [Phase 5](#phase-5--mixeddensitymatrix--densitymatrix--accumulate_estimators)).
Passed in the full `[stochastic_wfn]` tag at Phase 5 completion (**13 cases / 5843 assertions**, `mpirun -np 1`, Jun 2026; the current suite is **19 / 6657** after Phase 7).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_mixed_density_matrix_matches_nomsd` | The observable `MixedDensityMatrix(wset, G, Ov, compact)` in **both** layouts (compact `[nel·NMO]` and full `[NMO·NMO]`): (1) `inner_nwalkers` invariance — a static replicated ensemble (1 vs 3) gives an `inner_nwalkers`-independent `G` and `exp(Ov)`; (2) delegate limit — stochastic `G` and `exp(Ov)` equal NOMSD at `ndet == 1`. The DM is exposed on the `Wavefunction` variant, so `G` is compared directly (unlike `vbias`, where only the contracted force bias is observable). |
| `stochastic_accumulate_estimators_matches_nomsd` | `accumulate_estimators` driving a real `full1rdm` (one-body RDM, no rotation) observable — fed directly as `std::vector<full1rdm>` (the same `v.accumulate(...)` path the `Observable` variant takes). The accumulated 1-RDM is read back via the observable's HDF5 `print()` and compared to NOMSD for **both the mixed and the time-evolved path** (deterministic non-trivial X/Yc/M operators): (1) `inner_nwalkers` invariance (static replicated 1 vs 3); (2) delegate-limit equality at `ndet == 1`. |

### Phase 6-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`). Requires a **NOMSD** input (the production-order case also **CLOSED**/CPU); other inputs
skip silently. Passes in the full `[stochastic_wfn]` tag (**19 cases / 6657 assertions**, `mpirun -np 1`,
worker6035, Jun 2026; was 13 / 5843).

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_mean_field_matches_nomsd` | The **trial-only** mean-field quantities. Compares the mean-field bias `vMF` (= `L·G_MF`, a `[nCV]` vector) **and** the mean-field DM `G_MF` (`[nspin][npol·NMO][npol·NMO]`) directly against plain NOMSD: (1) `inner_nwalkers` invariance — a static replicated ensemble (1 vs 3) gives an `inner_nwalkers`-independent `vMF`/`G_MF`; (2) delegate-limit equality vs NOMSD at `ndet == 1`. There is no outer walker (these are `⟨Ψ_T|·|Ψ_T⟩/⟨Ψ_T|Ψ_T⟩`), so the test exercises the inner-ensemble-against-itself double-sum reduction; the `inner_nwalkers = 3` case runs it while `= 1` delegates, so the invariance check is the substantive one. |
| `stochastic_mean_field_production_order` | **Call-order regression** (CLOSED/CPU). Builds a **leapfrog** trial (`inner_conditioning = inner_leapfrog = true`, `inner_nsteps = 1`), calls `begin_inner_step(wset)` — which, in leapfrog mode, conditioned-resamples and **expands the inner ensemble to `nwalk·P`** — and only *then* calls `vMF`/`G_MF`, exactly as `Propagate` orders them (`begin_inner_step` before `generateP1`). **`REQUIRE`s `stochastic_inner_ensemble_size() == nwalk·P`** so the scenario is pinned (parity alone could pass on a still-`P` anchor ensemble). Asserts `vMF`/`G_MF` still equal NOMSD: with the ensemble out of `P`-sample form they must delegate to the anchor mean field, not reduce the conditioned `nwalk·P` block with the wrong `1/(nwalk·P)²` normalization. Guards `mean_field_uses_inner_ensemble` against the leapfrog timing bug. |

### Phase 7-specific tests (implemented)

***[overhaul] CPU-verified*** in `tests/test_wfn_factory.cpp` on `Ne_cc-pvdz` (`ham_chol_dense.h5` +
`wfn_rhf.h5`). Requires a **NOMSD** input (the production-order case also **CLOSED**/CPU); other inputs
skip silently. Part of the full `[stochastic_wfn]` tag (**19 cases / 6657 assertions**, `mpirun -np 1`,
worker6035, Jun 2026). **Scope:** reference-API + layout parity, plus static-limit BP integration
smokes through `EstimatorHandler` and `DriverFactory` (see table below). Dynamic (`inner_nsteps > 0`)
BP numerical stability remains a follow-up.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_back_propagation_matches_nomsd` | The Tier 6 back-propagation reference API under the **outer-NOMSD-delegate** semantics, plus a Tier 4–5 layout check. Verifies that for a stochastic trial the reference COUNT (`total_number_of_references`), per-reference WEIGHT (`getReferenceWeight`), and the reference Slater matrices themselves (`getReferences`, shape `[nrefs, npol·NMO, nel]`) all equal plain NOMSD's — **unconditionally** (not just `ndet==1`) and **independent of `inner_nwalkers`** (1 vs 3), since the delegate is inner-ensemble-agnostic. Also asserts Tier 4–5 parity (Cholesky count, Ham type, walker type). This is API + layout parity, *necessary but not sufficient* for full BP correctness (the estimator path beyond these methods is unexercised). |
| `stochastic_back_propagation_production_order` | **Call-order regression** (CLOSED/CPU), the Phase 6 lesson applied to Tier 6. BP captures references *during* the run; this builds a **leapfrog** trial (`inner_conditioning = inner_leapfrog = true`, `inner_nsteps = 1`), snapshots the references, calls `begin_inner_step(wset)` — which advances **and** expands the inner ensemble to `nwalk·P` (`REQUIRE`d) — and asserts the references (count, weights, Slater matrices) are **frozen at the outer trial**, unchanged by the inner resample and still `== NOMSD`. This pins the documented approximation: BP scores against the outer trial, not the evolving stochastic spread. |
| `stochastic_back_propagation_estimator_smoke` | **BP integration smoke** (CLOSED/CPU). Builds a **static** stochastic trial (`inner_nsteps = 0`, `inner_nwalkers = 4`), wires `BackPropagatedEstimator` through `EstimatorHandler` with a `back_propagation` / `onerdm` block, runs real hybrid `Propagate()` steps (BP history advanced by the propagator), and asserts the accumulated back-propagated one-RDM in `.stat.h5` is **finite**. Exercises backward propagation, `resize_bp`, path-restoration weighting, and `FullObsHandler` accumulation — not just the Tier 6 reference API. |
| `stochastic_back_propagation_driver_smoke` | **Driver integration smoke** (CLOSED/CPU). Minimal `DriverFactory::executeDriver` with inline `stochastic: true` (`inner_nsteps = 0`), `walker_type: closed`, and a `back_propagation` estimator child; asserts `executeDriver` succeeds, `.scalar.dat` is written, and the back-propagated one-RDM in `.stat.h5` is finite. Same static-limit scope as the estimator smoke; dynamic (`inner_nsteps > 0`) BP currently yields NaN accumulations (research follow-up). |

### Running the stochastic test suite

All stochastic tests share the Catch2 tag `[stochastic_wfn]` and require a NOMSD wavefunction input
(RHF/UHF/PHMSD inputs skip stochastic checks silently).

**`main` (overhaul API)** — target binary is the consolidated `test_afqmc`
(`tests/test_wfn_factory.cpp`); output under `${BUILD_DIR}/tests/bin/`. The static + 3b cases are
**ported and CPU-verified** (19 cases, 6657 assertions — incl. the Phase 3b-var anchor, 3c-i and 3c-ii smokes, the two Phase 5 observable cases, the two Phase 6 mean-field cases, and the four Phase 7 back-prop cases) on the
`Ne_cc-pvdz` dense+RHF fixture (the develop `ham_chol_sc.h5` / `wfn_msd.h5` fixtures are gone). Build is driven via `cmake --build` (Ninja
generator); on the Flatiron cluster build on a compute node, not the gateway:

```bash
HAMIL=$PWD/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
WFN=$PWD/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5

cd /path/to/SAFIRE/build
cmake --build . --target test_afqmc -j 16
mpirun -np 1 ./tests/bin/test_afqmc \
  --hamil "$HAMIL" --wfn "$WFN" \
  "[stochastic_wfn]"
# or: ctest -R stochastic_wfn -V
```

### Multi-rank (`-np > 1`) status (validated Jun 2026, worker6035)

The whole feature was developed and verified at **`-np 1`**; a `-np 2` sweep of `[stochastic_wfn]`
(`Ne_cc-pvdz`) surfaced latent multi-rank bugs (a single `MPI_ABORT` kills the whole tag, so cases were
run individually). **Four bugs were found and fixed (below); `-np 1` remains fully green (19 cases / 6657
assertions) and ALL 19 cases now pass at `-np 2`.** At `-np 2`:

| Case | `-np 2` | Note |
|------|---------|------|
| `overlap`, `vbias`, `mixed_density_matrix`, `mean_field`, `back_propagation_matches`/`production_order`, `wfn_matches_nomsd`, `build_smoke`, `inner_hamiltonian_same_as_true`, `dynamic_ensemble_smoke`, `propagator_step`, `conditioned`/`leapfrog_propagator_step`, `back_propagation_estimator_smoke` | ✅ pass | reductions/observables complete per rank |
| `energy_matches_nomsd` | ✅ pass **(after fix 1)** | see fix 1 |
| `accumulate_estimators_matches_nomsd` | ✅ pass **(after fix 2)** | see fix 2 |
| `full_g_matches_compact` | ✅ pass **(after fix 3)** | see fix 3 |
| `back_propagation_driver_smoke` | ✅ pass **(after fix 4)** | see fix 4 |

**Fix 1 (real production bug) — `StochasticWfn::Energy` double-counted at `-np > 1`.** `NOMSD::Energy`
does **no external `all_reduce`** — `HamOp.energy` returns the complete per-walker energy (any
Cholesky-distribution reduction is internal). The stochastic override `all_reduce`d `E` over `mpi_->comm`,
but `reduce_inner_cross_dm` replicates the full inner-walker loop on **every** rank (it is *not*
FairDivided), so `E` was already complete and the reduce **multiplied it by `comm.size()`** (a no-op at
`-np 1`, hence latent; doubled at `-np 2`, while the un-reduced overlap stayed correct). **Fixed**: removed
the `all_reduce` in `StochasticWfn::Energy` (`StochasticWfn.icc`), matching NOMSD's per-rank-complete
contract. Verified at `-np 2`.

**Fix 2 (test artifact) — `stochastic_accumulate_estimators_matches_nomsd` HDF5 round-trip.** The test
reads the accumulated 1-RDM back through a temp HDF5 file; at `-np 2` both ranks opened the **same**
filename (`H5Fopen … unable to lock the file, errno=11`), and `full1rdm::print` writes only on
`mpi->comm.root()`. **Fixed (test only)**: rank-unique temp filename + root-gate the read-back and the
`CHECK_THAT` comparisons (all ranks still call `accumulate_estimators` + `print`; only root reads/checks).
The **production** accumulate path was always multi-rank-safe (it uses the real observable HDF5 machinery);
only the test's manual round-trip wasn't.

**Fix 3 (pre-existing Phase 3b kernel bug) — `full_g::energy_closed`
(`HamiltonianOperations/full_g_estimators.hpp`) was multi-rank-broken.** The kernel **double-distributed**
with no reduction: walkers by `n % comm.size() == comm.rank()` (the GF build + the EXX/EJ loops) **and**
the combined `(i,nc)` index by `FairDivideBoundary` (only `Twban(:, i0:iN)` filled per rank). So at `-np 2`
the one-body `E1` was complete (its contraction is ungated, over all walkers) but **EXX/EJ came out both
walker-incomplete and CV-partial → garbage** (`full_g_matches_compact` failed on `full.exx`/`full.ej` with
values ~`2.5e4` vs the correct small numbers; `full.e1` matched). No single external `all_reduce` could fix
it — it would double the already-complete `E1` while only completing EXX/EJ (which is exactly why the case
failed on `e1` *before* fix 1's `all_reduce` removal and on `exx`/`ej` *after*). **Root cause:** the
distribution was spurious — the Cholesky (`Lank_full_flat_`) is **replicated** per rank (the caller passes
the full `nCV`, and `ensure_full_cholesky` builds the complete `[NMO·nCV][NMO]` un-rotated Cholesky on every
rank), exactly as the compact `Real3IndexFactorization::energy_impl` is (it iterates the full `nCV` with no
MPI reduction, which is why `NOMSD::Energy` needs none). **Fixed**: `energy_closed` now computes the
**complete** EXX/EJ for **all** walkers redundantly on every rank — removed the walker round-robin, the
`(i,nc)` `FairDivide`, and the barriers; the gemm contracts the full `Lankf` — so it returns a complete `E`
with no reduction, matching the compact path and consistent with fix 1 (the override does no reduce). Only
the `inner_nsteps > 0` full-G energy path is affected. Verified: `full_g_matches_compact` passes at `-np 1`
(no regression) and `-np 2`.

**Fix 4 (real production bug) — `StochasticWfn::Log_Overlap` corrupted overlaps for DISTRIBUTED walkers.**
`stochastic_back_propagation_driver_smoke` (a full `DriverFactory`/`AFQMCDriver` run) aborted at `-np 2`
inside `pair_branch` (`WalkerControl.hpp`) with *"Found 1 walkers with zero weight after branch"* — which,
because one rank `APP_ABORT`s mid-collective, manifests as a hang. **Not a generic population-control issue:
the manual `estimator_smoke` (same reductions, replicated walkers) passed.** Root cause: the stochastic
`Log_Overlap` distributed the `nw·P` pairs across `mpi_->comm` (`FairDivideBoundary`) and then
`all_reduce`d the per-walker `Ov(nw)`. That is correct only when every rank holds the **same** `wset`
(the unit tests, where the outer walkers are replicated). In the **real driver the outer walkers are
DISTRIBUTED** across ranks, so all-reducing `Ov(iw)` summed *different physical walkers* across ranks →
corrupted effective overlaps → bad hybrid step ratios → a walker collapsed to zero weight → `pair_branch`
abort. `NOMSD::Log_Overlap` is per-rank-local (no `all_reduce`); `Energy` (fix 1) and `vbias`
(`reduce_inner_cross_dm`) already are too — `Log_Overlap` was the last reduction still cross-reducing a
per-walker quantity. **Fixed**: `Log_Overlap` now loops the full `nw·P` pairs on every rank and does **no**
`all_reduce` — each rank computes the complete effective overlap for its **own** walkers, exactly like
`NOMSD::Log_Overlap`. (At the static inner ensemble used here the inner walkers are the replicated anchor,
so each rank scores its walkers against the same trial.) Verified: `driver_smoke` passes at `-np 2`, the
full `-np 1` tag stays 19 / 6657, and the full `-np 2` sweep is **all 19 cases passing**.

### Integration follow-ups (not yet validated)

**Overhaul port — done (Jun 2026):**

- ✅ Ported the static (1a–3a) + 3b `[stochastic_wfn]` cases to `tests/test_wfn_factory.cpp`; built
  `test_afqmc` and ran the full tag on a compute node (now 19 cases, 6657 assertions, `Ne_cc-pvdz`,
  incl. the Phase 3b-var anchor `stochastic_inner_hamiltonian_same_as_true`, the Phase 3c-i smoke
  `stochastic_conditioned_propagator_step`, the Phase 3c-ii leapfrog smoke
  `stochastic_leapfrog_propagator_step`, the two Phase 5 observable cases
  `stochastic_mixed_density_matrix_matches_nomsd` / `stochastic_accumulate_estimators_matches_nomsd`,
  the two Phase 6 mean-field cases `stochastic_mean_field_matches_nomsd` /
  `stochastic_mean_field_production_order`, and the four Phase 7 back-prop cases
  `stochastic_back_propagation_matches_nomsd` /
  `stochastic_back_propagation_production_order` /
  `stochastic_back_propagation_estimator_smoke` /
  `stochastic_back_propagation_driver_smoke`).
- ✅ Fixed the three overhaul-only bugs the port surfaced (log-overlap convention; full-G one-body
  rank mismatch; full-G EXX/EJ slice axis + `dotc`→`dot`).
- ✅ Full-G validated against the compact path on the dense `Real3IndexFactorization` route
  (`stochastic_full_g_matches_compact`). Note: the develop sparse `SparseTensor` full-G path no longer
  exists on overhaul, so only the dense route is exercised.

**Overhaul port — still open:**

- Port the Phase 1a/1b/1c **infrastructure** tests (need new `Wavefunction`-variant accessors).
- **`-np > 1`: validated (Jun 2026) — see [Multi-rank status](#multi-rank--np--1-status-validated-jun-2026-worker6035).** **The full `[stochastic_wfn]` suite passes at `-np 2`** (all 19 cases) after fixing four bugs (stochastic `Energy` double-count; accumulate-test HDF5; `full_g::energy_closed` kernel; stochastic `Log_Overlap` cross-rank reduce). Remaining: GPU and the develop-only Phase 1 infrastructure tests.

**Both code lines (`stochastic-wfn-develop` and `main`; longer term):**

- ✅ Full production driver run with `stochastic: true` and `inner_nsteps > 0` — **Ne cc-pVDZ Stages A/C** (3b-var) and **Stages D/E** (3c-i conditioning / 3c-ii leapfrog; the leapfrog tightens the weight band to deterministic-anchor quality and, with the generalized-`w_p` fix, gives an unbiased measurement energy) (Jun 2026); Stage B unstable at smoke parameters — see [Ne cc-pVDZ driver experiments](#ne-cc-pvdz-driver-experiments-jun-2026).
- Static-limit outer propagator step matching NOMSD at the delegate limit (dynamic path covered by `stochastic_propagator_step`; static delegate parity remains unit-tested per method, not through `Propagate()`).
- Mixed estimator (`MixedObsHandler`) forces and densities remain consistent.
- GPU build: all `[stochastic_wfn]` tests pass with `ENABLE_CUDA=ON`.
- GPU memory and task-group load acceptable with the additional inner ensemble and propagator.
- Research-level: dynamic-trial energies/overlaps vs analytic NOMSD within stochastic error bars; `P → ∞` convergence and `inner_seed` stability; **3c-ii leapfrog** energy parity and variance reduction vs 3b (unit tests are finiteness-only today).
- `KP3IndexFactorization` / other HamOps full-G stubs on overhaul until tests exercise those paths.
