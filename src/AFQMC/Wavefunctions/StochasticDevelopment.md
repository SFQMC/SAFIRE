# StochasticWfn Development Guide

This document tracks the current state of `StochasticWfn` and the methods that must be
overridden for it to function as a full-fledged stochastic trial wavefunction in SAFIRE,
rather than a thin delegate wrapper around `NOMSD`.

---

## Overhaul port status

The stochastic feature was first implemented on **develop** (`stochastic-wfn-phase-3b-dynamic-ensemble`
on `beevus77/SAFIRE`). It is being transplanted onto **`upstream/overhaul`** on branch
`stochastic-wfn-overhaul` (`nda`, `std::variant`, `memory::const_shared_array`, `Log_Overlap`, …).

| | Develop (reference) | Overhaul port (`stochastic-wfn-overhaul`) |
|---|---|---|
| Phases 1a–3b implementation | Complete | **Ported** (as of Jun 2026) |
| `safire_lib` build | Verified | **Compiles** |
| `[stochastic_wfn]` unit tests | Pass on CPU compute node | **Not ported** to `tests/test_wfn_factory.cpp` |
| Runtime / driver smoke | Partial (`stochastic_propagator_step`) | **Not run** |
| GPU build | CPU-only gate in dynamic path | **Not tested** |

**Next gate for the overhaul port:** port the `[stochastic_wfn]` Catch2 cases from develop, build
`test_afqmc`, and run them on a compute allocation. Until that passes, treat all "verified" claims
in this document as **develop-only** unless tagged **[overhaul]**.

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

### Known blocker: `stochastic_wfn_matches_nomsd` SIGSEGV (overhaul port)

**Status: failing on CPU (active investigation).** The keystone delegate-limit test in
`tests/test_wfn_factory.cpp` compiles and links but **SIGSEGVs** before any `CHECK_THAT` NOMSD-parity
assertions run.

**Repro (compute node):**

```bash
HAMIL=$PWD/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
WFN=$PWD/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
mpirun -np 1 ./build/tests/bin/test_afqmc --hamil "$HAMIL" --wfn "$WFN" stochastic_wfn_matches_nomsd
```

**Observed behavior:**

- `is_stochastic_wavefunction` and `stochastic_inner_walkers_initialized` `REQUIRE`s pass.
- Log shows **two** `WalkerSet` init banners (inner ensemble, then outer `wset_nomsd`).
- **No** `Building StochasticWfn inner propagator` line when `inner_nsteps = 0` (propagator build is
  correctly skipped after the delegate-limit fix).
- Catch reports a fatal `SIGSEGV` near the layout-parity / outer-walker setup `REQUIRE`s; the bisected
  test splits NOMSD `Log_Overlap` / `Energy` / `vbias` into stepwise checkpoints to narrow the fault.
- **GDB (RelWithDebInfo):** `RIP = 0`, empty `bt full` on the main thread — stack already destroyed;
  consistent with heap/stack corruption or a bad indirect call, not a clean null deref with intact
  frames. UCX/OpenMPI helper threads are idle; do not trust `catch signal SIGSEGV` in GDB here.

**Fixes already landed on the branch (delegate limit, not sufficient alone):**

- Skip inner `Propagator` construction when `inner_nsteps == 0` (`WavefunctionFactory.cpp`).
- At `inner_nwalkers == 1 && inner_nsteps == 0`, route `Log_Overlap`, `Energy`, `MixedDensityMatrix_for_vbias`,
  and `vbias` through `nomsd_` instead of the inner cross-reduction path (`StochasticWfn.icc`).
- `reduce_inner_cross_dm` uses owned ref buffers and `exp(ov_)` for log-overlap weighting.

**Leading hypothesis:** crash occurs on the **plain NOMSD outer-walker path** (`make_WalkerSet` →
`resize` → `perturb` → `Log_Overlap` / `Energy` / `vbias` on `wfn_nomsd`), *after* stochastic inner
init — not in the stochastic cross-reduction machinery. Confirm with: (1) bisection `REQUIRE` line
from a normal `mpirun` run (no GDB), (2) **AddressSanitizer** build (`-fsanitize=address`), (3) whether
`wfn_factory_sdet` passes on the same `ham_chol_dense` + `wfn_rhf` fixture.

---

## Current State

**Reading guide:** Bullets below describe the **intended** stochastic design and its verification on
develop. Unless tagged **[overhaul]**, "complete" / "CPU-verified" / "tests pass" refers to the
**develop reference branch**, not the current overhaul port.

`StochasticWfn` (`StochasticWfn.hpp` / `StochasticWfn.icc`) is a
`template<MEMORY_SPACE MEM, class devPsiT>` class that inherits from `AFQMCInfo` and owns:

```cpp
StochasticInnerEnsemble inner_ensemble_;  // inner WalkerSet + RNG (walker init only)
int inner_nwalkers_{1};
int inner_nsteps_{0};                     // parsed; > 0 drives inner propagation when Phase 3b is active
NOMSD<MEM, devPsiT> nomsd_;               // outer delegate (outer-walker-facing)
std::unique_ptr<StochasticInnerStack<MEM, devPsiT>> inner_stack_;  // inner NOMSD + Propagator + RNG
```

**Phase 1a is complete [develop]; ported [overhaul], untested.** The inner trial ensemble (`inner_wset()`, sized by `inner_nwalkers`)
is owned and initialized in a two-phase factory/driver workflow.

**Phase 1b is complete [develop]; ported [overhaul], untested.** Outer and inner each have their own `NOMSD` instance with
independently constructed `HamiltonianOperations` and `SlaterDetOperations` (cloned CI and
orbitals for the inner copy).

**Phase 1c is complete [develop]; ported [overhaul], untested.** The inner `NOMSD` (Phase 1b) is wrapped in a heap-allocated
`Wavefunction`, bound to an inner `Propagator` built via `PropagatorFactory` at wavefunction
construction time, with a dedicated device RNG. Default `inner_nsteps = 0`: the propagator
is wired but **not invoked** until Phase 3b enables `inner_nsteps > 0`. The Phase 2a/2b `Log_Overlap` and `Energy` overrides (and the Phase 3a
force bias) reduce the static ensemble at `inner_nsteps = 0`.

**Phase 2a (Overlap) is complete on develop (CPU-verified); ported to overhaul, untested.**
`Log_Overlap` reduces the inner ensemble into an effective stochastic-trial overlap per Eq. 24 of
arXiv:2505.18519 in the static-ensemble limit; see [Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a).
**[develop]** All `[stochastic_wfn]` tests pass on CPU. `stochastic_overlap_matches_nomsd` confirms **inner_nwalkers invariance**
(with `wfn_msd.h5`) and **delegate-limit equality** (`stochastic overlap == NOMSD`, with
`C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5`); see
[Phase 2a-specific tests](#phase-2a-specific-tests-implemented).

**Phase 2b (Energy) is complete on develop (CPU-verified); ported to overhaul, untested.** `Energy`
now overrides `nomsd_`: it reduces the inner ensemble into a per-outer-walker stochastic local
energy (E1, EXX, EJ) and overlap per Eq. 27 of arXiv:2505.18519 in the static-ensemble limit; see
[Stochastic local energy (Phase 2b)](#stochastic-local-energy-phase-2b). The returned `Ov[w]`
is the same reduction as the Phase 2a `Overlap` (the same per-pair terms, so equal up to
floating-point summation order — `stochastic_energy_matches_nomsd` asserts the agreement), and at
the single-determinant delegate limit both `E` and `Ov` reproduce `NOMSD` exactly. The dedicated
`stochastic_energy_matches_nomsd`
test mirrors `stochastic_overlap_matches_nomsd`; `stochastic_wfn_matches_nomsd` now gates its
energy/overlap assertions on `ndet == 1` (the delegate limit) since the override diverges from
NOMSD for multi-determinant trials by design. `stochastic_energy_matches_nomsd` confirms
**inner_nwalkers invariance** (with `wfn_msd.h5`) and **delegate-limit equality** (stochastic
energy/overlap == NOMSD at `ndet == 1`, with `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5`); see
[Phase 2b-specific tests](#phase-2b-specific-tests-implemented).

**Phase 3a (mixed DM + force bias) is complete on develop (CPU-verified); ported to overhaul, untested.**
`MixedDensityMatrix_for_vbias`
now overrides `nomsd_`: it reduces the inner ensemble into the per-outer-walker mixed density matrix
the force bias contracts against, `G[w] = Σ_p ⟨ψ_p|c†c|φ_w⟩ / Σ_p ⟨ψ_p|φ_w⟩` (estimator 3 of
arXiv:2505.18519, static-ensemble limit), returned in the outer `nomsd_` `vbias` layout; see
[Stochastic mixed DM + force bias (Phase 3a)](#stochastic-mixed-dm--force-bias-phase-3a). It mirrors
`NOMSD::MixedDensityMatrix_shared`'s accumulate-then-normalize with the inner-walker loop replacing
the trial-determinant loop and `1/P` replacing `conj(ci[nd])`, reusing the Phase 2b
`DensityMatrix(…, herm = false)` cross-DM machinery on the **outer (True Ham)** delegate. `vbias`
stays a **permanent delegate** to `nomsd_.vbias` — now fed the stochastic `G` — because the `L·G`
contraction is trial-independent given `G`. With `vHS` also a permanent delegate, **every Tier 1
hot-path method now has a stochastic implementation** (each unit-verified against NOMSD at the
delegate limit), so no method still needs overriding for the static-limit propagator. (End-to-end
outer propagation with a **dynamic** stochastic trial is covered by **`stochastic_propagator_step`**
(Phase 3b): real `AFQMCBasePropagator::Propagate()` on `inner_nsteps = 1`, asserting finite
weights/energies/overlaps — not NOMSD parity; see [Phase 3b-specific tests](#phase-3b-specific-tests-implemented).)
The dedicated
`stochastic_vbias_matches_nomsd` test mirrors
`stochastic_energy_matches_nomsd` (inner_nwalkers invariance with `wfn_msd.h5`; delegate-limit
`G`/force-bias equality vs NOMSD at `ndet == 1` with `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5`);
`stochastic_wfn_matches_nomsd` now gates its `MixedDensityMatrix_for_vbias`/`vbias` assertions on
`ndet == 1` too (the override diverges from NOMSD for multi-determinant trials by design).
**[develop]** All `[stochastic_wfn]` tests pass on CPU. The remaining Tier 1 hot-path method `vHS` is trial-independent
(`v = √Δτ·L·X`) and **delegates permanently** — it needs no override.

**All other Tier 2–6 visitor methods (and `vHS`) still delegate to `nomsd_` (outer) on outer
walkers**, so outer-facing behavior remains identical to plain `NOMSD` in the **single-determinant**
delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) for those methods. The full Tier 1
propagator hot path is now stochastic: hybrid propagation uses the Phase 2a `Overlap` override,
local-energy propagation the Phase 2b `Energy` override, and the force bias the Phase 3a
`MixedDensityMatrix_for_vbias` override (`vbias`/`vHS` stay trial-independent delegates).

Inner `NOMSD` is evaluated directly on `inner_wset()` via `inner_wfn()` / `inner_nomsd()` for
testing and future overrides. The inner propagator is accessed via `inner_propagator()` once
Phase 3b begins driving inner evolution.

**Phase 3b (dynamic ensemble + phase weights) is implemented on develop (CPU-verified); ported to overhaul, untested.**
**[develop]** All `[stochastic_wfn]` tests pass on a compute node with both `wfn_rhf.h5` (CLOSED) and `wfn_msd.h5`.
**[overhaul]** `energy_fullG` is wired through `full_g_estimators.hpp` and `Real3IndexFactorization`
(other HamOps types carry `energy_fullG` stubs until tests exercise them). The **full-G force bias has
no separate seam**: `Real3IndexFactorization::vbias` dispatches on the **G layout** (compact vs full
NMO×NMO) rather than the trial determinant count, so the un-rotated full-G contraction is reached
through the ordinary `vbias` / `vbias_from_G` path for a single-determinant stochastic trial. (The
former `vbias_fullG` seam — a near-duplicate of `vbias`'s `ndet>1` branch — was removed.)
`inner_nsteps > 0` now drives the inner propagator in **free-projection** mode: each outer step the
inner ensemble is reset to the anchor and advanced `inner_nsteps` `B̂_T` steps (`begin_inner_step()` +
`maybe_advance_inner_ensemble()`), the reductions use the Eq. 27 `S_p` phase weights (with `Overlap`
kept absolute), the shared cross-DM loop is factored into `reduce_inner_cross_dm`, and the True-Ham
energy for moved walkers uses the **un-rotated full-G contraction** (`energy_fullG`, CLOSED trials only
this phase); the force bias contracts the same full-G through `vbias`. The **Variational Hamiltonian `Ĥ_var` is deferred** — the
inner stack stays a True-Ham clone (`B̂_T` generated by the cloned True Cholesky). At `inner_nsteps = 0`
behavior is unchanged (Phases 1c–3a). See [Phase 3b](#phase-3b--dynamic-ensemble--phase-weights-implemented-cpu-ĥ_var-deferred).

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

## Dual-Hamiltonian architecture (True vs Variational) — the end goal

A full `StochasticWfn` carries **two** Hamiltonians, with two distinct jobs. This is the target
architecture; the current code is a static-limit placeholder that only ever sees one of them (see
[the placeholder note](#current-placeholder-one-hamiltonian-cloned) below).

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
**Phase 2b `Energy` already follows this** (its contraction was switched from `inner_nomsd()` to
`nomsd_` ahead of Phase 3b — behavior-preserving while the inner Ham is a True clone); the Phase 3a
`vbias` will do the same.

### Current placeholder: one Hamiltonian, cloned

Today the inner `NOMSD` is a **clone of the outer (True) Hamiltonian** (Phase 1b), and the inner
propagator is dormant (`inner_nsteps = 0`). In the static limit `B̂_T = 𝟙`, so `Ĥ_var` is never
exercised and the clone is harmless: Phase 2/3a are correct as static placeholders regardless. The
clone becomes wrong only when Phase 3b swaps the inner `HamOps` to `Ĥ_var`; at that point the inner
vs outer layout-parity tests (which assume equal Cholesky counts) no longer hold — they are
static-phase scaffolding.

---

## Phase 2 overview: `Overlap` (2a) vs `Energy` (2b)

Phase 2 splits the first stochastic reductions on the propagator hot path into two independent
overrides. They share the same static-ensemble limit (`inner_nsteps = 0`, `B̂_T = 𝟙`) and the
same inner/outer walker geometry, but they answer different questions and integrate with
different propagator modes.

| | **Phase 2a — `Overlap`** | **Phase 2b — `Energy`** |
|---|--------------------------|-------------------------|
| **Status** | **Complete [develop]**; ported [overhaul], untested | **Complete [develop]**; ported [overhaul], untested |
| **Quantity** | Effective trial overlap `Ov[w]` per outer walker | Local energy `E[w]` (E1, EXX, EJ) and overlap `Ov[w]` per outer walker |
| **Reduction** | Cross overlaps `(1/P) Σ_p ⟨ψ_p\|φ_w⟩` between inner walkers `ψ_p` and outer walkers `φ_w` | `E[w] = Σ_p ⟨ψ_p\|Ĥ\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩` from the same inner/outer cross pairs; `Ov[w]` identical to 2a |
| **Propagator consumer** | **Hybrid** path: `wfn.Overlap(wset, new_overlaps)` in `AFQMCBasePropagator::step` | **Local-energy** path: `wfn.Energy(wset, new_energies, new_overlaps)` |
| **Implementation** | `StochasticWfn.icc` — uses outer `SDetOp`, `FairDivideBoundary` over `nw×P` pairs | `StochasticWfn.icc` — mirrors `NOMSD::Energy_shared`, looping inner walkers as the determinant index (weight `1/P`); the cross DM + `energy_from_G` run on the **outer `nomsd_` (True Ham)** with inner-walker bras |
| **Phase 3 coupling** | Absolute overlap only; the `S_p` phase weight (Eq. 26) arrives in Phase 3b and the importance-sampling leapfrog (Eq. 25) in Phase 3c | Same, plus the per-inner-walker `nd` half-rotation in Phase 3b once `ψ_p ≠` anchor |

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
  transient overlap vector in the `Overlap(wset)` overload (mirrors NOMSD). The outer `nomsd_`'s
  `SlaterDetOperations` and `TG_` (= outer `TGwfn`) are reused.

### Why the visitor `Overlap` is overridden directly, and what stays in Phase 3

The hybrid propagation path (`AFQMCBasePropagator::step`) consumes the overlap **only as a step-to-step
ratio** `ratioOverlaps = new/old` in `hybrid_walker_update`: its argument drives the cosine (phase)
constraint and its log feeds the hybrid energy, and the stored `OVLP` property is overwritten each step.
This is exactly Eq. 25's structure, so overriding the visitor `Overlap` is the correct integration point.
**The exact `𝒩(φ)` cancellation of Eq. 25, however, requires numerator and denominator to share the inner
field samples conditioned on the old walker** — a coupling that only exists in the paper's
propagate-then-resample *leapfrog*, which lives in the propagator hot path. So the phase factor `S_p`
(Phase 3b), and walker-conditioned importance sampling + the leapfrog (Phase 3c), are deferred;
`Overlap` itself just produces the absolute reduction above, which is exact at `inner_nsteps = 0`.

### Delegate limit and the multi-determinant caveat

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`, inner walker = the
trial determinant) the reduction is exactly `⟨φ_T|φ_w⟩` = the NOMSD overlap. **This holds only for
single-determinant trials** (RHF/UHF files): the paper's anchor `|φ_T⟩` is a single determinant, and the
inner walkers are initialized from `getInitialGuess` (`Psi0`, the dominant determinant). For a
multi-determinant trial (e.g. `wfn_msd.h5`) the single-determinant inner ensemble cannot reproduce the
CI-weighted NOMSD overlap, so the overridden `Overlap` diverges from NOMSD there by design. The dedicated
`stochastic_overlap_matches_nomsd` test exercises `Overlap` in isolation and gates the NOMSD-equality
assertion on `ndet == 1`; the `stochastic_wfn_matches_nomsd` anchor likewise gates its overlap/energy
assertions on `ndet == 1` (since Phase 2b, `Energy` is also a stochastic override that diverges for
multi-determinant trials — it no longer delegates to `NOMSD`).

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
absolute overlap; the phase weights `S_p` arrive in Phase 3b and the `𝒩` cancellation (leapfrog)
in Phase 3c.

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
— together with the propagate-then-resample *leapfrog* that makes the `𝒩(φ_w)` cancellation in the
Eq. 25 ratio exact — is what Phase 3c adds (on top of the dynamic, walker-independent ensemble of
Phase 3b). Phases 2a/2b/3a implement the `B̂_T = 𝟙` static limit, where the ensemble is
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
  which are tied to that determinant's bra orbitals. In the static-ensemble limit every inner
  walker equals the inner trial's anchor determinant (`getInitialGuess == OrbMats[0]`), so `nd = 0`
  is the correct (and only valid) half-rotation. This single-determinant assumption is what
  **Phase 3b must revisit** once `inner_nsteps > 0` propagates the inner walkers away from the anchor.
- The reduction runs through the **outer `nomsd_`** (the True Hamiltonian), which shares `TG_` so the
  `DensityMatrix` / `energy_from_G` barriers stay consistent. A `LocalTGBufferManager
  shm_buffer_manager` member backs the shared `G` buffer; per-core energies are reduced with
  `all_reduce` over `TG_local`, then `E /= Ov`.
- **✓ Dual-Hamiltonian sourcing (revisit applied).** The contraction is scored against the **True**
  Hamiltonian via the outer `nomsd_`, *not* `inner_nomsd()` — see
  [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational--the-end-goal).
  This was switched from `inner_nomsd()` ahead of Phase 3b: it is behavior-preserving today (the inner
  NOMSD is still a True-Ham clone) but becomes a correctness requirement once Phase 3b makes the inner
  `HamOps` the Variational `Ĥ_var` — at which point `inner_nomsd().energy_from_G` would return the
  wrong operator. The cross **density matrix** is Hamiltonian-independent, so running it on `nomsd_`
  too just keeps Overlap (2a) and Energy (2b) on the same `SDetOp`. The Phase 3a `vbias` contraction
  follows the same rule.
- **Distributed Cholesky (`NGroupsPerTG > 1`) is guarded with `APP_ABORT`**: a single `energy_from_G`
  sees only one group's Cholesky vectors. The ring reduction (`NOMSD::Energy_distributed`) is a
  later phase; Phase 2b targets the shared limit (and the CPU test runs single-group).

### Delegate limit and the multi-determinant caveat

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`, inner walker =
the trial determinant) the reduction is exactly `⟨φ_T|Ĥ|φ_w⟩/⟨φ_T|φ_w⟩` = the NOMSD local energy,
and `Ov[w] = ⟨φ_T|φ_w⟩` = the NOMSD overlap. As with Phase 2a `Overlap`, this holds **only for
single-determinant trials**: a single-determinant inner ensemble cannot reproduce the CI-weighted
NOMSD energy, so for a multi-determinant trial (e.g. `wfn_msd.h5`) the overridden `Energy` diverges
from NOMSD by design. `stochastic_energy_matches_nomsd` gates the NOMSD-equality assertion on
`ndet == 1`; `stochastic_wfn_matches_nomsd` likewise gates its energy/overlap assertions on
`ndet == 1` (and, since Phase 3a, its `MixedDensityMatrix_for_vbias` / `vbias` comparisons too).

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

### Deferred to Phase 3b/3c (same as 2a)

Phase factor `S_p` (Eq. 26) arrives in **Phase 3b**; walker-conditioned importance sampling and the
propagate-then-resample leapfrog (Eq. 25) in **Phase 3c**. Phase 2b deliberately does **not** invoke
inner propagation (it reduces the static ensemble only), so the `inner_nsteps > 0` parse guard and
the `nd = 0` static-anchor half-rotation stay until **Phase 3b** drives `inner_propagator()` and
rebuilds the half-rotation per propagated inner walker.

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
- **✓ Dual-Hamiltonian sourcing.** Like the Phase 2b energy, the cross DM is run on the **outer `nomsd_`
  (True Ham)**, not `inner_nomsd()`. The cross DM is Hamiltonian-independent, so this just keeps Overlap
  (2a), Energy (2b), and the vbias DM (3a) on the same `SDetOp`; it becomes a correctness requirement at
  Phase 3b when the inner `HamOps` becomes the Variational `Ĥ_var`. The `nd = 0` static-anchor assumption
  is implicit: at `inner_nsteps = 0` every inner walker equals the trial anchor (`getInitialGuess ==
  OrbMats[0]`), so the compact DM's bra basis matches `nomsd_`'s half-rotated Cholesky (`vbias` carries no
  per-determinant `nd`). Phase 3b must revisit this once inner walkers leave the anchor.
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

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`, inner walker = the
trial determinant) the reduction is exactly the NOMSD mixed DM and force bias. As with Phases 2a/2b, this
holds **only for single-determinant trials**: a single-determinant inner ensemble cannot reproduce the
CI-weighted NOMSD DM, so for a multi-determinant trial (e.g. `wfn_msd.h5`) the override diverges from NOMSD
by design. `stochastic_vbias_matches_nomsd` gates the NOMSD-equality assertions on `ndet == 1`;
`stochastic_wfn_matches_nomsd` likewise now gates its `MixedDensityMatrix_for_vbias` / `vbias` assertions
on `ndet == 1`.

### Deferred to Phase 3b/3c (same as 2a/2b)

Phase factor `S_p` (Eq. 26) arrives in **Phase 3b**; walker-conditioned importance sampling and the
propagate-then-resample leapfrog (Eq. 25) in **Phase 3c**. Phase 3a reduces the static ensemble only, so
the `inner_nsteps > 0` parse guard and the `nd = 0` static-anchor assumption stay until **Phase 3b**.

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
| `inner_seed` | `777` | 1c | Seed for the inner propagator device RNG. `0` selects a time-based seed (same convention as the driver `seed`). Rank-decorrelated via `split_seed`. |
| `inner_propagator` | *(optional subtree)* | 1c/3b | Propagator input block for `PropagatorFactory`. If omitted, factory defaults apply. `system` and `name` are injected when missing (`name` suffix `_inner_propagator`). At `inner_nsteps > 0` the free-projection mode flags are forced; `timestep` (default `0.01`) sets the `B̂_T` step `dt`. |
| `inner_hamiltonian` *(deferred)* | *(required for Ĥ_var)* | 3b-var | Handle/filename for the pre-optimized **Variational** Cholesky Hamiltonian `Ĥ_var` that defines `B̂_T` (same basis/NMO as the True Ham). **Not yet parsed/built** — Phase 3b runs with the inner stack as a True-Ham clone; introducing `Ĥ_var` (a second `Hamiltonian` build + a variational Ham HDF5 file) is the deferred follow-up. See [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational--the-end-goal). |

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
| `NomsdHamOpsPair` / `makeOuterInnerHamOps()` | Two independent `getHamOps()` builds (in-place brace init; `HamOps` is move-only). **Today both are built from the True Hamiltonian (a clone); at Phase 3b the inner build switches to the Variational Hamiltonian** (read from `variational_hamiltonian`). |
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

> **Current and future expectation:** `StochasticWfn` owns **two** engines (see
> [Dual-Hamiltonian architecture](#dual-hamiltonian-architecture-true-vs-variational--the-end-goal)):
>
> - **Outer `nomsd_` (True Hamiltonian)** — scores every returned quantity and faces the outer driver.
> - **Inner AFQMC stack (Variational Hamiltonian)** — generates the stochastic-trial samples:
>   - **Walker set** — stochastic ensemble defining the trial wavefunction *(Phase 1a ✓)*
>   - **NOMSD** — single-determinant anchor + its own `HamOps`/`SDetOp`. Currently a True-Ham clone
>     *(Phase 1b ✓)*; **becomes the Variational `Ĥ_var` engine at Phase 3b**.
>   - **Propagator** — built and bound at construction; applies `B̂_T` (a free-projection step under
>     `Ĥ_var`) once Phase 3b enables `inner_nsteps > 0` *(Phase 1c ✓ infrastructure; invocation deferred)*
> - **Effective outer quantities** — overlap *(Phase 2a ✓)*, local energy *(Phase 2b ✓)*, mixed DM /
>   force bias *(Phase 3a ✓)*, all reduced from the inner ensemble and scored against the True Ham.
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

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `MixedDensityMatrix_for_vbias(wset, G)` | **Phase 3a ✓ (static limit):** reduces the per-pair cross mixed DMs over the inner ensemble, `G[w] = Σ_p ⟨ψ_p\|c†c\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩` (estimator 3 of arXiv:2505.18519 with `B̂_T = 𝟙`), returned in the outer `nomsd_` `vbias` layout (`compact_G_for_vbias`, `transposed_G_for_vbias_`); built via `nomsd_.DensityMatrix(…, herm=false)` on the **outer (True Ham)** delegate, as in Phase 2b. Matches NOMSD at the single-determinant delegate limit; diverges for multi-determinant trials by design. See [Stochastic mixed DM + force bias (Phase 3a)](#stochastic-mixed-dm--force-bias-phase-3a). | **Phase 3b:** replace `1/P` with the phase factor `S_p` (Eq. 26), drive inner propagation, and add the per-inner-walker `nd` half-rotation once `ψ_p ≠` anchor. **Phase 3c:** the leapfrog (Eq. 25). |
| `vbias(G, v, dt, a)` | **Phase 3a ✓:** permanent delegate to `nomsd_.vbias`, now fed the stochastic `G[w]` from `MixedDensityMatrix_for_vbias` — the `L·G` contraction (estimator 4, `x̄_γ[w] = L_γ · G[w]`) is trial-independent given `G`, scored against the **True Ham**, with the `nd = 0` static-anchor half-rotation (no per-determinant `nd` in `vbias`). The `√Δτ`/timestep prefactor stays in the propagator. | No override needed; consumes the Phase 3b/3c stochastic `G` unchanged. |
| `vHS(X, v, dt, a)` | Delegates to `nomsd_.HamOp.vHS`. | **No override — permanent delegate.** `vHS` is trial-independent: `v = √Δτ · Σ_γ X_γ L_γ` uses only the auxiliary fields `X` and the (trial-independent) Cholesky tensor `L` (no half-rotation, no inner-ensemble quantity). This is why Phase 3a completes the static propagator hot path. |
| `Overlap(wset)` / `Overlap(wset, Ov)` | **Phase 2a ✓ (static limit):** `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩`, a cross overlap between inner walkers `ψ_p` and each outer walker `φ_w` (Eq. 24 of arXiv:2505.18519 with `B̂_T = 𝟙`). Matches NOMSD at the single-determinant delegate limit; diverges for multi-determinant trials by design. Consumed by **hybrid** propagation. | **Phase 3b:** replace `1/P` with the phase factor `S_p` (Eq. 26) and drive inner propagation (relax the `inner_nsteps > 0` guard). **Phase 3c:** the importance-sampled propagate-then-resample leapfrog for the exact Eq. 25 `𝒩(φ)` cancellation. |
| `Energy(wset)` / `Energy(wset, E, Ov)` | **Phase 2b ✓ (static limit):** `E[w] = Σ_p ⟨ψ_p\|Ĥ\|φ_w⟩ / Σ_p ⟨ψ_p\|φ_w⟩` (E1, EXX, EJ) and `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩`, the same inner/outer cross pairs as Phase 2a (Eq. 27 of arXiv:2505.18519 with `B̂_T = 𝟙`). `Ov` matches the Phase 2a `Overlap`; matches NOMSD at the single-determinant delegate limit, diverges for multi-determinant trials by design. Consumed by **local-energy** propagation. | **Phase 3b:** `S_p` phase weight (Eq. 26) and inner propagation, including the **per-inner-walker** `nd` half-rotation once `ψ_p ≠` anchor (replacing the `nd = 0` static-anchor assumption); relax the `inner_nsteps > 0` and `NGroupsPerTG > 1` guards. **Phase 3c:** the leapfrog (Eq. 25). |

---

### Tier 2 — Mixed estimators and observable handlers

Called from `MixedObsHandler`, `FullObsHandler`, force estimators, and related code paths.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `MixedDensityMatrix(wset, G, ...)` | Delegates to `nomsd_`; full analytic mixed DM for outer walkers. | Stochastic mixed DM for **observable evaluation** (may differ in layout/options from the `for_vbias` variant). Average inner ensemble contributions; support `compact` and `transpose` flags as today. |
| `MixedDensityMatrix(wset, G, Ov, ...)` | Same as above, also returns overlaps per determinant/walker. | As above, plus return the overlap vector needed for weighted accumulation in multi-determinant or multi-reference estimators. |
| `DensityMatrix(wset, RefA, RefB, G, Ov, ...)` | Delegates to `nomsd_`; DM w.r.t. a specific reference determinant. | Stochastic DM relative to a chosen reference orbital set. Needed by `MixedObsHandler` for reference-resolved force and density estimators. |
| `accumulate_estimators(...)` | Delegates to `nomsd_`; analytic back-prop / correlated accumulation. | Accumulate estimator contributions using **stochastic** DMs and energies from the inner ensemble. Must remain consistent with `TimeEvolvedObsHandler` call signature. |

---

### Tier 3 — Mean-field subtraction

Called during propagator setup and mean-field initialization.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `G_MF(G)` | Delegates to `nomsd_`; builds mean-field Green's function from CI-weighted determinant DMs. | Build the **stochastic** mean-field Green's function used to subtract the trial mean-field potential. Average inner-walker MF contributions from the owned inner stack. |
| `vMF(v, dt)` | Delegates to `nomsd_`; mean-field contribution of Cholesky vectors. | Compute mean-field bias from the stochastic trial. May require inner `HamOps` and inner walker sampling. |

---

### Tier 4 — Layout and metadata queries

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
| `generalizedFockMatrix(...)` | Delegates to outer `HamOp`. | Used by `generalizedFockMatrix` observable; must use stochastic DM inputs (Phase 5) against the True Ham. |
| `getOneBodyPropagatorMatrix(...)` | Delegates to outer `HamOp`. | **Keep on `nomsd_` (True Ham)** — the outer propagator's one-body matrix. (`B̂_T`'s one-body matrix comes from `Ĥ_var` inside the inner stack.) |
| `vHS_sparse(...)` | Delegates to outer `HamOp`. | Sparse `vHS` of the **True** Ham for the outer propagator (`vHS` itself is a permanent delegate — Phase 4). |
| `getSlaterDetOperations()` | Returns `nomsd_` SlaterDetOperations pointer. | **Keep returning the outer `nomsd_` SDetOp** — it is Hamiltonian-independent (orbital algebra), so the reductions can use it for the cross DM regardless of which Ham scores the contraction. |

---

### Tier 6 — Back propagation

Called from `BackPropagatedEstimator` and related reference-tracking code.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `number_of_references_for_back_propagation()` | Returns `nomsd_` reference count. | Number of reference Slater matrices the stochastic trial exposes for back-propagation (may differ if inner ensemble uses a reduced reference set). |
| `getReferenceWeight(i)` | Returns `nomsd_.ci[i]`. | CI weight (or effective stochastic weight) for reference `i`. |
| `getReferencesForBackPropagation(A)` | Copies reference orbitals from `nomsd_`. | Provide reference Slater matrices for back-propagation, possibly averaged or selected from the inner ensemble. |

---

### Tier 7 — Construction and factory (infrastructure)

Not wavefunction visitor methods, but required for a full-fledged type.

| Component | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `StochasticWfn` constructor | Builds outer `nomsd_`; accepts pre-built `StochasticInnerStack` (inner NOMSD + propagator + RNG). Parses `inner_nwalkers`, `inner_nsteps`, `inner_seed`, `inner_propagator`. Defers inner `WalkerSet` resize. | **Build two `HamOps`** (Phase 3b): the True Ham for `nomsd_` and the **Variational** Ham for the inner stack — the factory must read the second Cholesky Hamiltonian (`variational_hamiltonian`) and build the inner stack around it instead of cloning the True Ham. Plus population control, first-class HDF5 type (Phase 8). |
| `interpret_inputs(pt)` | Validates NOMSD keys plus all stochastic keys listed above. Rejects `inner_nsteps > 0`. | Relax the `inner_nsteps` guard when Phase 3b invokes `inner_propagator()`. |
| `WavefunctionFactory` | `stochastic: true` on NOMSD HDF5 path; `buildStochasticInnerStack()` + `buildStochasticNomsdWavefunction*`; `maybe_initialize_stochastic_inner_walkers()` after build. | First-class `stochasticwfn` type with its own `fromHDF5` branch (Phase 8). |
| `getWavefunctionType()` | Not aware of `StochasticWfn`. | Detect stochastic trial wavefunction files on disk. |

---

## Suggested Implementation Order

Step 1 in the original plan ("own the full inner stack") bundles four heavyweight concerns —
walker set, `NOMSD`, `HamOps`, and propagator — each with its own factory wiring and test
surface. **Do not implement Step 1 as a single monolith.** Decompose it into the phases below.

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
> [dual-Hamiltonian end goal](#dual-hamiltonian-architecture-true-vs-variational--the-end-goal) it
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

**[develop] Verified:** factory constructs without error; inner propagator bound to inner `Wavefunction`;
Cholesky layout matches inner `NOMSD`; default hybrid propagation settings; all `[stochastic_wfn]`
tests pass on CPU build with `wfn_msd.h5`.

**Still deferred (post-1c):** invoking `inner_propagator()` from overrides and enabling
`inner_nsteps > 0` (Phase 3b). *(Note: under the dual-Hamiltonian design, outer-facing layout/SDetOp
queries deliberately stay on `nomsd_` (True Ham) — they are not rerouted to the inner stack; the
old "Phase 7 reroute" idea is superseded. See [Tier 4](#tier-4--layout-and-metadata-queries).)*

### Phase 2a — Stochastic `Log_Overlap` (**complete [develop]**; ported [overhaul], untested)

**Goal:** Override `Overlap` to reduce the inner ensemble into an effective stochastic-trial
overlap per outer walker. First Tier 1 method to diverge from the `nomsd_` delegate.

| Item | Status |
|------|--------|
| Method | `Overlap(wset)` / `Overlap(wset, Ov)` in `StochasticWfn.icc` |
| Reduction | `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩` (Eq. 24, static limit); see [Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a) |
| Infrastructure | `DeviceBufferManager buffer_manager`; outer `SDetOp` + `TG_`; `FairDivideBoundary` over `nw×P` |
| Propagator | **Hybrid** mode consumes this override directly |
| Tests | `stochastic_overlap_matches_nomsd` |

**[develop] Verified (CPU):** all `[stochastic_wfn]` tests pass. `stochastic_overlap_matches_nomsd`:
- **inner_nwalkers invariance** — `wfn_msd.h5` (`inner_nwalkers = 1` vs `= 3`, static replicated ensemble);
- **delegate limit** — `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5` (`stochastic overlap == NOMSD` at `ndet == 1`).

### Phase 2b — Stochastic `Energy` (**complete [develop]**; ported [overhaul], untested)

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

**[develop] Verified (CPU):** all `[stochastic_wfn]` tests pass. `stochastic_energy_matches_nomsd`:
- **inner_nwalkers invariance** — `wfn_msd.h5` (`inner_nwalkers = 1` vs `= 3`, static replicated ensemble);
- **delegate limit** — `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5` (stochastic E1/EXX/EJ/energy and overlap == NOMSD at `ndet == 1`);
- **Overlap/Energy consistency** — `Overlap(wset)` agrees with the overlap from `Energy(wset, E, Ov)` within `Approx`.

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
`stochastic_propagator_step`). The remaining Phase 3 work
(3b/3c) turns the static ensemble dynamic.

#### Phase 3a — Static stochastic mixed DM + force bias (**complete [develop]**, CPU-verified; **ported [overhaul], untested**)

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
| Tests | ✓ `stochastic_vbias_matches_nomsd` (mirrors `stochastic_energy_matches_nomsd`): `inner_nwalkers` invariance (`wfn_msd.h5`) + delegate-limit `G` / `vbias` (`x̄`) equality vs NOMSD (`ndet == 1`, `wfn_rhf.h5`). `stochastic_wfn_matches_nomsd` gates its `MixedDensityMatrix_for_vbias` / `vbias` assertions on `ndet == 1`. |
| Guards | `inner_nsteps > 0` stays rejected (still static); `NGroupsPerTG > 1` aborts in `MixedDensityMatrix_for_vbias`, deferred as in 2b. |

#### Phase 3b — Dynamic ensemble + phase weights (**complete [develop]**, CPU-verified; **ported [overhaul], untested**; Ĥ_var deferred)

**Goal:** make the inner ensemble genuinely stochastic by turning on `B̂_T`: the inner propagator
field-samples `{ψ_p = B̂_T(Y^[p])\|φ_T⟩}` (free projection), every reduction upgrades from `(1/P) Σ_p`
to the phase-weighted average of Eq. 27, and the True-Ham scoring is made correct for inner walkers
that have left the anchor. Reductions stay **scored against the True Hamiltonian** (outer `nomsd_`).

**Scope landed this phase ([develop], CPU-verified on a compute node):** dynamic free-projection sampling, the
`S_p` phase weights, the `reduce_inner_cross_dm` refactor, the un-rotated full-G True-Ham scoring, and
end-to-end outer propagator integration (`stochastic_propagator_step`). **Deferred:** the **Variational Hamiltonian `Ĥ_var`** — the inner stack stays a True-Ham clone, so
`B̂_T` is generated by the cloned True Cholesky; the machinery is fully exercised, only the generator differs
from the paper's intent. Building `Ĥ_var` needs new factory plumbing for a *second* `Hamiltonian` (the
wavefunction factory has only one) **and** a variational-Hamiltonian HDF5 file that does not exist in the
repo — a follow-up "Phase 3b-var". The deferred input key is **`inner_hamiltonian`**.

| Item | Status |
|------|--------|
| Sampling | ✓ Inner propagator forced to **free-projection** mode (`buildStochasticInnerStack`) when `inner_nsteps > 0`; the `inner_nsteps > 0` guard is relaxed. Each outer step the ensemble is **reset to the anchor `\|φ_T⟩` then advanced `inner_nsteps` free-projection `B̂_T` steps** (`maybe_advance_inner_ensemble`), driven by a per-step latch `begin_inner_step()` armed at the top of **every propagator `step()` entry point** (`AFQMCBasePropagator`, `AFQMCModelPropagator` dense/sparse, and the distributed variants — no-op for non-stochastic trials). Reset-then-resample (not a continuous free walk): the single-step trial is `B̂_T` applied to the anchor; recovers 3a exactly at `inner_nsteps = 0`. The `B̂_T` `dt` is `inner_propagator.timestep` (default 0.01); the inner propagator rebuilds its one-body/CV scaling for this `dt` on first `Propagate` (so it is the single effective source — no build-vs-runtime drift). |
| Fail-fast / limits | ✓ `inner_nsteps > 0` is rejected **at `StochasticWfn` construction** unless the trial is **CLOSED (RHF)** and the build is **CPU** (`!ENABLE_DEVICE`) — the un-rotated full-G kernels are CLOSED/CPU-only this phase, so misuse fails immediately, not deep in the first reduction's HamOp dispatch. |
| Phase weight | ✓ `⟨Ô⟩_w = Σ_p ⟨Ô⟩_{p,w} S_p / Σ_p S_p` with `S_p[w] = ⟨ψ_p\|φ_w⟩/\|⟨ψ_p\|φ_w⟩\|` (Eq. 26) in `Energy`/`MixedDensityMatrix_for_vbias`; degenerate `S_p` recovers 3a. **`Overlap` stays the absolute `(1/P) Σ_p ⟨ψ_p\|φ_w⟩`** (Eq. 24's normalization) — dropping the magnitude would break `Overlap == NOMSD` at the delegate limit, and the exact `𝒩(φ)` cancellation is the Phase 3c leapfrog. (The earlier Tier-1 "Overlap: replace 1/P with `S_p`" note was imprecise.) |
| Refactor | ✓ The shared per-inner-walker cross-DM loop is now a private `reduce_inner_cross_dm(wset, compact, transposed, Gsize, D, Ov, accumulate)` (the callback folds `G_{p,w}` into each estimator's `S_p`-weighted numerator); it also owns the `maybe_advance_inner_ensemble()` resample. `Overlap` keeps its own `nw×P` FairDivide loop. |
| Half-rotation | ✓ **[develop]** Resolved via the **un-rotated full-G contraction** (not per-walker re-rotation): once `ψ_p ≠` anchor the reductions form the **full** NMO×NMO cross `G` and contract it with the full (un-rotated) Cholesky/bare `hij`. The EXX/EJ/E1 kernel is a single shared `full_g::energy_closed` (`HamiltonianOperations/full_g_estimators.hpp`, reusing the proven `energy_impl` structure with an identity-rotated Cholesky) used by **both** Cholesky HamOps: `Real3IndexFactorization` (dense; `Likn` already dense) and **`SparseTensor`** (sparse; `Likn` densified once via `Matrix2MA`, as `getHSPotentials` does). **The CPU test Hamiltonian `ham_chol_sc.h5` is `FactorizedSparseHamiltonian` → `SparseTensor`, so SparseTensor is the path the develop tests exercise.** Both `energy_fullG` and `vbias_fullG` need the full path (the single-determinant `vbias`/energy are half-rotated). `ma_rotate::getLank` could not be reused (it reinterprets a real `Likn` as complex). Wired through NOMSD seams `energy_from_fullG`/`vbias_fullG` and HamOps dispatch. **[overhaul]** `Real3IndexFactorization` implements the full path; `THCOps`, `ModelHamOps`, `KPTHCOps`, and `KP3IndexFactorization` carry `energy_fullG` stubs pending test coverage. **The full-G force bias has no separate `vbias_fullG` seam on overhaul** — `Real3IndexFactorization::vbias` dispatches on the G layout (compact vs full NMO×NMO), so the un-rotated full-G contraction is reached through the ordinary `vbias`/`vbias_from_G` path; only `energy_fullG` remains a dedicated seam. The full-G energy G is requested as `[nwalk][NMO*NMO]` (transposed) uniformly so the kernel's layout is HamOp-independent. **CLOSED (RHF) trials only this phase** (rejected at `StochasticWfn` construction otherwise; also CPU-only); COLLINEAR/NONCOLLINEAR full-G are a follow-up. |
| Tests | ✓ **[develop]** **`stochastic_full_g_matches_compact`** — full-G at the anchor == compact `nd = 0` == NOMSD (new-kernel validation). ✓ **`stochastic_dynamic_ensemble_smoke`** — resample + **all four** stochastic overrides (`MixedDensityMatrix_for_vbias`/`vbias_fullG`, `Energy`/`energy_fullG`, `Overlap`) on the moved ensemble; asserts finite. ✓ **`stochastic_propagator_step`** — real OUTER `AFQMCBasePropagator::Propagate()` on a dynamic trial (`inner_nsteps = 1`, `inner_nwalkers = 4`); full hot path through the propagator (`begin_inner_step` → resample → `MixedDensityMatrix_for_vbias`/`vbias` (full-G via layout dispatch) → `vHS` → apply → `Overlap`), including G-buffer sizing from `size_of_G_for_vbias()`; asserts finite weights/energies/overlaps over 3 steps (smoke, not NOMSD parity). All three gated on **CLOSED (RHF)**. **CPU-verified** on a compute node with `wfn_rhf.h5` + `ham_chol_sc.h5`; the rest of `[stochastic_wfn]` also passes with `wfn_msd.h5`. **[overhaul]** Tests not yet ported. Remaining (research-level): `P → ∞` convergence and `inner_seed` stability. |

#### Phase 3c — Walker-conditioned sampling + propagate-then-resample leapfrog

**Goal:** importance-sample the inner fields conditioned on each outer walker (Eq. 23) and add the
leapfrog so the step-to-step overlap-ratio `𝒩(φ)` cancellation (Eq. 25) is exact — the variance
reduction and exactness the paper relies on to drive the outer walk.

| Item | Plan |
|------|------|
| Conditioning | `𝒫(Y; φ_w) = \|⟨φ_T\| p_T(Y) B̂_T(Y) \|φ_w⟩\| / 𝒩(φ_w)` (Eq. 23): the inner ensemble and `S_p` become `φ_w`-dependent. |
| Force bias (new code) | A **custom** inner force bias `x̄ ∝ √Δτ · L_γ^var · ⟨φ_T\|c†c\|φ_w⟩/⟨φ_T\|φ_w⟩` — the **Variational** Cholesky contracted with the `φ_T`–`φ_w` cross DM. It depends on the **outer** walker, so the stock inner `vbias` (self-conditioned on `φ_T`) cannot supply it; this is the genuinely new piece (still feeding the reused `assemble_X` → `vHS` → `apply_propagators`). Distinct from the Phase 3a outer force bias, which uses the **True** Cholesky and the `Ψ_T`–`φ_w` DM. No Variational *energy* is needed. |
| Leapfrog | Propagate-then-resample so numerator/denominator of the step-to-step overlap ratio share inner samples conditioned on the old walker, making the `𝒩(φ_w)` cancellation exact. |
| Coupling | Deepest integration into `AFQMCBasePropagator::step` — the inner walk is interleaved with the outer walk, not a standalone evolution. The most novel and propagator-invasive subphase. |
| Tests | Full stochastic-trial AFQMC run: total energy matches analytic-trial AFQMC within combined error bars; importance sampling reduces variance vs 3b. |

**Sequencing note:** 3a alone delivers a working (static) stochastic-trial propagator, delegate-limit
verified. 3b/3c can follow once 3a is stable; the observable/mean-field phases below reuse the same
reductions and inherit whichever depth (static after 3a, dynamic after 3b/3c) is in force.

### Phase 4 — `vHS` (resolved: no override needed)

`vHS` is **trial-independent** — `v = √Δτ · Σ_γ X_γ L_γ` uses only the auxiliary fields and the
(trial-independent) Cholesky tensor `L`, with no half-rotation or inner-ensemble quantity (verified
in `Real3IndexFactorization::vHS`). `StochasticWfn` delegates to `nomsd_` permanently; there is no
stochastic `vHS` work. Recorded here so the hot path is fully accounted for (see Phase 3a).

### Phase 5 — `MixedDensityMatrix` / `DensityMatrix` / `accumulate_estimators`

Mixed and back-propagated observables. The observable `MixedDensityMatrix` reuses the Phase 3a
mixed-DM reduction (estimator 3) with the observable layout/options; `DensityMatrix` and
`accumulate_estimators` extend it to reference-resolved and back-propagated estimators.

### Phase 6 — `G_MF` / `vMF`

Mean-field subtraction consistency: build the stochastic mean-field Green's function / Cholesky
contribution from the inner ensemble. Static-limit-first, like 3a.

### Phase 7 — Layout queries and back-prop references

Confirm the Tier 4–5 metadata matches what the stochastic Tier 1–3 methods actually produce. Under
the dual-Hamiltonian design these outer-facing queries **stay on `nomsd_` (True Ham)** — Phase 7 is
mostly *verification* that they remain consistent, not a reroute to the inner (Variational) stack
(see [Tier 4](#tier-4--layout-and-metadata-queries) / [Tier 5](#tier-5--hamiltonian-and-slater-infrastructure)).
The back-propagation references (Tier 6) are the substantive remaining work.

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

The checklist below defines the **regression targets** for stochastic work. Status bullets marked
**verified (CPU)** were achieved on **develop** only. On **overhaul**, these tests are **not yet
ported or run** — they are the acceptance criteria for the next port milestone.

### Delegate limit (primary regression anchor)

Compare against the current `nomsd_` delegate in the limit where the inner ensemble collapses
to a single deterministic state (`inner_nwalkers = 1`, `inner_nsteps = 0`):

- `stochastic_wfn_matches_nomsd` passes; since Phase 3a **all four** stochastic overrides (`Overlap`, `Energy`, `MixedDensityMatrix_for_vbias`, `vbias`) are exercised with their NOMSD-equality assertions gated on `ndet == 1` (the delegate limit), since each diverges from NOMSD for multi-determinant trials by design.
- `stochastic_inner_outer_infrastructure_independent` confirms dual infrastructure without breaking delegate-limit observables.
- `stochastic_inner_propagator_construction` confirms inner propagator wiring without changing outer behavior.
- `stochastic_overlap_matches_nomsd` — **[develop] verified (CPU):** inner_nwalkers invariance (`wfn_msd.h5`); delegate-limit overlap equality (`wfn_rhf.h5` + `ham_chol_sc.h5`).
- `stochastic_energy_matches_nomsd` — **[develop] verified (CPU):** inner_nwalkers invariance (`wfn_msd.h5`); delegate-limit energy/overlap equality (`wfn_rhf.h5` + `ham_chol_sc.h5`); Overlap/Energy `Ov` consistency.
- `stochastic_vbias_matches_nomsd` — **[develop] verified (CPU):** inner_nwalkers invariance (`wfn_msd.h5`); delegate-limit mixed-DM `G` and force-bias `x̄` equality vs NOMSD (`wfn_rhf.h5` + `ham_chol_sc.h5`).
- `stochastic_propagator_step` — **[develop] verified (CPU):** outer `AFQMCBasePropagator::Propagate()` completes without layout/runtime failures on a dynamic stochastic trial (`inner_nsteps = 1`; finiteness smoke, not NOMSD parity).
- Local energy and overlap match analytic `NOMSD` within stochastic error bars under dynamic sampling *(research-level; delegate-limit parity is unit-tested; `stochastic_propagator_step` asserts finiteness only)*.
- Mixed estimator (`MixedObsHandler`) forces and densities are consistent *(integration follow-up)*.
- GPU build and memory/task-group load with the additional inner ensemble *(integration follow-up)*.

### Phase 1a-specific tests (implemented)

*Implemented and verified on **develop** only. Target for overhaul port: `tests/test_wfn_factory.cpp` in `test_afqmc`.*

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_wfn_matches_nomsd` | Parity preserved at default inputs (delegate limit). **[overhaul] BLOCKED: SIGSEGV — see [Known blocker](#known-blocker-stochastic_wfn_matches_nomsd-sigsegv-overhaul-port).** |
| `stochastic_inner_walkers_init` | Inner walkers initialize to `inner_nwalkers`; Slater matrices match `getInitialGuess()`; inner `NOMSD` overlap/energy on `inner_wset()` matches reference; replicated ensemble gives identical observables. |
| `stochastic_inner_walkers_uninitialized_smoke` | Stochastic wavefunction built without `walker_pt` stays uninitialized until `maybe_initialize_stochastic_inner_walkers()`; then `stochastic_inner_wset()` is usable. (`inner_wset()` before init calls `APP_ABORT` — not exercised in-process.) |

### Phase 1b-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input (`getWavefunctionType() == "NOMSD"`); PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_outer_infrastructure_independent` | `outer_nomsd()` and `inner_nomsd()` are distinct objects with distinct `getSlaterDetOperations()` pointers; `StochasticWfn::getSlaterDetOperations()` returns outer; inner/outer layout metadata match (Cholesky distribution, transpose flags, `size_of_G_for_vbias`, walker/Ham types); inner overlap/energy on `inner_wset()` match plain `NOMSD` reference. |

### Phase 1c-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input; PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_propagator_construction` | `inner_nsteps() == 0`; `inner_propagator_built()`; `inner_wavefunction()` shares SDetOp with `inner_nomsd()` but not outer; layout parity between inner NOMSD and wrapped `Wavefunction`; propagator Cholesky count matches inner NOMSD; default hybrid (not free) propagation. |

### Phase 2a-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input. Checkpoint (2) additionally requires a **single-determinant**
trial (RHF/UHF); it is gated on `ndet == 1`, so on multi-determinant `wfn_msd.h5` only
checkpoint (1) runs.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_overlap_matches_nomsd` | `Overlap` in isolation (no following `Energy`): (1) **inner_nwalkers invariance**; (2) **delegate limit** — stochastic overlap equals NOMSD at `ndet == 1`. |

**[develop] Verified (CPU):**
- Checkpoint (1): `wfn_msd.h5` + `ham_chol_sc.h5`
- Checkpoint (2): `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5`

Example (develop — delegate-limit anchor):

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5

mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_overlap_matches_nomsd
```

### Phase 2b-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input. Checkpoint (2) additionally requires a **single-determinant**
trial (RHF/UHF); it is gated on `ndet == 1`, so on multi-determinant `wfn_msd.h5` only
checkpoint (1) runs.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_energy_matches_nomsd` | `Energy` in isolation: (1) **inner_nwalkers invariance** — a static replicated ensemble gives E1/EXX/EJ/energy/overlap independent of `inner_nwalkers`; (2) **delegate limit** — stochastic energy and overlap equal NOMSD at `ndet == 1`; (3) **Overlap/Energy consistency** — `Overlap(wset)` equals the overlap from `Energy(wset)` (locks the shared-reduction invariant across the two differently-computed code paths); (4) **propagator entry point** — the direct 3-arg `Energy(wset, E, Ov)` agrees with the property-setter form. Mirrors `stochastic_overlap_matches_nomsd`. |

`stochastic_wfn_matches_nomsd` now also exercises the `Energy` override (its energy/overlap
assertions are gated on `ndet == 1`; since Phase 3a the `MixedDensityMatrix_for_vbias` / `vbias`
assertions are gated on `ndet == 1` too — see [Phase 3a-specific tests](#phase-3a-specific-tests-implemented)).

**[develop] Verified (CPU):**
- Checkpoint (1): `wfn_msd.h5` + `ham_chol_sc.h5` (inner_nwalkers invariance)
- Checkpoint (2): `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5` (delegate limit)

Example (develop — delegate-limit anchor):

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5

mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_energy_matches_nomsd
```

### Phase 3a-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input. Checkpoint (2) additionally requires a **single-determinant**
trial (RHF/UHF); it is gated on `ndet == 1`, so on multi-determinant `wfn_msd.h5` only
checkpoint (1) runs.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_vbias_matches_nomsd` | `MixedDensityMatrix_for_vbias` + `vbias`: (1) **inner_nwalkers invariance** — a static replicated ensemble gives a mixed DM `G` and force bias `x̄` independent of `inner_nwalkers`; (2) **delegate limit** — stochastic `G` and `x̄` equal NOMSD at `ndet == 1`. Mirrors `stochastic_energy_matches_nomsd`. |

`stochastic_wfn_matches_nomsd` also exercises both Phase 3a methods, with its
`MixedDensityMatrix_for_vbias` / `vbias` assertions gated on `ndet == 1`.

**[develop] Verified (CPU):**
- Checkpoint (1): `wfn_msd.h5` + `ham_chol_sc.h5` (inner_nwalkers invariance)
- Checkpoint (2): `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5` (delegate limit)

Example (develop — delegate-limit anchor):

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5

mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_vbias_matches_nomsd
```

### Phase 3b-specific tests (implemented)

*Develop only; not yet ported to overhaul.*

Requires a **NOMSD** HDF5 input and a **CLOSED (RHF)** trial (`wfn_rhf.h5`); UHF/MSD/PHMSD inputs skip
the dynamic/full-G checks silently.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_full_g_matches_compact` | Un-rotated full-G energy and force bias at the anchor equal the compact `nd = 0` path and NOMSD (single-determinant delegate limit). |
| `stochastic_dynamic_ensemble_smoke` | With `inner_nsteps > 0`, inner resample + all four stochastic overrides on moved walkers; asserts finite energies/overlaps (and vbias path via MixedDM + `vbias`, full-G via layout dispatch). |
| `stochastic_propagator_step` | Real outer `AFQMCBasePropagator::Propagate()` on a dynamic trial (`inner_nsteps = 1`, `inner_nwalkers = 4`); full propagator hot path; asserts finite weights/energies/overlaps over 3 steps. |

**[develop] Verified (CPU):** `C_1x1x1_dzvp/wfn_rhf.h5` + `ham_chol_sc.h5`

Example (**develop** — end-to-end propagator step):

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5

mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_propagator_step
```

### Running the stochastic test suite

Stochastic helpers require a NOMSD wavefunction file (e.g. bundled `wfn_msd.h5`; RHF/UHF/PHMSD
inputs skip stochastic checks silently). All stochastic tests share the Catch2 tag `[stochastic_wfn]`.

**Develop (reference branch)** — build `test_afqmc_wavefunctions`, run from the build directory:

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_msd.h5

cd /path/to/SAFIRE/build
mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  "[stochastic_wfn]"
```

**Overhaul (`stochastic-wfn-overhaul`)** — target binary is the consolidated `test_afqmc`
(`tests/test_wfn_factory.cpp`); output under `${BUILD_DIR}/tests/bin/`. **Tests not yet ported**;
after porting, the equivalent command will be:

```bash
cd /path/to/SAFIRE/build
make test_afqmc
mpirun -np 1 ./tests/bin/test_afqmc \
  --hamil "$HAMIL" --wfn "$WFN" \
  "[stochastic_wfn]"
# or: ctest -R stochastic_wfn -V
```

Single test example (**develop**):

```bash
mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_inner_propagator_construction
```

### Integration follow-ups (not yet validated)

**Overhaul port (immediate):**

- Port all `[stochastic_wfn]` cases from develop's `test_wfn_factory.cpp` to `tests/test_wfn_factory.cpp`.
- Build `test_afqmc` and run the full `[stochastic_wfn]` tag on a compute allocation.
- Confirm `ham_chol_sc.h5` full-G path (sparse factorized Hamiltonian) matches develop behavior.
- Re-run delegate-limit parity tests (`stochastic_overlap_matches_nomsd`, etc.) before any upstream PR.

**Both branches (longer term):**

- Full production driver run (`DriverFactory` / input deck) with `stochastic: true` and `inner_nsteps > 0`.
- Static-limit outer propagator step matching NOMSD at the delegate limit (dynamic path covered by `stochastic_propagator_step`; static delegate parity remains unit-tested per method, not through `Propagate()`).
- Mixed estimator (`MixedObsHandler`) forces and densities remain consistent.
- GPU build: all `[stochastic_wfn]` tests pass with `ENABLE_CUDA=ON`.
- GPU memory and task-group load acceptable with the additional inner ensemble and propagator.
- Research-level: dynamic-trial energies/overlaps vs analytic NOMSD within stochastic error bars; `P → ∞` convergence and `inner_seed` stability.
- `KP3IndexFactorization` / other HamOps full-G stubs on overhaul until tests exercise those paths.
