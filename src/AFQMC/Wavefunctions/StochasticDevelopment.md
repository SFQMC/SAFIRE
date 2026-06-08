# StochasticWfn Development Guide

This document tracks the current state of `StochasticWfn` and the methods that must be
overridden for it to function as a full-fledged stochastic trial wavefunction in SAFIRE,
rather than a thin delegate wrapper around `NOMSD`.

---

## Current State

`StochasticWfn` (`StochasticWfn.hpp` / `StochasticWfn.icc`) is a `template<bool MP, class devPsiT>`
class that inherits from `AFQMCInfo` and owns a single member:

```cpp
NOMSD<MP, devPsiT> nomsd_;
```

**Today, every public method forwards to `nomsd_`.** Behavior is therefore identical to `NOMSD`
when the factory is invoked with `stochastic: true` on an NOMSD HDF5 file.

### Factory integration (scaffolding only)

- Selected via the XML input flag `stochastic: true` inside the existing `type == "nomsd"` path
  in `WavefunctionFactory::fromHDF5`.
- Reads the same NOMSD HDF5 data (`Wavefunction/NOMSD`, `PsiT_*`, CI coefficients, etc.).
- Construction is routed through `makeNomsdWavefunction()`, which chooses `StochasticWfn` or
  `NOMSD` at the last step.
- Registered as four explicit instantiations in the `Wavefunction` `boost::variant`.

This is intentionally **not** first-class like `PHMSD` (no separate HDF5 type, no dedicated
factory branch). That should change as `StochasticWfn` acquires its own initialization needs.

### Accessor

`inner_wfn()` exposes the wrapped `NOMSD` for debugging and incremental migration.

---

## Target Architecture

> **Future expectation:** Each `StochasticWfn` instance will own its own inner AFQMC stack:
>
> - **Walker set** — stochastic ensemble defining the trial wavefunction
> - **NOMSD** — analytic MSD structure evaluated on inner walkers
> - **Propagator(s)** — drives the inner stochastic evolution
> - **HamiltonianOperations (`HamOps`)** — inner Hamiltonian coupling for energy, bias, and forces
>
> The outer AFQMC driver will call `StochasticWfn` methods as it does for `NOMSD`/`PHMSD` today;
> `StochasticWfn` will orchestrate the inner calculation and return effective quantities
> (overlap, local energy, mixed density matrix, bias potentials) to the outer propagator.

---

## Methods to Override

The `Wavefunction` visitor dispatches all calls below. Any method not listed still delegates to
`nomsd_` today and may need overriding once the inner stack exists.

Methods are grouped by priority. **Tier 1** methods are on the propagator hot path (every
outer timestep). These are the hardest and most important to replace.

---

### Tier 1 — Propagator hot path (highest priority)

Called every outer propagation step from `AFQMCBasePropagator`, `AFQMCModelPropagator`, and
distributed propagator variants.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `MixedDensityMatrix_for_vbias(wset, G)` | Delegates to `nomsd_`; computes analytic mixed DM of outer walkers w.r.t. the deterministic MSD trial. | Compute the **stochastic** mixed density matrix used by `vbias`. Run (or sample from) the inner walker/propagator ensemble, evaluate inner `NOMSD` mixed DMs, and return an effective DM in the layout expected by `vbias` (`compact_G_for_vbias`, `transposed_G_for_vbias`). |
| `vbias(G, v, dt, a)` | Delegates to `nomsd_.HamOp.vbias` using the analytic `G` from above. | Apply Cholesky-vector bias using the **stochastic** mixed DM. May call inner `HamOps.vbias` on inner-walker quantities and reduce/average to the outer-walker buffer layout. |
| `vHS(X, v, dt, a)` | Delegates to `nomsd_.HamOp.vHS`. | Compute the spin-dependent or spin-independent one-body propagation matrix contribution from auxiliary fields, using inner stochastic information where the trial is not a single deterministic MSD. |
| `Overlap(wset)` / `Overlap(wset, Ov)` | Delegates to `nomsd_`; exact MSD overlap on outer walker Slater matrices. | Return **stochastic trial overlap** `<Ψ_T[inner]|Φ[outer]>`: average (or importance-sample) inner-walker overlaps from the owned inner `NOMSD`. Updates `wset` overlap properties when the `wset&` overload is used. |
| `Energy(wset)` / `Energy(wset, E, Ov)` | Delegates to `nomsd_`; exact local energy from analytic MSD. | Return **stochastic local energy** and overlap: run inner propagation or sampling, evaluate inner `NOMSD::Energy` on the inner walker set, and reduce to per-outer-walker `E` (E1, EXX, EJ) and `Ov`. |

---

### Tier 2 — Mixed estimators and observable handlers

Called from `MixedObsHandler`, `FullObsHandler`, force estimators, and related code paths.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `MixedDensityMatrix(wset, G, ...)` | Delegates to `nomsd_`; full analytic mixed DM for outer walkers. | Stochastic mixed DM for **observable evaluation** (may differ in layout/options from the `for_vbias` variant). Average inner ensemble contributions; support `compact` and `transpose` flags as today. |
| `MixedDensityMatrix(wset, G, Ov, ...)` | Same as above, also returns overlaps per determinant/walker. | As above, plus return the overlap vector needed for weighted accumulation in multi-determinant or multi-reference estimators. |
| `DensityMatrix(wset, RefA, RefB, G, Ov, ...)` | Delegates to `nomsd_`; DM w.r.t. a specific reference determinant. | Stochastic DM relative to a chosen reference orbital set. Needed by `MixedObsHandler` for reference-resolved force and density estimators. |
| `accumulate_estimators(...)` | Delegates to `nomsd_`; analytic back-prop / correlated accumulation. | Accumulate estimator contributions using **stochastic** DMs and energies from the inner ensemble. Must remain consistent with `TimeEvolvedObsHandler` call signature. |

---

### Tier 3 — Mean-field subtraction

Called during propagator setup and mean-field initialization.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `G_MF(G)` | Delegates to `nomsd_`; builds mean-field Green's function from CI-weighted determinant DMs. | Build the **stochastic** mean-field Green's function used to subtract the trial mean-field potential. Average inner-walker MF contributions from the owned inner stack. |
| `vMF(v, dt)` | Delegates to `nomsd_`; mean-field contribution of Cholesky vectors. | Compute mean-field bias from the stochastic trial. May require inner `HamOps` and inner walker sampling. |

---

### Tier 4 — Layout and metadata queries

Currently forwarded from `nomsd_` / inner `HamOp`. Must remain **consistent** with what the
overridden Tier 1–3 methods actually produce; may need custom logic once inner `HamOps` is owned
separately.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `size_of_G_for_vbias()` | Returns `nomsd_` DM dimension for `vbias` layout. | Return the DM dimension the stochastic `MixedDensityMatrix_for_vbias` / `vbias` pair actually uses. |
| `transposed_G_for_vbias()` | From inner `HamOp` flags. | Match the memory layout of stochastic `G` passed to `vbias`. |
| `transposed_G_for_E()` | From inner `HamOp` flags. | Match the memory layout of stochastic `G` used in energy evaluation. |
| `transposed_vHS()` | From inner `HamOp` flags. | Match the memory layout expected by `vHS`. |
| `getWalkerType()` | Returns outer `NOMSD` walker type. | Return the walker type the outer driver should use (likely unchanged, but must stay consistent with inner ensemble). |
| `local_number_of_cholesky_vectors()` | From inner `HamOp`. | Return local CV count from **inner** `HamOps` once owned by `StochasticWfn`. |
| `global_number_of_cholesky_vectors()` | From inner `HamOp`. | Global CV count for MPI distribution; must match inner Hamiltonian. |
| `global_origin_cholesky_vector()` | From inner `HamOp`. | Global CV offset for this task group. |
| `distribution_over_cholesky_vectors()` | From inner `HamOp`. | Whether CVs are distributed across the task group. |
| `spin_dependent_vHS()` | From inner `HamOp`. | Whether `vHS` expects spin-resolved output buffers. |

---

### Tier 5 — Hamiltonian and Slater infrastructure

May continue to delegate to inner objects early on, but ownership should eventually move into
`StochasticWfn`.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `getHamType()` | Returns `nomsd_.HamOp.getHamType()`. | Return Hamiltonian type from **inner** `HamOps`. |
| `getFieldTypes(...)` | Delegates to inner `HamOp`. | Expose auxiliary-field layout for inner Hamiltonian (used by `PropagatorFactory`). |
| `update_potentials(...)` | Delegates to inner `HamOp`. | Update inner Hamiltonian potentials after grid / ionic moves. |
| `generalizedFockMatrix(...)` | Delegates to inner `HamOp`. | Used by `generalizedFockMatrix` observable; must use stochastic DM inputs when overridden. |
| `getOneBodyPropagatorMatrix(...)` | Delegates to inner `HamOp`. | Return one-body propagator matrix from inner Hamiltonian. |
| `vHS_sparse(...)` | Delegates to inner `HamOp`. | Sparse `vHS` representation for memory-efficient propagation. |
| `getSlaterDetOperations()` | Returns `nomsd_` SlaterDetOperations pointer. | Return Slater determinant ops for the **inner** `NOMSD` (or a dedicated instance owned by `StochasticWfn`). |

---

### Tier 6 — Back propagation

Called from `BackPropagatedEstimator` and related reference-tracking code.

| Method | Current behavior | Desired functionality |
|--------|------------------|----------------------|
| `number_of_references_for_back_propagation()` | Returns `nomsd_` reference count. | Number of reference Slater matrices the stochastic trial exposes for back-propagation (may differ if inner ensemble uses a reduced reference set). |
| `getReferenceWeight(i)` | Returns `nomsd_.ci[i]`. | CI weight (or effective stochastic weight) for reference `i`. |
| `getReferencesForBackPropagation(A)` | Copies reference orbitals from `nomsd_`. | Provide reference Slater matrices for back-propagation, possibly averaged or selected from the inner ensemble. |

---

### Tier 7 — Construction and factory (infrastructure)

Not wavefunction visitor methods, but required for a full-fledged type.

| Component | Current behavior | Desired functionality |
|-----------|------------------|----------------------|
| `StochasticWfn` constructor | Accepts same arguments as `NOMSD`; builds inner `nomsd_` only. | Construct inner **walker set**, **NOMSD**, **propagator(s)**, and **HamOps** from input; wire task groups and memory managers. |
| `interpret_inputs(pt)` | Validates against `NOMSD` keys plus `stochastic`. | Parse inner-AFQMC parameters (inner walker count, inner steps, population control, etc.). |
| `WavefunctionFactory` | `stochastic: true` flag on NOMSD HDF5 path. | First-class `stochasticwfn` type (or dedicated HDF5 group) with its own `fromHDF5` branch, matching `NOMSD`/`PHMSD` treatment. |
| `getWavefunctionType()` | Not aware of `StochasticWfn`. | Detect stochastic trial wavefunction files on disk. |

---

## Suggested Implementation Order

1. **Constructor / factory** — own the inner stack (walker set, `NOMSD`, propagator, `HamOps`).
2. **`Overlap` and `Energy`** — simplest stochastic quantities; validate against analytic `NOMSD` in the delegate limit.
3. **`MixedDensityMatrix_for_vbias` and `vbias`** — unblock correct outer propagation (hardest coupling).
4. **`vHS`** — complete the per-step propagator chain.
5. **`MixedDensityMatrix` / `DensityMatrix` / `accumulate_estimators`** — mixed and back-propagated observables.
6. **`G_MF` / `vMF`** — mean-field subtraction consistency.
7. **Layout queries and back-prop references** — align metadata with actual stochastic outputs.
8. **Factory / HDF5 first-class treatment** — remove the `stochastic` boolean flag hack.

---

## Validation Checkpoints

When overriding each tier, compare against the current `nomsd_` delegate in the limit where the
inner ensemble collapses to a single deterministic state:

- Outer propagator completes without layout/runtime check failures.
- Local energy and overlap match analytic `NOMSD` within stochastic error bars.
- Mixed estimator (`MixedObsHandler`) forces and densities are consistent.
- GPU memory and task-group load remain acceptable with the additional inner walker ensemble.
