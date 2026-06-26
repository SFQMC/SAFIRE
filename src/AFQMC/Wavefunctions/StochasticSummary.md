# Stochastic Trial Wavefunction — Status Summary

Concise snapshot of the stochastic-trial implementation in SAFIRE (arXiv:2505.18519). For design detail and regression history, see `StochasticDevelopment.md`.

---

## What is implemented

`StochasticWfn` is a first-class trial type in the wavefunction factory. It presents the same outer interface as `NOMSD` to the AFQMC driver and propagator, but replaces deterministic trial overlaps and local energies with **inner-ensemble averages** (Eq. 27).

**Dual Hamiltonian.** The outer (True) Hamiltonian scores every returned quantity — local energy, force bias, observables. An optional **variational** Hamiltonian, supplied via `inner_hamiltonian`, drives the inner HS propagator $\hat{B}_T$ that generates stochastic trial samples from the anchor $|\phi_T\rangle$. If no variational file is given, the inner stack clones the True Hamiltonian.

**Inner ensemble.** Each outer walker carries $P$ inner samples (`inner_nwalkers`). At `inner_nsteps = 0` the ensemble is static (anchor copies). At `inner_nsteps > 0` one variational projection step $\hat{B}_T(Y^{[p]})|\phi_T\rangle$ is applied per sample before reduction.

**Sampling modes** (closed-shell, CPU today):

| Mode | Keys | Role |
|------|------|------|
| Static anchor | `inner_nsteps: 0` | $\hat{B}_T = I$; validates reductions without dynamics |
| Free projection | `inner_nsteps: 1` | Walker-independent Gaussian fields through inner $\hat{B}_T$ |
| Walker-conditioned | `+ inner_conditioning: true` | Importance-sampled fields biased toward each outer walker (Eq. 23) |
| Leapfrog | `+ inner_leapfrog: true` | Propagate-then-resample so the hybrid overlap ratio is exact (Eq. 25) |

**Quantities overridden** on the propagator hot path: log overlap, local energy, mixed DM for force bias. Observable mixed DM, `accumulate_estimators`, and mean-field subtraction ($G_{\mathrm{MF}}$, $v_{\mathrm{MF}}$) also use the inner ensemble. Trial-independent quantities (`vbias` layout, `vHS`, one-body density matrix) delegate to the outer `NOMSD`.

**Back propagation.** Reference API and static-limit estimator/driver smokes pass. Dynamic trials with `inner_nsteps > 0` can yield NaN back-propagated accumulations — treated as open.

**Input.** `type: stochasticwfn` (or deprecated `stochastic: true` on an NOMSD HDF5 block). HDF5 files may declare `Wavefunction/StochasticWfn` explicitly.

---

## Testing status

| Layer | Status | Notes |
|-------|--------|-------|
| Unit tests (`[stochastic_wfn]`, Ne cc-pVDZ RHF fixture) | **Pass** at `-np 1` and `-np 2` | 23 cases: delegate-limit parity with `NOMSD`, dynamic full-G vs compact path, conditioned/leapfrog smokes, observables, mean field, back-prop smokes, factory/HDF5 detection |
| Delegate limit (`P=1`, `inner_nsteps=0`) | **Anchored** | All overrides reproduce plain `NOMSD` at single-determinant trials |
| Production driver (`safire`) with VAFQMC-exported Ne cc-pVDZ | **Smoke-validated** | Static anchor, variational dynamic trial, conditioned, and leapfrog all complete 500 steps without NaN at small $P$ |
| Multi-rank (`-np > 1`) | **Pass** on unit suite | Four distributed-walker bugs found and fixed (Jun 2026) |
| GPU build / device memory | **Not tested** | Dynamic full-G path is CPU-gated |
| Dynamic back propagation | **Not validated** | Static-limit smokes only |
| Production statistics | **Not claimed** | Smoke trial (500 VAFQMC iterations, $P=4$, 4 walkers) |

**Known smoke-scale findings:** Dynamic trial with True-Ham inner clone (no `inner_hamiltonian`) showed population collapse — likely small-ensemble instability, not a factory bug. Variational Hamiltonian path is stable. Leapfrog improves weight stability to near-deterministic-trial levels; energy bias vs static anchor is modest at smoke parameters.

---

## End-to-end test: VAFQMC → SAFIRE HDF5 → driver

Canonical workflow for a **trained stochastic trial** (Ne cc-pVDZ example; generalize by changing the PySCF system in the build script).

### 1. VAFQMC side (`vafqmc` repo, Python 3.11)

```bash
cd /path/to/vafqmc
uv sync --frozen --python 3.11 && source .venv/bin/activate

python tools/build_ne_hamiltonian.py          # → hamiltonian.pkl
python tools/train_ne_smoke.py                # → checkpoints/checkpoint.pkl
python tools/export_safire.py \
    --hamiltonian hamiltonian.pkl \
    --checkpoint checkpoints/checkpoint.pkl \
    --out-dir safire_export \
    --variational                             # → ham.h5, wfn.h5, ham_var.h5
```

| Output | Content |
|--------|---------|
| `ham.h5` | True (physical) Hamiltonian |
| `wfn.h5` | Single-determinant anchor orbitals + CI |
| `ham_var.h5` | Variational Cholesky from optimized propagator |

### 2. SAFIRE driver

```bash
RUN=~/development/run_ne_stochastic   # or any run directory
mkdir -p "$RUN" && cd "$RUN"
cp /path/to/vafqmc/safire_export/{ham.h5,wfn.h5,ham_var.h5} .

SAFIRE=~/development/SAFIRE/build/bin/safire

# Representative decks (see that directory for all variants):
mpirun -np 1 "$SAFIRE" afqmc_static.json      # static anchor
mpirun -np 1 "$SAFIRE" afqmc_var.json         # dynamic + variational Ham
mpirun -np 1 "$SAFIRE" afqmc_cond.json        # + walker conditioning
mpirun -np 1 "$SAFIRE" afqmc_leapfrog.json    # + leapfrog
```

Minimal stochastic wavefunction block (dynamic + variational):

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

Add `"inner_conditioning": true` and/or `"inner_leapfrog": true` for the advanced sampling modes. Outer Hamiltonian, walker type, timestep, and population control are set in the usual AFQMC execute block.

### 3. Unit regression (no VAFQMC)

```bash
HAMIL=$PWD/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
WFN=$PWD/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
mpirun -np 1 ./build/tests/bin/test_afqmc \
  --hamil "$HAMIL" --wfn "$WFN" "[stochastic_wfn]"
```

---

## Next steps

**Hardening & coverage**
- GPU build and performance with the extra inner ensemble + propagator
- Dynamic back propagation through a full production run
- `KP3IndexFactorization` and other Hamiltonian-operation backends (full-G stubs today)
- Port deferred inner-stack infrastructure tests (need public accessors on the `Wavefunction` variant)
- Open-shell / non-collinear dynamic path (closed-shell only now)

**Science & integration**
- Longer VAFQMC training and larger $P$ / walker counts before trusting energies and variances
- Systematic comparison: static anchor vs free projection vs conditioning vs leapfrog on production trials
- $P \to \infty$ and `inner_seed` stability studies
- Multi-determinant trials (delegate-limit parity is intentionally single-det only)
- Observable forces and densities through `MixedObsHandler` under stochastic trials

**Operational**
- Prefer `type: stochasticwfn` over the deprecated `stochastic: true` flag in new input decks
- Export variational Hamiltonian (`--variational`) for any dynamic run intended to match the VAFQMC-trained propagator
