# safiretools — Design Decisions

Status: living document. Records decisions actually made, not the discussion that produced them.
Anything not listed here is still undecided — don't infer intent beyond what's written.

`safiretools` replaces the existing `afqmctools` + `stats` packages under `utils/`. `AutoHF` is a
separate package (its own git history) and is out of scope for this document.

## Scope

**Kept, ported now:**
- Hamiltonian construction/IO (lattice-model builder, molecular, k-point/supercell periodic).
- Wavefunction construction/IO (free-electron, molecular, PBC).
- 1-RDM statistics path (equilibration, autocorrelation-aware averaging).
- Execution-input (AFQMC run-config JSON) generation — redesigned, not just ported.
- `scalar_stats` — the only CLI entry point kept (`energy_stats`, a duplicate alias, is dropped).
- QE interop — bugs fixed, behavior preserved. `qe_driver.py` may be dropped outright (currently
  unimportable — references a nonexistent module — and superseded).
- AutoHF interop — kept, fragile unguarded import gets hardened.
- Dice-SHCI wavefunction import — kept, split into its own module, proper exceptions.
- `rhonk.py` (real-space observables) — kept for external callers; its vendored duplicate HDF5
  helpers get deduplicated onto the shared Core Library HDF5 utility.

**Deferred to a future, C++-spanning observables rewrite (not designed here):**
- The broader RDM/observable pipeline: 2RDM, generalized Fock matrix, spin-spin, pair-correlation,
  EKT/NOON analysis (currently `afqmctools/analysis/average.py` + `extraction.py`, essentially
  unintegrated today).
- `inputs/energy.py`-style Hamiltonian/HF-energy validation.

**Dropped entirely:**
- AIMBES interop (`aimbes_utils.py`, `aimbes_to_2nd_quant`/`aimbes_to_afqmc` CLIs) — AIMBES now
  generates its own SAFIRE-compatible inputs directly. Note current location in case this
  changes back; do not port.
- `analysis/new_rdm.py` — a third, weaker parallel 1-RDM implementation, fully superseded, no
  known external dependents.
- The entire CLI surface except `scalar_stats`.
- `FULLYPOLARIZED` as a distinct spin-symmetry value (removed C++-side too, this branch).

**Rule for all of the above:** "no callers found in `utils/`" is never sufficient reason to drop
something on its own — these are library packages with external callers writing their own AFQMC
workflows. Everything marked "dropped" above was an explicit user call, not an inference from
caller-count.

## Package layout (sketch — not fully confirmed, see Open Questions)

```
safiretools/
├── __init__.py            # curated flat re-exports: Hamiltonian, LatticeHamiltonian,
│                           #   Wavefunction, SpinSymm, HamiltonianBuilder, Lattice,
│                           #   mean_and_error, ... (concrete Lattice subclasses NOT re-exported —
│                           #   Lattice.from_dict() handles dispatch)
├── types.py                # canonical SpinSymm enum — dependency-free, no upward imports
├── hdf5.py                  # generic HDF5 read/write primitives — dependency-free, top-level
│                            #   like types.py (replaces utils/io.py's generic bits, rhonk.py's
│                            #   vendored copy, stats/config_h5.py).
├── stats.py                 # mean_and_error(), reblock(), autocorrelation estimator — generic,
│                            #   top-level (not nested under analysis/) since the deferred
│                            #   broader observables pipeline will need it too.
├── analysis/                # AFQMC-domain statistics built on top-level stats.py
│   ├── metadata.py           #   one typed metadata accessor (walker_type: SpinSymm, taus, ...)
│   ├── equilibration.py      #   Teq -> Neq (delta_tau = taus.max(), the corrected formula)
│   └── rdm.py                #   1-RDM extraction/averaging (from analysis/rdm.py)
├── hamiltonian/
│   ├── base.py               # Hamiltonian ABC: to_hdf5()/from_hdf5() pattern
│   ├── model/
│   │   ├── builder.py         # HamiltonianBuilder — public, used internally by from_dict()
│   │   ├── lattice_hamiltonian.py  # LatticeHamiltonian(Hamiltonian)
│   │   └── lattice.py         # Lattice (ABC) + SquareLattice/TriangularLattice/
│   │                          #   HoneycombLattice/KagomeLattice, moved from
│   │                          #   afqmctools/systems/lattice.py. Lattice.from_dict(params)
│   │                          #   classmethod replaces the free function get_lattice(params) —
│   │                          #   dispatches to the right concrete subclass. No standalone
│   │                          #   to_hdf5()/from_hdf5() — only ever persisted embedded in a
│   │                          #   LatticeHamiltonian's HDF5 file. Re-exported at top level
│   │                          #   (file stays here; only Lattice itself is re-exported, not
│   │                          #   the concrete subclasses).
│   ├── molecular.py           # MolecularHamiltonian(Hamiltonian) — from hamiltonian/mol.py
│   ├── periodic.py            # PeriodicHamiltonian(Hamiltonian) — merges kpoint.py+supercell.py
│   │                          #   (confirmed ~90% duplicated Cholesky-solver code today)
│   └── fcidump.py             # FCIDUMP external-format I/O
├── wavefunction/
│   ├── base.py                # Wavefunction ABC — spin_symm, nelec, nmo; implements
│   │                          #   to_hdf5()/from_hdf5() ONCE (not per-subclass — formats are
│   │                          #   identical across domains, unlike Hamiltonian's)
│   ├── nomsd.py                # NOMSDWavefunction(Wavefunction): coeffs + dets. Classmethods
│   │                           #   .from_free_electron()/.from_pyscf()/.from_pbc_scf() dispatch
│   │                           #   to the implementation modules below.
│   ├── phmsd.py                # PHMSDWavefunction(Wavefunction): coeffs + occa/occb.
│   │                           #   Classmethod .from_dice() dispatches to dice.py.
│   ├── free_electron.py       # from_free_electron() implementation; model.py's legacy
│   │                          #   duplicate retired
│   ├── pyscf.py                # from_pyscf() implementation (from wavefunction/mol.py)
│   ├── pbc.py                  # from_pbc_scf() implementation
│   ├── dice.py                 # from_dice() implementation, split out of wavefunction/converter.py
│   └── io.py                  # native SAFIRE HDF5 schema read/write (uses top-level hdf5.py),
│                               #   shared by both NOMSDWavefunction and PHMSDWavefunction
├── execution.py              # redesigned AFQMC JSON execution-parameter generator
│                              #   (from inputs/from_hdf.py) — naming/location still open
├── convert/
│   ├── pyscf.py                # thin orchestration: calls hamiltonian.molecular +
│   │                            #   wavefunction.pyscf (NOMSDWavefunction.from_pyscf)
│   └── autohf.py                # hardened AutoHF interop
└── qe/                        # relocated QE interop (qe_tools.py, qe_utils.py contents;
                                #   qe_driver.py likely dropped, see Scope)
```

`observables/` (rhonk.py et al.) stays close to its current shape for now, pending the
C++-spanning rewrite — only the HDF5-helper dedup happens in this pass.

## Class hierarchies

**`Hamiltonian`** splits by *source domain* — `LatticeHamiltonian`/`MolecularHamiltonian`/
`PeriodicHamiltonian`, because those are genuinely different storage formats. Each implements its
own `to_hdf5()`/`from_hdf5()`. `spin_symm` is a plain attribute on instances, not a subclass axis.

**`Wavefunction`** splits by *representation*, not domain — `NOMSDWavefunction` (coeffs + per-
determinant orbital matrices) and `PHMSDWavefunction` (coeffs + occa/occb occupation-number
strings). This matches what `write_wfn` already does today via ad hoc length-checking
(`len(wfn)==2` vs `==3`). Domain (free-electron / PySCF / PBC / Dice) becomes a factory
classmethod on the appropriate subclass (`NOMSDWavefunction.from_free_electron/.from_pyscf/
.from_pbc_scf`, `PHMSDWavefunction.from_dice`) rather than its own subclass — a free-electron
wavefunction and a PySCF UHF wavefunction are the same NOMSD representation, just built
differently. `spin_symm` is a plain attribute here too, for consistency with `Hamiltonian` —
RHF/UHF/GHF don't get their own subclasses. `Wavefunction`'s base class implements
`to_hdf5()`/`from_hdf5()` once (not per-subclass) since the format is identical across domains.

Both hierarchies follow the same shape: an ABC, concrete subclasses for what's *structurally*
different (storage format for Hamiltonian, mathematical representation for Wavefunction), and
plain attributes (not subclasses) for what's just *data* (spin_symm) or *provenance* (which
external tool/domain built it).

**`Lattice`** follows the same shape too: an ABC with concrete subclasses per lattice type
(`SquareLattice`/`TriangularLattice`/`HoneycombLattice`/`KagomeLattice`), a `Lattice.from_dict()`
classmethod replacing today's free-function `get_lattice()` factory, and no standalone
`to_hdf5()`/`from_hdf5()` — it's only ever persisted embedded in a `LatticeHamiltonian`'s HDF5
file. Exposed at the top level (`from safiretools import Lattice`) since users may want to
construct/inspect lattice geometry independent of building a full Hamiltonian; only the base
class is re-exported, not the concrete subclasses — `from_dict()` handles dispatch.

Known bugs in `afqmctools/systems/lattice.py` to fix during the port (independent of the above):
`get_lattice()`'s `a1`/`a2` overrides are silently discarded for every built-in lattice type
(absorbed into `**kwargs`, never applied); `_neighbor_distance_map`'s `min_distance` parameter is
passed by `__init__` but doesn't exist on the method signature (`TypeError` if ever non-`None`,
currently untested/unhit); `_is_allowed_site` is defined twice back-to-back (identical bodies);
dead rotation-group neighbor-generation code (`ROTATION_GROUP`, `_rotations()`,
`_check_add_image_neighbors`) whose docstring claims it's the live algorithm but isn't; and a
superseded `_is_valid_image_old` with an unresolved "check against next version" TODO.

## Statistics core

- One generic primitive, `mean_and_error(samples, axis=0)`, replaces `stat_h5.py::me2d`,
  `scalar_dat.py::single_column_from_array`/`error`, and the `scipy.stats.sem` fallback in
  `analysis/average.py`. Works over arbitrary-shape data (scalar traces and RDM matrices alike).
- Autocorrelation-time estimator: keep the current O(n²) + numba implementation — preserves
  validated numerics over switching to an FFT-based approach.
- Degenerate/constant data: detect up front via a variance check, not the current post-hoc
  NaN/Inf-then-replace-with-1.0 approach.
- Reblocking keeps a smaller trailing block for remainder samples (not discard-and-raise).
- Complex-valued samples (RDMs): track real and imaginary parts as independent real error bars.
- 1-RDM pipeline: the narrow, live path (`analysis/rdm.py` + `stat_h5.afobs`) is what's ported.

### Bug fixed along the way: equilibration formula

`analysis/common.py::_Neq_from_Teq` used `delta_tau = taus[1]-taus[0]` (spacing between
*configured BP-depth levels*) — wrong. Confirmed via `src/AFQMC/Estimators/
BackPropagatedEstimator.hpp` that `iblock` increments once per full `max_nback_prop`-step cycle,
uniformly across BP-depth levels, so the correct value is `taus.max()` (`= max_nback_prop * dt`).
The now-dropped CLI's `calc_nequil` had this right; the kept `_Neq_from_Teq` did not. Also: the
C++ side already discards equilibration blocks via `equil_multiplier` before anything reaches
`stat.h5` — but the Python-side `Teq`/`nequil` knob is still a wanted feature, for cases where
unequilibrated samples slip through despite the C++-side trim. `check_1rdm_convergence`'s
quadrature-sum indexing bug (uses only one of two BP-average endpoints' errors) gets fixed at the
same time.

## Spin-symmetry enum

- Canonical: `SpinSymm`, an `IntEnum` (not `IntFlag`) with exactly 3 members forming a hierarchy
  of increasing generality: `CLOSED < COLLINEAR < NONCOLLINEAR`. Matches `WALKER_TYPES` in
  `src/AFQMC/config.h` exactly (`FULLYPOLARIZED` removed there too, this branch).
- `_SlaterType`/`_slater_enum_map`/`_slater2dims` (`afqmctools/utils/slater_types.py`) are
  dropped entirely — `_SlaterType` needed a translation function to reach the C++ wire format;
  `SpinSymm`'s int values already are that format.
- The dead `SlaterDeterminant`/`MultiSlater`/`NonorthMSD`/`ParticleHoleMSD` stub classes are
  dropped — every method unconditionally raised `NotImplementedError`; nothing ever worked.
- Lives in `safiretools/types.py` — dependency-free, so hamiltonian/wavefunction/observables all
  import it downward rather than `observables` reaching up into `hamiltonian` for it (today's
  layering inversion).

## Known bug: `force_herm` diagonal-zeroing

`utils/matrix.py::force_herm`'s `'upper_triangular'` method (`np.triu(M,1)` then symmetrize)
zeroes the diagonal. Its only two call sites (`tband` in `nth_neighbor_hopping`, `epsilon_band`
in `onebody_onsite`) are one-body terms confirmed to be "read and used as the full matrix" (not
the U1/U2/J convention below) — so this has been silently discarding onsite/diagonal terms for
any caller using `force_herm=True` on a non-Hermitian `t`/`epsilon` input. Fix: reconstruct
including the diagonal (`np.triu(M,0) + np.triu(M,1).conj().T`).

**Distinct, correct convention — do not conflate:** only the U1 and U2 pieces of the interaction
matrix, and J, are meant to be upper-triangular, with the diagonal genuinely unused (e.g. U1's
diagonal is irrelevant because same-band interaction is already covered by the separate `U`
parameter). This is a different code path from `force_herm` and should stay a separate helper.

## Coding conventions

- **Errors**: library code raises typed exceptions (`ValueError`/`RuntimeError`), never
  `sys.exit()` or bare `assert(0)` for control flow. That style is confined to `scalar_stats`'s
  CLI entry point.
- **Mutation**: functions never silently mutate caller-supplied arguments.
- **Logging**: standard `logging` only — no separate always-on progress channel, no
  monkey-patching `logging.Logger` (drops the current `afqmctools/__init__.py` patch).

## Public API patterns

- **Import surface**: deep implementation tree, curated flat top-level `__init__.py` re-exports
  (numpy/scipy-style) — `from safiretools import Hamiltonian` without needing tree knowledge.
- **Construction**: classmethod factories (`from_dict`, `from_hdf5`, `from_fcidump`, ...) over
  parsing inside `__init__`. `__init__` is for already-fully-formed, validated in-memory data.
- **Serialization**: instance method + classmethod pair — `obj.to_hdf5(path) -> None`,
  `Cls.from_hdf5(path) -> Cls` (classmethod, dispatches to the right concrete subclass). Applies
  uniformly to both `Hamiltonian` and `Wavefunction`, for codebase-wide consistency. `Hamiltonian`
  subclasses (`LatticeHamiltonian`/`MolecularHamiltonian`/`PeriodicHamiltonian`) each implement
  it since their formats genuinely differ; `Wavefunction`'s base class implements it once and
  subclasses don't override, since wavefunction formats are identical across domains.
- **`HamiltonianBuilder`** stays public (importable, usable for advanced/custom term
  composition); `LatticeHamiltonian.from_dict(params)` wraps it internally for the common case.
- Avoid bare mutable-attribute access on builder-style objects (`builder.hamiltonian`) — prefer
  explicit accessor methods (`builder.get_hamiltonian()`).
- `AFQMC_EXEC` must never be required just to import safiretools (today's
  `RuntimeError: AFQMC_EXEC environment variable is not set` fires at import time in
  `tutorial_utils/helper.py` — becomes a lazy check, only triggered when something actually
  invokes SAFIRE). A future thin binding to invoke SAFIRE in-process (possibly nanobind) is
  planned — whatever "run SAFIRE" API safiretools exposes should not hard-code a
  subprocess/executable-path assumption that would preclude that later.
- Dependency cleanup: drop `pytables` (`stats/config_h5.py`'s only reason for it, rewritten onto
  `h5py`); merge the `LATTICE_HF` optional-dependency group into `AUTOHF` (exact duplicate).

## Open questions

- `execution.py`'s naming/location, and whether it eventually gets its own CLI entry point (the
  CLI is currently scoped to `scalar_stats` only) — explicitly deferred to a broader team
  discussion, not just this session.
