# safiretools — Design Decisions

Status: living document. Records decisions actually made, not the discussion that produced them.
Anything not listed here is still undecided. Don't infer intent beyond what's written.

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
- CAS/CI wavefunction import from a PySCF checkpoint (`write_cas_wfn`) — kept as
  `PHMSDWavefunction.from_pyscf_cas`; `ci_wavefunction` came with it, as `ci_expansion`.
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
- The AutoHF variational-energy measurement inside the free-electron wavefunction builder
  (`free_electron(measure_evar=...)`, `measure_spin`, `return_autohf`) — the wrong pathway for
  measuring an energy (user call), and an upward dependency from `wavefunction/` into the AutoHF
  interop that Phase 7 owns. Building a wavefunction and evaluating its energy are separate calls.
- `wavefunction/pbc.py::slater_gto2mo` — no callers anywhere, and unable to run as written (its
  `NONCOLLINEAR` branch returns `None`; its `'mol'` path reads `kwargs['cell']`). The conversion it
  describes is what `from_pyscf`'s basis transform already does.
- `wavefunction/pbc.py::write_wfn_pbc_old` — superseded by `write_wfn_pbc`.
- AIMBES interop (`aimbes_utils.py`, `aimbes_to_2nd_quant`/`aimbes_to_afqmc` CLIs) — AIMBES now
  generates its own SAFIRE-compatible inputs directly. Note current location in case this
  changes back; do not port.
- AIMBES was renames to CoQuí. Any references to AIMBES should be updated to CoQuí
- `analysis/new_rdm.py` — a third, weaker parallel 1-RDM implementation, fully superseded, no
  known external dependents.
- The entire CLI surface except `scalar_stats`.
- `FULLYPOLARIZED` as a distinct spin-symmetry value (removed C++-side too, this branch).

**Rule for all of the above:** "no callers found in `utils/`" is never sufficient reason to drop something on its own.
These are library packages with external callers writing their own AFQMC workflows. 
Everything marked "dropped" above was an explicit decision, not an inference from
caller-count.

## Transitional shim: `isinstance` gates in the un-ported packages

Until the packages that consume a lattice model Hamiltonian are themselves ported, three
`isinstance(source, Hamiltonian)` checks that tested only for afqmctools'
`ham_class.Hamiltonian` accept a `safiretools.LatticeHamiltonian` as well (user call):
`afqmctools/wavefunction/free_electron.py`, `afqmctools/inputs/from_autohf.py`, and
`AutoHF/autohf/hamiltonian.py`. 
`LatticeHamiltonian` already satisfies the accessor contract those
consumers use (`nsites`, `nbands`, `get_one_body()`, `get_U()`, `get_J()`, `get_heisenberg()`), so
only the type test needed widening; afqmctools' `Hamiltonian` is a plain class rather than an ABC,
so there was no `register()` route from the safiretools side.

The AutoHF change lives in a **separate submodule/repository** and needs its own commit. It is
guarded with `importlib.util.find_spec("safiretools")`, matching the pattern AutoHF already uses
for afqmctools, so AutoHF still imports without safiretools installed.

All three shims go away when `free_electron` (Phase 4) and the AutoHF interop (Phase 7) are ported,
and afqmctools is retired (Phase 9).

## Package layout (sketch — not fully confirmed, see Open Questions)

```
safiretools/
├── __init__.py            # curated flat re-exports, and the definition of what is
│                           #   user-facing: Hamiltonian, LatticeHamiltonian,
│                           #   MolecularHamiltonian, PeriodicHamiltonian, HamiltonianBuilder,
│                           #   Lattice, SpinSymm, the FCIDUMP I/O (read_fcidump,
│                           #   read_fcidump_header, write_fcidump, write_fcidump_kpoint,
│                           #   h1_spat2spin, h2_spat2spin), Wavefunction,
│                           #   NOMSDWavefunction, PHMSDWavefunction, mean_and_error, ...
│                           #   NOT re-exported: concrete Lattice subclasses (Lattice.from_dict()
│                           #   handles dispatch), HamiltonianComponent (build steps cover every
│                           #   custom term), and the FCIDUMP format internals (fcidump_header,
│                           #   check_sym, fmt_integral)
├── types.py                # canonical SpinSymm enum — dependency-free, no upward imports
├── hdf5.py                  # HDF5 read/write primitives — dependency-free, top-level
│                            #   like types.py (replaces utils/io.py's generic bits, rhonk.py's
│                            #   vendored copy, stats/config_h5.py). Also owns SAFIRE's
│                            #   file-level complex convention, to_complex/from_complex — the one
│                            #   definition, used by every schema (see "Complex arrays on disk").
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
│   │                          #   HoneycombLattice/KagomeLattice/CustomLattice, moved from
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
│   │                           #   Classmethods .from_dice(), .from_pyscf_cas() and
│   │                           #   .from_pbc_scf() dispatch to the modules below — the periodic
│   │                           #   path produces this representation whenever bands are
│   │                           #   partially occupied and more than one determinant is asked for.
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
.from_pbc_scf`, `PHMSDWavefunction.from_dice`) rather than its own subclass.
A free-electron wavefunction and a PySCF UHF wavefunction are the same NOMSD representation, 
just built differently. `spin_symm` is a plain attribute here too, for consistency with `Hamiltonian` —
RHF/UHF/GHF don't get their own subclasses. `Wavefunction`'s base class implements
`to_hdf5()`/`from_hdf5()` once (not per-subclass) since the format is identical across domains.

Both hierarchies follow the same shape: an ABC, concrete subclasses for what's *structurally*
different (storage format for Hamiltonian, mathematical representation for Wavefunction), and
plain attributes (not subclasses) for what's just *data* (spin_symm) or *provenance* (which
external tool/domain built it).

`Wavefunction`'s single `to_hdf5`/`from_hdf5` pair is implemented the way `Hamiltonian`'s dispatch
is: the base owns the file handling, the format detection (`wavefunction_format(path)` -> `nomsd` /
`phmsd`) and the shared header, and each subclass supplies only a `_write_payload`/`_read_payload`
hook for its own part. `PHMSDWavefunction` is always `SpinSymm.COLLINEAR` — the AFQMC executable's
`read_ph_wavefunction_hdf` rejects both closed-shell and noncollinear particle-hole wavefunctions.
Spin symmetry is a plain attribute there in the sense of "recorded", not "free".

### A wavefunction's determinants carry one block per independent spin channel

`NOMSDWavefunction.dets` is `(ndets, npol*nmo, sum(nelec_per_spin))`: `nup` columns for a
closed-shell wavefunction (the beta channel repeats alpha), `nup + ndown` when collinear, and
`nup + ndown` over `2*nmo` rows when noncollinear. `Wavefunction.nelec_per_spin` is the one place
that mapping lives, and every producer already builds exactly this shape.

`nelec` is the physical `(nup, ndown)` in memory; `nelec_on_disk` derives the `(nup + ndown, 0)`
pair a noncollinear file's `dims` records. Reading such a file back therefore reports
`(nup + ndown, 0)` — the split is not part of the format, and the executable does not use it.

### Writing never orthonormalizes

`orthonormalize()` is an explicit method returning a *new* instance, which the domain factories call
by default; `to_hdf5` warns about a non-orthonormal Slater matrix rather than quietly repairing it.
afqmctools orthonormalized inside `write_wfn`, and — because it applied its 1e-8 sparsification
threshold to the determinant array first — sometimes orthonormalized *because of* its own
thresholding.

**Every Slater matrix is covered, `psi0` included.** Both `orthonormalize()` and the write-time check
walk the same set, so there is no dense array that only one of them sees:

| matrix | `orthonormalize()` | checked by `to_hdf5` |
|---|---|---|
| `NOMSDWavefunction.dets[i]`, per spin channel | ✅ | ✅ as `dets[i] spin s` |
| `psi0` supplied explicitly, or assigned afterwards | ✅ | ✅ as `psi0 spin s` |
| `psi0` left to default | ✅ (re-derived from the fixed determinants) | ✅ — it *is* a copy of `dets[0]`'s blocks, so the determinant check covers it, and reports it under the determinant's name, which is where a caller would fix it |
| `PHMSDWavefunction.orbitals` | ✅ | ✅ as `orbitals[i]` |
| `PHMSDWavefunction`'s default `psi0` | ✅ | ✅ — identity columns, orthonormal by construction |

**Blocks are checked per independent spin channel**, which is the physically correct grain: a
collinear determinant's alpha and beta columns describe different spin sectors and need not be
orthogonal to each other — identical alpha and beta orbitals are an ordinary UHF-shaped determinant —
while a noncollinear determinant is a single `(2*nmo, nup + ndown)` block and is checked whole.
`Wavefunction.nelec_per_spin` supplies the split, as everywhere else.

**The `PsiT` blocks are checked again after sparsifying, because that is what reaches disk.** The
1e-8 threshold applies only to the sparse `PsiT` blocks it exists to sparsify, never to the dense
`Psi0` — that is the afqmctools bug above, and it is why `Psi0` goes to disk verbatim. But
sparsifying happens *after* `to_hdf5`'s check and can cost a determinant its orthonormality on its
own, when the columns' mutual orthogonality was carried by entries below the threshold: the
in-memory check then passes and the file still fails it on read-back. `write_nomsd` therefore
re-checks each thresholded block and warns naming the dataset (`PsiT_k`), so the warning describes
the bytes on disk rather than the array in memory. It still only warns — the two possible causes are
a determinant that was never orthonormal (call `orthonormalize()`) and sparsification damage (lower
`threshold`), and neither is something the writer should silently decide.

> The check tolerance is 1e-10 and the sparsification threshold is 1e-8, so this warning can fire on
> a wavefunction whose error is bounded by the threshold and therefore physically negligible for
> AFQMC. That is deliberate (user call): the predicate reported is the same one `is_orthonormal`
> applies everywhere else, rather than a second, looser one that would have to be explained.

**`Lattice`** follows the same shape too: an ABC with concrete subclasses per lattice type
(`SquareLattice`/`TriangularLattice`/`HoneycombLattice`/`KagomeLattice`, plus `CustomLattice`,
see below), a `Lattice.from_dict()` classmethod replacing today's free-function `get_lattice()` factory,
and no standalone `to_hdf5()`/`from_hdf5()`. It's only ever persisted embedded in a `LatticeHamiltonian`'s HDF5 file. Exposed at the top level (`from safiretools import Lattice`)
since users may want to construct/inspect lattice geometry independent of building a full Hamiltonian.
Only the base class is re-exported, not the concrete subclasses. 
`from_dict()` handles dispatch.

### Unit-cell geometry belongs to the lattice type

`a1`, `a2` and `basis` **always exist** on a `Lattice` instance, but for the built-in types they
are not caller-settable: they *are* the type. A `SquareLattice` is square precisely because its
lattice vectors are the unit x- and y-vectors, and a `HoneycombLattice` is a honeycomb precisely
because of its 2-site basis.

**`CustomLattice` is the one subclass that takes `a1`/`a2`/`basis`, and defining your own unit
cell is exactly what makes a lattice "custom."** It is therefore the supported way to build a
lattice whose geometry isn't one of the built-in types. 

Mechanically: each concrete type implements an abstract `_geometry() -> (a1, a2, basis)` hook, and
the `Lattice` constructor takes no geometry arguments at all,
so the built-in types cannot accept them even by accident. 
Passing `a1`/`a2`/`basis` to a built-in type raises `TypeError`; the equivalent keys in a `from_dict()` 
parameter dict raise `ValueError` (a key present but set to `None` is fine, so parameter templates 
carrying unused keys still work).

**The geometry is also immutable, not merely un-settable at construction.** `a1`/`a2`/`basis` are
read-only properties over private backing state; the arrays they return have `writeable=False` and
`basis` is a tuple, so the geometry can be neither replaced (`lattice.a1 = ...`) nor edited in place
(`lattice.a1[0] = ...`, `lattice.basis.append(...)`). Construction copies whatever `_geometry()`
returns before freezing it, so freezing never reaches an array the caller still holds. `cyl_mode`
reshaping the cell inside `build()` is the one place the geometry changes, it is derived from the
type's own `_geometry()` rather than from the caller, and building twice raises `RuntimeError`. 
(`L` also changes under `cyl_mode` and is left a plain attribute; only the three geometry members are locked down.)

### Basis validation

`CustomLattice`'s docstring carried a standing BUG note: "if the magnitude of the basis vectors is
too large, there are errors with computing direct neighbors and image neighbors."
`build()` now validates the basis and raises `ValueError` for the two failure cases:

1. **Two basis vectors differing by a lattice translation** describe the same site, so sites coincide and `_to_lattice_basis` cannot tell which sublattice a position belongs to.
2. **A basis offset reaching a full supercell or more.** `_build_image_distances` shifts by only one
   supercell in each direction, so beyond that the true image is never tested.

Both are checked in *fractional* (lattice-vector) coordinates, and both depend on the boundary
conditions and the lattice size, not just the basis: an open axis neither wraps nor contributes
image shifts, so a translation that lands outside a 1-cell-wide open lattice collides with nothing. 
Folding the basis into the unit cell satisfies both conditions for any lattice.

The rule was validated against a brute-force oracle over 660 (size, offset, boundary) combinations
at L>=3: 455 admitted, all agreeing with the oracle on both the nearest-neighbor distance and the
pair count; 0 false negatives, 0 false positives.

> The oracle has to count **directed crossings**, not distinct site pairs. Neighbor pairs are one
> per boundary crossing and are deliberately not deduplicated by minimum image, because the twist
> phase depends on which way the boundary is crossed — on a 2x2 periodic lattice site *i* reaches
> site *j* both inside the cell and across the boundary, with phases 0 and +theta. An oracle that
> collapses those reports half the pairs and makes correct lattices look broken.

`cyl_mode` itself is restricted to `TriangularLattice`. The XC/YC reshaping rotates `a2` onto
`-a1 + 2*a2` and doubles the basis along `a2`, which is only the correct cell for hexagonal
geometry. 
The old code said as much in a docstring but accepted the argument from any lattice type
and silently produced a wrong cell. It now raises `ValueError` for every other type, `CustomLattice`
included.

Known bugs in `afqmctools/systems/lattice.py` to fix during the port (independent of the above):
`get_lattice()`'s `a1`/`a2` overrides are silently discarded for every built-in lattice type
(absorbed into `**kwargs`, never applied). per the rule above the fix is to **reject** them
loudly, not to start honoring them, and the same applies to `basis`, which the two types that
define a default one (honeycomb/kagome) also discarded while square/triangular honored it;
`_neighbor_distance_map`'s `min_distance` parameter is
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
- **A wavefunction with no beta electrons is `COLLINEAR` with `ndown == 0`** (user call). That is
  what `FULLYPOLARIZED` encoded, and there is no separate value for it: `dims[3]` is 2, and the beta
  blocks go to disk with zero width (`Psi0_beta` of shape `(nmo, 0, 2)`, a `PsiT_1` whose `dims` is
  `[0, nmo, 0]`), because the executable's readers open them for any collinear file.

  > The C++ side of this branch is out of date relative to `main`, and its walker setup does not
  > yet accept an empty beta sector — see **Future changes**. That is a C++ item to revisit after
  > the sync, not a constraint on the Python format.
- Two members carry the coercion that `get_spin_symm_enum` used to: `SpinSymm.from_input(value)`
  accepts a `SpinSymm`, its int value, a spelling alias (`'closed'`/`'rhf'`, `'collinear'`/`'uhf'`,
  `'noncollinear'`/`'ghf'`, ...), or another enum whose value is one of those — so partly-migrated
  code can still hand over an `afqmctools._SlaterType`. `SpinSymm.label` is the lowercase name used
  on disk and in AFQMC input files. `utils/io.py` wrote `'closed'` but its reader looked up
  `'close'`, so a closed-shell Hamiltonian's `spin_type` could never be read back; `from_input`
  accepts both spellings and `label` always writes `'closed'`.

## Known bug: `force_herm` diagonal-zeroing

`utils/matrix.py::force_herm`'s `'upper_triangular'` method (`np.triu(M,1)` then symmetrize)
zeroes the diagonal. Its only two call sites (`tband` in `nth_neighbor_hopping`, `epsilon_band`
in `onebody_onsite`) are one-body terms confirmed to be "read and used as the full matrix" (not
the U1/U2/J convention below) — so this has been silently discarding onsite/diagonal terms for
any caller using `force_herm=True` on a non-Hermitian `t`/`epsilon` input. Fix: reconstruct
including the diagonal (`np.triu(M,0) + np.triu(M,1).conj().T`).

**Distinct, correct convention — do not conflate:** the U1 and U2 pieces of the interaction matrix,
and J, follow a triangularity rule of their own. This is a different code path from `force_herm` and
stays in separate helpers — `onsite_band_matrix` and `intersite_band_matrix`.

The AFQMC executable has **no notion of sites versus bands**: it reads a triangle of the whole
interaction matrix, indexed by the combined `mu = site * nbands + band`. Per
`ModelHamOpsGenerator.cpp`, the density-density block (rows `[0, NMO)` of `Uij`) keeps `i <= j` — its
diagonal *is* read, and that is where the onsite Hubbard U sits — while the spin-spin block (rows
`[NMO, 2*NMO)`) and `Jij` keep `i < j`. Anything else is dropped with a warning. So the requirement
is that U1, U2 and J land **strictly above the combined diagonal**, leaving it free for `U`; it is
not that "the diagonal is unused" in general.

A band matrix is only the second Kronecker factor of that, and **the two cases need different
matrices**, because the site matrix they pair with maps the band triangles differently:

| | site factor | band pair `(m, n)` lands at | which band pairs are read |
|---|---|---|---|
| onsite | identity, `I == J` | `(I*nb + m, I*nb + n)` | `m < n` only — `m > n` maps below the combined diagonal and is dropped |
| inter-site | strictly upper, `I < J` | `(I*nb + m, J*nb + n)` | **all of them** — every `(m, n)` is above the combined diagonal |

So the onsite band matrix is strictly upper-triangular (the unordered band pair `{m, n}` on one site
is one interaction, counted once; `m == n` is the separate Hubbard U), while the **inter-site band
matrix must be the full matrix, lower triangle included**. `(m, n)` there is band `m` on site `I`
interacting with band `n` on site `J`, which is a different interaction from `(n, m)`.

The fix was also unreachable as written: both call sites take a *boolean* parameter named
`force_herm` that shadows the imported function, so the branch would have raised
`TypeError: 'bool' object is not callable`. The function is `force_hermitian` in `safiretools`; the
boolean parameter keeps its documented name.

## Known bug: inverted `real_valued` flag

`ham_class.py::HamiltonianComponent` set `_real_valued = np.iscomplexobj(csr_array)` — true exactly
when the component is *complex* — and `Hamiltonian.__setitem__` cleared the Hamiltonian-level flag
whenever a component was real. Every real-valued model Hamiltonian was therefore upcast to complex
on write. `LatticeHamiltonian.real_valued` is a computed property over
`HamiltonianComponent.is_complex` instead, so a real Hamiltonian writes rank-1 real data. The AFQMC
executable already accepts both ranks and the real path is exercised by its own unit-test files.

## Complex arrays on disk

SAFIRE stores a complex array by interleaving the real and imaginary parts as a trailing length-2
axis. That convention is **defined once**, in `safiretools/hdf5.py` as `to_complex`/`from_complex`,
and every schema uses it — the dense `hcore`/Cholesky matrix, the k-point `H1_kp*`/`L*` blocks, and
a model component's CSR `data_`. (Phase 1 put the pair there, per the layout above; Phase 3 then
wrote three private copies — `molecular._to_complex`, `periodic._interleave`,
`lattice_hamiltonian._to_complex` — while the shared pair went uncalled. Consolidated back onto
`hdf5.py`.)

**Real and interleaved data are told apart by rank, not by the trailing axis.** `to_complex` appends
the axis, so an interleaved dataset has rank `real_ndim + 1`, where `real_ndim` is the rank the
dataset has when stored real — 2 for a matrix, 1 for a flat array like a CSR `data_`. Testing
"is the last axis length 2?" instead is ambiguous whenever a real dataset's own last axis happens to
be 2, and silently reads it as complex: a real `(2, 2)` `hcore` (`nmo == 2`), a Cholesky matrix with
`nchol == 2`. `from_complex(data, real_ndim=...)` therefore takes the expected real rank and raises
on anything that is neither rank.

Both ranks genuinely occur — the AFQMC executable accepts either for a model Hamiltonian's
components, and a real-valued Hamiltonian is written real so the file stays half the size (see
**Known bug: inverted `real_valued` flag**) — so the distinction cannot be avoided by always writing
complex.

## Hamiltonian on-disk formats

Each `Hamiltonian` subclass owns its format, and `Hamiltonian.from_hdf5` dispatches on what a file
actually contains via the public `hamiltonian_format(path)` (which replaces
`converter.py::read_hamil_type`): `model` -> `LatticeHamiltonian`, `dense` ->
`MolecularHamiltonian`, `kpoint` -> `PeriodicHamiltonian`. Subclasses implement
`_read_hdf5(path, fmt)` rather than overriding `from_hdf5`, so dispatch stays in one place.

**Lattice metadata is recorded, the `Lattice` object is not.** A `LatticeHamiltonian`'s file carries
the *shape* of the lattice it was built on under `Hamiltonian/ModelHamiltonian/Lattice`, written in
the dimension-agnostic form a future N-dimensional `Lattice` will need — `L` and `boundaries` and
`twist` as per-axis sequences, the unit cell as a `lattice_vectors` matrix — rather than as
`L1`/`L2`/`a1`/`a2` pairs. Today `ndim` is always 2. `nbands` is written alongside, since `dims[3]`
records only `nsites * nbands`. `LatticeHamiltonian.lattice_params` flattens the metadata back into
the keys `Lattice.from_dict` takes, so a lattice can be rebuilt from a Hamiltonian file. All of this
is additive — the executable ignores groups it does not read. (This resolves the question Phase 2
left open.)

**A supercell Hamiltonian is the Γ point of the supercell.** The sparse `Hamiltonian/Factorized`
format `write_hamil_supercell` wrote is gone — the AFQMC executable's `HamiltonianFactory` offers
only `KPTHC`, `KPFactorized`, `RealDenseFactorized`, `ModelHamiltonian` and `THC` — so the supercell
path emits the ordinary k-point format with a single k-point instead: `nkpts=1`,
`nmo_pk=[nmo_tot]`, `QKTok2=[[0]]`, `MinusK=[0]`, one `L0` of shape `(1, nmo_tot**2 * nchol)`. The
Cholesky vectors stay complex, as `KPFactorizedHamiltonian` requires. Lattice models stay sparse —
they are fundamentally sparse.

This is a simplification, not a workaround: `PeriodicHamiltonian` has one on-disk format and one
in-memory representation, `kpoint_symmetry` is only a knob on the *generator* rather than a property
of the result, and nothing downstream branches on it. It also sidesteps the fact that
`RealDenseHamiltonian` reads `DenseFactorized/L` into a *real* array and so could never have carried
a k-point-mesh supercell's complex Cholesky vectors.

> **Known, molecular-only.** `ComplexIntegrals` means "the Cholesky matrix is complex" to
> afqmctools' dense *writer* and "hcore is complex" to its *reader*, so
> `write_dense(real_chol=False)` has always produced a file that reader rejects. Preserved as-is;
> `safiretools`' reader ignores the flag and takes the dtypes from the datasets.

## Periodic Cholesky: one solver, one flag

`kpoint.py`'s `KPCholesky` and `supercell.py`'s `Cholesky` become a single `PeriodicCholesky` with a
`kp_sym` flag. The factorization loop is shared verbatim; only the k-point-pair enumeration, the
pivot bookkeeping index, and the momentum-conservation test differ, each behind a small method.
`run()` is a generator yielding one momentum block at a time, so the k-point path still streams to
disk rather than materializing every `L_Q`. The flag stays inside the solver: a `kp_sym=False`
result is recast as a Γ-point Hamiltonian (above), so both modes produce the same kind of object.

Two entry points, because the `to_hdf5` contract and the existing streaming behavior pull in
different directions: `PeriodicHamiltonian.from_pyscf(...)` builds in memory and is serial-only,
while `PeriodicHamiltonian.write_from_pyscf(comm, ..., path)` generates and streams over any
communicator. Both drive the same solver and produce identical files.

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
- **The top-level import surface defines what is user-facing.** A name re-exported from
  `safiretools/__init__.py` is user-facing; anything reachable only by a deeper path is an
  implementation detail. `__all__` there is the whole definition — there is no second list to keep
  in sync, and "is this public?" is decided by one lookup. It follows that:
  - **Docstrings of user-facing objects reference only other user-facing objects, by their
    top-level path** — `safiretools.Lattice`, never
    `safiretools.hamiltonian.model.lattice.Lattice`. Dev-facing code may reference anything,
    deep paths included, since its audience is reading the tree anyway.
  - **User-facing documentation — tutorials, examples, reference prose, and error messages a user
    can hit — shows only top-level imports.** If a doc or an exception needs to name something, that
    something gets promoted; writing the deep path is not an option. This is a forcing function, not
    a formatting rule: it turns "I'll just reference the module" into an explicit decision about
    whether the thing is public. It is what promoted the FCIDUMP I/O (`read_fcidump`,
    `read_fcidump_header`, `write_fcidump`, `write_fcidump_kpoint`, plus `h1_spat2spin`/
    `h2_spat2spin`, which `write_fcidump_kpoint`'s own `NotImplementedError` tells users to call).
  - **Reference docs document the public surface from `safiretools` itself**, so that the short
    paths resolve as cross-references. Verified: `automodule:: safiretools` with `:members:` and
    `:imported-members:` yields `safiretools.Lattice`, `safiretools.HamiltonianBuilder`, ... as
    documented targets. Autodoc'ing the same classes from their implementation modules instead makes
    the top-level path unresolvable — which is why the deep paths are in the docstrings today; see
    **Future changes**.
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
- **`HamiltonianComponent` is not part of the public surface, and no documentation tells a user to
  construct one** (user call). Every custom term a user needs is reachable through a build step —
  `builder.custom_one_body(...)` for a one-body term, and the interaction build steps
  (`onsite_hubbard`, `hubbard_U1_density_density`, `nth_order_hubbard_Vij`, ...) for everything
  else — or through the equivalent key in the input dict. The build steps also handle the spin
  structure and index mapping that hand-built components had to get right themselves.
- **`nelec` and `spin_symm` are Hamiltonian state, not write-time arguments.** They are set when
  the Hamiltonian is built (the `hamiltonian` input block, or the `HamiltonianBuilder` constructor)
  and `to_hdf5(path)` takes only a path — replacing
  `write_model_hamiltonian(ham, fname, nelec=..., spin_symm=...)`. In input files `nelec` therefore
  belongs in the `[hamiltonian]` block, not a separate `[misc_params]`/`[cli_params]` section.

  > **`nelec` is on its way out entirely** — see **Future changes**. It is Hamiltonian state only
  > because the on-disk formats still record it; the executable already ignores those fields. This
  > bullet describes where it lives *today*, and reduces to `spin_symm` alone once it is gone.
- **A wavefunction's `psi0` (the AFQMC initial walker) is instance state, not a write-time
  argument.** `Wavefunction.psi0` defaults to something derived from the wavefunction itself — the
  leading determinant's spin blocks for a NOMSD, identity columns at the leading occupations for a
  PHMSD — and can be assigned. This replaces `write_wfn(..., init=...)` and `orbmat=`, the latter
  becoming `PHMSDWavefunction(orbitals=...)`.
- **`to_hdf5()` replaces the Hamiltonian in its target file, not the whole file.** It opens the file
  in append mode (creating it if absent) and deletes an existing `Hamiltonian` group before writing.
  A SAFIRE input file holds **at most one Hamiltonian and at most one wavefunction**, so replacing
  rather than adding is the right semantics, and anything else in the file — notably `Wavefunction` —
  survives. This is what lets a Hamiltonian and a wavefunction share one file **in either order**,
  which is the common case; afqmctools' `write_wfn` already used exactly this pattern for
  `Wavefunction`, so the two are now symmetric. Applies to `PeriodicHamiltonian.write_from_pyscf`
  too, whose per-rank scratch files are still truncated since they hold only one run's partial
  blocks. (HDF5 unlinks rather than reclaims, so repeatedly rewriting into one file grows it.)
- `AFQMC_EXEC` must never be required just to import safiretools (today's
  `RuntimeError: AFQMC_EXEC environment variable is not set` fires at import time in
  `tutorial_utils/helper.py` — becomes a lazy check, only triggered when something actually
  invokes SAFIRE). A future thin binding to invoke SAFIRE in-process (possibly nanobind) is
  planned — whatever "run SAFIRE" API safiretools exposes should not hard-code a
  subprocess/executable-path assumption that would preclude that later.
- Dependency cleanup: drop `pytables` (`stats/config_h5.py`'s only reason for it, rewritten onto
  `h5py`); merge the `LATTICE_HF` optional-dependency group into `AUTOHF` (exact duplicate).

### Every factory dispatches from the base class

**Every construction factory is reachable from the ABC, which picks the concrete subclass** (user
call). The canonical safiretools script never names a subclass:

```python
from safiretools import Hamiltonian, Wavefunction

hamiltonian = Hamiltonian.from_hdf5("hamiltonian.h5")     # or any other factory
wavefunction = Wavefunction.from_hdf5("wavefunction.h5")  # or any other factory
```

The point is to minimize user friction and the chance of a mistake: picking the wrong subclass is an
error the library can simply not have. **The subclass factories stay** as aliases (user call) — a
caller who wants the type guarantee keeps it, and generic code can always be written against the
base class.

This matters more for `Wavefunction` than for `Hamiltonian`, because the two hierarchies split on
different things (see **Class hierarchies**). A `Hamiltonian` subclass is the *source domain*, which
the caller always knows — they ran a molecule or a solid. A `Wavefunction` subclass is the
*mathematical representation*, which often falls out of the **data**: `from_pbc_scf` cannot know
whether it will produce a particle-hole expansion until it sees whether bands came out partially
occupied. Dispatching is therefore not merely a convenience there — it is the only honest signature.

**Mechanism.** The base classmethod imports its subclasses *inside the method body*, which is how
`Hamiltonian.from_hdf5` and `Wavefunction.from_hdf5` already avoid the circular import (the
subclasses import the base module at module scope). **No `_Hamiltonian`/`_Wavefunction` split is
needed**, and none should be added.

**The aliases are inherited, not duplicated.** A factory defined on the ABC is already reachable as
`NOMSDWavefunction.from_pyscf(...)` — that *is* the alias, and it cannot drift from the base method
because it is the same function object. The cost is that inheritance offers **every** factory on
**every** subclass, including the ones that cannot produce that subclass: `Wavefunction` and
`Hamiltonian` each carry factories that are fixed to one representation or one domain. So every such
factory guards the class it was called on and raises a `ValueError` naming both classes rather than
quietly returning the wrong type — `PHMSDWavefunction.from_free_electron(...)` does not hand back an
`NOMSDWavefunction`.

**There are two guard mechanisms, because the target is known at two different times.** A
fixed-answer factory knows its target from its own definition and guards with `_check_representation`
(`wavefunction/base.py`) or `_check_domain` (`hamiltonian/base.py`). `from_hdf5` cannot: its target
comes from the *file*, so it resolves the format first and then applies the same test inline
(`issubclass(target, cls)`). That is why `NOMSDWavefunction.from_hdf5` on a particle-hole file raises
instead of returning a `PHMSDWavefunction`, while `Wavefunction.from_hdf5` on the same file returns
one.

| factory | guard | refuses |
|---|---|---|
| `Wavefunction.from_free_electron` / `.from_pyscf` | `_check_representation` | called on `PHMSDWavefunction` |
| `Wavefunction.from_pyscf_cas` / `.from_dice` | `_check_representation` | called on `NOMSDWavefunction` |
| `Hamiltonian.from_dict` | `_check_domain` | called on `Molecular`/`PeriodicHamiltonian` |
| `Hamiltonian.from_integrals` | `_check_domain` | called on `Lattice`/`PeriodicHamiltonian` |
| `Hamiltonian.write_from_pyscf` | `_check_domain` | called on `Lattice`/`MolecularHamiltonian` |
| `Hamiltonian.from_pyscf` | `_check_domain` on the resolved target | called on `LatticeHamiltonian` — but see the gap below |
| `Hamiltonian.from_hdf5` / `Wavefunction.from_hdf5` | inline `issubclass(target, cls)` | a stored format the subclass does not read |
| `Wavefunction.from_pbc_scf` | **none, deliberately** | — see below |

**A guard only runs when the call actually reaches the base method.** In the `Wavefunction` hierarchy
that is always: both subclasses *inherit* every fixed-answer factory, so every mismatched call is
refused. In the `Hamiltonian` hierarchy the implementations *are* the subclass classmethods (above),
so a subclass that defines a factory does not route through the base and is not guarded — which is
harmless wherever the defining class is the only right answer (`LatticeHamiltonian.from_dict`,
`MolecularHamiltonian.from_integrals`, `PeriodicHamiltonian.write_from_pyscf`).

> **Known gap: `from_pyscf` is the one factory two Hamiltonian subclasses both define**, so the
> cross-call is the one mismatch nothing catches. `Hamiltonian.from_pyscf` and
> `LatticeHamiltonian.from_pyscf` reach the guarded base method, but
> `MolecularHamiltonian.from_pyscf(periodic_scf_data)` runs the molecular implementation directly and
> dies on `KeyError: 'walker_type'`, and `PeriodicHamiltonian.from_pyscf(molecular_scf_data)` on
> `KeyError: 'cell'` — instead of the `ValueError` naming both classes that every other mismatch
> gets. The dispatching `Hamiltonian.from_pyscf` is unaffected and remains the recommended call.

`Wavefunction.from_pbc_scf` is the one base factory with no guard, and that is correct rather than an
oversight: it is the honest dispatcher, returning whichever representation the occupancies call for,
so there is no target to check it against. The subclasses do not inherit it — they **override** it
with narrowing forms that validate in their own way (below), which is where the equivalent refusal
lives.

**What each factory dispatches on:**

| factory | dispatches on |
|---|---|
| `Hamiltonian.from_hdf5` | `hamiltonian_format(path)` |
| `Wavefunction.from_hdf5` | `wavefunction_format(path)` |
| `Hamiltonian.from_pyscf` | `'cell' in scf_data` -> periodic, `'mol'` -> molecular |
| `Wavefunction.from_pbc_scf` | whatever `wavefunction/pbc.py::from_pbc_scf` returns |
| `Hamiltonian.from_dict` / `.from_integrals` / `.write_from_pyscf` | fixed (lattice / molecular / periodic) |
| `Wavefunction.from_free_electron` / `.from_pyscf` / `.from_dice` / `.from_pyscf_cas` | fixed (NOMSD / NOMSD / PHMSD / PHMSD) |

A fixed-answer factory still belongs on the base class: the caller should not have to know that
`from_dice` happens to produce a particle-hole expansion in order to ask for one.

**The two hierarchies differ in where the implementation lives, and the signatures follow.**
`Wavefunction`'s factories delegate to *free functions* in `wavefunction/{free_electron,pyscf,pbc,
dice}.py`, so the base method is the only wrapper and spells out **real parameters**; the subclasses
add nothing. `Hamiltonian`'s implementations *are* the subclass classmethods, so the base method
delegates to them and forwards `**kwargs`, leaving the concrete classmethod the single place the
defaults and the parameter documentation live.

**`scf_data` is told apart by key, not by type.** `pyscf.pbc.gto.Cell` is a *subclass* of
`gto.Mole`, so an `isinstance` test on the object would report a periodic cell as molecular. The
loaders are disjoint on the key — `load_from_pyscf_chk` stores `'cell'` (plus `'kpts'`, `'nmo_pk'`)
and `load_from_pyscf_chk_mol` stores `'mol'` — and that is the discriminator.

**`Hamiltonian.from_pyscf` forwards `**kwargs`** (user call). The two subclass signatures share only
`scf_data`, `chol_cut` and `verbose`; everything else is domain-specific
(`cas`/`ortho_ao`/`df`/`real_chol` molecular, `comm`/`kpoint_symmetry`/`maxvecs`/`exxdiv` periodic).
A merged union signature would silently accept `kpoint_symmetry=` for a molecular calculation, so the
base method forwards instead and lets the concrete classmethod raise its own `TypeError` naming the
real parameter. The docstring carries both parameter lists, and `inspect.signature` on the concrete
classmethod is still exact.

**`NOMSDWavefunction.from_pbc_scf` is not a pure alias** and stays documented as a *narrowing* form:
it forces `ndet_max=1` to guarantee a single determinant, where `Wavefunction.from_pbc_scf` passes
`ndet_max` through and returns whichever representation the occupancies call for.
`PHMSDWavefunction.from_pbc_scf` likewise still raises when a single determinant would describe the
system exactly.

## Future changes

Deliberately **not** in the current task list — recorded here so they are not lost, and so nobody
mistakes them for accidents. Add to these lists rather than widening a phase in progress.

### Things we will change

- **Shorten the deep paths in user-facing docstrings, and document the public surface from
  `safiretools`.** Four `lattice : ~safiretools.hamiltonian.model.lattice.Lattice` type fields
  (`HamiltonianBuilder` and its `from_input`, `LatticeHamiltonian.from_dict`, and the dev-facing
  `lattice_metadata_from`) violate the rule above. They are written that way because a bare
  `Lattice` is an **ambiguous cross-reference** while both `safiretools` and `afqmctools` are
  autodoc'd, and the short `safiretools.Lattice` only resolves once the public surface is documented
  at that path. The two halves therefore land together: a user-facing API page autodoc'ing
  `safiretools`, with those classes no longer autodoc'd from their implementation modules on
  user-facing pages, and the docstrings shortened. Belongs with **Phase 8** (top-level API surface);
  the underlying ambiguity disappears anyway at **Phase 9** when afqmctools is retired.
  The same page change decides whether `hamiltonian_format()` is promoted — it is genuinely useful
  as "what is in this file?", but it is not public today, so the reference prose no longer names it.
- **Remove `nelec` from Hamiltonians entirely — C++ and Python.** The electron count is a property
  of the *problem*, not of the Hamiltonian; it sits on `Hamiltonian` today only because the on-disk
  formats record it. The end state is that **nothing writes `nelec` to a Hamiltonian and nothing
  reads `nelec` from one**; `dims[4]`/`dims[5]` become unused.

  **The C++ side is already there.** Every Hamiltonian reader loads the 8-element `dims`
  (`HamiltonianFactory.cpp`, `RealDenseHamiltonian.cpp`, `KPFactorizedHamiltonian.cpp`,
  `ModelHamOpsGenerator.cpp`) but uses only `Idata[2]` (nkpts) and `Idata[3]` (NMO) — no reader
  touches `Idata[4]`/`Idata[5]`. Electron counts reach the executable from the wavefunction. Note
  the readers still declare `Idata(8)`, so **the two slots go unused rather than disappearing**: the
  array keeps its length and the fields are written as zero (or dropped from the writer while the
  readers keep skipping them). Changing the array length would be a format break, and is not part
  of this.

  **The Python side is what there is to do.** On all three subclasses: the `nelec` constructor
  argument and `.nelec` attribute; `MolecularHamiltonian.from_integrals`/`from_pyscf`;
  `LatticeHamiltonian`'s `nelec` key in the `hamiltonian` input block (`_parse_ham_input`'s
  `_known_params`) and `HamiltonianBuilder(nelec=)`; `PeriodicHamiltonian.from_pyscf` /
  `write_from_pyscf` and `_default_nelec`; the `dims` writes in `write_dense_hamiltonian`,
  `LatticeHamiltonian.to_hdf5` and `_write_kpoint_descriptors`; and the reads in
  `read_dense_hamiltonian` and each `_read_hdf5`.

  Two things this does **not** touch:

  - **FCIDUMP keeps its `nelec`.** `NELEC` is a field of that external format's own header, so
    `read_fcidump`, `read_fcidump_header`, `write_fcidump` and `write_fcidump_kpoint` are unaffected.
  - **The periodic generator still needs an electron count as an argument**, transiently:
    `from_pyscf`/`write_from_pyscf` pass `sum(nelec)` to `_zero_electron_energy`, which computes the
    Madelung/`exxdiv` correction that goes into `enuc`. Deleting that argument along with the
    attribute would silently change the constant energy. The rule is about *stored state and file
    fields*, not about arguments used to compute something else.

  Doc churn to expect: the `nelec` key currently in the `[hamiltonian]` block of the eleven
  `docs/snippets/01_setting_up/*/input*.toml` files, and in the Python parameter dicts across
  `docs/examples/models/*` and `docs/tutorials/models/*`, all comes back out — it went *in* during
  Phase 3b precisely because `to_hdf5` records it (see **Public API patterns**), so that bullet
  changes too.

- **Port `docs/tutorials/solids/04_computing_observables` off afqmctools.** It is the last doc source
  still calling `afqmctools.hamiltonian.converter.read_hamiltonian`, because its provided `hamil.h5`
  is in **CoQuí** format — `hamiltonian_format()` identifies it as `kpoint_coqui`, and safiretools
  has no reader for that, so `Hamiltonian.from_hdf5` raises `NotImplementedError`. Blocked on the
  observables rewrite, but it **must** be ported before afqmctools is removed. A comment in the
  tutorial cell records the same.

- **Let a collinear walker set have an empty beta sector (C++), after the C++ side is synced with
  `main`.** This is what the `FULLYPOLARIZED`-removal decision above implies, and it is *not* done
  on this branch — but the C++ here is out of date relative to `main`, so **re-check it against
  `main` before acting on any of the following.** As observed on this branch (2026-09):
  `walker::SlaterMatrix(Beta)` guards on `desc[2] > 0` while
  `WalkerSetBase::populate_from_guess` writes the beta block unconditionally, so a `COLLINEAR`
  wavefunction with `ndown == 0` is read correctly and the run then aborts with
  ``error:walker spin out of range in SlaterMatrix(SpinType)``. Every downstream consumer of a
  collinear walker's beta sector wants the same audit, so this is a deliberate change rather than a
  one-line guard.

  Two things wait on it: retiring `FULLYPOLARIZED` from `src/AFQMC/config.h` and
  `WalkerSetBase::parse_walker_type` (it is still parsed from input today), and porting
  `docs/examples/molecules/04_V-fully_polarized`, whose `afqmc.json` still asks for
  `"walker_type": "FULLYPOLARIZED"` — a file safiretools cannot write.

### Things we might change

- **The periodic multi-determinant expansion keeps the *least* probable determinants.**
  `reoccupy` selects with `probabilities.argsort()[:ndets]`, and `argsort` is ascending, so
  determinant 0 — which the executable takes as its reference configuration — is the least likely
  configuration rather than the most likely. Almost certainly a bug, fixed by one `[::-1]`, but it
  changes numerics on a path with established behavior (and would break the `ndet_max=4` periodic
  equivalence check), so it is preserved verbatim pending a decision.
- **Direct use of NOMSDWavefunction and PHMSDWavefunction in tutorials is potentially confusing.**
  (This applies mostly to the Molecules writting a Wavefunction tutorial) We added to factories to the 
  Wavefunction baseclass to specifically avoid users needed to do this; however, one could argue that
  it is better for pedagogical reasons to work with explicit classes here; in that case, we can add a
  section near the end (or maybe beginning?) that demonstrates that we could have written all of the
  wavefunctions using `Wavefunction.from_[X]` etc.

## Open questions

- `execution.py`'s naming/location, and whether it eventually gets its own CLI entry point (the
  CLI is currently scoped to `scalar_stats` only) — explicitly deferred to a broader team
  discussion, not just this session.
