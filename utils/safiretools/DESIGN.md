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

## Transitional shim: `isinstance` gates in the un-ported packages

Until the packages that consume a lattice model Hamiltonian are themselves ported, three
`isinstance(source, Hamiltonian)` checks that tested only for afqmctools'
`ham_class.Hamiltonian` accept a `safiretools.LatticeHamiltonian` as well (user call):
`afqmctools/wavefunction/free_electron.py`, `afqmctools/inputs/from_autohf.py`, and
`AutoHF/autohf/hamiltonian.py`. `LatticeHamiltonian` already satisfies the accessor contract those
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
│                           #   h1_spat2spin, h2_spat2spin), Wavefunction, mean_and_error, ...
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
(`SquareLattice`/`TriangularLattice`/`HoneycombLattice`/`KagomeLattice`, plus `CustomLattice` —
see below), a `Lattice.from_dict()` classmethod replacing today's free-function `get_lattice()`
factory, and no standalone `to_hdf5()`/`from_hdf5()` — it's only ever persisted embedded in a
`LatticeHamiltonian`'s HDF5 file. Exposed at the top level (`from safiretools import Lattice`)
since users may want to construct/inspect lattice geometry independent of building a full
Hamiltonian; only the base class is re-exported, not the concrete subclasses — `from_dict()`
handles dispatch.

### Unit-cell geometry belongs to the lattice type

`a1`, `a2` and `basis` **always exist** on a `Lattice` instance, but for the built-in types they
are not caller-settable: they *are* the type. A `SquareLattice` is square precisely because its
lattice vectors are the unit x- and y-vectors, and a `HoneycombLattice` is a honeycomb precisely
because of its 2-site basis — handing either a different `a1` would produce something that is no
longer the type it claims to be.

**`CustomLattice` is the one subclass that takes `a1`/`a2`/`basis`, and defining your own unit
cell is exactly what makes a lattice "custom."** It is therefore the supported way to build a
lattice whose geometry isn't one of the built-in types, not a redundant alias for them. (It is
also why `CustomLattice` is ported rather than dropped, even though the port fixes the bug that
made the built-in types ignore geometry arguments.)

Mechanically: each concrete type implements an abstract `_geometry() -> (a1, a2, basis)` hook, and
the `Lattice` constructor — the only one, with no `**kwargs` — takes no geometry arguments at all,
so the built-in types cannot accept them even by accident. Passing `a1`/`a2`/`basis` to a built-in
type raises `TypeError`; the equivalent keys in a `from_dict()` parameter dict raise `ValueError`
(a key present but set to `None` is fine, so parameter templates carrying unused keys still work).

**The geometry is also immutable, not merely un-settable at construction.** `a1`/`a2`/`basis` are
read-only properties over private backing state; the arrays they return have `writeable=False` and
`basis` is a tuple, so the geometry can be neither replaced (`lattice.a1 = ...`) nor edited in place
(`lattice.a1[0] = ...`, `lattice.basis.append(...)`). Construction copies whatever `_geometry()`
returns before freezing it, so freezing never reaches an array the caller still holds. `cyl_mode`
reshaping the cell inside `build()` is the one place the geometry changes, it is derived from the
type's own `_geometry()` rather than from the caller, and building twice raises `RuntimeError` — so
once a lattice is built its geometry is fixed. (`L` also changes under `cyl_mode` and is left a
plain attribute; only the three geometry members are locked down.)

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
geometry — the old code said as much in a docstring but accepted the argument from any lattice type
and silently produced a wrong cell. It now raises `ValueError` for every other type, `CustomLattice`
included.

Known bugs in `afqmctools/systems/lattice.py` to fix during the port (independent of the above):
`get_lattice()`'s `a1`/`a2` overrides are silently discarded for every built-in lattice type
(absorbed into `**kwargs`, never applied) — per the rule above the fix is to **reject** them
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

### Things we might change

*(nothing recorded here at the moment.)*

## Open questions

- `execution.py`'s naming/location, and whether it eventually gets its own CLI entry point (the
  CLI is currently scoped to `scalar_stats` only) — explicitly deferred to a broader team
  discussion, not just this session.
