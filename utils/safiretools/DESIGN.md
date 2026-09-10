# safiretools — Design Decisions

Status: living document. Records decisions actually made, not the discussion that produced them.
Anything not listed here is still undecided. Don't infer intent beyond what's written.

`safiretools` replaces the existing `afqmctools` + `stats` packages under `utils/`. `AutoHF` is a
separate package (its own git history) and is out of scope for this document.

## Key design points

One short statement per decision. The sections below carry the detail; nothing here is a new
decision.

**Package and API surface**

- One package, `safiretools`, replaces `afqmctools` + `stats`; `AutoHF` stays separate.
- Deep implementation tree, curated flat re-exports from `safiretools/__init__.py`.
- `__all__` in `safiretools/__init__.py` is the sole definition of what is public.
- User-facing docstrings, documentation and error messages name only top-level paths.
- `scalar_stats` is the only CLI entry point kept.
- Importing safiretools never requires `AFQMC_EXEC`, and never pulls in pyscf or mpi4py.
- "No internal callers found in `utils/`" is never on its own a reason to drop something.

**Hamiltonian and Wavefunction**

- `Hamiltonian` splits by source domain: lattice / molecular / periodic.
- `Wavefunction` splits by representation: NOMSD / PHMSD.
- `spin_symm` is a plain attribute on both hierarchies, never a subclass axis.
- Construction is by classmethod factory; `__init__` takes already-formed, validated data.
- A factory lives on the class that can produce it: `Hamiltonian` keeps only the shared `from_hdf5`
  and defines each domain factory on its subclass; every `Wavefunction` factory is on the ABC.
- `Wavefunction`'s inherited factories guard the class they were called on; a `Hamiltonian` domain
  factory needs no guard, because a sibling's factory is simply not there to call.
- `nelec` and `spin_symm` are Hamiltonian state, not write-time arguments.
- `psi0` is wavefunction state with a derived default, not a write-time argument.
- A determinant carries one column block per independent spin channel.
- Writing never orthonormalizes — `orthonormalize()` is explicit and `to_hdf5` only warns.

**On-disk formats**

- Serialization is the pair `obj.to_hdf5(path)` / `Cls.from_hdf5(path)`.
- `Hamiltonian` subclasses each implement the pair; `Wavefunction` implements it once, in the base.
- `from_hdf5` dispatches on the file's own contents, via `hamiltonian_format` /
  `wavefunction_format`.
- A Hamiltonian file records its own format in `Hamiltonian/type`; the layout heuristic is the
  fallback for files written before that key existed.
- `HamiltonianFormat` pairs each format's safiretools name with its on-disk tag, and is not public.
- `to_hdf5` replaces its own group in the target file, so a Hamiltonian and a wavefunction can
  share one file in either order.
- A complex array goes to disk as an interleaved trailing length-2 axis, defined once in `hdf5.py`.
- Real and interleaved data are told apart by rank, never by the trailing axis length.
- `SpinSymm` is a 3-member `IntEnum` whose values are the C++ `WALKER_TYPES` wire format.
- A wavefunction with no beta electrons is `COLLINEAR` with `ndown == 0`.
- A `LatticeHamiltonian` file records the lattice's metadata dimension-agnostically, not the
  `Lattice` object.
- One `PeriodicCholesky` solver with a `kp_sym` flag, and a supercell is the Γ point of the
  supercell.

**Lattice**

- Lattice geometry (`a1`/`a2`/`basis`) *is* the lattice type: immutable, and caller-supplied only
  on `CustomLattice`.
- `Lattice.from_dict()` is the factory; concrete subclasses are not re-exported.
- `build()` validates the basis and runs once per instance.

**Statistics and conventions**

- One generic `mean_and_error(samples, axis=0)` serves every statistics path.
- Library code raises typed exceptions; `sys.exit()` is confined to the `scalar_stats` CLI.
- Functions never mutate caller-supplied arguments.
- Logging is the standard `logging` module only, with no always-on progress channel.

## Scope

**Kept, ported now:**
- Hamiltonian construction/IO (lattice-model builder, molecular, k-point/supercell periodic).
- Wavefunction construction/IO (free-electron, molecular, PBC).
- 1-RDM statistics path (equilibration, autocorrelation-aware averaging).
- Execution-input (AFQMC run-config JSON) generation — redesigned, not just ported.
- `scalar_stats` — the only CLI entry point kept (`energy_stats`, a duplicate alias, is dropped).
- QE interop — bugs fixed, behavior preserved.
- CAS/CI wavefunction import from a PySCF checkpoint (`write_cas_wfn`) — kept as
  `PHMSDWavefunction.from_pyscf_cas`; `ci_wavefunction` came with it, as `ci_expansion`.
- AutoHF interop — kept, fragile unguarded import gets hardened.
- Dice-SHCI wavefunction import — kept, split into its own module, proper exceptions.
- `rhonk.py` (real-space observables) — kept for external callers; its vendored duplicate HDF5
  helpers get deduplicated onto the shared Core Library HDF5 utility.
- `write_rhoG_kpoints`/`write_rhoG_supercell` — kept only as a **raising stub**, the single
  `periodic.write_rhoG(comm, scf_data, path, gcut, ...)`, which explains why in its
  `NotImplementedError` (user call). Neither original could run, so there is no behavior to
  preserve; the stub gives a working implementation an obvious place to land.

**Deferred to a future, C++-spanning observables rewrite (not designed here):**
- The broader RDM/observable pipeline: 2RDM, generalized Fock matrix, spin-spin, pair-correlation,
  EKT/NOON analysis (currently `afqmctools/analysis/average.py` + `extraction.py`, essentially
  unintegrated today).
- `inputs/energy.py`-style Hamiltonian/HF-energy validation.

**Dropped entirely:**
- The AutoHF variational-energy measurement inside the free-electron wavefunction builder
  (`free_electron(measure_evar=...)`, `measure_spin`, `return_autohf`) — building a wavefunction and
  evaluating its energy are separate calls, and `wavefunction/` must not depend on the AutoHF
  interop that Phase 7 owns.
- `wavefunction/pbc.py::slater_gto2mo` — no callers anywhere, unable to run as written, and the
  conversion it describes is what `from_pyscf`'s basis transform already does.
- `wavefunction/pbc.py::write_wfn_pbc_old` — superseded by `write_wfn_pbc`.
- `qe_driver.py` — unimportable as it stands (references a nonexistent module) and superseded.
- AIMBES interop (`aimbes_utils.py`, `aimbes_to_2nd_quant`/`aimbes_to_afqmc` CLIs) — CoQuí
  (formerly AIMBES) now generates its own SAFIRE-compatible inputs directly. Note the current
  location in case this changes back; do not port. References to AIMBES should say CoQuí.
- `analysis/new_rdm.py` — a third, weaker parallel 1-RDM implementation, fully superseded, no
  known external dependents.
- The entire CLI surface except `scalar_stats`.
- `FULLYPOLARIZED` as a distinct spin-symmetry value; a wavefunction with no beta electrons is
  `COLLINEAR` with `ndown == 0` instead as in the current C++ (see **Spin-symmetry enum**).

**Rule for all of the above:** "no callers found in `utils/`" is never sufficient reason to drop
something on its own. These are library packages with external callers writing their own AFQMC
workflows. Everything marked "dropped" above was an explicit decision, not an inference from
caller-count.

## Accepting a `LatticeHamiltonian` in afqmctools and AutoHF

Until the packages that consume a lattice model Hamiltonian are themselves ported, three
`isinstance(source, Hamiltonian)` checks that tested only for afqmctools'
`ham_class.Hamiltonian` accept a `safiretools.LatticeHamiltonian` as well (user call):
`afqmctools/wavefunction/free_electron.py`, `afqmctools/inputs/from_autohf.py`, and
`AutoHF/autohf/hamiltonian.py`. `LatticeHamiltonian` already satisfies the accessor contract those
consumers use (`nsites`, `nbands`, `get_one_body()`, `get_U()`, `get_J()`, `get_heisenberg()`), so
only the type test needed widening.

The AutoHF change lives in a **separate submodule/repository** and needs its own commit. It is
guarded with `importlib.util.find_spec("safiretools")`, matching the pattern AutoHF already uses
for afqmctools, so AutoHF still imports without safiretools installed.

The two afqmctools gates go away with the package at **Phase 9**. `AutoHF`'s gate is not
transitional: AutoHF must keep importing without safiretools installed, so its `find_spec`-guarded
check stays — at Phase 9 it simply stops testing for afqmctools' type as well.

## Package layout (sketch — not fully confirmed, see Open Questions)

```
safiretools/
├── __init__.py            # curated flat re-exports; this list defines what is user-facing.
│                           #   Deliberately NOT re-exported: the concrete Lattice subclasses,
│                           #   HamiltonianComponent, and the FCIDUMP format internals
│                           #   (fcidump_header, check_sym, fmt_integral)
├── types.py                # canonical SpinSymm and HamiltonianFormat enums — dependency-free,
│                           #   no upward imports
├── hdf5.py                  # HDF5 read/write primitives — dependency-free, top-level like
│                            #   types.py (replaces utils/io.py's generic bits, rhonk.py's
│                            #   vendored copy, stats/config_h5.py). Also owns to_complex/
│                            #   from_complex (see "Complex arrays on disk").
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
│   │   ├── lattice_hamiltonian.py  # LatticeHamiltonian(Hamiltonian), HamiltonianComponent
│   │   └── lattice.py         # Lattice (ABC) + SquareLattice/TriangularLattice/
│   │                          #   HoneycombLattice/KagomeLattice/CustomLattice, moved from
│   │                          #   afqmctools/systems/lattice.py
│   ├── molecular.py           # MolecularHamiltonian(Hamiltonian) — from hamiltonian/mol.py
│   ├── periodic.py            # PeriodicHamiltonian(Hamiltonian) — merges kpoint.py+supercell.py
│   └── fcidump.py             # FCIDUMP external-format I/O
├── wavefunction/
│   ├── base.py                # Wavefunction ABC — spin_symm, nelec, nmo; implements
│   │                          #   to_hdf5()/from_hdf5() ONCE, not per-subclass
│   ├── nomsd.py                # NOMSDWavefunction(Wavefunction): coeffs + dets
│   ├── phmsd.py                # PHMSDWavefunction(Wavefunction): coeffs + occa/occb
│   ├── free_electron.py       # from_free_electron() implementation; model.py's legacy
│   │                          #   duplicate retired
│   ├── pyscf.py                # from_pyscf() implementation (from wavefunction/mol.py)
│   ├── pbc.py                  # from_pbc_scf() implementation
│   ├── dice.py                 # from_dice() implementation, split out of wavefunction/converter.py
│   ├── slater.py               # domain-independent Slater-matrix operations: make_slater(),
│   │                            #   transform_slater(), spin_blocks(), modified_gram_schmidt(),
│   │                            #   is_orthonormal()
│   └── io.py                  # native SAFIRE HDF5 schema read/write (uses top-level hdf5.py),
│                               #   shared by both NOMSDWavefunction and PHMSDWavefunction
├── execution.py              # redesigned AFQMC JSON execution-parameter generator
│                              #   (from inputs/from_hdf.py) — naming/location still open
├── convert/
│   ├── pyscf.py                # thin orchestration: calls hamiltonian.molecular +
│   │                            #   wavefunction.pyscf (NOMSDWavefunction.from_pyscf)
│   └── autohf.py                # hardened AutoHF interop
└── qe/                        # relocated QE interop (qe_tools.py, qe_utils.py contents)
```

`observables/` (rhonk.py et al.) stays close to its current shape for now, pending the
C++-spanning rewrite — only the HDF5-helper dedup happens in this pass.

## Class hierarchies

**`Hamiltonian`** splits by *source domain* — `LatticeHamiltonian`/`MolecularHamiltonian`/
`PeriodicHamiltonian`, because those are genuinely different storage formats. Each implements its
own `to_hdf5()`/`from_hdf5()`. `spin_symm` is a plain attribute on instances, not a subclass axis.

**`Wavefunction`** splits by *representation*, not domain — `NOMSDWavefunction` (coeffs + per-
determinant orbital matrices) and `PHMSDWavefunction` (coeffs + occa/occb occupation-number
strings). Domain (free-electron / PySCF / PBC / Dice) becomes a factory classmethod on the
appropriate subclass rather than its own subclass: a free-electron wavefunction and a PySCF UHF
wavefunction are the same NOMSD representation, just built differently. `spin_symm` is a plain
attribute here too.

So both hierarchies have an ABC, concrete subclasses for what is *structurally* different (storage
format for Hamiltonian, mathematical representation for Wavefunction), and plain attributes for what
is just *data* (spin_symm) or *provenance* (which external tool built it).

`Wavefunction`'s base class implements `to_hdf5()`/`from_hdf5()` once, since the format is identical
across domains. It is implemented the way `Hamiltonian`'s dispatch is: the base owns the file
handling, the format detection (`wavefunction_format(path)` -> `nomsd` / `phmsd`) and the shared
header, and each subclass supplies only a `_write_payload`/`_read_payload` hook for its own part.
`PHMSDWavefunction` is always `SpinSymm.COLLINEAR` — the AFQMC executable's
`read_ph_wavefunction_hdf` rejects both closed-shell and noncollinear particle-hole wavefunctions —
so spin symmetry is a plain attribute there in the sense of "recorded", not "free".

### A wavefunction's determinants carry one block per independent spin channel

`NOMSDWavefunction.dets` is `(ndets, npol*nmo, sum(nelec_per_spin))`: `nup` columns for a
closed-shell wavefunction (the beta channel repeats alpha), `nup + ndown` when collinear, and
`nup + ndown` over `2*nmo` rows when noncollinear. `Wavefunction.nelec_per_spin` is the one place
that mapping lives.

`nelec` is the physical `(nup, ndown)` in memory; `nelec_on_disk` derives the `(nup + ndown, 0)`
pair a noncollinear file's `dims` records. Reading such a file back therefore reports
`(nup + ndown, 0)` — the split is not part of the format, and the executable does not use it.

### Operations on that layout live in `wavefunction/slater.py`

Building a Slater matrix, transforming it into another basis, splitting it into spin blocks, and
checking or restoring its orthonormality are all independent of where the orbitals came from, so
they live in one module rather than in whichever domain module needed them first. A domain module
keeps only what decodes *its own* conventions — `pyscf.py` keeps the `mo_occ` readers (which
orbitals PySCF calls occupied, and how an ROHF reference packs both channels into one vector),
`pbc.py` keeps its fractional per-k-point occupancy logic.

`slater.py` therefore imports nothing from the package but `types.py`, which is what lets `io.py`
and `base.py` both use it without an import cycle.

### Writing never orthonormalizes

`orthonormalize()` is an explicit method returning a *new* instance, which the domain factories call
by default; `to_hdf5` warns about a non-orthonormal Slater matrix rather than quietly repairing it.

**Every Slater matrix is covered, `psi0` included.** Both `orthonormalize()` and the write-time check
walk the same set, so there is no dense array that only one of them sees:

| matrix | `orthonormalize()` | checked by `to_hdf5` |
|---|---|---|
| `NOMSDWavefunction.dets[i]`, per spin channel | ✅ | ✅ as `dets[i] spin s` |
| `psi0` supplied explicitly, or assigned afterwards | ✅ | ✅ as `psi0 spin s` |
| `psi0` left to default | ✅ (re-derived from the fixed determinants) | ✅ — it *is* a copy of `dets[0]`'s blocks, so the determinant check covers it, and reports it under the determinant's name |
| `PHMSDWavefunction.orbitals` | ✅ | ✅ as `orbitals[i]` |
| `PHMSDWavefunction`'s default `psi0` | ✅ | ✅ — identity columns, orthonormal by construction |

**Blocks are checked per independent spin channel**, which is the physically correct grain: a
collinear determinant's alpha and beta columns describe different spin sectors and need not be
orthogonal to each other, while a noncollinear determinant is a single `(2*nmo, nup + ndown)` block
and is checked whole. `Wavefunction.nelec_per_spin` supplies the split, as everywhere else.

**The `PsiT` blocks are checked again after sparsifying, because that is what reaches disk.** The
1e-8 threshold applies only to the sparse `PsiT` blocks it exists to sparsify, never to the dense
`Psi0`, which goes to disk verbatim. Sparsifying happens *after* `to_hdf5`'s check and can cost a
determinant its orthonormality on its own, when the columns' mutual orthogonality was carried by
entries below the threshold. `write_nomsd` therefore re-checks each thresholded block and warns
naming the dataset (`PsiT_k`), so the warning describes the bytes on disk rather than the array in
memory. It still only warns: the caller has to choose between `orthonormalize()` and a lower
`threshold`, and the writer should not decide that silently.

> The check tolerance is 1e-10 and the sparsification threshold is 1e-8, so this warning can fire on
> a wavefunction whose error is bounded by the threshold and therefore physically negligible for
> AFQMC. That is deliberate (user call): the predicate reported is the same one `is_orthonormal`
> applies everywhere else, rather than a second, looser one that would have to be explained.

**`Lattice`** follows the same shape too: an ABC with concrete subclasses per lattice type
(`SquareLattice`/`TriangularLattice`/`HoneycombLattice`/`KagomeLattice`, plus `CustomLattice`), a
`Lattice.from_dict()` classmethod replacing today's free-function `get_lattice()` factory, and no
standalone `to_hdf5()`/`from_hdf5()` — a lattice is only ever persisted embedded in a
`LatticeHamiltonian`'s file. Only the base class is re-exported (`from safiretools import Lattice`),
since users may want to construct or inspect lattice geometry independent of building a full
Hamiltonian; `from_dict()` handles dispatch.

### Unit-cell geometry belongs to the lattice type

`a1`, `a2` and `basis` **always exist** on a `Lattice` instance, but for the built-in types they
are not caller-settable: they *are* the type. A `SquareLattice` is square precisely because its
lattice vectors are the unit x- and y-vectors, and a `HoneycombLattice` is a honeycomb precisely
because of its 2-site basis.

**`CustomLattice` is the one subclass that takes `a1`/`a2`/`basis`**, and it is therefore the
supported way to build a lattice whose geometry isn't one of the built-in types.

Mechanically: each concrete type implements an abstract `_unitcell() -> (a1, a2, basis)` hook, and
the `Lattice` constructor takes no geometry arguments at all, so the built-in types cannot accept
them even by accident. Passing `a1`/`a2`/`basis` to a built-in type raises `TypeError`; the
equivalent keys in a `from_dict()` parameter dict raise `ValueError` (a key present but set to
`None` is fine, so parameter templates carrying unused keys still work).

**The geometry is also immutable, not merely un-settable at construction.** `a1`/`a2`/`basis` are
read-only properties over private backing state; the arrays they return have `writeable=False` and
`basis` is a tuple, so the geometry can be neither replaced nor edited in place. Construction copies
whatever `_unitcell()` returns before freezing it, so freezing never reaches an array the caller
still holds. `cyl_mode` reshaping the cell inside `build()` is the one place the geometry changes,
it is derived from the type's own `_unitcell()` rather than from the caller, and building twice
raises `RuntimeError`. (`L` also changes under `cyl_mode` and is left a plain attribute; only the
three geometry members are locked down.)

### Basis validation

`build()` validates the basis and raises `ValueError` for two failure cases:

1. **Two basis vectors differing by a lattice translation** describe the same site, so sites
   coincide and `_to_lattice_basis` cannot tell which sublattice a position belongs to.
2. **A basis offset reaching a full supercell or more.** `_build_image_distances` shifts by only one
   supercell in each direction, so beyond that the true image is never tested.

Both are checked in *fractional* (lattice-vector) coordinates, and both depend on the boundary
conditions and the lattice size, not just the basis: an open axis neither wraps nor contributes
image shifts, so a translation that lands outside a 1-cell-wide open lattice collides with nothing.
Folding the basis into the unit cell satisfies both conditions for any lattice. This replaces
`CustomLattice`'s standing BUG note about basis-vector magnitude, whose framing was wrong — the
admissible offset depends on the lattice, not on a magnitude.

> Anything re-deriving or re-validating this rule has to count **directed crossings**, not distinct
> site pairs. Neighbor pairs are one
> per boundary crossing and are deliberately not deduplicated by minimum image, because the twist
> phase depends on which way the boundary is crossed — on a 2x2 periodic lattice site *i* reaches
> site *j* both inside the cell and across the boundary, with phases 0 and +theta.

`cyl_mode` is restricted to `TriangularLattice`. The XC/YC reshaping rotates `a2` onto `-a1 + 2*a2`
and doubles the basis along `a2`, which is only the correct cell for hexagonal geometry. It raises
`ValueError` for every other type, `CustomLattice` included.

`get_lattice()`'s `a1`/`a2` overrides were silently discarded for every built-in lattice type
(absorbed into `**kwargs`, never applied), and the same held for `basis` on the two types that
define a default one. Per the rule above the fix is to **reject** them loudly, not to start
honoring them.

### `min_distance` filters the neighbor-distance map but keeps the self-distance

`min_distance` drops nonzero separations below it and **keeps the zero self-distance at index 0**,
so `_dist_map[n]` still means the nth-neighbor shell. It is applied lazily, when the distance map is
first built, which also makes it work with `build=False`. These semantics were chosen rather than
preserved: the parameter was passed by `__init__` to a method whose signature did not accept it, so
it could never run.

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
*configured BP-depth levels*). The correct value is `taus.max()` (`= max_nback_prop * dt`), since
`iblock` increments once per full `max_nback_prop`-step cycle, uniformly across BP-depth levels
(confirmed in `src/AFQMC/Estimators/BackPropagatedEstimator.hpp`). Also: the C++ side already
discards equilibration blocks via `equil_multiplier` before anything reaches `stat.h5`, but the
Python-side `Teq`/`nequil` knob is still a wanted feature, for cases where unequilibrated samples
slip through despite the C++-side trim. `check_1rdm_convergence`'s quadrature-sum indexing bug (uses
only one of two BP-average endpoints' errors) gets fixed at the same time.

## Spin-symmetry enum

- Canonical: `SpinSymm`, an `IntEnum` (not `IntFlag`) with exactly 3 members forming a hierarchy
  of increasing generality: `CLOSED < COLLINEAR < NONCOLLINEAR`. Matches `WALKER_TYPES` in
  `src/AFQMC/config.h`.
- `_SlaterType`/`_slater_enum_map`/`_slater2dims` (`afqmctools/utils/slater_types.py`) are
  dropped entirely — `SpinSymm`'s int values already are the C++ wire format, so no translation
  function is needed.
- The dead `SlaterDeterminant`/`MultiSlater`/`NonorthMSD`/`ParticleHoleMSD` stub classes are
  dropped — every method unconditionally raised `NotImplementedError`.
- Lives in `safiretools/types.py` — dependency-free, so hamiltonian/wavefunction/observables all
  import it downward rather than `observables` reaching up into `hamiltonian` for it (today's
  layering inversion).
- **A wavefunction with no beta electrons is `COLLINEAR` with `ndown == 0`** (user call). There is no
  separate value for that case: `dims[3]` is 2, and the beta blocks go to disk with zero width (`Psi0_beta` of shape `(nmo, 0, 2)`, a `PsiT_1` whose `dims` is
  `[0, nmo, 0]`), because the executable's readers open them for any collinear file.

  > **On the C++ side:**` WALKER_TYPES` is exactly
  > `CLOSED`/`COLLINEAR`/`NONCOLLINEAR` ([config.h:48](../../src/AFQMC/config.h#L48)), and a
  > collinear walker set accepts an empty beta sector: `walker::SlaterMatrix(Beta)` returns a
  > zero-width view rather than raising, the zero-extent beta operations are skipped where they
  > would trap on GPU, and
  > [tests/test_polarized_consistency.cpp](../../tests/test_polarized_consistency.cpp) pins a
  > `ndown == 0` collinear run against an up-only noncollinear reference. Its
  > `derive_polarized_wfn` writes exactly the layout above — `dims = [nmo, nup, 0, 2, ndets]`,
  > `Psi0_beta` of shape `(nmo, 0)`, `PsiT_1` a `(0, nmo)` CSR — so the format here is the one the
  > executable's own test data uses.
- Two members carry the coercion that `get_spin_symm_enum` used to: `SpinSymm.from_input(value)`
  accepts a `SpinSymm`, its int value, a spelling alias (`'closed'`/`'rhf'`, `'collinear'`/`'uhf'`,
  `'noncollinear'`/`'ghf'`, ...), or another enum whose value is one of those — so partly-migrated
  code can still hand over an `afqmctools._SlaterType`. `SpinSymm.label` is the lowercase name used
  on disk and in AFQMC input files. `from_input` accepts both `'closed'` and `'close'` — `utils/io.py`
  wrote the first and read the second, so a closed-shell `spin_type` could never round-trip — and
  `label` always writes `'closed'`.

## Known bug: `force_herm` diagonal-zeroing

`utils/matrix.py::force_herm`'s `'upper_triangular'` method (`np.triu(M,1)` then symmetrize)
zeroes the diagonal. Its only two call sites (`tband` in `nth_neighbor_hopping`, `epsilon_band`
in `onebody_onsite`) are one-body terms confirmed to be "read and used as the full matrix" (not
the U1/U2/J convention below) — so this has been silently discarding onsite/diagonal terms for
any caller using `force_herm=True` on a non-Hermitian `t`/`epsilon` input. Fix: reconstruct
including the diagonal (`np.triu(M,0) + np.triu(M,1).conj().T`). The function is `force_hermitian`
in `safiretools`, since both call sites take a *boolean* parameter of the same name that shadowed
it — the boolean parameter keeps its documented name.

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

## Known bug: inverted `real_valued` flag

`ham_class.py::HamiltonianComponent` set `_real_valued = np.iscomplexobj(csr_array)` — true exactly
when the component is *complex* — and `Hamiltonian.__setitem__` cleared the Hamiltonian-level flag
whenever a component was real, so every real-valued model Hamiltonian was upcast to complex on
write. `LatticeHamiltonian.real_valued` is a computed property over
`HamiltonianComponent.is_complex` instead, so a real Hamiltonian writes rank-1 real data. The AFQMC
executable already accepts both ranks.

## Complex arrays on disk

SAFIRE stores a complex array by interleaving the real and imaginary parts as a trailing length-2
axis. That convention is **defined once**, in `safiretools/hdf5.py` as `to_complex`/`from_complex`,
and every schema uses it — the dense `hcore`/Cholesky matrix, the k-point `H1_kp*`/`L*` blocks, and
a model component's CSR `data_`.

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
`MolecularHamiltonian`, `kpoint` -> `PeriodicHamiltonian`. It also recognizes `thc` and
`kpoint_coqui`, which safiretools has no reader for — `from_hdf5` raises `NotImplementedError` for
those. Subclasses implement `_read_hdf5(path, fmt)` rather than overriding `from_hdf5`, so dispatch
stays in one place. Those format names are `HamiltonianFormat` members, which are strings too, so
they read and compare as the bare names throughout.

**A file says which format it is; the layout heuristic is the fallback.** Every writer records its
format in `Hamiltonian/type` via `write_hamiltonian_format`, and `hamiltonian_format` reads that key
in preference to guessing. Guessing is what it did for every file originally — `model` if
`ModelHamiltonian/number_of_components` is there, `dense` if `DenseFactorized/L` is, and so on — and
that path stays, because files written before the key existed are still perfectly good input. A tag
that contradicts the layout wins: it is what the writer said. The value stored is the executable's
own `HamiltonianTypes` spelling (`ModelHamiltonian`, `RealDenseFactorized`, `KPFactorized`, `THC`)
rather than safiretools' shorter name, so that the C++ side can eventually read this key instead of
running the same heuristic in `peekHamType`. `kpoint_coqui` has no tag — a CoQuí file has no
`Hamiltonian` group to put one in — and neither do the hand-rolled writers in `afqmctools`/`cli`,
which is exactly what the fallback is for. 
It is desirable to update CoQuí to write a tag as well.
The dataset is a variable-length string, the same as
`spin_type`, so the C++ side reads it the way it already reads that; parallel HDF5 cannot write
variable-length data, so `write_from_pyscf` — the one Hamiltonian written in parallel — tags itself
from rank 0 after every rank has closed the file, rather than the tag costing the file a second
string convention of its own.

**The two names for a format live together in `HamiltonianFormat`.** Each format has a safiretools
name and, usually, an on-disk tag, and `types.HamiltonianFormat` (beside `SpinSymm`) is the single
place that pairs them: `MODEL = 'model', 'ModelHamiltonian'`, with `.tag` reading the second and
`.from_tag()` going back the other way. Members subclass `str`, so a format still compares, hashes
and formats as its safiretools name — `hamiltonian_format(path) == 'model'` holds, `_READERS` stays
keyed on plain strings, and nothing that formats a format into a message had to change. That mixin
needs one guard: Python 3.11 made a mixin `Enum`'s `str()` its *member* name, so the class sets
`__str__ = str.__str__` to keep `f"{fmt}"` rendering `model` rather than `HamiltonianFormat.MODEL`.
Keeping the pairing there leaves `TYPE_DATASET` as the only global the tag itself needs in
`hamiltonian/base.py`, alongside the `_READERS` table that was already there.

**`HamiltonianFormat` is deliberately not re-exported.** Nothing in the public API takes or returns
a format — `hamiltonian_format()` is not public either (see **Future changes**) — so the enum is an
implementation detail that `types.py` happens to be the right home for, next to `SpinSymm`, rather
than a second public enum. That is also what lets its docstring name
`hamiltonian.base.write_hamiltonian_format` by its real path.

**Lattice metadata is recorded, the `Lattice` object is not.** A `LatticeHamiltonian`'s file carries
the *shape* of the lattice it was built on under `Hamiltonian/ModelHamiltonian/Lattice`, written in
the dimension-agnostic form a future N-dimensional `Lattice` will need — `L` and `boundaries` and
`twist` as per-axis sequences, the unit cell as a `lattice_vectors` matrix — rather than as
`L1`/`L2`/`a1`/`a2` pairs. Today `ndim` is always 2. `nbands` is written alongside, since `dims[3]`
records only `nsites * nbands`. `LatticeHamiltonian.lattice_params` flattens the metadata back into
the keys `Lattice.from_dict` takes, so a lattice can be rebuilt from a Hamiltonian file. All of this
is additive — the executable ignores groups it does not read.

**A model file's `Uij` splits back into `U`/`U1`/`U2` exactly.** The three terms share one dataset
on disk, and `from_hdf5` has to separate them: U2 is the lower `nbasis` rows of the `(2M, M)`
matrix, the onsite Hubbard U is the diagonal of the upper block, and U1 is what is left off that
diagonal. The split is invertible because the onsite U is built diagonal and U1 never has diagonal
entries — intrasite U1 is strictly inter-band, and intersite U1 only connects distinct sites.

**A supercell Hamiltonian is the Γ point of the supercell.** The sparse `Hamiltonian/Factorized`
format `write_hamil_supercell` wrote is gone — the AFQMC executable's `HamiltonianFactory` offers
only `KPTHC`, `KPFactorized`, `RealDenseFactorized`, `ModelHamiltonian` and `THC` — so the supercell
path emits the ordinary k-point format with a single k-point instead: `nkpts=1`,
`nmo_pk=[nmo_tot]`, `QKTok2=[[0]]`, `MinusK=[0]`, one `L0` of shape `(1, nmo_tot**2 * nchol)`. The
Cholesky vectors stay complex, as `KPFactorizedHamiltonian` requires. Lattice models stay sparse —
they are fundamentally sparse.

This is a simplification rather than a workaround: `PeriodicHamiltonian` ends up with one on-disk
format and one in-memory representation, `kpoint_symmetry` is only a knob on the *generator*, and
nothing downstream branches on it. The dense format was never an option here anyway —
`RealDenseHamiltonian` reads `DenseFactorized/L` into a *real* array, so it could not carry a
supercell's complex Cholesky vectors.

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

**Writing a supercell factorization in parallel raises `NotImplementedError`, and that is settled,
not a gap** (user call): it is not needed. Each rank's Cholesky vectors scatter across the combined
orbital basis rather than filling a contiguous slice, so the per-rank merge the k-point path uses
does not apply — and `kpoint_symmetry=True` is the path that runs in parallel.

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
  in sync. It follows that:
  - **Docstrings of user-facing objects reference only other user-facing objects, by their
    top-level path** — `safiretools.Lattice`, never
    `safiretools.hamiltonian.model.lattice.Lattice`. Dev-facing code may reference anything,
    deep paths included, since its audience is reading the tree anyway.
  - **User-facing documentation — tutorials, examples, reference prose, and error messages a user
    can hit — shows only top-level imports.** If a doc or an exception needs to name something, that
    something gets promoted; writing the deep path is not an option. This is a forcing function: it
    turns "I'll just reference the module" into an explicit decision about whether the thing is
    public. It is what promoted the FCIDUMP I/O (`read_fcidump`, `read_fcidump_header`,
    `write_fcidump`, `write_fcidump_kpoint`, plus `h1_spat2spin`/`h2_spat2spin`, which
    `write_fcidump_kpoint`'s own `NotImplementedError` tells users to call).
  - **Reference docs document the public surface from `safiretools` itself** (`automodule::
    safiretools` with `:members:` and `:imported-members:`), so that the short paths resolve as
    cross-references. Autodoc'ing the same classes from their implementation modules instead makes
    the top-level path unresolvable — which is why the deep paths are in the docstrings today; see
    **Future changes**.
- **Construction**: classmethod factories (`from_dict`, `from_hdf5`, `from_fcidump`, ...) over
  parsing inside `__init__`. `__init__` is for already-fully-formed, validated in-memory data.
- **Serialization**: instance method + classmethod pair — `obj.to_hdf5(path) -> None`,
  `Cls.from_hdf5(path) -> Cls` (classmethod, dispatches to the right concrete subclass). Applies
  uniformly to both `Hamiltonian` and `Wavefunction`. `Hamiltonian` subclasses each implement it
  since their formats genuinely differ; `Wavefunction`'s base class implements it once and
  subclasses don't override.
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
- **The default-`psi0` warning is NOMSD-specific.** A NOMSD's default `psi0` is the leading
  determinant's spin blocks, which for a UHF-shaped determinant is worth warning about; a PHMSD's is
  identity columns at the leading occupations — the recommended ROHF-like choice — so warning there
  would be noise. It hangs off an overridable `_warn_about_default_psi0` hook.
- **`to_hdf5()` replaces the Hamiltonian in its target file, not the whole file.** It opens the file
  in append mode (creating it if absent) and deletes an existing `Hamiltonian` group before writing.
  A SAFIRE input file holds **at most one Hamiltonian and at most one wavefunction**, so replacing
  rather than adding is the right semantics, and anything else in the file — notably `Wavefunction` —
  survives. This is what lets a Hamiltonian and a wavefunction share one file **in either order**,
  which is the common case. Applies to `PeriodicHamiltonian.write_from_pyscf` too, whose per-rank
  scratch files are still truncated since they hold only one run's partial blocks. (HDF5 unlinks
  rather than reclaims, so repeatedly rewriting into one file grows it.)
- `AFQMC_EXEC` must never be required just to import safiretools (today's
  `RuntimeError: AFQMC_EXEC environment variable is not set` fires at import time in
  `tutorial_utils/helper.py` — becomes a lazy check, only triggered when something actually
  invokes SAFIRE). A future thin binding to invoke SAFIRE in-process (possibly nanobind) is
  planned — whatever "run SAFIRE" API safiretools exposes should not hard-code a
  subprocess/executable-path assumption that would preclude that later.
- Dependency cleanup: drop `pytables` (`stats/config_h5.py`'s only reason for it, rewritten onto
  `h5py`); merge the `LATTICE_HF` optional-dependency group into `AUTOHF` (exact duplicate).
- **Python 3.10 is the floor**, in `requires-python` and in ruff's `target-version`. It is what the
  code already required rather than a new minimum: shipped `afqmctools` uses PEP 604 unions in
  runtime annotations (`utils/io.py`, `hamiltonian/model/builder.py`, `observables/rhonk.py`,
  `utils/aimbes_utils.py`) and `zip(strict=)` (`rhonk.py`), so 3.9 could not import it. Nothing
  needs 3.11 or 3.12 — the `itertools.batched` import in `analysis/common.py` sits behind a
  disabled guard with a local fallback. The previous `">3.9"` was doubly wrong: it admitted every
  3.9.x above 3.9.0 too, since `3.9.1 > 3.9`. Raising ruff's target turns on `B905` (bare `zip()`
  without `strict=`), which is left unaddressed — those findings are all pre-existing calls, and
  adding `strict=` changes behavior.

### Where a factory lives

**A factory lives on the class that can actually produce the result, and the two hierarchies answer
that differently** (user call):

- **`Hamiltonian`** keeps only `from_hdf5` on the base class. Every domain factory is defined on the
  subclass that builds it — `LatticeHamiltonian.from_dict`,
  `MolecularHamiltonian.from_integrals`/`.from_pyscf`,
  `PeriodicHamiltonian.from_pyscf`/`.write_from_pyscf`.
- **`Wavefunction`** puts *every* factory on the base class, which picks the representation.

```python
from safiretools import Hamiltonian, MolecularHamiltonian, Wavefunction

hamiltonian = Hamiltonian.from_hdf5("hamiltonian.h5")     # shared: dispatches on the file
hamiltonian = MolecularHamiltonian.from_pyscf(scf_data)   # domain factory: name the class
wavefunction = Wavefunction.from_hdf5("wavefunction.h5")  # or any other Wavefunction factory
```

**Why the two differ: what the subclass axis means** (see **Class hierarchies**). A `Hamiltonian`
subclass is the *source domain* — lattice, molecular, periodic — which the caller always knows,
because they are holding the `scf_data` or the parameter dict that only one domain can consume.
Dispatch there would resolve a question nobody was asking. A `Wavefunction` subclass is the
*mathematical representation*, which often falls out of the **data**: `from_pbc_scf` cannot know
whether it will produce a particle-hole expansion until it sees whether bands came out partially
occupied. Dispatching is the only honest signature there.

**`from_hdf5` is a genuinely shared factory**. Its target comes from the *file*, not from the call site, so the caller
cannot name the subclass without peeking at file contents. 
It resolves the format first (`hamiltonian_format(path)`)
and then checks the resolved class against the class it was called on, so
`MolecularHamiltonian.from_hdf5` on a k-point file raises instead of returning a
`PeriodicHamiltonian`, while `Hamiltonian.from_hdf5` on the same file returns one. Subclasses
implement `_read_hdf5(path, fmt)` rather than overriding `from_hdf5`, so dispatch stays in one place.

`Wavefunction`'s base-class factories still need their guard, because inheritance does offer every
one of them on every subclass. A fixed-answer factory knows its target from its own definition and
guards with `_check_representation` (`wavefunction/base.py`), raising a `ValueError` naming both
classes rather than quietly returning the wrong type — `PHMSDWavefunction.from_free_electron(...)`
does not hand back an `NOMSDWavefunction`. `Wavefunction.from_hdf5` cannot guard that way, since its
target comes from the file, so it applies the same test inline (`issubclass(target, cls)`) after
resolving the format.

| factory | lives on | target decided by |
|---|---|---|
| `Hamiltonian.from_hdf5` | base — shared | `hamiltonian_format(path)`, then `issubclass(target, cls)` |
| `LatticeHamiltonian.from_dict` | subclass | the class named at the call site |
| `MolecularHamiltonian.from_integrals` / `.from_pyscf` | subclass | the class named at the call site |
| `PeriodicHamiltonian.from_pyscf` / `.write_from_pyscf` | subclass | the class named at the call site |
| `Wavefunction.from_hdf5` | base | `wavefunction_format(path)`, then `issubclass(target, cls)` |
| `Wavefunction.from_free_electron` / `.from_pyscf` | base | fixed NOMSD, `_check_representation` |
| `Wavefunction.from_pyscf_cas` / `.from_dice` | base | fixed PHMSD, `_check_representation` |
| `Wavefunction.from_pbc_scf` | base | the occupancies — **no guard, deliberately** |

`Wavefunction.from_pbc_scf` is the one base factory with no guard, and that is correct rather than an
oversight: it is the honest dispatcher, returning whichever representation the occupancies call for,
so there is no target to check it against. The subclasses do not inherit it — they **override** it
with narrowing forms that validate in their own way (below).

A fixed-answer `Wavefunction` factory still belongs on the base class: the caller should not have to
know that `from_dice` happens to produce a particle-hole expansion in order to ask for one.

**Mechanism, where a base factory dispatches.** The base classmethod imports its subclasses *inside
the method body*, which is how both `from_hdf5` methods avoid the circular import. **No
`_Hamiltonian`/`_Wavefunction` split is needed**, and none should be added.

**Each hierarchy's signatures follow from where the implementation lives.** `Wavefunction`'s base
factories delegate to *free functions* in `wavefunction/{free_electron,pyscf,pbc,dice}.py`, so the
base method is the only wrapper and spells out **real parameters**; the subclasses add nothing. A
`Hamiltonian` domain factory *is* the subclass classmethod, so it spells out its own real parameters
and documents them in the one place they apply with no `**kwargs` forwarding layer in between.
That is the practical payoff of pushing them down: the two `from_pyscf` signatures share only
`scf_data`, `chol_cut` and `verbose`, and everything else is domain-specific
(`cas`/`ortho_ao`/`df`/`real_chol` molecular; `comm`/`kpoint_symmetry`/`maxvecs`/`exxdiv` periodic),
so `inspect.signature` is exact and a keyword aimed at the wrong domain is a plain `TypeError` from
the method the caller actually named.

**`scf_data` is told apart by key, not by type** wherever something still has to tell them apart
(`PeriodicHamiltonian.from_pyscf` reads `'cell'`). `pyscf.pbc.gto.Cell` is a *subclass* of
`gto.Mole`, so an `isinstance` test on the object would report a periodic cell as molecular. The
loaders are disjoint on the key — `load_from_pyscf_chk` stores `'cell'` (plus `'kpts'`, `'nmo_pk'`)
and `load_from_pyscf_chk_mol` stores `'mol'` — and that is the discriminator.

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
    - a Blocking item for this is computing the exchange divergence correction energy from the madelung
    constant (found in the CoQuí Hamiltonian format) and electron number (from the Wavefunction) instead of 
    reading the energy directly from HDF5. **Would require adding the madelung constant in the periodice PySCF**
    **to SAFIRE route as well**.
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

- **Read `Hamiltonian/type` on the C++ side instead of guessing from the layout.** `peekHamType`
  ([hdf5_helpers.hpp](../../src/AFQMC/Hamiltonians/hdf5_helpers.hpp)) runs the same subgroup-name
  heuristic safiretools now only falls back to, so the two implementations have to stay in step.
  Writing the key is the half that had to come first: the executable cannot rely on it until enough
  files carry it. The stored value is already the `HamiltonianTypes` spelling precisely so this step
  is a lookup rather than a translation table, and `h5::h5_read(grp, "type", std::string&)` reads
  the variable-length string as-is — the same call that already reads `spin_type`. It must keep the
  layout fallback for untagged files, exactly as the Python side does, and the `format`
  (`"std"`/`"coqui"`) axis is unaffected: a CoQuí file has no `Hamiltonian` group and so no tag.

- **`from_free_electron` should not overwrite a parameter dict's own `lattice.twist`.** It
  substitutes `DEFAULT_TWIST` unless `twist=` is passed explicitly, so a twist supplied in an input
  file is silently ignored and every caller has to re-pass it — which is why the explicit `twist=`
  in `docs/snippets/01_setting_up/06_twist_angle` is load-bearing rather than decorative. Preserved
  for now because doc and user code is written against it.

- **Port `docs/tutorials/solids/04_computing_observables` off afqmctools.** It is the last doc source
  still calling `afqmctools.hamiltonian.converter.read_hamiltonian`, because its provided `hamil.h5`
  is in **CoQuí** format — `hamiltonian_format()` identifies it as `kpoint_coqui`, and safiretools
  has no reader for that, so `Hamiltonian.from_hdf5` raises `NotImplementedError`. Blocked on the
  observables rewrite, but it **must** be ported before afqmctools is removed. A comment in the
  tutorial cell records the same.

### Things we might change

- **The periodic multi-determinant expansion keeps the *least* probable determinants.**
  `reoccupy` selects with `probabilities.argsort()[:ndets]`, and `argsort` is ascending, so
  determinant 0 — which the executable takes as its reference configuration — is the least likely
  configuration rather than the most likely. Almost certainly a bug, fixed by one `[::-1]`, but it
  changes numerics on a path with established behavior (and would break the `ndet_max=4` periodic
  equivalence check), so it is preserved verbatim pending a decision.
    - **This was flagged by AI** It is likely wrong here.
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
