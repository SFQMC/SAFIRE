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
- Nothing in the package uses MPI; the periodic Cholesky factorization runs serially.
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
- A trial file stores no initial walker; the AFQMC input's `walker_set.from` chooses the
  wavefunction the walkers start from.
- A determinant's shape carries its spin symmetry publicly, and is one matrix per independent spin
  channel internally.
- Writing never repairs: every Slater matrix reaching disk has its overlap's condition number
  checked, and an ill-conditioned one is warned about. `orthonormalize()` is explicit, and the
  domain factories do not call it.
- `from_pyscf` takes PySCF objects and reads them by duck typing; the molecular path imports no
  PySCF. The helpers live in `convert/pyscf.py`.
- The source object (the physics, including any spin-orbit term) and the basis are separate
  arguments; the spin symmetry is deduced, never passed.

**On-disk formats**

- Serialization is the pair `obj.to_hdf5(path)` / `Cls.from_hdf5(path)`.
- `Hamiltonian` subclasses each implement the pair; `Wavefunction` implements it once, in the base.
- `from_hdf5` dispatches on the file's own contents, via `hamiltonian_format` /
  `wavefunction_format`.
- The Hamiltonian format is inferred from the layout. Writers also stamp the format name as the
  `type` attribute of the `Hamiltonian` group, which nothing reads yet.
- `HamiltonianFormat` enumerates the format names, and is not public.
- `to_hdf5` replaces its own group in the target file, so a Hamiltonian and a wavefunction can
  share one file in either order.
- A complex array goes to disk as an interleaved trailing length-2 axis, defined once in `hdf5.py`.
- Real and interleaved data are told apart by rank, never by the trailing axis length.
- The dense format blocks its arrays by spin: `hcore` is `(nspin, npol, nmo, npol, nmo)` and
  `DenseFactorized/L` is `(nspin, npol, nmo, npol, nmo, nchol)`. L's `nspin` is 1 when every spin
  sector shares the vectors, and the executable reads only `npol == 1`.
- `SpinSymm` is a 3-member `IntEnum` whose values are the C++ `WALKER_TYPES` wire format.
- A wavefunction with no beta electrons is `COLLINEAR` with `ndown == 0`.
- A `LatticeHamiltonian` file records the lattice's metadata dimension-agnostically, not the
  `Lattice` object.
- One `PeriodicCholesky` solver with a `kp_sym` flag, and a supercell is the Γ point of the
  supercell.
- FCIDUMP is reached only through the Hamiltonian classes: `MolecularHamiltonian.from_fcidump` /
  `.to_fcidump` and `PeriodicHamiltonian.to_fcidump`. `hamiltonian/fcidump.py` is dev-facing.

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
- CAS/CI wavefunction import from PySCF (`write_cas_wfn`) — kept as
  `PHMSDWavefunction.from_pyscf_cas`; `ci_wavefunction` came with it, as `ci_expansion`.
- AutoHF interop — kept, fragile unguarded import gets hardened.
- Dice-SHCI wavefunction import — kept, split into its own module, proper exceptions.
- `rhonk.py` (real-space observables) — kept for external callers; its vendored duplicate HDF5
  helpers get deduplicated onto the shared Core Library HDF5 utility.
- `write_rhoG_kpoints`/`write_rhoG_supercell` — kept only as a **raising stub**, the single
  `periodic.write_rhoG(kmf, path, gcut, ...)`, which explains why in its
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
│                           #   HamiltonianComponent, and all of hamiltonian/fcidump.py,
│                           #   which is reached through from_fcidump/to_fcidump
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
│   └── fcidump.py             # FCIDUMP external-format I/O — dev-facing, driven by the
│                              #   Hamiltonian classes' from_fcidump/to_fcidump
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
│   │                            #   transform_slater(), parse_spin_layout(),
│   │                            #   format_spin_layout(), orthonormalize(), is_orthonormal()
│   └── io.py                  # native SAFIRE HDF5 schema read/write (uses top-level hdf5.py),
│                               #   shared by both NOMSDWavefunction and PHMSDWavefunction
├── execution.py              # redesigned AFQMC JSON execution-parameter generator
│                              #   (from inputs/from_hdf.py) — naming/location still open
├── convert/
│   ├── pyscf.py                # duck-typed reading of PySCF objects: working_basis,
│   │                            #   one_body, periodic_solution, determine_spin_symm.
│   │                            #   Shared by hamiltonian/{molecular,periodic} and
│   │                            #   wavefunction/{pyscf,pbc}, so it can live in neither.
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

### The public spin layout: the shape carries the spin symmetry

Every determinant-shaped value the wavefunction classes take or return — the
`from_single_determinant` argument, `NOMSDWavefunction(dets=...)` and `.dets`, `.determinant(i)`,
and `PHMSDWavefunction(orbitals=...)` — uses one layout whose *type and rank* say which spin
symmetry it is:

| spin symmetry | layout                                            |
|---------------|---------------------------------------------------|
| closed        | `ndarray (nmo, nup)`                              |
| collinear     | `tuple` of `ndarray (nmo, nup)`, `ndarray (nmo, ndown)` |
| noncollinear  | `ndarray (2, nmo, nelec)`, spin-up components first |

A multi-determinant value (`dets`) adds a leading `ndets` axis to each array. A PHMSD reference
shared by both spins takes the single-array form even when the wavefunction is collinear; the tuple
means spin-resolved (on-disk `type` 2).

So `NOMSDWavefunction(coeffs, dets)` and `from_single_determinant(det)` take no
`nelec`, `spin_symm` or `nmo`: all three follow from `dets`, and can therefore never contradict
it. The parsing and its inverse are `slater.parse_spin_layout`/`format_spin_layout`.

**A collinear value must be a `tuple`**, and a list is a `TypeError`. With `nup == ndown`, a pair of
equally shaped arrays is indistinguishable by shape from a noncollinear `(2, nmo, n)` array, so
the container type is what decides; NumPy never produces a tuple from slicing or stacking, which
makes it a reliable signal. What it cannot catch: a collinear determinant *already stacked into one
array*, such as PySCF's UHF `mo_coeff[:, :, :nocc]`, is read as noncollinear. The
`from_single_determinant` docstring says so.

A noncollinear value records only its total electron count, so such a wavefunction reports
`nelec == (nelec, 0)` — the same as the file gives back, since its single `PsiT_0` block holds
every electron, and the executable does not use the split either.

### Internally, one matrix per spin channel

The public layout is converted at the class boundary into its uniform core: a tuple with one
`(npol*nmo, n)` matrix per independent spin channel — one entry when closed or noncollinear, two
when collinear. `NOMSDWavefunction._dets` holds one `(ndets, npol*nmo, n)` stack per channel, and
PHMSD references, `make_slater`'s result and `io.write_nomsd`/`read_nomsd` all use the same
tuple. The channels are never concatenated into one matrix, so nothing splits them back apart;
the executable's `PsiT_k` groups are already one per determinant and channel.

A noncollinear matrix stays `(2*nmo, n)` internally: QR, the overlap checks, `transform_slater`'s
`kron(eye(2), X)` promotion and the sparse writer all need the spinor matrix, not its `(2, nmo, n)`
view.

### Operations on that layout live in `wavefunction/slater.py`

Building a Slater matrix, transforming it into another basis, converting it to and from the public
layout, and checking or restoring its orthonormality are all independent of where the orbitals came from, so
they live in one module rather than in whichever domain module needed them first. A domain module
keeps only what decodes *its own* conventions — `pyscf.py` keeps the `mo_occ` readers (which
orbitals PySCF calls occupied, and how an ROHF reference packs both channels into one vector),
`pbc.py` keeps its fractional per-k-point occupancy logic.

`slater.py` therefore imports nothing from the package but `types.py`, which is what lets `io.py`
and `base.py` both use it without an import cycle.

### Writing never repairs: the overlap's condition number is what is checked

**Every Slater matrix that reaches disk has the condition number of its overlap :math:`S = M^\dagger M` checked**,
and an ill-conditioned one is warned about and written as it stands (user call).

`overlap_condition_number` computes it from `M`'s singular values as
:math:`(\sigma_{max}/\sigma_{min})^2`, never forming `S`. Two degenerate cases matter and are both
pinned by tests: a **rank-deficient matrix (an all-zero one included) is `inf`**, and a matrix
with **no columns is 1.0**, since a wavefunction with no beta electrons writes a zero-width beta
block whose overlap is the empty identity. The limit is `CONDITION_MAX`, :math:`1/\sqrt{\epsilon}`
≈ 6.7e7, because forming `S` squares the conditioning of `M`.

`orthonormalize()` remains as an explicit, non-mutating method returning a *new* instance, for a
caller who wants orthonormal columns; it uses `is_orthonormal` to leave already-orthonormal blocks
exactly alone. **The domain factories no longer call it**, because every source they read from is
orthonormal by construction: PySCF's orbitals are orthonormal in the basis they are expressed in,
and the free-electron and periodic paths occupy eigenvectors of a Hermitian matrix.

**`PsiT` blocks are checked after sparsifying.** The 1e-8
threshold applies to the sparse `PsiT` blocks it exists to sparsify, so the state checked is the
state written.

**Blocks are checked per independent spin channel**, which is the physically correct grain: a
collinear determinant's alpha and beta columns describe different spin sectors and need not be
orthogonal to each other, while a noncollinear determinant is a single `(2*nmo, nup + ndown)` block
and is checked whole. `Wavefunction.nelec_per_spin` supplies the split, as everywhere else.

`slater.orthonormalize` returns a block that is already orthonormal to within the tolerance
untouched, and otherwise takes the `Q` of a reduced QR decomposition. The sign convention is pinned
so `R`'s diagonal is real and non-negative, which makes `Q` unique.

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
- Autocorrelation-time estimator: pure NumPy, one dot product per lag, keeping the early break at
  the first non-positive lag — same O(n·*l*) cost in the autocorrelation length *l* as the previous
  nested-loop version, and the same numerics (agreement to 1.2e-15 relative over 400 randomized
  `mean_and_error` cases, with identical inf/NaN patterns). Chosen over the numba implementation
  because it **drops `numba` as a `safiretools` dependency** and is *faster in practice*: numba
  charges ~0.6 s to import and JIT-compile before the first result, which exceeds the entire
  computation in every realistic case, so a fresh interprete is
  3–19× faster end-to-end without it.
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
discards equilibration steps via `equilibration_steps` before anything reaches `stat.h5`, but the
Python-side `Teq`/`nequil` knob is still a wanted feature, for cases where unequilibrated samples
slip through despite the C++-side trim. `check_1rdm_convergence`'s quadrature-sum indexing bug (uses
only one of two BP-average endpoints' errors) gets fixed at the same time.

## Spin-symmetry enum

- Canonical: `SpinSymm`, an `IntEnum` (not `IntFlag`) with exactly 3 members forming a hierarchy
  of increasing generality: `CLOSED < COLLINEAR < NONCOLLINEAR`. Matches `WALKER_TYPES` in
  `src/AFQMC/config.h`.
- `_SlaterType`/`_slater_enum_map`/`_slater2dims` (`afqmctools/utils/slater_types.py`) are
  dropped entirely. On disk the spin symmetry is the `spin_type` string attribute, spelled as
  `SpinSymm.label` and read by the C++ through the `WALKER_TYPES` json names, so no translation
  function is needed.
- The dead `SlaterDeterminant`/`MultiSlater`/`NonorthMSD`/`ParticleHoleMSD` stub classes are
  dropped — every method unconditionally raised `NotImplementedError`.
- Lives in `safiretools/types.py` — dependency-free, so hamiltonian/wavefunction/observables all
  import it downward rather than `observables` reaching up into `hamiltonian` for it (today's
  layering inversion).
- **A wavefunction with no beta electrons is `COLLINEAR` with `ndown == 0`** (user call). There is no
  separate value for that case: `spin_type` is `"collinear"`, and the beta blocks go to disk with
  zero width (a `PsiT_1` whose CSR `shape` is `[0, nmo]`), because the executable's readers open
  it for any collinear file and take `ndown` from its row count.

  > **On the C++ side:**` WALKER_TYPES` is exactly
  > `CLOSED`/`COLLINEAR`/`NONCOLLINEAR` ([config.h:48](../../src/AFQMC/config.h#L48)), and a
  > collinear walker set accepts an empty beta sector: `walker::SlaterMatrix(Beta)` returns a
  > zero-width view rather than raising, the zero-extent beta operations are skipped where they
  > would trap on GPU, and
  > [tests/test_polarized_consistency.cpp](../../tests/test_polarized_consistency.cpp) pins a
  > `ndown == 0` collinear run against an up-only noncollinear reference. Its
  > `derive_polarized_wfn` writes exactly the layout above — `spin_type = "collinear"`,
  > `PsiT_1` a `(0, nmo)` CSR — so the format here is the one the executable's own test data uses.
- Two members carry the coercion that `get_spin_symm_enum` used to: `SpinSymm.from_input(value)`
  accepts a `SpinSymm`, its int value, a spelling alias (`'closed'`/`'rhf'`, `'collinear'`/`'uhf'`,
  `'noncollinear'`/`'ghf'`, ...), or another enum whose value is one of those — so partly-migrated
  code can still hand over an `afqmctools._SlaterType`. `SpinSymm.label` is the lowercase name used
  on disk and in AFQMC input files. `from_input` accepts both `'closed'` and `'close'` — `utils/io.py`
  wrote the first and read the second, so a closed-shell `spin_type` could never round-trip — and
  `label` always writes `'closed'`.

## Non-Hermitian one-body band matrices are an error

`afqmctools`' `force_herm=True` option on `nth_neighbor_hopping` and `onebody_onsite` symmetrized a
non-Hermitian `t`/`epsilon` band matrix from its upper triangle (and zeroed the diagonal while doing
so). `safiretools` drops the option: a non-Hermitian band matrix raises `ValueError`, since silently
discarding half of the user's input is never what they meant.

## Interaction-matrix triangle convention

The U1 and U2 pieces of the interaction matrix, and J, follow a triangularity rule of their own,
implemented in `onsite_band_matrix` and `intersite_band_matrix`.

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
and every schema uses it — the dense `hcore`/Cholesky matrix, the k-point `hcore`/`L*` blocks, and
a model component's CSR `values`.

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

## Format version

The group holding each input header — `Hamiltonian`, and `Wavefunction/NOMSD` or
`Wavefunction/PHMSD` — and the root of a results.h5 carry an int32 `format_version` attribute. It
is one number for all of them, `FORMAT_VERSION` in
`safiretools/hdf5.py` and in `src/AFQMC/Utilities/format_version.hpp`, bumped together with every
incompatible layout change. Readers on both sides require an exact match and name regeneration as
the fix; nothing migrates an old layout on read.

It sits on the header group rather than the file root because a Hamiltonian and a wavefunction are
written independently and may share one file. A results.h5 has a single writer, the executable,
so there it sits on the root; every stage of a run appends to the file, and the first stamps it.

CoQuí output is the one unversioned input, since we cannot change what it writes: a CoQuí
Hamiltonian is recognized by its `System`/`Interaction` groups, and a CoQuí wavefunction by a
`dims` array in place of `format_version`. That wavefunction layout is simply the legacy SAFIRE
one, so an old SAFIRE wavefunction is read through the same path, and the committed legacy
wavefunctions were left in it, apart from the finite-temperature one in
`square_2x2_hubbard_Beta3_nt100`, which was converted to a versioned header.

## Sparse matrices on disk

A CSR matrix is a group of four datasets: `shape` (rows, columns), `row_pointers` (`rows + 1`
offsets), `column_indices` and `values` — scipy's `indptr`/`indices`/`data` under descriptive
names. Written by `write_csr` and `math::sparse::CSR2HDF`, read by `read_csr` and
`math::sparse::HDF2CSR`.

It replaced a layout that recorded the nonzero count in a `dims` array next to the shape and kept
a separate begin and end offset per row. Both readers still take that one, told apart by the
absence of `shape`, because it is what CoQuí writes. Nothing writes it any more.

## Hamiltonian on-disk formats

Each `Hamiltonian` subclass owns its format, and `Hamiltonian.from_hdf5` dispatches on what a file
actually contains via the public `hamiltonian_format(path)` (which replaces
`converter.py::read_hamil_type`): `model` -> `LatticeHamiltonian`, `dense` ->
`MolecularHamiltonian`, `kpoint` -> `PeriodicHamiltonian`. It also recognizes `thc` and
`kpoint_coqui`, which safiretools has no reader for — `from_hdf5` raises `NotImplementedError` for
those. Subclasses implement `_read_hdf5(path, fmt)` rather than overriding `from_hdf5`, so dispatch
stays in one place. Those format names are `HamiltonianFormat` members, which are strings too, so
they read and compare as the bare names throughout.

**The layout decides the format.** `hamiltonian_format` goes by which datasets are present —
`model` if a `ModelHamiltonian` group is there, `dense` if `DenseFactorized/L` is, and
so on — which is the only thing that works for every file: older ones, CoQuí files (no
`Hamiltonian` group) and those from the hand-rolled writers in `afqmctools`/`cli`. The
`write_hamiltonian_header` shared by the safiretools writers (which also writes the
`nuclear_energy` every format shares) stamps the format name as the `type` attribute of the
`Hamiltonian` group, but nothing reads it yet.

**Scalars are attributes, arrays are datasets** — the rule CoQuí's files follow, which we cannot
change, so ours follow it too. That makes `spin_type`, `model_type`, `hst_type`, the lattice
`type` and `cyl_mode`, and the counts attributes, named `number_of_*` as CoQuí names its counts
(`number_of_sites`, `number_of_bands`). The constant energy uses CoQuí's names as well: optional
`nuclear_energy` and `frozen_core_energy` attributes of `Hamiltonian`, where CoQuí puts them on
`System`, replacing the two-slot `Energies` array. The executable reads both with one function,
`read_energy_offset`, which differs only in the group it opens; safiretools writes only
`nuclear_energy`, since it has no frozen core.

**Sizes are not recorded separately.** The file used to carry an 8-slot `dims` array duplicating
the orbital, k-point and Cholesky counts. Now each is read off the arrays that hold the data —
`hcore` and `DenseFactorized/L` for dense, `hcore` and each `KPFactorized/L{Q}` for k-point
(which drops `NMOPerKP` and `NCholPerKP`) — so the two cannot disagree.
The one exception is the lattice model, whose matrix shapes don't give the basis size
unambiguously; it writes `number_of_sites` and `number_of_bands` attributes on
`ModelHamiltonian`. The same goes for
the model's other derived values: the number of components is where the gap-free
`ModelComponent_<n>` numbering stops, and the executable sizes the sparse matrices it collects the
interaction terms into from the components' own row counts, where it used to trust a
`maximum_connectivity` hint.

**`HamiltonianFormat` members are strings.** They subclass `str`, so a format compares, hashes and
formats as its safiretools name — `hamiltonian_format(path) == 'model'` holds and `_READERS` stays
keyed on plain strings. That mixin needs one guard: Python 3.11 made a mixin `Enum`'s `str()` its
*member* name, so the class sets `__str__ = str.__str__` to keep `f"{fmt}"` rendering `model`
rather than `HamiltonianFormat.MODEL`.

**`HamiltonianFormat` is deliberately not re-exported.** Nothing in the public API takes or returns
a format — `hamiltonian_format()` is not public either (see **Future changes**) — so the enum is an
implementation detail that `types.py` happens to be the right home for, next to `SpinSymm`, rather
than a second public enum. That is also what lets its docstring name
`hamiltonian.base.write_hamiltonian_header` by its real path.

**Lattice metadata is recorded, the `Lattice` object is not.** A `LatticeHamiltonian`'s file carries
the *shape* of the lattice it was built on under `Hamiltonian/ModelHamiltonian/Lattice`, written in
the dimension-agnostic form a future N-dimensional `Lattice` will need — `L` and `boundaries` and
`twist` as per-axis sequences, the unit cell as a `lattice_vectors` matrix — rather than as
`L1`/`L2`/`a1`/`a2` pairs. Today `ndim` is always 2. `LatticeHamiltonian.lattice_params` flattens the metadata back into
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
`QKTok2=[[0]]`, `MinusK=[0]`, one `L0` of shape `(1, 1, 1, nmo_tot, 1, nmo_tot, nchol)`. The
Cholesky vectors stay complex, as `KPFactorizedHamiltonian` requires. Lattice models stay sparse —
they are fundamentally sparse.

This is a simplification rather than a workaround: `PeriodicHamiltonian` ends up with one on-disk
format and one in-memory representation, `kpoint_symmetry` is only a knob on the *generator*, and
nothing downstream branches on it. The dense format was never an option here anyway —
`RealDenseHamiltonian` reads `DenseFactorized/L` into a *real* array, so it could not carry a
supercell's complex Cholesky vectors.

**The k-point layout is the dense one with a leading k-point axis.** `hcore` is
`(nkpts, nspin, npol, nmo, npol, nmo)` and each `KPFactorized/L{Q}` is
`(nkpts, nspin, npol, nmo, npol, nmo, nchol_Q)`, replacing one `H1_kp{k}` dataset per k-point and
flat `(nkpts, nmo**2 * nchol_Q)` vectors. `nspin` and `npol` are always 1: the format holds only
spin-independent Hamiltonians, as before. One `nmo` for all k-points is what the executable already
required, so `to_hdf5` refuses a Hamiltonian whose k-points carry different orbital counts rather
than padding it. `PeriodicHamiltonian` keeps its in-memory representation — per-k-point `hcore`
blocks and flat `chol` — only the file changed.

**No `ComplexIntegrals` flag.** afqmctools wrote one, meaning "the Cholesky matrix is complex" to
its dense *writer* and "hcore is complex" to its *reader*, and the executable never read it. Every
reader takes the dtype from the dataset itself, so format version 1 drops it.

## FCIDUMP is an external format the Hamiltonian classes own

FCIDUMP is a Hamiltonian on disk, so it is read and written the way every other Hamiltonian format
is, through the class that holds those integrals.

```python
hamiltonian = MolecularHamiltonian.from_fcidump("FCIDUMP", cholesky_tol=1e-5)
hamiltonian.to_fcidump("FCIDUMP.out")     # and PeriodicHamiltonian.to_fcidump likewise
```

`hamiltonian/fcidump.py` keeps the format itself — parsing, index symmetry, line formatting, the
spatial-to-spinor conversion — and is **dev-facing in its entirety**.

**There is no `PeriodicHamiltonian.from_fcidump`.** A
FCIDUMP records one combined orbital basis and no k-point structure at all, so there is nothing in
the file to rebuild a k-point factorization from; a FCIDUMP written from a periodic Hamiltonian
reads back as a molecular one.

**The write-time arguments that were Hamiltonian state are gone**, exactly as for `to_hdf5`:
`hcore`, `chol`, `enuc`, `nmo`, `nelec` all come from the instance, and `chol_is_eri` disappears
because a `MolecularHamiltonian` always holds a factorization. What is left is the format's own
knobs — `tol`, `ctol`, `sym`, `cplx`, `paren`, `use_spinor`.

**`to_fcidump` fills in the momentum transfers a k-point file does not store.** `chol` holds only
`Q <= minus_k[Q]`; the FCIDUMP needs all of them, so `_chol_all_momenta` reconstructs each
partner as
`L[-Q][k1][i,j,n] == conj(L[Q][k2][j,i,n])` with `k2 = qk_to_k2[-Q, k1]`; `nchol_pk`, read off
the stored blocks, gives an unstored `Q` its partner's count. This is the same expansion
any reader of the on-disk format performs — afqmctools did it in `get_kpoint_chol` at read time —
so it is not new behavior, only relocated to the one place that needs the full set.

**Uneven per-k-point orbital counts are rejected rather than written wrongly.** The FCIDUMP orbital
index is the combined `k * nmo_pk + i`, and `write_fcidump_kpoint` walks it as though every k-point
carried `nmo_max` orbitals, so a mesh with different counts per k-point (linear dependencies
removed per k-point) silently produced garbage. `PeriodicHamiltonian.to_fcidump` raises `ValueError`
instead. The underlying free function is left as it is.

**A spinor-basis FCIDUMP is still not available for a k-point Hamiltonian**, and `to_fcidump` keeps
`use_spinor` for the sake of saying so. Its `NotImplementedError` no longer names
`h1_spat2spin`/`h2_spat2spin` (a user-facing message may not name something the user cannot
import) and points at `use_spinor=False` instead. On the molecular side `use_spinor` works, and is
refused only for a Hamiltonian already in a spin-orbital basis, where converting again would
silently double the basis a second time.

### The orbital-pair order between the two is load-bearing

`read_fcidump` returns the chemists' `(ik|jl)` at `[i, k, j, l]`, while the Cholesky decomposition
`from_integrals` performs needs the *hermitian* pair matrix `{(ik), (lj)}`.
`from_fcidump` therefore transposes `(0, 1, 3, 2)` on the way in.

> The same transpose was sitting in `tutorials/molecules/03` as a bare
> `np.transpose(H2_ijkl, (0,1,3,2))` with the comment "match the eri convention from_integrals()
> expects". It was correct and load-bearing, not decorative. `from_integrals`' own `eri`
> documentation still says only "chemists' notation `(ij|kl)`", which is the ambiguous half of the
> story; it is left as it is for now.

## Reading PySCF objects: `convert/pyscf.py`

**Every `from_pyscf` factory takes PySCF objects, not checkpoint files**, and reads them by duck
typing: only methods and attributes of the `mf`/`mc`/`kmf` it is handed are used (`get_hcore()`,
`mo_coeff`, `mol.intor`, `fcisolver.large_ci`, ...), so the molecular path never imports PySCF.
The periodic Cholesky solver still imports `pyscf.pbc` lazily for `FFTDF`, `get_coulG` and
`madelung`; reimplementing those buys nothing, since whoever holds a `Cell` has PySCF, and CoQuí is
the supported route for solids. The extraction helpers live in `convert/pyscf.py` because four
modules share them.

The checkpoint format pushed work onto users — hand-written `scf/hcore`, `scf/fock`,
`scf/orthoAORot` and `scf/nmo_per_kpt` for a solid, a `j3c` key for density fitting,
`base='mcscf'` for a CASSCF basis — that the objects answer directly. Someone holding only a
checkpoint rebuilds the object with PySCF's own `scf.chkfile.load_scf`.

**Two roles, two arguments.** The source object says what the physics is: its `get_hcore()` is the
one-body Hamiltonian and its `mol` supplies the two-electron integrals. `basis` says which orbitals
everything is expressed in — `None` for the source's own, `'ortho_ao'`, another SCF or CASSCF
object, or an `(nao, nmo)` array — and has to be the same for the Hamiltonian and the wavefunction:

```python
rohf = scf.ROHF(mol).run()
ghf_soc = scf.GHF(mol); ghf_soc.with_soc = True; ghf_soc.kernel()

MolecularHamiltonian.from_pyscf(ghf_soc, basis=rohf).to_hdf5("afqmc_soc.h5")
Wavefunction.from_pyscf(ghf_soc, basis=rohf).to_hdf5("afqmc_soc.h5")
```

**Spin-orbit coupling is a property of the source, not of the basis.** It used to be a `soc_type`
option on *loading the basis*, which attached a one-body term the ROHF calculation never used to
the ROHF orbitals. PySCF already builds every variant as some object's `get_hcore()`:

| treatment | source object | `get_hcore()` |
|---|---|---|
| none | `scf.RHF/ROHF/UHF(mol)` | `(nao, nao)` |
| spin-free X2C | `mol.RHF().sfx2c1e()` | `(nao, nao)` |
| spin-free, spinor basis | `scf.GHF(mol)`, `rhf.to_ghf()` | `(2 nao, 2 nao)`, block diagonal |
| ECP spin-orbit | GHF with `with_soc = True` | `(2 nao, 2 nao)` |
| X2C | `mol.GHF().x2c1e()` | `(2 nao, 2 nao)` |

The source need not be converged when it only supplies the hcore. A molecule carrying a spin-orbit
ECP whose hcore has no spin-orbit term warns, since a forgotten `with_soc` is otherwise silent.

**Nothing the data already says is an argument.** `ortho_ao` became `basis='ortho_ao'`,
`real_chol` went (the Cholesky dtype follows the data), and so did `spin_symm` (user call):

- A molecular Hamiltonian's spin symmetry is read off the hcore: noncollinear for a spinor matrix,
  closed otherwise. It describes the operator, not the reference state — a spin-independent
  one-body term gains nothing from two identical spin sectors, and the executable pairs a closed
  Hamiltonian with a collinear trial. A collinear molecular Hamiltonian is still reachable through
  the raw constructor.
- A wavefunction's spin symmetry is read off the solution by `determine_spin_symm`: a spinor
  `mo_coeff` is noncollinear; spin-resolved orbitals or a fractionally/singly occupied orbital are
  collinear; occupancies that are all 0 or 2 are closed shell. The order matters — a GHF solution
  has one `mo_coeff` matrix like an RHF one, and only the basis size separates them.

The constructors, `from_integrals`, `from_fcidump` and the model builders keep `spin_symm`: raw
arrays cannot say whether they are closed or collinear.

## Periodic Cholesky: one solver, one flag, serially

`kpoint.py`'s `KPCholesky` and `supercell.py`'s `Cholesky` become a single `PeriodicCholesky` with a
`kp_sym` flag. The factorization loop is shared; only the k-point-pair enumeration, the pivot
bookkeeping index, and the momentum-conservation test differ, each behind a small method. `run()` is
a generator yielding one momentum block at a time. The flag stays inside the solver: a
`kp_sym=False` result is recast as a Γ-point Hamiltonian (above), so both modes produce the same kind
of object.

**The solver is serial and there is one entry point**: `PeriodicHamiltonian.from_pyscf`
builds the whole factorization in memory and `to_hdf5` writes it, like every other `Hamiltonian`
subclass. The MPI machinery is gone. `Partition`, `fair_share`, `bisect`, `FileHandler`,
`rank_filename`, `_SerialComm`, the per-rank scratch files and their merge, the parallel-HDF5 branch,
the pivot `Allgather`/`Bcast`, and the second `write_from_pyscf` entry point that existed to stream
over a communicator. It complicated the module out of proportion to what it bought, and **CoQuí is
the supported route for production-sized solids**; see *Things we might change*.

The cost is that a large k-point mesh now needs every `L_Q` resident at once, where the streaming
writer held one block. The `run()` generator is still block-at-a-time internally, so a streaming
writer could be reintroduced without touching the solver.

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
    something gets promoted or the thing it names is reworked; writing the deep path is not an
    option. This is a forcing function: it turns "I'll just reference the module" into an explicit
    decision about whether the thing is public. The FCIDUMP I/O is where it bit hardest. Its six
    functions were re-exported for exactly this reason — a tutorial and a `NotImplementedError`
    named them — and the answer was to give the *Hamiltonian classes* the FCIDUMP entry points
    instead (see **FCIDUMP is an external format the Hamiltonian classes own**), which took the whole
    module off the public surface and made both the tutorial and the exception shorter.
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
- **The AFQMC initial walker is not part of a wavefunction.** A trial file used to carry one
  (`Psi0_alpha`/`Psi0_beta`, and `write_wfn(..., init=...)` before that); now the AFQMC input's
  `walker_set.from` names the wavefunction a walker set starts from, and the executable derives the
  determinant from it — the largest-coefficient determinant of a NOMSD, the reference of a PHMSD.
  Starting a UHF trial from an ROHF determinant is therefore a second, single-determinant
  wavefunction file, not state on the first one. `orbmat=` became `PHMSDWavefunction(orbitals=...)`.
- **A PHMSD records `number_of_orbitals`.** It is the one size the format stores explicitly:
  occupation numbers alone do not span the orbitals, and with no initial walker on disk nothing
  else does either. The name follows CoQuí's.
- **`to_hdf5()` replaces the Hamiltonian in its target file, not the whole file.** It opens the file
  in append mode (creating it if absent) and deletes an existing `Hamiltonian` group before writing.
  A SAFIRE input file holds **at most one Hamiltonian and at most one wavefunction**, so replacing
  rather than adding is the right semantics, and anything else in the file — notably `Wavefunction` —
  survives. This is what lets a Hamiltonian and a wavefunction share one file **in either order**,
  which is the common case. (HDF5 unlinks rather than reclaims, so repeatedly rewriting into one
  file grows it.)
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
  `MolecularHamiltonian.from_integrals`/`.from_pyscf`/`.from_fcidump`,
  `PeriodicHamiltonian.from_pyscf`/`.write_from_pyscf`.
- **`Wavefunction`** puts *every* factory on the base class, which picks the representation.

```python
from safiretools import Hamiltonian, MolecularHamiltonian, Wavefunction

hamiltonian = Hamiltonian.from_hdf5("hamiltonian.h5")     # shared: dispatches on the file
hamiltonian = MolecularHamiltonian.from_pyscf(mf)         # domain factory: name the class
wavefunction = Wavefunction.from_hdf5("wavefunction.h5")  # or any other Wavefunction factory
```

**Why the two differ: what the subclass axis means** (see **Class hierarchies**). A `Hamiltonian`
subclass is the *source domain* — lattice, molecular, periodic — which the caller always knows,
because they are holding the SCF object or the parameter dict that only one domain can consume.
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
| `MolecularHamiltonian.from_integrals` / `.from_pyscf` / `.from_fcidump` | subclass | the class named at the call site |
| `PeriodicHamiltonian.from_pyscf` | subclass | the class named at the call site |
| `Wavefunction.from_hdf5` | base | `wavefunction_format(path)`, then `issubclass(target, cls)` |
| `Wavefunction.from_free_electron` / `.from_pyscf` | base | fixed NOMSD, `_check_representation` |
| `Wavefunction.from_pyscf_cas` / `.from_dice` | base | fixed PHMSD, `_check_representation` |
| `Wavefunction.from_pbc_scf` | base | fixed NOMSD, `_check_representation` |

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
the SCF object, `basis`, `chol_cut` and `verbose`, and everything else is domain-specific
(`active_space`/`df` molecular; `kpoint_symmetry`/`maxvecs` periodic),
so `inspect.signature` is exact and a keyword aimed at the wrong domain is a plain `TypeError` from
the method the caller actually named.

**No subclass overrides a base factory.** Every one returns a fixed representation, so each is a
plain inherited alias guarded by `_check_representation`, which is what stops
`PHMSDWavefunction.from_free_electron(...)` from quietly handing back a `NOMSDWavefunction`.

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
  of the *problem*, not of the Hamiltonian. The end state is that **nothing writes `nelec` to a
  Hamiltonian and nothing reads `nelec` from one**. The code side is done: the file formats carry
  no electron count, the C++ readers take it from the wavefunction, and no Python `Hamiltonian`
  takes or holds one.
  The last holdout was the periodic PySCF route, whose `from_pyscf(nelec=)` only fed the
  Madelung/`exxdiv` correction baked into `enuc`. It now records a `madelung_constant` attribute
  instead, in CoQuí's convention (`0.5 * pyscf.pbc.tools.madelung(cell, kpts)`), and the
  executable's `read_energy_offset` subtracts `madelung_constant * nelec` with the trial
  wavefunction's count — so the correction can no longer disagree with the electrons actually
  simulated. The treatment is `kmf.exxdiv`, not a separate argument that could disagree with it;
  only `'ewald'` and `None` are accepted, since the Cholesky vectors use the plain Coulomb kernel
  with `G = 0` dropped, which the truncated `'vcut_sph'`/`'vcut_ws'` kernels are not.
  **FCIDUMP keeps its `nelec`**, as an argument of `to_fcidump`: `NELEC` is a field of that
  external format's own header, and FCIDUMP has only one constant, so a periodic Hamiltonian folds
  `madelung_constant * sum(nelec)` into it there and refuses a nonzero constant with no count.

  Doc churn still to do: the `nelec` key currently in the `[hamiltonian]` block of the eleven
  `docs/snippets/01_setting_up/*/input*.toml` files, and in the Python parameter dicts across
  `docs/examples/models/*` and `docs/tutorials/models/*`, all comes back out — it went *in* during
  Phase 3b precisely because `to_hdf5` records it (see **Public API patterns**), so that bullet
  changes too.


- ~~**`from_free_electron` should not overwrite a parameter dict's own `lattice.twist`.**~~ Resolved.
  `from_free_electron` now takes only a built `LatticeHamiltonian` and applies no twist at all: the
  twist is a property of the `Lattice` the Hamiltonian was built on, so there is nothing left to
  overwrite. The `twist=` and `lattice=` parameters are gone from both the free function and
  `Wavefunction.from_free_electron`. Callers that want a twisted trial wavefunction build a second
  Hamiltonian on a twisted lattice and hand AFQMC the untwisted one; the docs under
  `docs/snippets/01_setting_up` and `docs/examples/models` all follow that pattern.

- **The open-shell warning and `SHELL_TOL` are sized against each other.** `from_free_electron`
  warns when the fill stops part-way through a degenerate shell, which is exactly the case a twist
  is meant to fix — so a twisted lattice silences it without the code ever reading a twist
  attribute. That balance depends on `SHELL_TOL` (`1e-10`) sitting between the splitting
  `DEFAULT_TWIST` produces (`4e-9`–`3e-8`, falling as the square of the twist and with lattice size)
  and the eigensolver noise floor (`~1e-13`). Raising `SHELL_TOL` back toward `1e-6`, or shrinking
  `DEFAULT_TWIST`, makes the warning fire on lattices that are in fact fine. `SHELL_TOL` is defined
  once and `Wavefunction.from_free_electron` defers to it via `shell_tol=None` rather than
  restating the number.

- **Port `docs/tutorials/solids/04_computing_observables` off afqmctools.** It is the last doc source
  still calling `afqmctools.hamiltonian.converter.read_hamiltonian`, because its provided `hamil.h5`
  is in **CoQuí** format — `hamiltonian_format()` identifies it as `kpoint_coqui`, and safiretools
  has no reader for that, so `Hamiltonian.from_hdf5` raises `NotImplementedError`. Blocked on the
  observables rewrite, but it **must** be ported before afqmctools is removed. A comment in the
  tutorial cell records the same.

### Things we might change

- **Drop PySCF support for periodic systems entirely.** `PeriodicHamiltonian.from_pyscf` and
  `NOMSDWavefunction.from_pbc_scf` are the only two things left that build a solid from a PySCF
  mean-field reference, and **CoQuí is the supported route for solids**. If that stays true, both
  can go, along with `convert/pyscf.py`'s `periodic_solution`, and `PeriodicHamiltonian` reduces to
  reading CoQuí's k-point format. The serial-only Cholesky factorization is sized for the same
  judgment: it is fine for a test case and not meant for a production mesh.
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
