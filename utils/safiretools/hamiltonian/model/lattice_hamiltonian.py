# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
`LatticeHamiltonian` — the lattice-model Hamiltonian, stored as a collection of
sparse terms, plus the sparse on-disk format the AFQMC executable's
``ModelHamOpsGenerator`` reads.

A `LatticeHamiltonian` is a *container* for terms built by
`HamiltonianBuilder`; it never reaches back into the `Lattice` it was built on.
It does record the lattice's shape as metadata, so a file can say what lattice
it came from (see `LatticeHamiltonian.lattice_params`).
"""

import itertools

import numpy as np
import scipy.sparse as sps
import h5py as h5

from safiretools.hamiltonian.base import (
    Hamiltonian,
    write_hamiltonian_header,
)
from safiretools.hdf5 import read_csr, replace_group, write_csr
from safiretools.types import SpinSymm


class HamiltonianComponent:
    """
    One sparse term of a lattice-model Hamiltonian, with the metadata the AFQMC
    executable needs to interpret it.

    Parameters
    ----------
    csr_array : scipy.sparse.csr_array or array_like
        Matrix elements. Converted to `scipy.sparse.csr_array` if needed.
    model_type : str
        Term type as the AFQMC executable names it, e.g. ``'one_body'``,
        ``'hubbard_u'``, ``'hubbard_j'``, ``'heisenberg_j'``.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry this term is expressed in. Default `SpinSymm.CLOSED`.
    hst_type : str, optional
        Hubbard-Stratonovich transformation of an interaction term, written
        alongside the matrix.
    """

    def __init__(self, csr_array, model_type, spin_symm=SpinSymm.CLOSED,
                 hst_type=None) -> None:
        if not isinstance(csr_array, sps.sparray):
            csr_array = sps.csr_array(csr_array)

        self.csr_array = csr_array
        self.model_type = model_type
        self.spin_symm = SpinSymm.from_input(spin_symm)
        self.hubbard_strat_type = hst_type

    @property
    def is_complex(self) -> bool:
        """
        True if this term has complex matrix elements.

        See `LatticeHamiltonian.real_valued`, which decides the on-disk dtype
        for the whole Hamiltonian from this.
        """
        return np.iscomplexobj(self.csr_array)

    def __add__(self, other):
        """
        Add two terms of the same `model_type` and Hubbard-Stratonovich type.

        The result's spin symmetry is the *lower* of the two (the
        larger `SpinSymm` value), since the sum is only as symmetric as its
        least symmetric part.
        """
        if other is None or (np.isscalar(other) and other == 0):
            return self

        if not isinstance(other, HamiltonianComponent) or self.model_type != other.model_type:
            raise ValueError(
                f"Can't add a HamiltonianComponent and {other!r} of type {type(other)}"
            )

        if self.hubbard_strat_type != other.hubbard_strat_type:
            raise ValueError(
                "Addition between Hamiltonian components with differing "
                "Hubbard-Stratonovich types is not allowed "
                f"('{self.hubbard_strat_type}' vs '{other.hubbard_strat_type}')"
            )

        return HamiltonianComponent(
            csr_array=self.csr_array + other.csr_array,
            model_type=self.model_type,
            spin_symm=max(self.spin_symm, other.spin_symm),
            hst_type=self.hubbard_strat_type,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self) -> str:
        return f"{self.csr_array}"

    @property
    def shape(self):
        return self.csr_array.shape

    def toarray(self):
        return self.csr_array.toarray()


class LatticeHamiltonian(Hamiltonian):
    """
    A lattice-model Hamiltonian: a dictionary of sparse terms keyed by the names
    the AFQMC executable uses.

    An *example* of the internal structure::

        terms = {
            'tij'  : [tij_component_1, tij_component_2, ...],
            'Uij'  : [U_component],
            'Jij'  : [Jij_component],
        }

    Each key maps to a *list* of `HamiltonianComponent`. Keeping terms separate
    is what makes different Hubbard-Stratonovich transformations possible within
    one Hamiltonian; `HamiltonianBuilder.finalize` collapses the ones that can
    be combined.

    Parameters
    ----------
    nsites : int
        Number of lattice sites.
    nbands : int, optional
        Number of bands per site. Default 1.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry of the Hamiltonian. Default `SpinSymm.CLOSED`.
    lattice_metadata : dict, optional
        Shape of the lattice this Hamiltonian was built on — see
        `lattice_metadata_from`. Recorded on disk; not used for any computation.

    Notes
    -----
    This class is a container only. It holds no reference to a `Lattice`, and
    terms are built by `HamiltonianBuilder`.
    """

    def __init__(
            self,
            nsites: int,
            nbands: int = 1,
            spin_symm=SpinSymm.CLOSED,
            lattice_metadata=None,
    ) -> None:
        super().__init__(spin_symm=spin_symm)

        self.terms = dict()
        self.nsites = nsites
        self.nbands = nbands
        self.lattice_metadata = dict(lattice_metadata) if lattice_metadata else {}

    # ------------------------------------------------------------------
    # container interface
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        return self.terms[key]

    def keys(self):
        return self.terms.keys()

    def get(self, key, default=None):
        return self.terms.get(key, default)

    def add_term(self, key: str, component: HamiltonianComponent) -> None:
        """Append `component` to the list of terms stored under `key`."""
        self.terms.setdefault(key, []).append(component)

    def pop_term(self, key: str):
        """
        Remove and return every component stored under `key`.

        .. warning:: Destructive. `HamiltonianBuilder.finalize` uses this to
                     move terms out before writing combined ones back.
        """
        return self.terms.pop(key)

    @property
    def num_components(self) -> int:
        """Total number of components across all terms."""
        return sum(len(components) for components in self.terms.values())

    @property
    def real_valued(self) -> bool:
        """
        True when every component has real matrix elements.

        The AFQMC executable requires all components of a file to be either real
        or complex, so this decides the on-disk dtype for the whole Hamiltonian.
        """
        return not any(component.is_complex
                       for components in self.terms.values()
                       for component in components)

    @property
    def nbasis(self) -> int:
        """Single-particle basis size, ``nsites * nbands``."""
        return self.nsites * self.nbands

    @property
    def basis_rotation(self) -> np.ndarray:
        """The identity: a lattice model is computed in the basis it is given in."""
        return np.eye(self.spin_symm.npol * self.nbasis)

    # ------------------------------------------------------------------
    # assembled views of the terms
    # ------------------------------------------------------------------

    def get_one_body(self):
        """
        The one-body part of the Hamiltonian summed into a single
        `scipy.sparse.csr_array`, or None if there is no one-body term.
        """
        one_body_terms = self.get("tij")
        if one_body_terms is None:
            return None
        return sum(one_body_terms).csr_array

    def get_U(self):
        """
        The Hubbard interaction (U, U1 and U2 together) as a single
        `scipy.sparse.csr_array`.

        U and U1 share a shape and are summed; U2 occupies the lower half of a
        ``(2*nbasis, nbasis)`` matrix, so the two families are placed separately
        rather than summed directly.
        """
        M = self.nbasis

        density_density = []
        spin_spin = []
        for term in self.get('Uij', []):
            if term.shape == (M, M):
                density_density.append(term.csr_array)
            elif term.shape == (2 * M, M):
                spin_spin.append(term.csr_array)
            else:
                raise ValueError(
                    f"Invalid U term shape {term.shape} for a basis of size {M}"
                )

        U1 = sum(density_density)
        U2 = sum(spin_spin)

        # U1 may be absent entirely, in which case sum() returned the int 0
        U1_shape = getattr(U1, "shape", (0, 0))
        U2_shape = getattr(U2, "shape", (0, 0))

        U = sps.lil_matrix(max(U1_shape, U2_shape))
        U[:U1_shape[0], :U1_shape[1]] = U1
        U[:U2_shape[0], :U2_shape[1]] += U2

        return U.tocsr()

    def get_J(self):
        """The Hund's J part of the Hamiltonian as a single sparse matrix."""
        return sum(term.csr_array for term in self.get("Jij", []))

    def get_heisenberg(self):
        """The Heisenberg part of the Hamiltonian as a single sparse matrix."""
        return sum(term.csr_array for term in self.get("J_heisenberg", []))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, source, lattice=None) -> "LatticeHamiltonian":
        """
        Build a fully finalized `LatticeHamiltonian` from a declarative input.

        This is the common-case wrapper around `HamiltonianBuilder`: it parses
        the input, runs the build steps it names, and finalizes. Use
        `HamiltonianBuilder` directly to compose terms that no input key covers.

        Parameters
        ----------
        source : dict or str or pathlib.Path
            Input parameters, as a dict with a ``'hamiltonian'`` section (and a
            ``'lattice'`` section unless `lattice` is given), or the path to a
            TOML file holding the same.
        lattice : ~safiretools.hamiltonian.model.lattice.Lattice, optional
            Lattice to build on. Built from ``source['lattice']`` if omitted.

        Returns
        -------
        LatticeHamiltonian

        Examples
        --------
        >>> hamiltonian = LatticeHamiltonian.from_dict({
        ...     'lattice': dict(L1=4, L2=4, boundary1='pbc', boundary2='pbc'),
        ...     'hamiltonian': dict(t=1.0, U=4.0),
        ... })
        """
        from safiretools.hamiltonian.model.builder import HamiltonianBuilder

        builder = HamiltonianBuilder.from_input(source, lattice=lattice)
        return builder.get_hamiltonian()

    @property
    def lattice_params(self) -> dict:
        """
        The recorded lattice shape as a parameter dict that
        ``Lattice.from_dict`` accepts, or ``{}`` if no lattice metadata was
        recorded.

        Notes
        -----
        `lattice_metadata` is stored in the dimension-agnostic form
        (``L``/``boundaries`` as per-axis sequences) that a future N-dimensional
        `Lattice` will use. This property flattens it back into today's
        two-dimensional ``L1``/``L2``/``boundary1``/``boundary2`` keys.
        """
        metadata = self.lattice_metadata
        if not metadata:
            return {}

        params = {'type': metadata['type']}
        for axis, (size, boundary) in enumerate(
                zip(metadata['L'], metadata['boundaries']), start=1):
            params[f'L{axis}'] = int(size)
            params[f'boundary{axis}'] = str(boundary)

        twist = metadata.get('twist')
        if twist is not None and np.any(np.asarray(twist) != 0.0):
            params['twist'] = tuple(float(t) for t in twist)

        cyl_mode = metadata.get('cyl_mode')
        if cyl_mode and cyl_mode != 'none':
            params['cyl_mode'] = cyl_mode

        if metadata['type'] == 'custom':
            a1, a2 = metadata['lattice_vectors']
            params['a1'] = list(a1)
            params['a2'] = list(a2)
            params['basis'] = [list(b) for b in metadata['basis']]

        return params

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_hdf5(self, path) -> None:
        """
        Write this Hamiltonian in the sparse component format the AFQMC
        executable's ``ModelHamOpsGenerator`` reads.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist. A Hamiltonian
            already in the file is replaced; everything else — notably a
            ``Wavefunction`` — is left alone, so a Hamiltonian and a
            wavefunction can share one file in either order.

        Notes
        -----
        Set `spin_symm` on the instance before calling; it is recorded in the
        file. No electron count is recorded: the AFQMC executable takes it
        from the wavefunction.
        """
        if self.spin_symm is None:
            raise ValueError("Cannot write a Hamiltonian with no spin symmetry set")

        real_valued = self.real_valued

        with h5.File(path, 'a') as fh5:
            group = replace_group(fh5, 'Hamiltonian')
            write_hamiltonian_header(group, 'model')
            group.attrs['spin_type'] = self.spin_symm.label

            model = group.create_group('ModelHamiltonian')
            # int32, as TRIQS/h5 writes an int: the executable reads attributes by exact type
            model.attrs['number_of_sites'] = np.int32(self.nsites)
            model.attrs['number_of_bands'] = np.int32(self.nbands)

            self._write_lattice_metadata(model)

            for n, (key, component) in enumerate(
                    (key, component) for key in self.keys() for component in self[key]):
                component_group = model.create_group(f'ModelComponent_{n}')
                component_group.attrs['model_type'] = component.model_type
                component_group.attrs['spin_type'] = component.spin_symm.label

                if component.hubbard_strat_type is not None:
                    component_group.attrs['hst_type'] = component.hubbard_strat_type

                csr_array = component.csr_array
                if not real_valued:
                    csr_array = csr_array.astype(np.complex128)

                write_csr(component_group, key, csr_array)

    def _write_lattice_metadata(self, model) -> None:
        """Write `lattice_metadata`, if any, into the ``Lattice`` subgroup of `model`."""
        metadata = self.lattice_metadata
        if not metadata:
            return

        group = model.create_group('Lattice')
        group.attrs['type'] = metadata['type']
        group.attrs['cyl_mode'] = metadata.get('cyl_mode') or 'none'
        group.create_dataset('L', data=np.asarray(metadata['L'], dtype=np.int64))
        group.create_dataset('boundaries',
                             data=np.array(metadata['boundaries'], dtype=h5.string_dtype()))
        group.create_dataset('twist',
                             data=np.asarray(metadata['twist'], dtype=np.float64))
        group.create_dataset('lattice_vectors',
                             data=np.asarray(metadata['lattice_vectors'], dtype=np.float64))
        group.create_dataset('basis',
                             data=np.asarray(metadata['basis'], dtype=np.float64))

    @classmethod
    def _read_hdf5(cls, path) -> "LatticeHamiltonian":
        """
        Read a model Hamiltonian written by `to_hdf5`.

        The Hubbard components are split back out of the single combined ``Uij``
        matrix that `HamiltonianBuilder.finalize` produced — see
        `_split_hubbard_u`.
        """
        with h5.File(path, 'r') as fh5:
            spin_symm = SpinSymm.from_input(fh5['Hamiltonian'].attrs['spin_type'])

            group = fh5['Hamiltonian/ModelHamiltonian']
            nsites = int(group.attrs['number_of_sites'])
            nbands = int(group.attrs['number_of_bands'])

            hamiltonian = cls(
                nsites=nsites,
                nbands=nbands,
                spin_symm=spin_symm,
                lattice_metadata=_read_lattice_metadata(group),
            )

            # numbered without gaps, so the count is where they stop
            for n in itertools.count():
                if f'ModelComponent_{n}' not in group:
                    break
                component_group = group[f'ModelComponent_{n}']
                key = _component_key(component_group)
                component = HamiltonianComponent(
                    csr_array=read_csr(component_group[key]),
                    model_type=component_group.attrs['model_type'],
                    spin_symm=SpinSymm.from_input(component_group.attrs['spin_type']),
                    hst_type=component_group.attrs.get('hst_type'),
                )

                if key == 'Uij':
                    for split_key, split in _split_hubbard_u(component,
                                                             hamiltonian.nbasis):
                        hamiltonian.add_term(split_key, split)
                else:
                    hamiltonian.add_term(key, component)

        return hamiltonian


def lattice_metadata_from(lattice) -> dict:
    """
    Extract the recordable shape of `lattice` as a plain dictionary.

    Parameters
    ----------
    lattice : ~safiretools.hamiltonian.model.lattice.Lattice
        The lattice a `LatticeHamiltonian` is being built on.

    Returns
    -------
    dict
        Keys ``type``, ``L``, ``boundaries``, ``twist``, ``lattice_vectors``,
        ``basis`` and ``cyl_mode``.

    Notes
    -----
    Per-axis quantities are stored as sequences (``L = [L1, L2]``,
    ``boundaries = ['pbc', 'pbc']``) rather than as ``L1``/``L2`` pairs, and the
    unit cell as a ``lattice_vectors`` matrix rather than ``a1``/``a2``. Today's
    `Lattice` is two-dimensional, so those sequences always have length 2; the
    form is the dimension-agnostic one a future N-dimensional `Lattice` will
    need, so files written now stay readable then.
    """
    boundaries = [lattice.axis1_boundary, lattice.axis2_boundary]

    return {
        'type': lattice._type,
        'L': [int(size) for size in lattice.L],
        'boundaries': [_boundary_name(boundary) for boundary in boundaries],
        'twist': [float(boundary.phase) for boundary in boundaries],
        'lattice_vectors': [list(lattice.a1), list(lattice.a2)],
        'basis': [list(b) for b in lattice.basis],
        'cyl_mode': lattice.cyl_mode or 'none',
    }


def _boundary_name(boundary) -> str:
    """The `Lattice.from_dict` spelling of a `Boundary` instance's type."""
    from safiretools.hamiltonian.model.lattice import PBCBoundary

    return 'pbc' if isinstance(boundary, PBCBoundary) else 'open'


def _read_lattice_metadata(group):
    """Read back what `LatticeHamiltonian._write_lattice_metadata` wrote."""
    if 'Lattice' not in group:
        return {}

    lattice_group = group['Lattice']
    return {
        'type': lattice_group.attrs['type'],
        'L': [int(size) for size in lattice_group['L'][...]],
        'boundaries': [str(name) for name in lattice_group['boundaries'].asstr()[...]],
        'twist': [float(angle) for angle in lattice_group['twist'][...]],
        'lattice_vectors': [list(vector) for vector in lattice_group['lattice_vectors'][...]],
        'basis': [list(b) for b in lattice_group['basis'][...]],
        'cyl_mode': lattice_group.attrs['cyl_mode'],
    }


def _component_key(component_group) -> str:
    """
    The term key (``'tij'``, ``'Uij'``, ...) a stored component was written
    under: the one subgroup holding its CSR datasets.
    """
    keys = [name for name, item in component_group.items() if isinstance(item, h5.Group)]
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one term subgroup in '{component_group.name}', found {keys}"
        )
    return keys[0]


def _split_hubbard_u(component: HamiltonianComponent, nbasis: int):
    """
    Split a stored, combined ``Uij`` component back into the terms it was built
    from.

    `HamiltonianBuilder.finalize` merges the onsite Hubbard U, the
    density-density U1 and the spin-spin U2 of a given Hubbard-Stratonovich type
    into one matrix, so a file holds no separate U/U1/U2. The merge is
    invertible: U2 is the lower ``nbasis`` rows of a ``(2*nbasis, nbasis)``
    matrix, the onsite U is the diagonal of the upper block, and U1 is what is
    left off the diagonal (onsite U is built diagonal, while U1 never has
    diagonal entries — intrasite U1 is strictly inter-band, and intersite U1
    only connects distinct sites).

    Yields
    ------
    (str, HamiltonianComponent)
        Term key and component, for each non-empty part. A U2 part is yielded
        whenever the stored shape says one is present, even if it is numerically
        zero, so that re-combining reproduces the stored shape.
    """
    matrix = component.csr_array

    def part(csr, model_type=None):
        return HamiltonianComponent(
            csr_array=csr,
            model_type=model_type or component.model_type,
            spin_symm=component.spin_symm,
            hst_type=component.hubbard_strat_type,
        )

    if matrix.shape == (2 * nbasis, nbasis):
        density_density = sps.csr_array(matrix[:nbasis, :])
        spin_spin = sps.csr_array(matrix[nbasis:, :])
    elif matrix.shape == (nbasis, nbasis):
        density_density = matrix
        spin_spin = None
    else:
        raise ValueError(
            f"Stored Uij has shape {matrix.shape}, which is neither "
            f"({nbasis}, {nbasis}) nor ({2 * nbasis}, {nbasis})"
        )

    onsite = sps.csr_array(sps.diags(density_density.diagonal(), format='csr'))
    interaction = density_density - onsite

    if onsite.nnz:
        yield 'Uij', part(onsite)
    if interaction.nnz:
        yield 'U1ij', part(interaction)
    if spin_spin is not None:
        yield 'U2ij', part(spin_spin)
