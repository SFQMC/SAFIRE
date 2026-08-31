# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

r"""
`HamiltonianBuilder` — composes a `LatticeHamiltonian` term by term on a
`Lattice`.

Each build-step method (`nth_neighbor_hopping`, `onsite_hubbard`,
`hubbard_Jij`, ...) appends one `HamiltonianComponent`; `finalize` then combines
the components that can be combined, keeping terms with different
Hubbard-Stratonovich transformations separate.

`LatticeHamiltonian.from_dict` wraps this class for the common case of a
declarative input. Use the builder directly to compose terms no input key
covers.
"""

import functools
import itertools
import logging
from pathlib import Path
from warnings import warn

import numpy as np
import scipy.sparse as sps
import toml

from safiretools.hamiltonian.model.lattice import Lattice
from safiretools.hamiltonian.model.lattice_hamiltonian import (
    HamiltonianComponent,
    LatticeHamiltonian,
    lattice_metadata_from,
)
from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

HST_TYPES = (
    "discrete_spin",
    "discrete_charge",
    "continuous_spin",
    "continuous_charge",
)
"""Hubbard-Stratonovich transformation types the AFQMC executable understands.
The order also fixes the order combined Hubbard components are written in."""


def is_hermitian(M, tol=1e-10) -> bool:
    """
    True if the dense or sparse matrix `M` equals its conjugate transpose.

    Parameters
    ----------
    M : numpy.ndarray or scipy.sparse matrix
        Matrix to check.
    tol : float, optional
        Absolute tolerance, dense inputs only. Default 1e-10.
    """
    if sps.issparse(M):
        return (M - M.conj().T).nnz == 0
    return np.allclose(M, M.conj().T, atol=tol)


def force_hermitian(M, method='upper_triangular'):
    r"""
    Force a matrix to be Hermitian.

    Parameters
    ----------
    M : numpy.ndarray or scipy.sparse matrix
        Matrix to symmetrize.
    method : {'upper_triangular', 'triu', 'average', 'avg'}, optional
        How to symmetrize. Default ``'upper_triangular'``.

    Returns
    -------
    numpy.ndarray or scipy.sparse matrix
        A Hermitian matrix.

    Notes
    -----
    ``'upper_triangular'`` (``'triu'``) discards the lower triangle and mirrors
    the upper one, keeping the diagonal:
    :math:`M \rightarrow \mathrm{triu}(M, 0) + \mathrm{triu}(M, 1)^\dagger`.

    ``'average'`` (``'avg'``) takes :math:`\frac{1}{2}(M + M^\dagger)`.
    """
    if method in ('upper_triangular', 'triu'):
        logger.info("forcing Hermiticity from the upper triangle; "
                    "the lower triangle is ignored")
        if sps.issparse(M):
            return sps.triu(M, 0) + sps.triu(M, 1).conj().T
        return np.triu(M, 0) + np.triu(M, 1).conj().T

    if method in ('average', 'avg'):
        logger.info("forcing Hermiticity by averaging M and its conjugate transpose")
        return 0.5 * (M + M.conj().T)

    raise ValueError(f"Unknown force_hermitian method: {method}")


def onsite_band_matrix(value, nbands: int):
    r"""
    Build the ``(nbands, nbands)`` band matrix carrying a single *onsite*
    interaction amplitude: strictly upper-triangular, ``m < n``.

    Parameters
    ----------
    value : float
        Amplitude placed in every band pair ``m < n``.
    nbands : int
        Number of bands.

    Returns
    -------
    scipy.sparse.csr_array
        Band matrix of shape ``(nbands, nbands)``.

    See Also
    --------
    intersite_band_matrix : the corresponding inter-site form, which is *full*.

    Notes
    -----
    **This convention is not `force_hermitian`, and the two must not be
    merged.**

    The AFQMC executable has no notion of sites versus bands: it reads a
    *triangle of the whole interaction matrix*, indexed by the combined
    ``mu = site * nbands + band``. Per ``ModelHamOpsGenerator``, the
    density-density block of ``Uij`` keeps ``i <= j`` (so its diagonal *is* read
    — that is where the onsite Hubbard U sits), while the spin-spin block and
    ``Jij`` keep ``i < j``. Anything else is dropped with a warning.

    An onsite term kroneckers this band matrix with the *identity* over sites,
    so band pair ``(m, n)`` lands at combined ``(i*nbands + m, i*nbands + n)``
    and the two triangles map onto each other: ``m < n`` is read, ``m > n``
    falls in the combined lower triangle and is discarded. Filling only ``m < n``
    is therefore both complete and non-duplicating — the unordered band pair
    ``{m, n}`` on one site is one interaction, counted once. ``m == n`` is
    excluded because same-band onsite interaction is the separate Hubbard U.
    """
    pairs = list(itertools.combinations(range(nbands), r=2))

    row = [m for m, _ in pairs]
    column = [n for _, n in pairs]
    data = [value] * len(pairs)

    return sps.csr_array((data, (row, column)), shape=(nbands, nbands))


def intersite_band_matrix(value, nbands: int):
    r"""
    Build the ``(nbands, nbands)`` band matrix carrying a single *inter-site*
    interaction amplitude: the **full** matrix, every ``(m, n)``.

    Parameters
    ----------
    value : float
        Amplitude placed in every band pair.
    nbands : int
        Number of bands.

    Returns
    -------
    scipy.sparse.csr_array
        Dense-valued band matrix of shape ``(nbands, nbands)``.

    See Also
    --------
    onsite_band_matrix : the corresponding onsite form, which is upper-triangular.

    Notes
    -----
    Unlike the onsite case this must be the full matrix, **including its lower
    triangle**. An inter-site term kroneckers it with a site matrix that already
    holds only ``I < J``, so band pair ``(m, n)`` lands at combined
    ``(I*nbands + m, J*nbands + n)``, which is strictly above the combined
    diagonal for *every* ``(m, n)`` — the whole band matrix is in the range the
    executable reads.

    Those pairs are also physically distinct: ``(m, n)`` is band ``m`` on site
    ``I`` interacting with band ``n`` on site ``J``, which is not the same
    interaction as ``(n, m)``. Restricting to ``m <= n`` silently drops half of
    them, which is what afqmctools' `_build_intersite_band_matrix` did — on a
    2-band model it omitted the ``s0b1``–``s1b0`` interaction entirely.
    """
    return sps.csr_array(np.full((nbands, nbands), value))


def skip_empty_params(func):
    """
    Skip the decorated build step when every parameter is zero.

    ``None`` counts as zero, so an input template carrying unset keys does not
    build empty components. The decorated function's docstring is preserved.
    """

    @functools.wraps(func)
    def wrapper(self, params, *args, **kwargs):
        params = np.array(params)
        # The following line looks wrong, but is correct!
        #   numpy properly handles the comparison with None behind the scenes!
        params[params == None] = 0.0  # noqa: E711
        if not np.allclose(params, 0.0):
            func(self, params, *args, **kwargs)
        else:
            logger.debug("skipping empty param in %s", func.__name__)

    return wrapper


def iterate_nth_order(start_n=1):
    """
    Iterate a build step over a sequence of per-neighbor-order parameters.

    Decorates functions with the signature ::

        func(self, params, nth_neighbor=n, *args, **kwargs)

    A scalar or a single 2-d matrix is passed straight through with
    ``nth_neighbor`` defaulting to `start_n`. A 1-d sequence (or a 3-d stack of
    matrices) is iterated, with ``nth_neighbor`` taken from the index.

    Parameters
    ----------
    start_n : int, optional
        Neighbor order the first element corresponds to. Default 1.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, params, *args, **kwargs):
            params = np.array(params)
            if params.ndim in {0, 2}:
                kwargs["nth_neighbor"] = kwargs.get("nth_neighbor", start_n)
                func(self, params, *args, **kwargs)
            elif params.ndim in {1, 3}:
                for n, param in enumerate(params, start=start_n):
                    logger.debug("calling %s with %s for nth_neighbor=%s",
                                 func.__name__, param, n)
                    kwargs["nth_neighbor"] = n
                    func(self, param, *args, **kwargs)
            else:
                raise ValueError(
                    f"could not iterate over {func.__name__} with params = {params}: "
                    "np.array(params) must have dimension 0, 1, 2, or 3"
                )
        return wrapper
    return decorator


def _parse_ham_input(source: dict):
    """
    Translate a ``'hamiltonian'`` input mapping into the Hamiltonian attributes
    to set and the sequence of build steps to run.

    Returns
    -------
    ham_params : dict
        Attribute/value pairs to set on the built `LatticeHamiltonian`.
    build_steps : list[tuple[str, list]]
        ``(builder method name, positional args)`` pairs, in order.
    """
    source = dict(source)

    _known_params = {
        'nbands': 1,
        'twist': None,
        'nelec': (0, 0),
        'afm_pin_type': "staggered",
        'fm_pin_type': "staggered",
    }

    _supported_steps = {
        'nth_neighbor_hopping': 't',
        'custom_one_body': 'custom_one_body',
        'onsite_hubbard': 'U',
        'hubbard_U1_density_density': 'U1',
        'hubbard_U2_spin_spin': 'U2',
        'hubbard_Jij': 'J',
        'heisenberg_J': 'J_heisenberg',
        'nth_order_hubbard_Vij': "V",
        'afm_pinning': 'h_afm_pin',
        'fm_pinning': 'h_fm_pin',
        'charge_pinning': 'h_charge_pin',
    }

    hst_type_overrides = source.get("hst_types", None)

    # TODO: consider how we want to handle defaulting to 1.0 AND allowing there
    #        to be no hopping!
    if source.get('t') is None:
        source['t'] = 1.0

    ham_params = {param: source.get(param, default)
                  for param, default in _known_params.items()}

    build_steps = []
    for step, key in _supported_steps.items():
        if key in source:
            args = [source[key]]
            if hst_type_overrides is not None and key in hst_type_overrides:
                args.append(hst_type_overrides[key])
            build_steps.append((step, args))

    return ham_params, build_steps


class HamiltonianBuilder:
    """
    Builds a `LatticeHamiltonian` on a `Lattice`, one term at a time.

    Parameters
    ----------
    lattice : Lattice
        Lattice describing the geometry the Hamiltonian is defined on.
    nbands : int, optional
        Number of bands. Default 1.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry the Hamiltonian is expressed in. Default
        `SpinSymm.COLLINEAR`.
    nelec : tuple(int, int), optional
        ``(nup, ndown)``, recorded on the Hamiltonian for writing.

    Raises
    ------
    ValueError
        When a build step is given an amplitude it can't build from, or when no
        Hubbard-Stratonovich type can be unambiguously inferred.

    Notes
    -----
    See the User Documentation for the definition and input conventions of each
    Hamiltonian term.
    """

    def __init__(
            self,
            lattice: Lattice,
            nbands: int = 1,
            spin_symm=SpinSymm.COLLINEAR,
            nelec=(0, 0),
            **kwargs
    ) -> None:
        if lattice is None:
            raise ValueError("A Hamiltonian must be defined on a 'Lattice' instance.")

        self._lattice = lattice
        self._hamiltonian = LatticeHamiltonian(
            nsites=lattice.N_sites,
            nbands=nbands,
            spin_symm=spin_symm,
            nelec=nelec,
            lattice_metadata=lattice_metadata_from(lattice),
        )

        for param in ('afm_pin_type', 'fm_pin_type', 'twist'):
            if param in kwargs:
                setattr(self._hamiltonian, param, kwargs.pop(param))

        if kwargs:
            raise ValueError(
                f"Unknown HamiltonianBuilder parameters: {sorted(kwargs)}"
            )

    def get_hamiltonian(self) -> LatticeHamiltonian:
        """The `LatticeHamiltonian` being built."""
        return self._hamiltonian

    def get_lattice(self) -> Lattice:
        """The `Lattice` this Hamiltonian is defined on."""
        return self._lattice

    @classmethod
    def from_input(cls, source, lattice: Lattice = None) -> "HamiltonianBuilder":
        r"""
        Construct and fully build a `HamiltonianBuilder` from a declarative
        input.

        Parses a ``'hamiltonian'`` (and optionally ``'lattice'``) description,
        runs the corresponding build steps, and finalizes. Use
        `get_hamiltonian` on the result.

        Parameters
        ----------
        source : dict or str or pathlib.Path
            Hamiltonian (and possibly lattice) parameters. A str/Path is read as
            a TOML input file.
        lattice : Lattice, optional
            Lattice describing the geometry. Built from ``source['lattice']`` if
            omitted.

        Notes
        -----
        The Hamiltonian is built by a sequence of build steps named as key-value
        pairs in the ``'hamiltonian'`` section. Nearest-neighbor hopping is
        included by default; every interaction term must be requested
        explicitly. The build-step methods document finer control and the full
        input conventions.

        The following build steps are supported, listed as `key : description`:

        - t : nth-order neighbor hopping term; input convention are as follows:

          - if t is a scalar, then nearest-neighbor hopping is included with strength t
          - if t is a 1-dimensional list (i.e. [t1,t2,...,tn]), then up to nth-order hopping is included
          - if t is a 2-dimensional with shape (nbands, nbands), then
            the t is interpreted as an on-site inter-band hopping matrix
          - if t is a list of 2-dimensional arrays, then each element is interpreted as an on-site
            inter-band hopping matrix. This is functionally the same as :math:`t_{mn} = \sum t^{(i)}_{mn}`
            where :math:`t^{(i)}` is the ith element of the list and :math:`m,n` are band indices.

        - U : onsite Hubbard interaction term; input convention are as follows:

          - if U is a scalar, then the onsite Hubbard interaction term is included with strength U
            and applied to all sites and bands.
          - if U is 1-dimensional with length nbands (i.e. [U1,U2,...,Um] where for an m-band model),
            then the onsite Hubbard interaction term is included with strength U_i applied to band i
            and uniformly across sites.
          - if U is 1-dimensional with length nsites (i.e. [U1,U2,...,Un] where n is the number of sites),
            then the onsite Hubbard interaction term is included with strength U_i applied to site i
            and uniformly across bands.

        - U1 : Hubbard density-density interaction term. Input convention are as follows:

          - if U1 is a scalar, then the Hubbard density-density interaction term is included with strength U1
            and is applied to all sites, and is uniform across bands.
          - if U1 is a 2-dimensional with shape (nbands, nbands), then U1 is used as an intrasite density-density
            interaction, and is applied uniformly across sites.

        - U2 : Hubbard spin-spin interaction term. Input convention are as follows:

          - if U2 is a scalar, then the Hubbard spin-spin interaction term is included with strength U2
            and is applied to all sites, and is uniform across bands.
          - if U2 is a 2-dimensional with shape (nbands, nbands), then U2 is used as an intrasite spin-spin
            interaction, and is applied uniformly across sites.

        - J : Hund's coupling interaction term. Input convention are as follows:

          - if J is a scalar, then the Hund's coupling interaction term is included with strength J
            and is applied to all sites, and is uniform across bands.
          - if J is a 2-dimensional with shape (nbands, nbands), then J is used as an intrasite
            Hund's coupling interaction, and is applied uniformly across sites.

        Examples
        --------
        >>> builder = HamiltonianBuilder.from_input("input.toml")
        >>> hamiltonian = builder.get_hamiltonian()
        """
        if isinstance(source, (str, Path)):
            with open(source, 'r') as f:
                source_dict = toml.loads(f.read())
        elif isinstance(source, dict):
            source_dict = source
        else:
            raise ValueError(
                "Invalid parameter source: must be a dict, or the file name of a "
                f"TOML input file, not {type(source).__name__}"
            )

        ham_input = source_dict['hamiltonian']
        ham_params, build_steps = _parse_ham_input(ham_input)

        if lattice is None:
            logger.info("no lattice instance supplied: building from parameters")
            lattice = Lattice.from_dict(params=source_dict['lattice'])

        builder = cls(
            lattice=lattice,
            nbands=ham_params['nbands'],
            spin_symm=ham_input.get('spin_symm', SpinSymm.COLLINEAR),
            nelec=ham_params['nelec'],
        )

        hamiltonian = builder.get_hamiltonian()
        for param in ('twist', 'afm_pin_type', 'fm_pin_type'):
            setattr(hamiltonian, param, ham_params[param])

        for step, args in build_steps:
            logger.info("running build step %s(%s)", step, args)
            getattr(builder, step)(*args)

        builder.finalize()

        return builder

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _add_term(self, key: str, term: HamiltonianComponent) -> None:
        """Add `term` to the Hamiltonian under `key`."""
        self._hamiltonian.add_term(key, term)

    def _find_max_spin_symm(self) -> None:
        """
        Lower the Hamiltonian's spin symmetry to that of its least symmetric
        component. A Hamiltonian is only as symmetric as the term that breaks
        the most symmetry.
        """
        hamiltonian = self._hamiltonian
        current = hamiltonian.spin_symm or SpinSymm.CLOSED

        for components in hamiltonian.terms.values():
            for component in components:
                current = max(current, component.spin_symm)

        hamiltonian.spin_symm = current
        logger.info("max spin symmetry is %s", current)

    def _index_map(self, lattice_i: int, band_m: int) -> int:
        """
        Basis index for site `lattice_i` and band `band_m`: ``mu = (i, m)`` with
        the band as the fast index.
        """
        return lattice_i * self._hamiltonian.nbands + band_m

    def _is_valid_band_matrix(self, input: np.ndarray) -> bool:
        """True if `input` has shape ``(nbands, nbands)``."""
        nbands = self._hamiltonian.nbands
        return input.shape == (nbands, nbands)

    def _band_matrix(self, amplitude, name: str, force_herm: bool = False):
        """
        Interpret a one-body amplitude as an ``(nbands, nbands)`` band matrix.

        A scalar becomes ``amplitude * I``; a ``(nbands, nbands)`` matrix is used
        as given, after a Hermiticity check.

        Parameters
        ----------
        amplitude : numpy.ndarray
            Scalar or band-matrix amplitude.
        name : str
            Term name, used in error messages.
        force_herm : bool, optional
            Symmetrize a non-Hermitian band matrix instead of raising.

        Raises
        ------
        ValueError
            If `amplitude` is neither a scalar nor a band matrix, or if it is a
            non-Hermitian band matrix and `force_herm` is False.
        """
        if self._is_valid_band_matrix(amplitude):
            if is_hermitian(amplitude):
                return amplitude
            if not force_herm:
                raise ValueError(
                    f"{name} is not hermitian, and force_herm is False. "
                    "Rerun with force_herm=True to continue."
                )
            logger.warning("%s is not hermitian; forcing Hermiticity", name)
            return force_hermitian(amplitude, method='upper_triangular')

        if amplitude.shape == ():
            return amplitude * np.eye(self._hamiltonian.nbands)

        raise ValueError(f"could not build {name} from {amplitude}")

    def _upgrade_one_body_shape(self, in_mat: np.ndarray, target_spin_symm, nbasis: int):
        """
        Upgrade a one-body matrix's shape to match the target spin symmetry.

        Performs automatic shape conversions:

        - CLOSED ``(nbasis, nbasis)`` -> COLLINEAR ``(2*nbasis, nbasis)`` by vstacking
        - CLOSED ``(nbasis, nbasis)`` -> NONCOLLINEAR ``(2*nbasis, 2*nbasis)`` as block diagonal
        - COLLINEAR ``(2*nbasis, nbasis)`` -> NONCOLLINEAR ``(2*nbasis, 2*nbasis)`` by
          splitting the up/down sectors into diagonal blocks

        Parameters
        ----------
        in_mat : numpy.ndarray
            One-body matrix to upgrade.
        target_spin_symm : SpinSymm
            Spin symmetry to upgrade to.
        nbasis : int
            Basis size, ``nbands * N_sites``.

        Returns
        -------
        numpy.ndarray
            The matrix, shaped for `target_spin_symm`.

        Raises
        ------
        ValueError
            When `in_mat` has no valid shape for the target spin symmetry.
        """
        if target_spin_symm is SpinSymm.NONCOLLINEAR:
            if in_mat.shape == (2 * nbasis, 2 * nbasis):
                return in_mat
            if in_mat.shape == (nbasis, nbasis):
                return sps.block_diag([in_mat, in_mat], format='csr').toarray()
            if in_mat.shape == (2 * nbasis, nbasis):
                return sps.block_diag([in_mat[:nbasis, :], in_mat[nbasis:, :]],
                                      format='csr').toarray()
            raise ValueError(
                "custom one-body in_mat has invalid shape. Must have shape "
                "(nbasis,nbasis), (2*nbasis,nbasis), or (2*nbasis,2*nbasis) for "
                f"NONCOLLINEAR spin symmetry, but has shape {in_mat.shape}."
            )

        if target_spin_symm is SpinSymm.COLLINEAR:
            if in_mat.shape == (2 * nbasis, nbasis):
                return in_mat
            if in_mat.shape == (nbasis, nbasis):
                return np.vstack([in_mat, in_mat])
            raise ValueError(
                "custom one-body in_mat has invalid shape. Must have shape "
                "(nbasis,nbasis) or (2*nbasis,nbasis) for COLLINEAR spin "
                f"symmetry, but has shape {in_mat.shape}."
            )

        if target_spin_symm is SpinSymm.CLOSED:
            if in_mat.shape == (nbasis, nbasis):
                return in_mat
            raise ValueError(
                "custom one-body in_mat has invalid shape. Must have shape "
                f"(nbasis,nbasis) for CLOSED spin symmetry, but has shape "
                f"{in_mat.shape}."
            )

        raise ValueError(
            "invalid spin symmetry. Only CLOSED, COLLINEAR and NONCOLLINEAR are supported"
        )

    def _wrap_one_body_spin_structure(self, mat_up, target_spin_symm, mat_down=None):
        """
        Wrap up- and down-sector matrices into the spin structure
        `target_spin_symm` calls for.

        Parameters
        ----------
        mat_up : sparse or dense array
            Spin-up sector, ``(nbasis, nbasis)``.
        target_spin_symm : SpinSymm
            Target spin symmetry.
        mat_down : sparse or dense array, optional
            Spin-down sector. Defaults to `mat_up`.

        Returns
        -------
        sparse array
            Stacked ``(2*nbasis, nbasis)`` for COLLINEAR, block-diagonal
            ``(2*nbasis, 2*nbasis)`` for NONCOLLINEAR, `mat_up` unchanged for
            CLOSED.
        """
        if mat_down is None:
            mat_down = mat_up

        if target_spin_symm is SpinSymm.COLLINEAR:
            return sps.bmat(blocks=[[mat_up], [mat_down]], format='csr')

        if target_spin_symm is SpinSymm.NONCOLLINEAR:
            return sps.block_diag(mats=[mat_up, mat_down], format='csr')

        return mat_up

    # ------------------------------------------------------------------
    # one-body build steps
    # ------------------------------------------------------------------

    @iterate_nth_order(1)
    @skip_empty_params
    def nth_neighbor_hopping(self, t=1.0, nth_neighbor: int = 1, spin_symm=None,
                             opposite_twists=False, force_herm=False):
        r"""adds an nth-order neighbor hopping term to the Hamiltonian

        .. math:: \sum_{\langle ij\rangle^n} (-t) \hat{c}^\dagger_i \hat{c}_j

        Parameters
        ----------
        t : float | numpy.ndarray, default: 1.0
            the hopping strength. By convention a minus sign (-) is applied to the hopping.
            i.e. :math:`\sum_{<ij>} (-t) \hat{c}^\dagger_i \hat{c}_j`
            If t is a float, hoping between sites but not between bands, is used. If t is an
            numpy.ndarray, it must have shape nbands x nbands and is interpreted as a band
            dependent hopping.
        nth_neighbor : int, optional, default: 1
            the order of neighbor to use for hopping. `nth_neighbor` equal to 1 is nearest-neighbors.
            If `nth_neighbor` is not given, and `t` is an iterable, `t` is iterated over and `nth_neighbor` is
            inferred from the array index, `i`, of each entry in `t` as `nth_neighbor=i+1`
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerated type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
        opposite_twists : bool, optional, default: False
            if True, the hopping matrix is constructed using opposite twists, for the up and down
            spins (i.e. twist_down = -twist_up ). If False, the hopping matrix is constructed
            using the same twist for both spins.
        force_herm : bool, optional, default: False
            symmetrize a non-Hermitian band-hopping matrix rather than raising. See
            `force_hermitian`.

        Raises
        ------
        ValueError
            when the hopping matrix can't be constructed for the combination of `t` and `nth_neighbor`

        Examples
        --------
        adds default hopping (nearest-neighbor with t=1.0)

        >>> builder.nth_neighbor_hopping()

        add nearest-, and next-nearest-neighbor hopping

        >>> builder.nth_neighbor_hopping(t=[1.0,0.5])
        """
        if spin_symm is None:
            spin_symm = self._hamiltonian.spin_symm

        t = np.array(t)
        if np.allclose(t, 0.0):
            logger.info("no hopping t provided, skipping build")
            return

        row, column, data = [], [], []
        for pair in self._lattice.get_nth_neighbors(n=nth_neighbor,
                                                    twist=self._hamiltonian.twist):
            if pair.phase != 0.0:
                data.append(-1 * np.exp(-1j * pair.phase))
            else:
                data.append(-1)
            row.append(pair.i)
            column.append(pair.j)

        neighbor_graph = sps.csr_array(
            (data, (row, column)),
            shape=(self._lattice.N_sites, self._lattice.N_sites)
        )

        tband = self._band_matrix(t, f"{nth_neighbor}th-neighbor band hopping",
                                  force_herm=force_herm)

        Hhop = sps.kron(A=neighbor_graph, B=tband, format='csr')

        if spin_symm is SpinSymm.CLOSED and opposite_twists:
            warn(
                "Requested opposite twists for each spin sector and closed spin symmetry 1-body term."
                " Only the direct angle will be used. To use opposite twists, set spin_symm to "
                "COLLINEAR or NONCOLLINEAR."
            )

        if opposite_twists:
            logger.debug("using opposite twists for up and down spins")
            Hhop_down = Hhop.conj().T
        else:
            logger.debug("using the same twist for up and down spins")
            Hhop_down = Hhop

        Hhop = self._wrap_one_body_spin_structure(Hhop, spin_symm, Hhop_down)

        self._add_term('tij', HamiltonianComponent(
            csr_array=Hhop,
            model_type='one_body',
            spin_symm=spin_symm,
        ))

    def custom_one_body(self, in_mat: np.ndarray, spin_symm=None):
        """add a custom one-body term to the Hamiltonian

        Parameters
        ----------
        in_mat : numpy.ndarray
            the one-body term to add to the Hamiltonian
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerated type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.

        Notes
        -----
        The `in_mat` must have a valid shape for the Hamiltonian and spin symmetry. For
        COLLINEAR spin symmetry, the matrix must have shape (nbasis,nbasis) or
        (2*nbasis,nbasis), and for NONCOLLINEAR spin symmetry, the matrix must have shape
        (nbasis,nbasis), (2*nbasis,nbasis) or (2*nbasis,2*nbasis).

        Examples
        --------
        >>> import numpy as np
        >>> from safiretools import Lattice
        >>> from safiretools.hamiltonian.model.builder import HamiltonianBuilder
        >>> lattice = Lattice.from_dict(
        ...     params=dict(L1=3, L2=2, boundary1="PBC", boundary2="PBC")
        ... )
        >>> builder = HamiltonianBuilder(lattice=lattice)
        >>> nbasis = lattice.N_sites
        >>> one_body_matrix = 0.0001*np.random.rand(nbasis, nbasis)
        >>> one_body_matrix = 0.5*(one_body_matrix + one_body_matrix.T)
        >>> builder.custom_one_body(one_body_matrix)
        >>> builder.finalize()
        """
        if spin_symm is None:
            spin_symm = self._hamiltonian.spin_symm

        upgraded = self._upgrade_one_body_shape(
            in_mat, spin_symm, self._hamiltonian.nbasis)

        self._add_term('tij', HamiltonianComponent(
            csr_array=sps.csr_array(upgraded),
            model_type='one_body',
            spin_symm=spin_symm,
        ))

    @skip_empty_params
    def onebody_onsite(self, epsilon, spin_symm=None, force_herm=False):
        r"""
        Adds an onsite one-body term to the Hamiltonian (for example, a chemical potential,
            band energies, interband hopping, etc.)

        .. math::

            \hat{H}_{onsite} = \sum_i \sum_{m m'} \epsilon_{m m'} \hat{c}^\dagger_{i,m} \hat{c}_{i,m'}

        where :math:`\hat{c}^\dagger_{i,m}`/:math:`\hat{c}_{i,m}` are the creation / annihilation operator
        for site i and band m.

        Parameters
        ----------
        epsilon : float | numpy.ndarray
            the onsite energy. If epsilon is a float, it is interpreted as a band-independent onsite energy.
            If epsilon is a numpy.ndarray, it must have shape (nbands,nbands) and is interpreted as a band-dependent
            onsite energy. In all cases, epsilon is applied uniformly to all sites.
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry of the term.
        force_herm : bool, optional, default: False
            symmetrize a non-Hermitian `epsilon` rather than raising. See `force_hermitian`.
        """
        if spin_symm is None:
            spin_symm = self._hamiltonian.spin_symm

        epsilon = np.array(epsilon)
        epsilon_band = self._band_matrix(epsilon, "onsite one-body epsilon",
                                         force_herm=force_herm)

        epsilon_up = sps.kron(
            A=sps.eye(self._lattice.N_sites),
            B=epsilon_band,
            format='csr'
        )

        # for now, assume that the band energies are the same in the up and down sectors
        self._add_term('tij', HamiltonianComponent(
            csr_array=self._wrap_one_body_spin_structure(epsilon_up, spin_symm),
            model_type='one_body',
            spin_symm=spin_symm,
        ))

    @skip_empty_params
    def rashba_soc(self, rashba_lambda: float = 1.0, t=1.0, n: int = 1, spin_symm=None):
        r"""
        Builds a Rashba spin-orbit coupling term of the form:

        .. math::

            \hat{H}_{rashba SOC} = i \lambda_{rashba} \sum_{ij,\sigma\sigma'} t_{ij}
                                    (\vec{\sigma} \times \hat{r}_{ij})^{\sigma\sigma'}_z
                                    \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma'}

        where :math:`\vec{\sigma}` is the vector of Pauli matrices, and :math:`\vec{r}_{ij}`
        is the relative position between sites i and j.
        """
        if not rashba_lambda:
            logger.info("no Rashba SOC lambda provided, skipping build")
            return

        if spin_symm is None:
            spin_symm = self._hamiltonian.spin_symm

        if spin_symm is not SpinSymm.NONCOLLINEAR:
            raise ValueError("Rashba SOC is only valid for non-collinear spin symmetry")

        rashba_lambda = np.array(rashba_lambda)
        t = np.array(t)

        if t.ndim == 1:
            for order, tval in enumerate(t, start=1):
                self.rashba_soc(rashba_lambda=rashba_lambda, t=tval, n=order,
                                spin_symm=spin_symm)
            return

        if rashba_lambda.shape != ():
            raise ValueError(
                f"could not build Rashba SOC from lambda = {rashba_lambda} "
                f"with {n}th neighbor hopping"
            )

        nbands = self._hamiltonian.nbands
        if nbands > 1:
            warn(f"using {nbands} bands while Rashba SOC is only tested for nbands=1")

        shape = (self._hamiltonian.nbasis, self._hamiltonian.nbasis)

        def soc_matrix(axis):
            """The site-pair matrix carrying the `axis` component of r_ij."""
            row, column, data = [], [], []
            for pair in self._lattice.get_nth_neighbors(n=n,
                                                        twist=self._hamiltonian.twist):
                for m in range(nbands):
                    row.append(self._index_map(pair.i, m))
                    column.append(self._index_map(pair.j, m))

                    if pair.phase != 0.0:  # phase is the twist
                        prefactor = rashba_lambda * t * np.exp(-1j * pair.phase)
                    else:
                        prefactor = rashba_lambda * t

                    data.append(prefactor * pair.r_relative[axis])

            return sps.csr_array((data, (row, column)), shape=shape)

        H_rashba_up_down = 1j * soc_matrix(1) - soc_matrix(0)

        zero = sps.csr_array(H_rashba_up_down.shape, dtype=H_rashba_up_down.dtype)
        H_rashba = sps.bmat([
            [zero, H_rashba_up_down],
            [H_rashba_up_down.conj().T, zero]
        ], format='csr')

        self._add_term('tij', HamiltonianComponent(
            csr_array=H_rashba,
            model_type='one_body',
            spin_symm=spin_symm,
        ))

    def afm_pinning(self, h_afm_pin, axis=0, spin_symm=None, pin_type=None):
        r"""adds anti-ferromagnetic (AFM) pinning to the Hamiltonian

        Builds an anti-ferromagnetic (AFM) pinning term of the type:
        :math:`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i \downarrow} = - v_{i \uparrow} = 1/2(-1)^(h(i)) h_afm_pin`
        and h_afm_pin is the pinning field strength,
        and adds it to the Hamiltonian.

        Parameters
        ----------
        h_afm_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
        pin_type : str, optional
            overrides the Hamiltonian's ``afm_pin_type``.

        Raises
        ------
        ValueError
            when the Hamiltonian has an invalid `afm_pin_type`

        Notes
        -----
        available pinning functions:

        - staggered: :math:`h(i) = i_1+i_2`
        - fm: :math:`h(i) = i_2`

        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.
        """
        if pin_type is None:
            pin_type = self._hamiltonian.afm_pin_type
        pin_type = pin_type.lower()
        logger.info("using afm pin type: %s", pin_type)

        if pin_type in ("staggered", "afm"):
            def pinning_func(coord):
                return 0.5 * (-1)**(sum(coord))
        elif pin_type in ("same", "matching", "fm"):
            def pinning_func(coord):
                return 0.5 * (-1)**(coord[(axis + 1) % 2])
        else:
            raise ValueError(
                f"Attempted to build AFM pinning with invalid pin_type '{pin_type}'"
            )

        self.edge_pinning(pinning_func, h_afm_pin, False, axis, spin_symm)

    def fm_pinning(self, h_fm_pin, axis=0, spin_symm=None, pin_type=None):
        r"""adds ferromagnetic (FM) pinning to the Hamiltonian

        builds an FM pinning term of the type:
        :math:`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i \downarrow} = - v_{i \uparrow} = 1/2 h(i) h_fm_pin`
        and h_fm_pin is the pinning field strength.

        Parameters
        ----------
        h_fm_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
        pin_type : str, optional
            overrides the Hamiltonian's ``fm_pin_type``.

        Raises
        ------
        ValueError
            when the Hamiltonian has an invalid `fm_pin_type`

        Notes
        -----
        available functions:

        - staggered: :math:`h(i) = (-1)^{(i_1==0)}`
        - fm: :math:`h(i) = +0.5`

        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.
        """
        if pin_type is None:
            # afqmctools read afm_pin_type here, which made fm_pin_type dead input
            pin_type = self._hamiltonian.fm_pin_type
        pin_type = pin_type.lower()
        logger.info("using fm pin type: %s", pin_type)

        if pin_type in ("staggered", "afm", "opposite"):
            def pinning_func(coord):
                return 0.5 * (-1)**(coord[axis] == 0)
        elif pin_type in ("same", "matching", "fm"):
            def pinning_func(coord):
                return 0.5
        else:
            raise ValueError(
                f"Attempted to build FM pinning with invalid pin_type '{pin_type}'"
            )

        self.edge_pinning(pinning_func, h_fm_pin, False, axis, spin_symm)

    def charge_pinning(self, h_charge_pin, axis=0, spin_symm=None):
        r"""Adds charge pinning to the Hamiltonian

        builds a charge pinning term of the type:
        :math:`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i \downarrow} = v_{i \uparrow} = h_charge_pin`
        and h_charge_pin is the pinning field strength.

        Parameters
        ----------
        h_charge_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.

        Notes
        -----
        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.
        """
        self.edge_pinning(lambda coord: 0.5, h_charge_pin, True, axis, spin_symm)

    def edge_pinning(self, pinning_func, h_pin, same_sign=False, axis=0, spin_symm=None):
        """Add general pinning, at the edge, to the Hamiltonian

        Generalized pinning function: apply `pinning_func` to sites at the edge.
        `pinning_func` takes a lattice coordinate.

        Parameters
        ----------
        pinning_func : callable(coord) -> float
            a modulation function for the pinning field
        h_pin : float
            the amplitude of the pinning field
        same_sign : bool
            if True, spin up and spin down sectors have the same sign.
            if False, spin up and spin down sectors have opposite signs.
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.

        Notes
        -----
        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.
        """
        if spin_symm is None:
            spin_symm = self._hamiltonian.spin_symm

        nbands = self._hamiltonian.nbands
        shape = (self._hamiltonian.nbasis, self._hamiltonian.nbasis)

        row, column, data = [], [], []
        # TODO: a little inefficient, i.e. could loop over just the sites
        #           on the desired edge, not a problem for now
        for site in self._lattice.get_sites():
            r = site.coord
            if r[axis] == 0 or r[axis] == self._lattice.L[axis] - 1:
                for m in range(nbands):
                    row.append(self._index_map(site.index, m))
                    column.append(self._index_map(site.index, m))
                    data.append(pinning_func(site.coord) * h_pin)

        H_pin_up = sps.csr_array((data, (row, column)), shape=shape)
        H_pin_down = H_pin_up if same_sign else -1 * H_pin_up

        self._add_term('tij', HamiltonianComponent(
            csr_array=self._wrap_one_body_spin_structure(H_pin_up, spin_symm, H_pin_down),
            model_type='one_body',
            spin_symm=spin_symm,
        ))

    # ------------------------------------------------------------------
    # interaction build steps
    # ------------------------------------------------------------------

    def _get_hst_type(self, U, is_discrete=True) -> str:
        """
        Choose the default Hubbard-Stratonovich transformation type from the
        sign of `U` (any of Ui, U1ij, U2ij; also J, though J should use the
        continuous versions).

        Parameters
        ----------
        U : float or numpy.ndarray
            The interaction amplitude.
        is_discrete : bool, optional
            Select the discrete rather than continuous family. Default True.

        Raises
        ------
        ValueError
            If `U` has both signs, so no single type applies.
        """
        prefix = "discrete_" if is_discrete else "continuous_"

        if np.all(U <= 0):
            return prefix + "charge"
        if np.all(U >= 0):
            return prefix + "spin"

        raise ValueError(
            "Unable to unambiguously choose a Hubbard-Stratonovich "
            f"transformation type for U = {U}"
        )

    def _clean_hst_type(self, hst_type: str) -> str:
        """
        Normalize a Hubbard-Stratonovich transformation type string to one the
        AFQMC executable understands.

        Raises
        ------
        ValueError
            If `hst_type` names no known transformation.
        """
        cleaned = hst_type.lower().replace(" ", "_")
        if cleaned not in HST_TYPES:
            raise ValueError(
                f"Invalid hst_type '{hst_type}': valid options are: {', '.join(HST_TYPES)}"
            )
        return cleaned

    def _add_interaction(self, band_U, _key, model_type=None, is_discrete=True,
                         spin_symm=SpinSymm.CLOSED, hst_type=None, nth_neighbor=0):
        """
        Add an interaction term, splitting it by sign first.

        Positive and negative amplitudes call for different Hubbard-Stratonovich
        transformations, so each sign becomes its own component and the two are
        never combined by `finalize`.

        Parameters
        ----------
        band_U : scipy.sparse array
            The possibly band-dependent interaction matrix.
        _key : str
            Term key to store under.
        model_type : str, optional
            Model type recorded on the component.
        is_discrete : bool, optional
            Use a discrete Hubbard-Stratonovich type. Default True.
        spin_symm : SpinSymm, optional
            Spin symmetry of the term. Default `SpinSymm.CLOSED`.
        hst_type : str, optional
            Explicit Hubbard-Stratonovich type. Inferred from the sign of the
            interaction when omitted, which keeps constrained-path AFQMC rather
            than phaseless AFQMC.
        nth_neighbor : int, optional
            Neighbor order for the interaction. 0 (default) means on-site.
        """
        positive = band_U.copy()
        positive[band_U < 0] = 0.0

        negative = band_U.copy()
        negative[band_U > 0] = 0.0

        for part in (negative, positive):
            # count_nonzero(), not nnz: masking a sparse matrix to zero can leave
            #   the entries stored explicitly, so nnz would count them.
            if part.count_nonzero() == 0:
                continue
            logger.debug("building %s with U values: %s", _key, part)
            self._add_interaction_impl(
                band_U=part,
                _key=_key,
                model_type=model_type,
                is_discrete=is_discrete,
                spin_symm=spin_symm,
                hst_type=hst_type,
                nth_neighbor=nth_neighbor,
            )

    def _add_interaction_impl(self, band_U, _key, model_type=None, is_discrete=True,
                              spin_symm=SpinSymm.CLOSED, hst_type=None, nth_neighbor=0):
        r"""
        Build a CSR interaction from the inter-band matrix `band_U` and add it.

        Explicitly, builds

        .. math:: U_{(i,m),(j,m')} = S_{ij} U^{band}_{m,m'}

        where :math:`S` is the identity for an on-site interaction, or the
        ``i < j`` neighbor graph of order `nth_neighbor` otherwise.

        Notes
        -----
        Agnostic to the type of U term (U, U1, U2, ...); `band_U` is assumed to
        already follow the upper-triangular convention (see
        `upper_triangular_band_matrix`).
        """
        if nth_neighbor == 0:
            site_matrix = sps.identity(self._lattice.N_sites)
        else:
            site_matrix = sps.lil_matrix((self._lattice.N_sites, self._lattice.N_sites))
            for neighbor in self._lattice.get_nth_neighbors(n=nth_neighbor):
                # convention for interactions is i < j!
                if neighbor.i < neighbor.j:
                    site_matrix[neighbor.i, neighbor.j] += 1

        H_interaction = sps.kron(A=site_matrix, B=band_U, format='csr')

        if hst_type is None:
            hst_type = self._get_hst_type(band_U.toarray(), is_discrete=is_discrete)

        self._add_term(_key, HamiltonianComponent(
            csr_array=H_interaction,
            model_type=model_type,
            spin_symm=spin_symm,
            hst_type=self._clean_hst_type(hst_type),
        ))

    @skip_empty_params
    def onsite_hubbard(self, U, hst_type=None):
        """adds onsite Hubbard U to the Hamiltonian with
        possibly band-dependent amplitude U. See Notes for conventions.

        Parameters
        ----------
        U : float | iterable | numpy.ndarray
            the amplitude of the onsite Hubbard interaction. See Notes
            for the conventions on how U is interpreted.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U.

        Notes
        -----
        conventions:

        - a single number given for U implies the same U for all sites
        - a 1-Dimensional array with length nbands gives a different U for each
          band, but U is independent of site index.
        - a 1-Dimensional array with length nsites gives a different U for each
          site, uniform across bands.

        Advanced Notes:

        - in case of an array of U values, the U values are separated into
          positive and negative values. Up to two terms may be generated (i.e. one positive
          and one negative) to allow different Hubbard-Stratonovich Transformations to be
          used for each case.
        """
        U = np.array(U)

        if np.allclose(U, 0.0):
            logger.info("no Hubbard U provided, skipping build")
            return

        logger.info("building Hubbard U term with U=%s", U)

        U_positive = U.copy()
        U_positive[U < 0] = 0.0

        U_negative = U.copy()
        U_negative[U > 0] = 0.0

        for part in (U_positive, U_negative):
            if np.any(part):
                logger.debug("building onsite hubbard with U values: %s", part)
                self._onsite_hubbard_impl(U=part, hst_type=hst_type)

    def _onsite_hubbard_impl(self, U, hst_type=None):
        """
        Build the onsite Hubbard U matrix from the amplitude `U` and add it to
        the Hamiltonian.
        """
        nbands = self._hamiltonian.nbands
        nsites = self._lattice.N_sites

        if U.shape == ():
            H_U = sps.csr_array(U * sps.eye(nsites * nbands, format='csr'))
        elif U.ndim == 1 and U.shape[0] == nbands:
            H_U = sps.kron(A=sps.identity(nsites), B=np.diag(U), format='csr')
        elif U.ndim == 1 and U.shape[0] == nsites:
            H_U = sps.kron(A=np.diag(U), B=sps.identity(nbands), format='csr')
        else:
            raise ValueError(f"could not build hubbard U from U = {U}")

        if hst_type is None:
            hst_type = self._get_hst_type(U)

        self._add_term('Uij', HamiltonianComponent(
            csr_array=H_U,
            model_type='hubbard_u',
            hst_type=self._clean_hst_type(hst_type),
        ))

    def _interaction_band_matrix(self, amplitude, nth_neighbor: int, name: str):
        """
        Interpret a U1/U2/J amplitude as an ``(nbands, nbands)`` band matrix.

        A scalar is expanded by `onsite_band_matrix` for an on-site term
        (strictly upper-triangular) or `intersite_band_matrix` for an inter-site
        one (the full matrix) — see those for why the two differ. A
        ``(nbands, nbands)`` matrix is used as given.

        Raises
        ------
        ValueError
            If `amplitude` has neither shape.
        """
        if amplitude.shape == ():
            nbands = self._hamiltonian.nbands
            if nth_neighbor > 0:
                return intersite_band_matrix(amplitude, nbands)
            return onsite_band_matrix(amplitude, nbands)

        if self._is_valid_band_matrix(amplitude):
            return sps.csr_array(amplitude)

        raise ValueError(f"could not build {name} from {amplitude}: invalid shape")

    def _require_multiband_onsite(self, nth_neighbor: int, name: str) -> None:
        """
        Reject an on-site inter-band interaction on a single-band model, where
        there is no second band for it to act between.
        """
        if self._hamiltonian.nbands == 1 and nth_neighbor == 0:
            raise ValueError(
                f"Onsite {name} is not supported for nbands=1; try setting nth_neighbor > 0"
            )

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_U1_density_density(self, U1, hst_type=None, nth_neighbor=0):
        r"""
        Adds a density-density Hubbard U1 term to the Hamiltonian.

        .. math:: H_{U_1} = \sum_{ij} U^1_{ij} (n_{i\uparrow} n_{j\downarrow} + n_{i\downarrow} n_{j\uparrow})

        Parameters
        ----------
        U1 : float | numpy.ndarray
            the amplitude of the Hubbard U1 term. See Notes for conventions.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U1.
        nth_neighbor : int, optional
            the order of neighbor to use for the interaction term. If not given, the interaction is
            assumed to be on-site.

        Notes
        -----
        conventions:

        - a single number given for U1 implies the same U1 for all sites, and is uniform across bands
        """
        self._require_multiband_onsite(nth_neighbor, "Hubbard U1")

        U1 = np.array(U1)
        if np.allclose(U1, 0.0):
            logger.info("no Hubbard-Kanamori U1 provided, skipping build")
            return

        logger.info("building Hubbard-Kanamori density-density U1 term with U1=%s", U1)

        self._add_interaction(
            band_U=self._interaction_band_matrix(U1, nth_neighbor, "hubbard U1"),
            _key='U1ij',
            model_type='hubbard_u',
            hst_type=hst_type,
            nth_neighbor=nth_neighbor,
        )

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_U2_spin_spin(self, U2, hst_type=None, nth_neighbor=0):
        r"""
        Adds a spin-spin Hubbard U2 term to the Hamiltonian.

        .. math:: H_{U_2} = \sum_{ij} U^2_{ij} (n_{i\uparrow} n_{j\uparrow} + n_{i\downarrow} n_{j\downarrow})

        Parameters
        ----------
        U2 : float | numpy.ndarray
            the amplitude of the Hubbard U2 term. See Notes for conventions.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U2.
        nth_neighbor : int, optional
            the order of neighbor to use for the interaction term. If not given, the interaction is
            assumed to be on-site.

        Notes
        -----
        conventions:

        - a single number given for U2 implies the same U2 for all sites, and is uniform across bands
        """
        self._require_multiband_onsite(nth_neighbor, "Hubbard U2")

        U2 = np.array(U2)
        if np.allclose(U2, 0.0):
            logger.info("empty Hubbard-Kanamori U2 provided, skipping build")
            return

        logger.info("building Hubbard-Kanamori spin-spin U2 term with U2=%s", U2)

        self._add_interaction(
            band_U=self._interaction_band_matrix(U2, nth_neighbor, "hubbard U2"),
            _key='U2ij',
            model_type='hubbard_u',
            spin_symm=SpinSymm.COLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor,
        )

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_Jij(self, J, hst_type=None, nth_neighbor=0):
        r"""
        Adds a Hund's J term to the Hamiltonian.

        The general form of the Hund's J term is:

        .. math::
            \sum_{i<j} J_{ij} (
                \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
                +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow}
                +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
                +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow}
            ),

        where i,j are combined site and band indices.

        Here, we generate the :math:`J_{ij}` matrix which
        will be interpreted as above.
        """
        self._require_multiband_onsite(nth_neighbor, "Hubbard J")

        J = np.array(J)
        if np.allclose(J, 0.0):
            logger.info("no Hubbard-Kanamori J provided, skipping build")
            return

        logger.info("building Hubbard-Kanamori J term with J=%s", J)

        self._add_interaction(
            band_U=self._interaction_band_matrix(J, nth_neighbor, "hubbard J"),
            _key='Jij',
            model_type='hubbard_j',
            is_discrete=False,
            spin_symm=SpinSymm.COLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor,
        )

    @iterate_nth_order(1)
    @skip_empty_params
    def heisenberg_J(self, J, hst_type=None, nth_neighbor=1):
        r"""
        Adds an isotropic Heisenberg spin-term.

        The isotropic Heisenberg Hamiltonian can be expressed
        in terms of the Hubbard-Kanamori terms as:

        .. math:: H_{Heisenberg}[J] = H_{U_1}[-J/2] + H_{U_2}[J/2]
                          - J \sum_{i < j}( C^\dagger_{i \uparrow}C^\dagger_{j \downarrow} C_{i \downarrow}C_{j \uparrow}
                                        + C^\dagger_{i \downarrow}C^\dagger_{j \uparrow} C_{i \uparrow}C_{j \downarrow} )

        The last two terms are two of the four terms of the Hund's J.
        """
        self._require_multiband_onsite(nth_neighbor, "Heisenberg J")

        self.hubbard_U1_density_density(-J / 4, hst_type=hst_type, nth_neighbor=nth_neighbor)
        self.hubbard_U2_spin_spin(J / 4, hst_type=hst_type, nth_neighbor=nth_neighbor)

        J = np.array(-J)  # by convention
        if np.allclose(J, 0.0):
            logger.info("empty Heisenberg J provided, skipping build")
            return

        logger.info("building Heisenberg J term with J=%s", J)

        self._add_interaction(
            band_U=self._interaction_band_matrix(J, nth_neighbor, "heisenberg J"),
            _key='J_heisenberg',
            model_type='heisenberg_j',
            is_discrete=False,
            spin_symm=SpinSymm.NONCOLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor,
        )

    @iterate_nth_order(1)
    @skip_empty_params
    def nth_order_hubbard_Vij(self, V, hst_type=None, nth_neighbor=1):
        r"""
        Builds an extended Hubbard model V term and adds it to the Hamiltonian.

        The conventions used are, if :math:`i = (\mu, m)` is a combined lattice and band index,

        .. math:: H_V = \sum_{\langle(i<j)\rangle} \sum_{\sigma \sigma'} V_{n m} ( \hat{n}_{i \sigma \sigma'} \hat{n}_{j \sigma \sigma'}  )

        which is equivalent to :math:`H_V = H_{U_1}(U_1=V) + H_{U_2}(U_2=V)`.

        .. TODO: Be careful about the conventions on defining 'V', there could be a
           factor of 2 (or 1/2)
        """
        self.hubbard_U1_density_density(V, hst_type=hst_type, nth_neighbor=nth_neighbor)
        self.hubbard_U2_spin_spin(V, hst_type=hst_type, nth_neighbor=nth_neighbor)

    # ------------------------------------------------------------------
    # finalization
    # ------------------------------------------------------------------

    def _components_by_hst(self, components) -> dict:
        """Group `components` by their Hubbard-Stratonovich transformation type."""
        by_hst = {}
        for component in components:
            by_hst.setdefault(component.hubbard_strat_type, []).append(component)
        return by_hst

    def _combine_components(self, _key='tij') -> None:
        """
        Combine every term stored under `_key` into as few components as
        possible.

        Terms are kept separate while building so that different kinds of
        one-body term share one interface. Two-body terms are only combined when
        they share a Hubbard-Stratonovich type.
        """
        if _key not in self._hamiltonian.terms:
            return

        new_terms = []
        for _, components in self._components_by_hst(self._hamiltonian[_key]).items():
            if len(components) == 1:
                new_terms.append(components[0])
            else:
                logger.debug("combining %s terms", _key)
                new_terms.append(sum(components))

        self._hamiltonian.terms[_key] = new_terms

    def _combine_hubbard_u(self) -> None:
        """
        Merge the Hubbard U, U1 and U2 terms into the single ``Uij`` matrix the
        AFQMC executable reads, preserving separation by Hubbard-Stratonovich
        type.

        Two cases:

        1. U2 is non-zero, so the result is a ``(2M, M)`` matrix (M the basis
           size) whose top block is ``U + U1`` and whose bottom block is ``U2``.
        2. U2 is zero, so the result is just ``U + U1``.

        `LatticeHamiltonian._split_hubbard_u` inverts this on read.
        """
        def terms_by_hst(local_terms, hst_type, keys):
            """Every term of Hubbard-Stratonovich type `hst_type` under `keys`."""
            return [term
                    for key in keys
                    for term in local_terms.get(key, [])
                    if term.hubbard_strat_type == hst_type]

        logger.debug("combining Hubbard U, U1, and U2 terms where possible")

        onsite_key, density_key, spin_key = 'Uij', 'U1ij', 'U2ij'
        hamiltonian = self._hamiltonian

        # Be careful: all U, U1 and U2 terms are *moved* into a local dictionary,
        #   and the combined results are added back as we iterate over it.
        local_terms = {}
        for key in (onsite_key, density_key, spin_key):
            self._combine_components(key)
            if key in hamiltonian.keys():
                local_terms[key] = hamiltonian.pop_term(key)

        for hst_type in HST_TYPES:
            density_density = terms_by_hst(local_terms, hst_type, [onsite_key, density_key])
            density_density_term = sum(density_density) if density_density else None

            spin_spin = terms_by_hst(local_terms, hst_type, [spin_key])
            if len(spin_spin) > 1:
                raise RuntimeError(
                    "[Error for Developers] HamiltonianBuilder._combine_hubbard_u found "
                    f"more than one spin-spin term with hst_type={hst_type}"
                )

            if spin_spin:
                spin_spin_term = spin_spin[0]
                if density_density_term is None:
                    density_density_csr = sps.csr_array(spin_spin_term.shape)
                else:
                    density_density_csr = density_density_term.csr_array

                hubbard_matrix = sps.vstack([density_density_csr,
                                             spin_spin_term.csr_array])
                spin_symm = spin_spin_term.spin_symm
            else:
                if density_density_term is None:
                    continue
                hubbard_matrix = density_density_term.csr_array
                spin_symm = density_density_term.spin_symm

            self._add_term('Uij', HamiltonianComponent(
                csr_array=hubbard_matrix,
                model_type='hubbard_u',
                spin_symm=spin_symm,
                hst_type=hst_type,
            ))

    def log_components(self, verbose=False) -> None:
        """Log each current Hamiltonian component, and its matrix if `verbose`."""
        for key, components in self._hamiltonian.terms.items():
            logger.info(" === %s terms ===", key)
            for i, component in enumerate(components):
                logger.info("%s component %d with hst_type = %s",
                            key, i, component.hubbard_strat_type)
                if verbose:
                    logger.info("csr matrix: %s", component.csr_array)

    def finalize(self, verbose=False) -> None:
        """
        Combine Hamiltonian terms wherever possible, keeping terms with
        different `hst_type` separate, and settle the Hamiltonian's spin
        symmetry.

        Parameters
        ----------
        verbose : bool
            Log each component's matrix before and after combining.
        """
        if verbose:
            logger.info(" ====== Hamiltonian terms before combining components ====== ")
            self.log_components(verbose)

        logger.debug("combining terms of the same type")
        self._combine_components('tij')
        self._combine_hubbard_u()
        self._find_max_spin_symm()

        if verbose:
            logger.info(" ====== Hamiltonian terms after combining components ====== ")
            self.log_components(verbose)
