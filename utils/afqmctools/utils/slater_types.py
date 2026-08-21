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
Define an enumerated type to simplify specifying a Slater determinant type.
"""

from enum import Enum

from afqmctools.hamiltonian.model.ham_class import SpinSymm

class _SlaterType(Enum):
    """
    An enumerated class/type in order to
        document the different types of
        Slater determinant.

    Design Notes:

    - the '_slater_enum_map(type)' function should be used 
       internally to convert all user input to a specific 
       Slater determinant type
    - the `_slater2dims(slater_type)` function should be used
       internally to map from a SlaterType enumeration to
       the integer value used by the C++ side of the AFQMC code.
    
       
    # TODO: use integers or auto() below to enforce non-reliance
    #        on the specific enumerated values - see design notes
    """

    CLOSED='rhf'
    COLLINEAR='uhf'
    NONCOLLINEAR='ghf'


def _slater_enum_map(type):
    """
    Maps from a slater determint 'type' description ( possibly
      an int, or one of several strings ), to a specific 
      _SlaterType enumeration.

    this function should be called internally whenever user input
       must be mapped onto a specific Slater determinant type.
    """
    if isinstance(type,_SlaterType):
        return type

    # TODO: for Python 3.11+ used match-case
    if type in (1,'closed','rhf',SpinSymm.CLOSED):
        return _SlaterType.CLOSED
    elif type in (2,'collinear','rohf','uhf',SpinSymm.COLLINEAR):
        return _SlaterType.COLLINEAR
    elif type in (3,'noncollinear','ghf',SpinSymm.NONCOLLINEAR):
        return _SlaterType.NONCOLLINEAR
    else:
        raise ValueError(f"Invalid Slater type: {type}")


def _slater2dims(slater_type:_SlaterType):
    """
    Map from a specific _SlaterType the appropriate integer
       value used in writing 'dims'

    This function should be called internally whenever user input
       must be mapped onto a specific Slater determinant type.

    Parameters
    ----------
    slater_type : _SlaterType
        the Slater determinant type to be converted

    Returns
    -------
    int
        the integer value used in the C++ code to represent
        the Slater determinant type
    """
    assert isinstance(slater_type,_SlaterType)

    # TODO: for Python 3.11+ used match-case
    if slater_type == _SlaterType.CLOSED:
        return 1
    elif slater_type == _SlaterType.COLLINEAR:
        return 2
    elif slater_type == _SlaterType.NONCOLLINEAR:
        return 3
    else:
        return 0


# TODO: use a SlaterDeterminant class to document the type of object that we have
class SlaterDeterminant:

    def __init__(self,
        type:_SlaterType,
        slater_matrix=None
    ) -> None:

        self.type = _SlaterType(type)
        self.slater_matrix = slater_matrix

    def __matmul__(self,other):
        raise NotImplementedError

    def __mul__(self,other):
        raise NotImplementedError

    def __add__(self,other):
        # build and return a NOMSD
        raise NotImplementedError


class MultiSlater:

    def __init__(self) -> None:
        pass

class NonorthMSD(MultiSlater):

    def __init__(self) -> None:
        pass


class ParticleHoleMSD(MultiSlater):

    def __init__(self) -> None:
        pass


def _is_collinear(phi,nelec,M):
    """
    Determine if the Slater matrix is collinear-like
    (i.e. UHF or ROHF-like) based on the shape of the
    Slater matrix and the number of electrons.

    Collinear determinants come in two equivalent layouts:
    the spin-resolved `(2,M,*)` form, and the 2-D form with
    the alpha and beta blocks concatenated column-wise,
    `(M,nalpha+nbeta)`.
    """

    # UHF-like
    if len(phi.shape) == 3:
        return phi.shape[0] == 2 and phi.shape[1] == M
    elif len(phi.shape) != 2:
        return False
    return phi.shape[0] == M and phi.shape[1] == nelec[0] + nelec[1]

def _is_closed(phi,nelec,M):
    """
    Determine if the Slater matrix is closed-shell-like (i.e. RHF-like):
    a single spin block of doubly-occupied orbitals, so phi is 2-D with
    M rows and exactly nalpha == nbeta columns.
    """
    if len(phi.shape) != 2 or nelec[0] != nelec[1]:
        return False
    return phi.shape[0] == M and phi.shape[1] == nelec[0]

def _is_noncollinear(phi,nelec,M):
    """
    Determine if the Slater matrix is noncollinear-like (i.e. GHF-like):
    every column spans both spin blocks, so phi is 2-D with 2*M rows
    and nalpha+nbeta columns.
    """
    if len(phi.shape) != 2:
        return False
    return phi.shape[0] == 2*M and phi.shape[1] == nelec[0] + nelec[1]

def _get_slater_type(phi,nelec,M):
    """
    Determine the Slater determinant type based
      on the given Slater matrix dimensions, and
      the number of electrons (nelec) if provided.

    Parameters
    ----------
    phi : np.ndarray
        the Slater matrix of the wavefunction, holding only the *occupied*
        orbitals. The expected shapes are `(M,nalpha)` for Closed,
        `(M,nalpha+nbeta)` or `(2,M,*)` for Collinear, and
        `(2*M,nalpha+nbeta)` for Noncollinear.
    nelec : tuple
        the number of electrons in the system.
    M : int
        the number of spatial orbitals in the system.

    Returns
    -------
    _SlaterType
        the type of the Slater determinant.

    Raises
    ------
    ValueError
        if the shape of `phi` is not consistent with any Slater determinant
        type for the given `nelec` and `M`.

    Notes
    -----
    This expects a Slater matrix over the occupied orbitals only, *not* a full
      molecular orbital coefficient matrix. The two conventions are
      indistinguishable by shape alone (an RHF coefficient matrix `(M,M)` at
      half filling looks exactly like a Collinear `(M,nalpha+nbeta)` block), so
      callers holding a full coefficient matrix must determine the type from
      the occupations instead - see `afqmctools.utils.pyscf_utils`.

    A collinear determinant in the `(2,M,*)` layout is unambiguous, since it is
      the only Slater matrix with 3 dimensions.

    Closed and Collinear are distinguished by the number of columns: a Closed
      determinant carries one spin block (nalpha columns), a Collinear
      determinant carries both (nalpha+nbeta columns).

    Collinear and Noncollinear are distinguished by the number of rows, and so
      require M (i.e. the number of spatial orbitals).

    The three predicates below are mutually exclusive for any M >= 1 and any
      nelec other than the (0,0) vacuum, so the order of the checks does not
      affect the result. Shapes matching none of them are rejected.
    """

    if _is_noncollinear(phi,nelec=nelec,M=M):
        return _SlaterType.NONCOLLINEAR
    elif _is_closed(phi,nelec=nelec,M=M):
        return _SlaterType.CLOSED
    elif _is_collinear(phi,nelec=nelec,M=M):
        return _SlaterType.COLLINEAR
    else:
        raise ValueError(
            f"Unable to determine a valid Slater determinant type: phi has shape "
            f"{phi.shape}, which is not consistent with nelec={tuple(nelec)} and "
            f"M={M}. Expected (M,nalpha), (M,nalpha+nbeta), (2,M,*) or "
            f"(2*M,nalpha+nbeta)."
        )
