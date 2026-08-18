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


def _is_collinear(phi,nelec):
    """
    Determine if the Slater matrix is collinear-like
    (i.e. UHF or ROHF-like) based on the shape of the
    Slater matrix and the number of electrons.
    """

    # UHF-like
    if len(phi.shape) == 3:
        return True
    # ROHF-like
    elif len(phi.shape) == 2 and nelec[0] != nelec[1]:
        return True
    else:
        return False

def _is_closed(phi,nelec):
    if len(phi.shape) == 2 and nelec[0] == nelec[1]:
        return True
    else:
        return False

def _is_noncollinear(phi,M):
    if len(phi.shape) == 2 and phi.shape[0] == 2*M:
        return True
    else:
        return False

def _get_slater_type(phi,nelec,M):
    """
    Determine the Slater determinant type based
      on the given Slater matrix dimensions, and 
      the number of electrons (nelec) if provided.

    Parameters
    ----------
    phi : np.ndarray
        the Slater matrix of the wavefunction with shape (Nmo,Nelec)
    nelec : tuple
        the number of electrons in the system.
    M : int
        the number of spatial orbitals in the system.
        
    Notes
    -----
    A collinear determinant can be unambigiously determined
      by the shape of phi since it is the only Slater matrix with 3 dimensions.

    For Closed / Noncollinear determinants, we can't distinguish between the two without
       using nelec.

    There is no way to distinguish between ROHF-like Collinear and Noncollinear without
       also using M (i.e. the number of spatial orbitals!)
    
    """

    # NOTE: the order of checks is important!
    if _is_noncollinear(phi,M=M):
        return _SlaterType.NONCOLLINEAR
    elif _is_closed(phi,nelec=nelec):
        return _SlaterType.CLOSED
    elif _is_collinear(phi,nelec=nelec):
        return _SlaterType.COLLINEAR
    else:
        raise ValueError("Unable to determine a valid Slater determinant type.")
