# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from enum import Enum, IntEnum


class SpinSymm(IntEnum):
    """
    Spin symmetry of a Hamiltonian, Wavefunction, or Walker ordered by increasing
    generality. Values match WALKER_TYPES in src/AFQMC/config.h exactly.
    """

    CLOSED = 1
    COLLINEAR = 2
    NONCOLLINEAR = 3

    @classmethod
    def from_input(cls, value) -> "SpinSymm":
        """
        Coerce user/on-disk input to a `SpinSymm`.

        Accepts a `SpinSymm`, its integer value, or one of the recognized
        spelling aliases (``'closed'``/``'rhf'``, ``'collinear'``/``'uhf'``,
        ``'noncollinear'``/``'ghf'``, ...), case-insensitively. Another
        enumeration whose value is one of those (afqmctools' ``_SlaterType``,
        say) is unwrapped, so partly-migrated code can hand its own enum over.

        Raises
        ------
        ValueError
            If `value` names no known spin symmetry.
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, Enum) and not isinstance(value, int):
            value = value.value

        if isinstance(value, str):
            key = value.strip().lower()
            if key in _SPIN_SYMM_ALIASES:
                return _SPIN_SYMM_ALIASES[key]
            raise ValueError(
                f"Unknown spin symmetry '{value}': supported values are "
                f"{sorted(_SPIN_SYMM_ALIASES)}"
            )

        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown spin symmetry {value!r}: supported values are "
                f"{[member.value for member in cls]} or "
                f"{sorted(_SPIN_SYMM_ALIASES)}"
            ) from None

    @property
    def label(self) -> str:
        """
        The lowercase name used for this spin symmetry on disk and in AFQMC
        input files (``'closed'``, ``'collinear'``, ``'noncollinear'``).
        """
        return self.name.lower()


_SPIN_SYMM_ALIASES = {
    'closed': SpinSymm.CLOSED,
    # 'close' is not a typo here: afqmctools' writer wrote 'closed' while its
    #   reader looked up 'close', so closed-shell files could never be read
    #   back. We write 'closed' and accept both.
    'close': SpinSymm.CLOSED,
    'rhf': SpinSymm.CLOSED,
    'collinear': SpinSymm.COLLINEAR,
    'col': SpinSymm.COLLINEAR,
    'uhf': SpinSymm.COLLINEAR,
    'noncollinear': SpinSymm.NONCOLLINEAR,
    'nc': SpinSymm.NONCOLLINEAR,
    'ghf': SpinSymm.NONCOLLINEAR,
}
"""Recognized spelling aliases, mapped onto the canonical `SpinSymm` members."""


class HamiltonianFormat(str, Enum):
    """
    An on-disk Hamiltonian format, paired with the name the AFQMC executable
    knows it by.

    Members are strings, so a format compares, hashes and formats as its
    safiretools name (``'model'``, ``'dense'``, ...) anywhere one of those is
    expected. `tag` is the executable's own ``HamiltonianTypes`` spelling, which
    is what a Hamiltonian file records — see
    `safiretools.hamiltonian.base.write_hamiltonian_format`.
    """

    # Python 3.11 made a mixin Enum's str() its member name; this keeps both
    #   str() and f-strings rendering the value, as the pre-enum strings did.
    __str__ = str.__str__

    def __new__(cls, value, tag=''):
        """Build a member from its safiretools name and, if it has one, its tag."""
        member = str.__new__(cls, value)
        member._value_ = value
        member._tag = tag
        return member

    MODEL = 'model', 'ModelHamiltonian'
    DENSE = 'dense', 'RealDenseFactorized'
    KPOINT = 'kpoint', 'KPFactorized'
    THC = 'thc', 'THC'
    KPOINT_COQUI = 'kpoint_coqui'

    @property
    def tag(self) -> str:
        """
        The ``HamiltonianTypes`` name a file records this format as, or ``''``
        for a format that is never recorded: a CoQuí file has no ``Hamiltonian``
        group to record a type in, so `KPOINT_COQUI` has no tag.
        """
        return self._tag

    @classmethod
    def from_tag(cls, tag: str) -> "HamiltonianFormat":
        """
        The format a file recorded as `tag`.

        Raises
        ------
        ValueError
            If `tag` names no format safiretools recognizes.
        """
        for member in cls:
            if member.tag and member.tag == tag:
                return member

        raise ValueError(
            f"unknown Hamiltonian type '{tag}': safiretools recognizes "
            f"{sorted(member.tag for member in cls if member.tag)}"
        )
