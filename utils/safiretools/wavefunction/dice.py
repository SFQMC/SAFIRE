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
Selected-CI (SHCI) trial wavefunctions from Dice's output.

Dice writes its determinants either to an HDF5 file or to its text log, in both
cases as one occupation character per orbital: ``'2'`` doubly occupied, ``'a'``
or ``'b'`` singly occupied in that spin channel, ``'0'`` empty.

Reached through `PHMSDWavefunction.from_dice`, which detects the format.
"""

import ast
import logging
import re

import numpy as np
import h5py as h5

logger = logging.getLogger(__name__)

_OCCUPATION_CHARACTERS = ('0', 'a', 'b', '2')
"""Dice's per-orbital occupation characters."""

_STATE_HEADER = 'Printing most important determinants'
"""The line in Dice's text log after which the determinants are listed."""


def from_dice(path, ndets, state=0):
    """
    Read a selected-CI expansion from Dice's output.

    Parameters
    ----------
    path : str or pathlib.Path
        Dice output to read: an HDF5 file, or its text log.
    ndets : int
        Number of determinants to read, largest coefficient first.
    state : int, optional
        Index of the state to read. Default 0.

    Returns
    -------
    PHMSDWavefunction
        The expansion, as `SpinSymm.COLLINEAR` — with ``ndown == 0`` for a fully
        spin-polarized one.

    Raises
    ------
    ValueError
        If `ndets` is not a positive integer, `state` is out of range, the file
        holds fewer determinants than requested, or an occupation string is
        malformed.

    Notes
    -----
    afqmctools reported every one of these conditions with a bare
    ``assert(0)``, after printing an explanation that a caller could not catch.
    """
    from safiretools.wavefunction.phmsd import PHMSDWavefunction

    if not isinstance(ndets, (int, np.integer)) or ndets < 1:
        raise ValueError(f"ndets must be a positive integer, got {ndets!r}")

    if h5.is_hdf5(path):
        coeffs, occa, occb, nmo = _read_hdf5(path, ndets, state)
    else:
        coeffs, occa, occb, nmo = _read_ascii(path, ndets, state)

    nelec = (occa.shape[1], occb.shape[1])
    logger.info("read %d determinant(s) of state %d from %s: nelec=%s, nmo=%d",
                len(coeffs), state, path, nelec, nmo)

    return PHMSDWavefunction(coeffs=coeffs, occa=occa, occb=occb, nmo=nmo,
                             nelec=nelec)


def parse_coefficient(text):
    """
    Parse one Dice coefficient, which is either a plain float or a
    ``(real, imag)`` pair.

    Parameters
    ----------
    text : str
        The coefficient as Dice printed it.

    Returns
    -------
    complex

    Raises
    ------
    ValueError
        If `text` is neither.
    """
    try:
        return complex(text)
    except ValueError:
        pass

    try:
        real, imaginary = ast.literal_eval(text)
    except (SyntaxError, ValueError, TypeError):
        raise ValueError(
            f"could not parse '{text}' as a Dice CI coefficient: expected a "
            "number or a (real, imaginary) pair"
        ) from None

    return real + 1j * imaginary


def _occupations_from_string(characters, determinant: int):
    """
    Split one determinant's occupation characters into its alpha and beta
    occupied-orbital indices.

    Parameters
    ----------
    characters : sequence of str
        One character per orbital.
    determinant : int
        Index of this determinant, for error messages.

    Returns
    -------
    tuple(list, list)
        Alpha and beta occupied-orbital indices, ascending.
    """
    occa, occb = [], []

    for orbital, character in enumerate(characters):
        if character not in _OCCUPATION_CHARACTERS:
            raise ValueError(
                f"unknown occupation character '{character}' at orbital "
                f"{orbital} of determinant {determinant}: expected one of "
                f"{list(_OCCUPATION_CHARACTERS)}"
            )
        if character in ('2', 'a'):
            occa.append(orbital)
        if character in ('2', 'b'):
            occb.append(orbital)

    return occa, occb


def _stack_occupations(occupations):
    """
    Stack per-determinant ``(occa, occb)`` pairs into two ``(ndets, nelec)``
    arrays, keeping the determinant axis when a spin channel is empty.
    """
    ndets = len(occupations)
    nup = len(occupations[0][0])
    ndown = len(occupations[0][1])

    for index, (occa, occb) in enumerate(occupations):
        if (len(occa), len(occb)) != (nup, ndown):
            raise ValueError(
                f"determinant {index} has {len(occa)} alpha and {len(occb)} "
                f"beta electrons, but the leading determinant has {nup} and "
                f"{ndown}"
            )

    return (np.array([occa for occa, _ in occupations],
                     dtype=np.int64).reshape(ndets, nup),
            np.array([occb for _, occb in occupations],
                     dtype=np.int64).reshape(ndets, ndown))


# ----------------------------------------------------------------------
# Dice's HDF5 output
# ----------------------------------------------------------------------

def _read_hdf5(path, ndets: int, state: int):
    """Read `ndets` determinants of `state` from Dice's HDF5 output."""
    with h5.File(path, 'r') as fh5:
        nroots = int(fh5['/nroots'][0])
        available = int(fh5['/ndets'][0])
        nmo = int(fh5['/norbs'][0])

        if not 0 <= state < nroots:
            raise ValueError(
                f"'{path}' holds {nroots} state(s), so state {state} does not "
                "exist"
            )
        if ndets > available:
            raise ValueError(
                f"'{path}' holds {available} determinant(s), fewer than the "
                f"{ndets} requested"
            )

        coefficients = fh5[f'/coeff_r{state}'][:]
        configurations = fh5[f'/confg_r{state}'][:, :]

    if configurations.shape != (available, nmo):
        raise ValueError(
            f"'{path}' holds configurations of shape {configurations.shape}, "
            f"expected ({available}, {nmo})"
        )

    # Dice already prints in alpha-beta parity, so no reordering is needed
    order = np.argsort(np.abs(coefficients))[::-1][:ndets]

    coeffs = np.zeros(ndets, dtype=np.complex128)
    occupations = []
    for kept, index in enumerate(order):
        occupations.append(_occupations_from_string(
            [chr(character) for character in configurations[index, :]], index))
        coeffs[kept] = parse_coefficient(coefficients[index])

    return (coeffs, *_stack_occupations(occupations), nmo)


# ----------------------------------------------------------------------
# Dice's text output
# ----------------------------------------------------------------------

def has_real_coefficients(fields) -> bool:
    """
    Whether a split determinant line from Dice's text log carries a real
    coefficient.

    Dice prints one column for a real coefficient and two for a complex one, so
    the third field is either the first occupation character or the imaginary
    part::

        0    -0.5032009288     2 2 2 2 0   0 0 0 0 0   0     # real
        0    -0.5032009288     0.7487247473 2 2 2 2 0  ...   # complex

    Parameters
    ----------
    fields : list of str
        One determinant line, split on whitespace.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the third field is neither an occupation character nor a number.
    """
    if re.match(r'^[0ab2]$', fields[2]):
        return True
    if re.match(r'^[-+]?[0-9]+\.[0-9]+$', fields[2]):
        return False

    raise ValueError(
        f"could not parse '{fields[2]}' in a Dice determinant line: expected "
        "either an occupation character or the imaginary part of a coefficient"
    )


def _read_line(handle, path):
    """Read one line, raising rather than returning at end of file."""
    line = handle.readline()
    if line == '':
        raise ValueError(f"reached the end of '{path}' while parsing it")
    return line


def _seek_state(handle, path, state: int):
    """Advance `handle` to the first determinant line of `state`."""
    line = _read_line(handle, path)
    while line.find(_STATE_HEADER) < 0:
        line = _read_line(handle, path)

    fields = _read_line(handle, path).split()
    while (len(fields) != 3 or fields[0].find('State') < 0
           or int(fields[2]) != state):
        fields = _read_line(handle, path).split()


def _read_ascii(path, ndets: int, state: int):
    """Read `ndets` determinants of `state` from Dice's text log."""
    with open(path) as handle:
        _seek_state(handle, path, state)

        fields = _read_line(handle, path).split()
        if len(fields) < 3:
            raise ValueError(
                f"'{path}' holds no determinant for state {state}"
            )

        real_coefficients = has_real_coefficients(fields)
        first_occupation = 2 if real_coefficients else 3

        # afqmctools computed this as len(fields) - 2 before deciding whether
        #   the coefficient was complex, so it counted the imaginary column as
        #   an orbital and reported nmo one too large for complex output.
        nmo = len(fields) - first_occupation

        coeffs = np.zeros(ndets, dtype=np.complex128)
        occupations = []

        while len(fields) == nmo + first_occupation:
            index = len(occupations)

            if real_coefficients:
                coeffs[index] = parse_coefficient(fields[1])
            else:
                coeffs[index] = (parse_coefficient(fields[1])
                                 + 1j * parse_coefficient(fields[2]))

            occupations.append(
                _occupations_from_string(fields[first_occupation:], index))

            if len(occupations) == ndets:
                break
            fields = _read_line(handle, path).split()

    return (coeffs[:len(occupations)], *_stack_occupations(occupations), nmo)
