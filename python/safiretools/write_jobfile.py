# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Writing the JSON input file the AFQMC executable is run on, together with the
HDF5 files holding the Hamiltonians and wavefunctions it refers to."""

import json
from pathlib import Path

from safiretools.hamiltonian.base import Hamiltonian
from safiretools.wavefunction.base import Wavefunction

_OBJECTS = (Hamiltonian, Wavefunction)


def write_jobfile(filename, parameters) -> None:
    """
    Write the SAFIRE input file `filename` from `parameters`.

    Parameters
    ----------
    filename : str or pathlib.Path
        JSON file to write. Its directory must exist.
    parameters : dict
        The input parameters, in the format of the SAFIRE afqmc.json input file,
        except that a `Hamiltonian` or `Wavefunction` object may stand in for the
        ``filename`` of a ``hamiltonian`` or ``wavefunction`` block, inline or
        in the ``hamiltonians`` / ``wavefunctions`` list, or for the whole
        block, which is then ``{'filename': obj}``. Every such object is
        written to ``<filename without suffix>_<name>.h5`` next to the input
        file, and its ``filename`` becomes the name of that file. ``<name>`` is
        the ``"name"`` of the block, or else ``hamiltonian<i>`` /
        ``wavefunction<i>``, counting the unnamed objects of that kind in order
        of appearance. An object in several unnamed blocks is written once.

    Raises
    ------
    ValueError
        If two blocks of the same name hold different Hamiltonians, or
        different wavefunctions.

    Examples
    --------
    >>> write_jobfile(run_dir / 'afqmc.json', {
    ...     'execute': {
    ...         'wavefunction': {'name': 'uhf', 'filename': uhf, 'ndets_to_read': 10},
    ...         'walker_set': {'from': {'wavefunction': rohf}},
    ...         'hamiltonian': ham,
    ...         'timestep': 0.01,
    ...         'steps': 10000,
    ...         'num_walkers': 200,
    ...     },
    ... })

    writes ``afqmc_uhf.h5``, ``afqmc_wavefunction0.h5`` and
    ``afqmc_hamiltonian0.h5``.
    """
    filename = Path(filename)
    with open(filename, 'w') as f:
        json.dump(_resolve_inputs(parameters, filename), f, indent=2)


def _kind(obj) -> str:
    return 'hamiltonian' if isinstance(obj, Hamiltonian) else 'wavefunction'


def _resolve_inputs(parameters: dict, jobfile: Path) -> dict:
    """`parameters` with every Hamiltonian and wavefunction in it written to its
    own HDF5 file next to `jobfile` and replaced by that file's name. The
    original is left untouched."""
    names = {'hamiltonian': {}, 'wavefunction': {}}
    # keyed by id: every object stays referenced by `parameters` for the whole
    #   call, so no id can be reused by another object meanwhile
    chosen = {}
    written = set()

    # every name the input gives has to be known before the first one is
    #   generated, since it may appear after the unnamed block that would
    #   otherwise be given it
    def claim(node):
        if isinstance(node, dict):
            obj = node.get('filename')
            if isinstance(obj, _OBJECTS) and 'name' in node:
                kind, name = _kind(obj), node['name']
                if names[kind].setdefault(name, obj) is not obj:
                    raise ValueError(f"two different {kind}s are both named '{name}'")
                chosen.setdefault(id(obj), name)
            for value in node.values():
                claim(value)
        elif isinstance(node, list):
            for item in node:
                claim(item)

    def write(obj, name):
        kind = _kind(obj)
        if name is None:
            name = chosen.get(id(obj))
        if name is None:
            index = 0
            while f'{kind}{index}' in names[kind]:
                index += 1
            name = f'{kind}{index}'
            names[kind][name] = obj
            chosen[id(obj)] = name

        path = jobfile.with_name(f'{jobfile.stem}_{name}.h5')
        if (kind, name) not in written:
            obj.to_hdf5(path)
            written.add((kind, name))
        return path.name

    def resolve(node):
        if isinstance(node, _OBJECTS):
            return {'filename': write(node, None)}
        if isinstance(node, dict):
            name = node.get('name')
            return {key: write(value, name)
                    if key == 'filename' and isinstance(value, _OBJECTS) else resolve(value)
                    for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    claim(parameters)
    return resolve(parameters)
