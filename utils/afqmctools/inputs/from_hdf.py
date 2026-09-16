# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import json
import os
from types import SimpleNamespace

import h5py as h5

JSON_EXECUTE_INPUT_BLOCKS = ("walker_set", "wavefunction", "hamiltonian", "propagator", "estimators")

# parameters of the run as a whole rather than of one execute block. They are accepted in
# exec_opts and hoisted to the top level, because that is where callers used to pass them.
JSON_TOP_LEVEL_KEYS = ("seed",)

# the default of the execute block's population_control_interval, in steps
DEFAULT_POPULATION_CONTROL_INTERVAL = 10


def bp_measure_interval_multipliers(tbp, timestep, population_control_interval, num_bp,
                                    verbose=False):
    """The back propagation lengths that reach `tbp`, in units of the population control
    interval: `num_bp` evenly spaced averages, the longest of them the requested time.

    The spacing is what gets rounded, so that every average is a whole multiple of the
    shortest one, as it was when the input took a step count and a number of averages.
    """
    spacing = max(1, round(tbp / (timestep * population_control_interval * num_bp)))
    multipliers = [spacing * k for k in range(1, num_bp + 1)]

    realized = multipliers[-1] * population_control_interval * timestep
    msg = f"tbp = {realized} (requested {tbp})"
    if verbose:
        print(msg)
    if abs((realized - tbp) / tbp) > 0.1:
        raise RuntimeError(msg)
    return multipliers


def get_estimator_settings(args, population_control_interval=DEFAULT_POPULATION_CONTROL_INTERVAL):
    """The "estimators" block the given arguments ask for, empty if they ask for none.

    The energy estimator is not in it: it is measured unless the input removes it with
    "energy": null.
    """
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    # an observable is measured if and only if its block is present, and one that takes no
    # parameters is requested by an empty block
    observables = {'onerdm': {}}  # TODO: add more observables options

    estimators = {}

    if args.mixed_est:
        estimators['mixed'] = dict(observables)

    if args.time_bp is not None:
        est = dict(observables)
        est['measure_interval_multiplier'] = bp_measure_interval_multipliers(
            args.time_bp, args.timestep, population_control_interval, args.num_bp, args.verbose)
        pr = args.path_restoration
        if pr == '0':
            est['path_restoration'] = False
        elif pr == '1':
            est['path_restoration'] = True
        elif pr == 'e':
            est['path_restoration'] = True
            est['extra_path_restoration'] = True
        else:
            msg = 'path_restoration must be one of "0", "1", "e"'
            msg += ' not "%s"' % pr
            raise RuntimeError(msg)
        estimators['backprop'] = est

    return estimators


def write_json(fout, fwfn0, fham0=None, relpath=True, exec_opts=dict(), args_namespace=None, **kwargs):
    r"""
    Write JSON input file for AuxiliaryFiles.

    Parameters
    ----------
    fout : str
        Output JSON file name.
    fwfn0 : str
        Name of HDF5 file containing the trial wavefunction.
    fham0 : str, optional
        Name of HDF5 file containing the Hamiltonian.
    relpath : bool, optional
        If True, use relative path for the wavefunction and Hamiltonian files.
    exec_opts : dict, optional
        Dictionary containing execution options using the same keys as in the
        JSON file. This will be written into an "execute" node in the JSON file,
        except for the parameters of the run as a whole listed in
        JSON_TOP_LEVEL_KEYS, which are written next to it.
    args_namespace : argparse.Namespace, optional
        Command line arguments parsed by argparse.
    
    Examples
    --------
    >>> write_json("afqmc.json", "afqmc.h5")

    Notes
    -----
    For parameters that may be set in both args_namespace and kwargs, the values in
    args_namespace will take precedence. The only parameter to which this applies is
    output_name, which names the results file. When it is left out, AFQMC names the
    results after the input file itself.

    For input blocks provided in exec_opts, each parameter is appended into the
    corresponding block in the JSON file. For example, if exec_opts contains
    ``{"execute": {"walker_set": {"min_weight": 0.01}}}``, the resulting JSON
    file will contain the following::

        {
            "driver": "afqmc",
            "execute": {
                "walker_set": {
                    "walker_type": "COLLINEAR",
                    "min_weight": 0.01
                },
                ...
            }
        }

    since COLLINEAR is the default walker type.
    """
    if fham0 is None:
        fham0 = fwfn0
    inps = default_inputs(fwfn0, fham0=fham0)
    if relpath:  # use relative file path
        path = os.path.dirname(fout)
        fwfn = os.path.relpath(fwfn0, path)
        fham = os.path.relpath(fham0, path)
        inps["execute"]["wavefunction"]["filename"] = fwfn
        if fham != fwfn:
            inps["execute"]["hamiltonian"] = dict(
                filename = fham
            )
    if args_namespace is not None:
        # a block exec_opts names itself is more specific than one the arguments imply
        estimators = exec_opts.setdefault("estimators", {})
        for name, block in get_estimator_settings(args_namespace, exec_opts.get(
                "population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL)).items():
            estimators.setdefault(name, block)
        if not estimators:
            exec_opts.pop("estimators")

    if exec_opts is not None:
         # we want to append the settings in each known input block to what (may) exist in inps
        input_block_generator = ( (key,exec_opts[key]) for key in JSON_EXECUTE_INPUT_BLOCKS if key in exec_opts )
        for key,input_block in input_block_generator:
            input_block_dict = inps["execute"].setdefault(key, {})
            for subkey,val in input_block.items():
                input_block_dict[subkey] = val
            # remove the key from exec_opts so it doesn't get passed to .update(exec_opts)
            exec_opts.pop(key)

        # a parameter of the whole run goes next to "execute", not inside it
        for key in JSON_TOP_LEVEL_KEYS:
            if key in exec_opts:
                inps[key] = exec_opts.pop(key)

        # set all other options directly
        inps["execute"].update(exec_opts)

    # the output name is optional: not every caller's namespace carries it, and AFQMC falls
    # back to the name of the input file when it is absent
    if args_namespace is not None and getattr(args_namespace, 'output_name', None):
        inps["output_name"] = getattr(args_namespace, 'output_name')
    elif kwargs.get('output_name'):
        inps["output_name"] = kwargs['output_name']


    with open(fout, 'w') as f:
        json.dump(inps, f, indent=2)

def read_info(fwfn):
    walker_types = ['NONE', 'CLOSED', 'COLLINEAR', 'NONCOLLINEAR']
    info = dict()
    with h5.File(fwfn, 'r') as f:
        has_wfn = False
        has_ham = False
        for key in f.keys():
            if key == 'Wavefunction':
                has_wfn = True
            elif key == 'Hamiltonian':
                has_ham = True
        if not has_wfn:
          msg = 'no Wavefunction in %s' % fwfn
          raise RuntimeError(msg)
        info['has_ham'] = has_ham
        wfkeys = list(f['Wavefunction'].keys())
        if len(wfkeys) != 1:
            msg = f'cannot handle wf {wfkeys}'
            raise RuntimeError(msg)
        wf = wfkeys[0]
        dims = f['Wavefunction/%s/dims' % wf][()]
        info['NMO'] = int(dims[0])
        info['NAEA'] = int(dims[1])
        info['NAEB'] = int(dims[2])
        info['walker_type'] = walker_types[dims[3]]
    return info

def default_inputs(fwfn0, fham0=None):
    info = read_info(fwfn0)
    wfn_has_ham = info.pop('has_ham')
    if fham0 is None:
        if wfn_has_ham:
            fham0 = fwfn0
        else:
            msg = 'must provide initial walker --fham_exe'
            raise RuntimeError(msg)
    inps = {
        "driver": "afqmc",
        "execute": {
            "walker_set": {
                'walker_type': info['walker_type']
            },
            "wavefunction": {
                "filename": fwfn0,
            },
            "timestep": 0.01,
            "steps": 10000,
            "n_walkers_per_mpi_task": 10,
        }
    }
    use_wfn_ham = fham0 == fwfn0
    if not use_wfn_ham:
        inps["execute"]["hamiltonian"] = {
          "name": "ham0",
          "filename": fham0,
        }
    return inps
