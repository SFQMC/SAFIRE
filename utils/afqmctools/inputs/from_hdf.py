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

JSON_EXECUTE_INPUT_BLOCKS = ("walker_set", "wavefunction", "hamiltonian", "propagator")

def get_estimator_settings(exec_opts,args=None):

    if args is None:
        return sum(k.startswith("estimator") for k in exec_opts.keys())

    if isinstance(args,dict):
        args = SimpleNamespace(args)

    iest = 1
    observables = ['onerdm']  # TODO: add more observables options

    if args.mixed_est:
        # add mixed estimator
        est = dict(
            name = "mixed",
        )
        for obs in observables:
            est[obs] = {"name" : obs}
        exec_opts['estimator%d' % iest] = est
        iest += 1

    if args.time_bp is not None:
        tbp = args.time_bp
        dt = args.timestep
        # determine back propagation steps
        nstep_bp_ideal = tbp/dt
        #nbase = args.steps*args.num_bp
        nbase = args.num_bp
        nmult = nstep_bp_ideal//nbase
        nstep_bp = int(round(nmult*nbase))
        tbp1 = nstep_bp*dt
        msg = f"tbp = {tbp1} (requested {tbp})"
        if args.verbose:
            print(msg)
        if abs((tbp1-tbp)/tbp) > 0.1:
            raise RuntimeError(msg)
        # add BP estimator
        est = dict(
            name = "back_propagation",
            nsteps = nstep_bp,
            naverages = args.num_bp,
        )
        for obs in observables:
            est[obs] = {"name" : obs}
        pr = args.path_restoration
        if pr == '0':
            pass  # no PR is the default
        elif pr == '1':
            est['path_restoration'] = True
        elif pr == 'e':
            est['path_restoration'] = True
            est['extra_path_restoration'] = True
        else:
            msg = 'path_restoration must be one of "0", "1", "e"'
            msg += ' not "%s"' % pr
            raise RuntimeError(msg)
        exec_opts['estimator%d' % iest] = est
        iest += 1

    return iest


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
        JSON file. This will be written into an "execute" node in the JSON file.
    args_namespace : argparse.Namespace, optional
        Command line arguments parsed by argparse.
    
    Examples
    --------
    >>> write_json("afqmc.json", "afqmc.h5")

    Notes
    -----
    For parameters that may be set in both args_namespace and kwargs, the values in
    args_namespace will take precedence. The parameters to which this applies are:
    - id
    - series

    For input blocks provided in exec_opts, each parameter is appended into the
    corresponding block in the JSON file. For example, if exec_opts contains
    ``{"execute": {"walker_set": {"min_weight": 0.01}}}``, the resulting JSON
    file will contain the following::
    
        {
            "afqmc": {
                "execute": {
                    "walker_set": {
                        "walker_type": "COLLINEAR",
                        "min_weight": 0.01
                    },
                    ...
                }
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
        inps["afqmc"]["execute"]["wavefunction"]["filename"] = fwfn
        if fham != fwfn:
            inps["afqmc"]["execute"]["hamiltonian"] = dict(
                filename = fham
            )
    num_est = get_estimator_settings(exec_opts,args=args_namespace)
    if exec_opts is not None:
         # we want to append the settings in each known input block to what (may) exist in inps
        input_block_generator = ( (key,exec_opts[key]) for key in JSON_EXECUTE_INPUT_BLOCKS if key in exec_opts )
        for key,input_block in input_block_generator:
            input_block_dict = inps["afqmc"]["execute"].setdefault(key, {})
            for subkey,val in input_block.items():
                input_block_dict[subkey] = val
            # remove the key from exec_opts so it doesn't get passed to .update(exec_opts)
            exec_opts.pop(key)

        # set all other options directly
        inps["afqmc"]["execute"].update(exec_opts)

    # get project settings
    if args_namespace is not None and getattr(args_namespace, 'series'):
        inps["afqmc"]["project"]["series"] = getattr(args_namespace, 'series')
    elif kwargs.get('series'):
        inps["afqmc"]["project"]["series"] = kwargs['series']

    if args_namespace is not None and getattr(args_namespace, 'id'):
        inps["afqmc"]["project"]["id"] = getattr(args_namespace, 'id')
    elif kwargs.get('id'):
        inps["afqmc"]["project"]["id"] = kwargs['id']
    
    with open(fout, 'w') as f:
        json.dump(inps, f, indent=2)

    # !!!! HACK to introduce more than 1 estimator node in JSON
    #   KE: in principle, one could use a custom class based on dict along
    #      with a custom json.JSONEncoder class to do this "correctly"
    if num_est > 1:
        with open(fout, 'r') as f:
            text = f.read()
        for i in range(1, num_est+1):
            text = text.replace('estimator%d' % i, 'estimator')
        with open(fout, 'w') as f:
            f.write(text)

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
    inps = dict(afqmc={
        "project": {
            "id": "qmc",
            "series": 0,
        },
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
    })
    use_wfn_ham = fham0 == fwfn0
    if not use_wfn_ham:
        inps["afqmc"]["execute"]["hamiltonian"] = {
          "name": "ham0",
          "filename": fham0,
        }
    return inps
