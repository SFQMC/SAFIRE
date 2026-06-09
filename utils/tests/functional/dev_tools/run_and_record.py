# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path
import subprocess as sp
import re
from time import perf_counter
from warnings import warn

import h5py as h5
import numpy as np

from stats.scalar_dat import analyze_scalar_data
from afqmctools.analysis.rdm import average_afqmc_rdm
from dev_tools.results import Results

AFQMC_EXEC = os.environ.get("AFQMC_EXEC",None)
if AFQMC_EXEC is None:
    raise RuntimeError(
        "AFQMC_EXEC environment variable is not set. "
        "Set it to the path to the AFQMC executable."
    )

# TODO: use write_json instead of an f-string
def make_afqmc_input(fname,hamil_file,wfn_file,walker_type):

    with open(fname,'w') as f:
        line = '''{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0,
      "mixed_precision": false
    },
    "execute": {
      "walker_set": {''' f'''
        "walker_type": "{walker_type}"''' + '''
      },
      "wavefunction": {''' + f'''
        "filename": "{wfn_file}" ''' + '''
      },
      "hamiltonian": {''' + f'''
        "filename": "{hamil_file}" ''' + '''
      },
      "timestep": 0.01,
      "steps": 10000,
      "n_walkers_per_mpi_task": 40,
      "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "nsteps": 400,
        "equil": 4000,
        "onerdm": {
          "name": "one_rdm"
        }
      },
      "population_control_interval": 2,
      "seed": 42
    }
  }
}
'''
        f.write(line)

def afqmc_warning_message(fname):
    if not afqmc_raised_warning(fname):
        return set()
    else:
        with open(fname,'r') as f:
            results = re.findall(r"(\[warning\]) (.+)",f.read())
        results = [ w[1].lstrip().rstrip() for w in results]
        return set(results)

def afqmc_error_message(fname):
    if not afqmc_raised_error(fname):
        return set()
    else:
        with open(fname,'r') as f:
            # filter unwanted characters
            text = re.sub(r'\x1b\[[0-9;]*m', '', f.read())
            results = re.findall(r"(\[error\]) (.+)",text)

        # get rid of duplicates
        results = set(results)

        # redundant filter out unwanted characters
        results = [
            re.sub(r'\x1b\[[0-9;]*m', '', str(w[1])).lstrip().rstrip() for w in results
        ]

        # remove formatting
        results = [
            w for w in results
            if not re.match(r"\*{10,}", w) and not re.match(r"APPLICATION ABORT: Fatal Error\.", w)
        ]
        return set(results)

def get_and_save_warnings(fname:str,fh5:h5.File):
    """
    will find the warning messages printed in 'fname' and save
    them to the h5py File instance fh5
    """
    warning_groups = afqmc_warning_message(fname)
    if warning_groups is not None:
        warning_h5_group = fh5.create_group("warning_messages")
        warning_h5_group.create_dataset("num_messages",data=len(warning_groups))
        for i,warning in enumerate(warning_groups):
            warning_h5_group.create_dataset(f"warning_{i}",data=str(warning))
    

def get_and_save_errors(fname:str,fh5:h5.File):
    """
    will find the error messages printed in 'fname' and save
    them to the h5py File instance fh5
    """
    error_groups = afqmc_error_message(fname)
    if error_groups is not None:
        error_h5_group = fh5.create_group("error_messages")
        error_h5_group.create_dataset("num_messages",data=len(error_groups))
        for i,warning in enumerate(error_groups):
            error_h5_group.create_dataset(f"error_{i}",data=str(warning))


def get_afqmc_results(return_code:int,run_time_secs=-1.0,nequil=5.0,run_bp=False):
    """
    Will extract the following information and save
      to `results.h5`

    0. did it run or not
    1. initial variational energy
    2. AFQMC energy with set equilbration length
    3. AFQMC avg. walker weight (set equil)
    4. AFQMC LogOvlpFActor (set equil)
    5. Averaged 1-rdm
    
    This function now uses the Results class internally for better modularity.
    """
    # Create Results object from AFQMC run output files
    results = Results.from_afqmc_run(
        return_code=return_code,
        run_time_secs=run_time_secs,
        nequil=nequil,
        run_bp=run_bp
    )
    
    # Write to HDF5 file
    results.to_hdf5("results.h5")


def afqmc_raised_error(fname):
    """
    returns true if an error is raised by the AFQMC code in the AFQMC format
    
    use a regex to find: '[error]' on one or more lines
    """
    # TODO: use regex groups to identify what error is raised
    with open(fname,'r') as f:
        result = re.search(r"\[error\]",f.read())
    if result is None:
        return False
    else:
        return True


def afqmc_raised_warning(fname):
    """
    returns true if a warning is raised by the AFQMC code in the AFQMC format
    
    use a regex to find: '[warning]' on one or more lines
    """
    # TODO: use regex groups to identify what warning is raised
    with open(fname,'r') as f:
        result = re.search(r"\[warning\]",f.read())
    if result is None:
        return False
    else:
        return True

def afqmc_out_is_finite(fname):
    """
    Check that the results in the afqmc ascii output are finite

    Here, finite means not "nan"

    Parameters
    ----------
    fname : str
        The name of the file containing afqmc ascii output.

    Returns
    -------
    bool
        True if the results are finite, False otherwise.
    """
    with open(fname,'r') as f:
        result = re.search(r"\([-]?nan,|[-]nan\)",f.read())
    if result is None:
        return True
    else:
        return False

class AFQMCHelper:
    """
    A class to run AFQMC and record the results in an HDF5 file.

    The AFQMCHelper's external interface is `AFQMCHelper.run_afqmc()`.
    See the documentation for `run_afmqc()` for more.
    """

    def __init__(self,afqmc_exec,num_mpi_tasks,timeout_mins=None) -> None:
        self.afqmc_exec = afqmc_exec
        if num_mpi_tasks is None:
            num_mpi_tasks = os.environ.get("AFQMC_NUM_MPI_TASKS", 1)
        self.num_mpi_tasks = int(num_mpi_tasks)

        if timeout_mins is not None:
            timeout_mins = float(timeout_mins)
        else:
            timeout_mins = 15
        self.timeout_mins = timeout_mins

    def __repr__(self):
        return (
            f"AFQMCHelper(afqmc_exec={self.afqmc_exec}, "
            f"num_mpi_tasks={self.num_mpi_tasks}, "
            f"timeout_mins={self.timeout_mins})"
        )

    # TODO: update to use the afqmctools to generate the input file
    def make_afqmc_input(
            self,
            fname,
            hamil_file,
            wfn_file,
            walker_type,
            num_mpi_tasks=None,
            n_walkers_per_mpi_task=None,
            run_bp=False,
            gpu=False,
            total_walkers=1600,
            steps=10000,
            timestep=0.01
        ):

        num_mpi_tasks = self.num_mpi_tasks if num_mpi_tasks is None else num_mpi_tasks

        if n_walkers_per_mpi_task is None and not gpu:
            n_walkers_per_mpi_task = total_walkers // num_mpi_tasks
        elif n_walkers_per_mpi_task is None and gpu:
            n_walkers_per_mpi_task = total_walkers

        with open(fname,'w') as f:
            input_text = '''{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0,
      "mixed_precision": false
    },
    "execute": {
      "walker_set": {''' f'''
        "walker_type": "{walker_type}"''' + '''
      },
      "wavefunction": {''' + f'''
        "filename": "{wfn_file}" ''' + '''
      },
      "hamiltonian": {''' + f'''
        "filename": "{hamil_file}" ''' + '''
      }, ''' + f'''
      "timestep": {timestep},
      "steps": {steps},
      "n_walkers_per_mpi_task": {n_walkers_per_mpi_task},'''

            if run_bp:
                input_text += '''
      "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "measure_interval_multiplier": 40,
        "equil_multiplier": 200,
        "onerdm": {
          "name": "one_rdm"
        }
      }, ''' 
            input_text += '''
      "population_control_interval": "10",
      "measure_interval_multiplier": "1",
      "walker_ortho_interval": "10",
      "seed": 42
    }
  }
}
'''
            f.write(input_text)

    def get_afqmc_results(self,return_code:int,run_time_secs=-1.0,run_bp=False):
        get_afqmc_results(
            return_code=return_code,
            run_time_secs=run_time_secs,
            run_bp=run_bp
        )

    def run_afqmc(self,run_path,timeout_mins=None,afqmc_runmode=None,run_bp=False,**kwargs):
        """
        Runs AFQMC and records the results in a 'results.h5' file.
    
        The main interface for interacting with the AFQMCHelper class.
    
        Inputs:
        - "run_path" pathlib.Path or str : the path where AFQMC will run and where `results.h5` will be written
        - "timeout_mins" int or float : the number of minutes before an AFQMC run times out and is halted.
        - "afqmc_runmode" : str : the mode to run AFQMC in. Options are (default) "slurm_cpu", "local_cpu".
    
        TODO:
        - add run modes for "batch_gpu" and "local_gpu"
        """
        
        if timeout_mins is None:
            timeout_mins = self.timeout_mins

        _runparams = kwargs.pop("runparams", {}) # do not forward  allrunparams to make_afqmc_input!
        num_mpi_tasks = _runparams.setdefault("num_mpi_tasks",self.num_mpi_tasks)
        max_num_mpi_ranks = _runparams.setdefault("max_num_mpi_ranks",None)
        total_walkers = _runparams.setdefault("total_walkers",1600)
        _runparams.pop("num_mpi_tasks")
        _runparams.pop("max_num_mpi_ranks")
        _runparams.pop("total_walkers")
        kwargs.update(_runparams)

        if max_num_mpi_ranks is not None and num_mpi_tasks > max_num_mpi_ranks:
            warn(
                f"Requested {num_mpi_tasks} MPI tasks, but the maximum allowed is {max_num_mpi_ranks}. "
                f"Setting num_mpi_tasks to {max_num_mpi_ranks}."
            )
            num_mpi_tasks = max_num_mpi_ranks

        current_dir = Path(os.getcwd())
        
        if afqmc_runmode is None:
            afqmc_runmode = "slurm_cpu"
    
        print("Running AFQMC in temporary path:", run_path)
        os.chdir(run_path)
        print("  AFMQC run mode is: ", afqmc_runmode )
    
        output_file = "afqmc.out"
        self.make_afqmc_input(
            num_mpi_tasks=num_mpi_tasks,
            run_bp=run_bp,
            gpu=(afqmc_runmode.endswith("gpu")),
            total_walkers=total_walkers,
            **kwargs
        )
    
        try:
            with open(output_file,"w") as f:
                ts = perf_counter()
                # TODO: rewrite with structural pattern matching
                if afqmc_runmode == "slurm_cpu":
                    run_string = ["srun","--cpu-bind=cores"]
                    if num_mpi_tasks != self.num_mpi_tasks:
                        run_string.extend(["-n",str(max_num_mpi_ranks)])
                    run_string.extend([self.afqmc_exec,"--filenames","afqmc.json"])
                elif afqmc_runmode == "local_cpu":
                    run_string = [self.afqmc_exec,"--filenames","afqmc.json"]
                    if timeout_mins is None:
                        warn("By defualt, AFQMC calculations will not timeout in local_cpu mode.")
                elif afqmc_runmode == "slurm_gpu":
                    run_string = ["srun","--cpu-bind=cores","--gpu-bind=none",self.afqmc_exec,"--filenames","afqmc.json"]
                elif afqmc_runmode == "local_gpu":
                    run_string = [self.afqmc_exec,"--filenames","afqmc.json"]
                elif afqmc_runmode == "mpi_cpu":
                    run_string = ["mpirun","-np",str(num_mpi_tasks),self.afqmc_exec,"--filenames","afqmc.json"]
                    if timeout_mins is not None and timeout_mins < 15: # NOTE: the 15 minute threshold is arbitrary, should benchmark timing!
                        warn(
                            "AFQMC calculations may be very slow in mpi_cpu mode (especially inside containers)."
                            " This may lead to artifical test failures due to timeout."
                            " if this happens, increase the timeout_mins value using the `--afqmc-timeout` command line" 
                            " option when invoking pytest"
                        )
                else:
                    raise ValueError("[for developers] Invalid AFQMC run mode")
    
                if timeout_mins is not None:
                    timeout_mins = timeout_mins*60
    
                if timeout_mins < 0:
                    timeout_mins = None
    
                afqmc_instance = sp.run(
                    run_string,
                    stdout=f,
                    stderr=f,
                    env=os.environ, # NOTE: this is necessary since PyTest modifies the environment
                    timeout=timeout_mins
                )
                tf = perf_counter()
            self.get_afqmc_results(afqmc_instance.returncode,run_time_secs=tf-ts,run_bp=run_bp)
            os.chdir(current_dir)
        except Exception as e:
            print(f"An exception was raised: ", e)
            os.chdir(current_dir)
