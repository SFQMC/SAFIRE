# use only Python built-in modules
from pathlib import Path
import os
from time import perf_counter
import subprocess as sp
from warnings import warn

AFQMC_EXEC = os.environ.get("AFQMC_EXEC",None)  # provided by the modulefile
AFQMC_SCRATCH_DIR_ROOT = os.environ.get("AFQMC_SCRATCH_DIR",Path("./"))

if AFQMC_EXEC is None:
    warn(
        "AFQMC_EXEC environment variable is not set. "
        "Set it to the path to the AFQMC executable."
    )

if not Path(AFQMC_EXEC).exists():
    warn(
        f"AFQMC_EXEC is set to {AFQMC_EXEC} which does not exist. "
        "Set it to the path to the AFQMC executable."
    )

def get_scratch_dir(run_name:str,root_dir:Path=AFQMC_SCRATCH_DIR_ROOT,create:bool=True):
    """
    Get the scratch directory for a given run name. The scratch directory is
    root_dir/run_name.

    Parameters
    ----------
    run_name: str
        The name of the run.
    root_dir: Path
        The root directory for the scratch directory. Defaults to the value of
        the environment variable AFQMC_SCRATCH_DIR.
    create: bool
        Whether to create the scratch directory if it does not exist. Defaults
        to True.
    """
    scratch_dir = Path(root_dir) / run_name
    scratch_dir.mkdir(parents=True, exist_ok=create)
    if not os.access(scratch_dir, os.W_OK):
        warn(
            f"You don't have write to scratch directory {scratch_dir} "
            "Please use a different scratch directory."
        )
    return scratch_dir

def run_afqmc(run_dir:Path=None,timeout_mins=None,run_mode="local_cpu",output_file="afqmc.out",**kwargs):
    """
    Run AFQMC in a given directory. The directory is changed to run_dir before
    running AFQMC and is changed back to the original directory after running
    AFQMC.

    Parameters
    ----------
    run_dir: Path
        The directory in which to run AFQMC. Defaults to the current directory.
    timeout_mins: int
        The maximum amount of time to allow AFQMC to run in minutes. Default is no timeout.
    run_mode: str
        The mode in which to run AFQMC. Options are "slurm_cpu" and "local_cpu".
        Defaults to "local_cpu".
    output_file: str
        The name of the output file. Defaults to "afqmc.out".
    """

    current_dir = Path(os.getcwd())
    if run_dir is None:
        run_dir = current_dir
    os.chdir(run_dir)

    input_file = kwargs.get("input_file","afqmc.json")
    
    if timeout_mins is not None:
        timeout_mins = timeout_mins*60
    
    if run_mode == "slurm_cpu":
        sp_list = ["srun","--cpu-bind=cores",AFQMC_EXEC,"--filenames",input_file]
    elif run_mode == 'local_cpu':
        np = str(kwargs.get("np",16))
        print(f"Running AFQMC locally with {np} mpi tasks")
        sp_list = ["mpirun","-np",np,AFQMC_EXEC,"--filenames",input_file]
    else:
        
        raise ValueError(f"run_mode {run_mode} not recognized")
    try:
        if output_file is not None:
            with open(output_file,"w") as f:
                ts = perf_counter()
                sp.run(
                    sp_list,
                    stdout=f,
                    stderr=f,
                    timeout=timeout_mins
                )
                tf = perf_counter()
        else:
            ts = perf_counter()
            sp.run(
                sp_list,
                stderr=sp.STDOUT,
                timeout=timeout_mins
            )
            tf = perf_counter()
        print(f"AFQMC ran in {tf-ts:.6f} seconds")
        os.chdir(current_dir)
        return tf-ts
    except Exception as e:
        print(f"An exception was raised: ", e)
        os.chdir(current_dir)
        return None
