---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: ipython3
  name: python3
---

# Hello SAFIRE

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-intro -->
:end-before: <!-- END hello-intro -->
```

For this tutorial, we have supplied an HDF5 file containing the Hamiltonian
corresponding to the Hubbard model on a 4x4 lattice with periodic boundary conditions,
and $U/t = 4$ in `hamil.h5`.

![](files/Hubbard_4x4_tutorial.png)

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-wavefunction-and-outputs -->
:end-before: <!-- END hello-wavefunction-and-outputs -->
```

You will learn more about this in {doc}`../08_computing_observables/08_computing_observables`.

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-running -->
:end-before: <!-- END hello-running -->
```

```{code-cell} ipython3
!safire files/input.json --verbosity 1
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-parallel -->
:end-before: <!-- END hello-parallel -->
```

### Your Turn

Try running SAFIRE with MPI! To get the same numbers as us, you will need to use the same number of MPI processes as we did, 16.

```{code-cell} ipython3
!mpirun -np 16 safire files/input.json --verbosity 1
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-postprocessing -->
:end-before: <!-- END hello-postprocessing -->
```

```bash
====== [analyze_scalar_data Settings] ======

[+] fname            = qmc.s000.scalar.dat
[+] mark_header      = #               
[+] series_column    = time            
[+] nequil           = 2.0             
[+] estimate_equil   = False           
[+] column           = LocalEnergy     
[+] reblock          = 1               
[+] ndiscard         = None            
[+] list             = False           
[+] trace            = False           
[+] append           = None            
[+] dump             = False           
[+] dump_fname       = trace.dat       
[+] verbose          = True            
[+] autocorr         = None            
[+] savefig          = None            
[+] dump_avail_columns = False           

AFQMC Energy   -19.580848 +/-   0.000669 3.56  2.0/100.0
```

```{code-cell} ipython3
!scalar_stats qmc.s000.scalar.dat -s time -e 2.0
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-helpers -->
:end-before: <!-- END hello-helpers -->
```

```{code-cell} ipython3
from pathlib import Path
import shutil

import numpy as np

from tutorial_utils import run_afqmc
from stats.scalar_dat import analyze_scalar_data

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

# copy the provided files
files_dir = scratch_dir / "files"
files_dir.mkdir(exist_ok=True)
shutil.copy("files/input.json", files_dir)
shutil.copy("files/wfn.h5", files_dir)
shutil.copy("files/hamil.h5", files_dir)

run_afqmc(
    run_dir=scratch_dir,
    input_file="files/input.json",   # the location of the sample input file.
    np=16,                                         # to run in serial just as before!
)

# post process!
_ = analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",
    nequil = 5,
    trace = True
))
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-summary -->
:end-before: <!-- END hello-summary -->
```
