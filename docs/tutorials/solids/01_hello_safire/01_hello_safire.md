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

For this tutorial, we have supplied an HDF5 file containing the Born-Oppenheimer
Hamiltonian for Si in its primitive cell at equilibrium
in `hamiltonian.h5`.

![](files/Si_primitive.png)

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-wavefunction -->
:end-before: <!-- END hello-wavefunction -->
```

For this tutorial, we have supplied an HDF5 file containing the Kohn-Sham Slater determinant
of a Quantum ESPRESSO DFT calculation as the trial wavefunction.

$$
| \Psi_\mathrm{T} \rangle = | \Phi_\mathrm{DFT} \rangle
$$

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-jobfile-and-running -->
:end-before: <!-- END hello-jobfile-and-running -->
```

```{code-cell} ipython3
!echo Path to SAFIRE: $AFQMC_EXEC
!$AFQMC_EXEC --help
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-running -->
:end-before: <!-- END hello-running -->
```

```{code-cell} ipython3
:tags: [scroll-output]
:scrolled: true

!mpirun -n 4 $AFQMC_EXEC files/afqmc.json
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-run-output -->
:end-before: <!-- END hello-run-output -->
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-postprocessing -->
:end-before: <!-- END hello-postprocessing -->
```

```{code-cell} ipython3
from safiretools import Results

results = Results("afqmc.results.h5")

energy, energy_err = results.average("Energy")

print(f"E = {energy.real} ± {energy_err.real}")
```

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-timeseries -->
:end-before: <!-- END hello-timeseries -->
```

```{code-cell} ipython3
# manual
import matplotlib.pyplot as plt
plt.plot(results.timeseries("ProjectionTime"), results.timeseries("Energy").real)
plt.show()

# visualization helper
from safiretools import vis
vis.plot_timeseries(results, skip=0)
plt.show()
```

You should get an output compatible with `Energy = -0.7299 ± 0.0017`.

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-equilibration -->
:end-before: <!-- END hello-equilibration -->
```

You will learn how to measure observables other than the energy in {doc}`../04_computing_observables/04_computing_observables`.

+++

```{include} /tutorials/molecules/01_hello_safire/01_hello_safire.md
:start-after: <!-- BEGIN hello-summary -->
:end-before: <!-- END hello-summary -->
```
