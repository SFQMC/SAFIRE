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

+++ {"id": "NudwCmJzZWxz"}

# Writing a Hamiltonian file and a Trial Wavefunction using CoQuí

<b>Goal:</b>
Become acquainted with how to write a Hamiltonian to the SAFIRE HDF5 format.

## What you will learn

1.  How to write a Hamiltonian to the SAFIRE HDF5 format, using CoQui, starting from a Quantum Espresso calculation
2.  How to write a Trial wavefunction to the SAFIRE HDF5 format, using CoQui, starting from a Quantum Espresso calculation

## External Software

In addition to SAFIRE and afqmctools, we will be using the following software packages:

- [Quantum Espresso (QE)](https://www.quantum-espresso.org/) for density functional theory (DFT) calculations, and some integrals via its post-processing utilities
- [CoQuí](https://github.com/AbInitioQHub/coqui) for generating the SAFIRE Hamiltonian and Trial wavefunction HDF5 files from the output of QE.
- [pw2coqui.x](https://github.com/AbInitioQHub/coqui/tree/main/qe_converter) see the instructions there for adding this to your QE build.

<div class="alert alert-block alert-info">
  <b>Note:</b> This tutorial assumes familiarity with Quantum Espresso (QE); we will point out some
    specific settings that are necessary for the workflow, but a general tutorial on QE
    is beyond the scope of this tutorial.
</div>

## Tutorial Files

We provide sample input files for each step of this tutorial
within the directory of this tutorial.
A list is files is as follows.

```bash
$ tree -a
.
├── 03_Si_writing_H_and_wfn_coqui.ipynb
└── files
    ├── coqui
    │   ├── hamil.toml
    │   └── wfn.toml
    └── qe
        ├── nscf.in
        ├── pw2coqui.in
        ├── scf.in
        └── Si_ONCV_PBE-1.2.upf
```

## Introduction

In previous tutorials, we learned how to run SAFIRE, and how
to post process the results to arrive at the AFQMC energy.
Now, we will learn how to write a Hamiltonian and a trial wavefunction in SAFIRE's HDF5 format.
CoQuí (Correlated Quantum ínterface), is a software project designed for ab initio electronic structure beyond density functional theory (DFT).
It implements several many-body perturbation theory (MBPT) approaches as well as performing supporting operations for other methods such as dynamical mean-field theory (DMFT) as well as AFQMC.
For AFQMC, and SAFIRE specifically, CoQuí can be used to directly generate
a second-quantized Hamiltonian from an electronic structure calculation.

<!--
<div>
<img src="./figs/QChemWorkflow_v2.png" width="1000"/>
</div>
-->

![](files/TopLevelFlowChart_v3.png)

In this tutorial,
we will compute the ground state energy of
the minimal basis hydrogen dimer with a bondlength of 1.4 Bohr radii. 

<!--
<div>
<img src="./figs/HyrdogenMolecule.png" width="500"/>
</div>
--->

![](files/Si_primitive.png)

In this tutorial, we will generate the Hamiltonian and trial wavefunction for
Si in the primitive cell.

+++ {"id": "TGWpFn73aPAi"}

## Setup

First, we need an orthonormal basis to express the second quantized Hamiltonian in.
We will run Quantum Espresso (QE) to generate a set of Kohn-Sham orbitals to use as a basis.
We assume knowledge of the basic use of QE in this example, but we'll point out a few details that are relevant to the workflow here.
All input files are provided in `files/qe`.

<div class="alert alert-block alert-warning">
 <b>⚠️ Imporant: </b> you will need to install the pw2coqui.x converter in your QE build. See the above!
</div>

We will perform the following calculations with QE:

1. self-consistent DFT with a large k-point grid, but a minimal number of bands
2. a one-shot DFT calculation starting from the converged result in step 1., but at the $\Gamma$-point and using a large number of bands.
3. run the pw2coqui.x post-processing utility to generate data that we will need later.

<div class="alert alert-block alert-info">
  <b>Note:</b> CoQuí expects to find `[prefix].coqui.h5` in the QE output directory. Double check that the output path of the post-processing tools are all set to the same directory.
</div>

### Step 1 : self-consistent DFT

To generate the Hamiltonian for SAFIRE, we need
to set `force_symmorphic=.true.` in the `&system` input card.
Run self-consistent DFT with the provided input file and pseudopotential.

From the `files/qe` directory, or a working directory of your choice where you have copied the inputs to,

```bash
$ pw.x -inp scf.in > scf.out
```

where we have redirected output to scf.out.
You should see a final energy of,

```
highest occupied, lowest unoccupied level (ev):     6.2533    6.8167

!    total energy              =     -15.75925947 Ry
     estimated scf accuracy    <          3.5E-11 Ry

     The total energy is the sum of the following terms:
     one-electron contribution =       4.74553084 Ry
     hartree contribution      =       1.11063812 Ry
     xc contribution           =      -4.81955233 Ry
     ewald contribution        =     -16.79587610 Ry

```

### Step 2 : One-shot DFT

The goal here is generate the orbitals we need to generate the second
quantized Hamiltonian for AFQMC.
**In general, AFQMC needs to be converged in the number of bands included in the basis**.
Coquí will allow us to select a subset of bands.
One possible strategy at this point is to include a relatively large number of bands here and we use Coquí to write a subset of them.
This avoids re-running the one-shot DFT calculation at the cost of saving more data.

**Run the one-shot QE calculation.**

```bash
$ pw.x -inp nscf.in > nscf.out
```

You should see something like the following,

```bash
     Band Structure Calculation
     Davidson diagonalization with overlap

     Computing kpt #:     1  of     1
     total cpu time spent up to now is        0.2 secs

     ethr =  1.25E-12,  avg # of iterations = 40.0

     total cpu time spent up to now is        0.2 secs

     End of band structure calculation

          k = 0.0000 0.0000 0.0000 (   537 PWs)   bands (ev):

    -5.7232   6.2533   6.2533   6.2533   8.8090   8.8090   8.8090   9.5704
    14.0110  14.0110  14.1787  17.4871  17.4871  17.4871  21.6280  29.4183
    29.4183  29.4183  30.5351  30.5351

     occupation numbers
     1.0000   1.0000   1.0000   1.0000   0.0000   0.0000   0.0000   0.0000
     0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
     0.0000   0.0000   0.0000   0.0000

     highest occupied, lowest unoccupied level (ev):     6.2533    8.8090

     Writing all to output data dir OUT/pwscf.save/ :
```

**Note the location of the output data**; we will use the orbitals saved there when we compute the real-space charge density.

+++ {"id": "_uliHCdtig7m"}

### Step 3: QE post-processing utilities

CoQuí reads some of the details of the pseudopotential from QE.
We need to dump the relevant data to a file using the pw2coqui.x QE post-processing utility.
As described above, you will need to install this utility into
your QE build using the instructions found [here](https://github.com/AbInitioQHub/coqui/tree/main/qe_converter).
See the provided input files for more.

pw2coqui.inp

```
&input_pw2coqui
  prefix = "pwscf"
  outdir = "OUT"
/
```

Run the following.

```bash
$ pw2coqui.x < pw2coqui.in > pw2coqui.out
```

<div class="alert alert-block alert-info">
  <b>Note:</b> Unlike many of the QE post-processing tools,
   the pw2coqui.x converter does not support MPI.
   Be sure to run the convert in serial as shown above!
</div>

If everything worked, you should see the following files in the "OUT" directory
that should have been generated by QE.

```bash
$ ls OUT
pwscf.coqui.h5 pwscf.save
```

**We're ready to move on to CoQuí.**

+++ {"jp-MarkdownHeadingCollapsed": true, "id": "H2qrKFX29jMg"}

### Your Turn

If you have not already, run Quantum Espresso with the sample inputs provided in the `files/qe` directory
within this tutorial's directory before moving on.

+++ {"id": "qELxzaWcjjlh"}

## Writing the second-quantized Hamiltonian and Trial wavefunction

CoQuí is able to directly generate a Hamiltonian HDF5 file for SAFIRE using the data output by QE.
Additionally, it can write a trial wavefunction based on the DFT solution from QE.
These steps can be performed either separately or in a single input file.
We will use the former approach and generate each separately for pedagogical reasons.
In general, it is recommended to to use a single CoQuí input file.

### Writing the Hamiltonian

CoQuí is controlled via a TOML-based input file.
**A sample CoQuí input file demonstrating how to generate a Hamiltonian is provided**.

```toml
[mean_field.qe]
name = "mf"
prefix = "pwscf"
outdir = "../qe/OUT"
nbnd = 20

[interaction.cholesky]
name        = "eri"
mean_field  = "mf"
output      = "hamil.h5"
write_type  = "single"
tol         = 1e-4  # you may need to converge this value!
ecut        = 50

[hamiltonian]
mean_field  = "mf"
interaction = "eri"
output      = "hamil.h5"
```

Here, we are using the Cholesky form of interaction (see the [User manual for details on the K-point factorized Cholesky Hamiltonian](https://users.flatironinstitute.org/~beskridge/safire/user_manual/hamiltonians.html#k-point-factorized-kp).

The key details to note are:

1. In `[mean_field.qe]`, we need to set the `outdir` to the directory where both the QE `pwscf.xml` file is saved, and where `[prefix].coqui.h5` is.
2. Also in `[mean_field.qe]`, we can set the number of bands to use with `nbnd`. Of course, we can only use as many bands as we output in the one-shot DFT calculation previously.
3. In the `interaction` input block, we are using the "cholesky" decomposed form for the interaction and have set a tolerance of `tol = 1e-4`. We have set the `output` to a file called "hamil.h5` to tell CoQuí to save the interaction there. **In general, one must converge the AFQMC energy in this parameter!**
4. The `hamiltonian` block is used to write the one-body part of the Hamiltonian to HDF5. **We need to set this to the same file as the interaction.**

.. warning

Now, run Coquí.

```bash
$ /path/to/coqui --verbosity=2 --filenames hamil.toml &> hamil.out
```

You should see the following output in `hamil.out`.

```
 ---------------------------------
     ____ ___   ___  _   _ ___
    / ___/ _ \ / _ \| | | |_ _|
   | |  | | | | | | | | | || |
   | |__| |_| | |_| | |_| || |
    \____\___/ \__\_\\___/|___|
  --------------------------------
 |  Correlated Quantum Interface  |
  --------------------------------

Input Parameters
----------------

[mean_field.qe]
name = 'mf'
nbnd = 20
outdir = '../qe/OUT'
prefix = 'pwscf'

[interaction.cholesky]
ecut = 50
mean_field = 'mf'
name = 'eri'
output = 'hamil.h5'
tol = 0.0001
write_type = 'single'

[hamiltonian]
interaction = 'eri'
mean_field = 'mf'
output = 'hamil.h5'

-- End of Input Parameters --

  Brillouin zone symmetry info
  ----------------------------
  Q-points in the irreducible zone = 1
  Symmetries applied to Q-points   = 1
  Time-reversal k-point pairs      = 0

  Quantum ESPRESSO reader
  -----------------------
  Number of spins                = 1
  Number of polarizations        = 1
  Number of bands                = 20
  Monkhorst-Pack mesh            = (1,1,1)
  K-points                       = 1 total, 1 in the IBZ
  Number of electrons            = 8.0
  Electron density energy cutoff = 100.000 a.u. | FFT mesh = (24,24,24)
  Wavefunction energy cutoff     = 11.994 a.u. | FFT mesh = (11,11,11), Number of PWs = 537

  Electron-electron interaction kernel
  ------------------------------------
  type          = coulomb
  ndim          = 3
  cutoff        = 1e-08
  screen_type   = none

*******************************
 ERI::cholesky:
*******************************
  -pw cutoff (Ha): 50.0
  -size of PW basis: 4573
  -cholesky truncation: 0.0001
  -number of k-point pools: 1
  -number of processors per pools: 1
  -default block size: 32


  iq:0  nchol:135
Writing distributed Vq at iq = 0 to .//hamil.h5
*******************************
 Cholesky ERI Reader:
*******************************
    - Np max  = 135
    - accuracy = 0.0001
    - read mode = each_q
    - eri storage: outcore
    - ERI dir = ./
    - ERI output = hamil.h5

*******************************
 Second-quantized 1-Body Hamiltonian  
*******************************
output: hamil.h5
format: qmc
type: bare
add_wfn:
************************************************
 Initializing External Potential:
************************************************
  input type: coqui::h5
  type: NCPP
  # of species: 1
  # of atoms: 2
  max # of projectors per atom: 8
  # of projectors: 16
  # of polarizations: 1
  spin orbit: false

 Memory usage:
   Overlaps:                      4.76837158203125e-06 MB
   Dion:                          9.5367431640625e-07 MB
************************************************

  Electron-electron interaction kernel
  ------------------------------------
  type          = coulomb
  ndim          = 3
  cutoff        = 1e-08
  screen_type   = none
```

The HDF5 file, "hamil.h5", that CoQuí just generated can be
directly read in SAFIRE to get the Hamiltonian.

+++ {"id": "233lFNEV9jMg"}

### More Information

For larger supercells, it is highly recommended that you use the [tensor hypercontraction (THC) form of the interaction](https://users.flatironinstitute.org/~beskridge/safire/user_manual/hamiltonians.html#tensor-hyper-contraction-thc) instead.
**CoQuí can also generate these integrals** by using an input block such as the following.

```toml
# Constructing THC-ERI by specifying the accuracy of the pivoted Cholesky
[interaction.thc]
name            = "eri"
mean_field      = "mf"
thresh          = 1e-6            
ecut            = 60
save            = "hamil.h5"
storage         = "incore"
```

<div class="alert alert-warning">
    <b>Important</b> SAFIRE currently only supports Gamma-point THC.
</div>

+++ {"id": "t0fL-8B49jMg"}

## Writing the Trial Wavefunction

Finally, we can save a single Slater determinant wavefunction to HDF5 using a "wavefunction" block.
The Slater determinant is constructed by occupying the Kohn-Sham orbitals with the largest occupancy.
The sample input file, `wfn.toml` is as follows:

```toml
[mean_field.qe]
name = "mf"
prefix = "pwscf"
outdir = "../qe/OUT"
nbnd = 20

[wavefunction.mf]
mean_field  = "mf"
output      = "wfn.h5"
```

The key points are that `mean_field` parameter in the `[wavefunction.mf]` must point to
the `name` of the `[mean_field.qe]` block.
You can use any `*.h5` file name for the output parameter.
This includes the HDF5 file containing the Hamiltonian if desired.

Now, run CoQuí on this input file.

```bash
$ coqui wfn.toml &> wfn.out
```

You should see the following output

```
 ---------------------------------
     ____ ___   ___  _   _ ___
    / ___/ _ \ / _ \| | | |_ _|
   | |  | | | | | | | | | || |
   | |__| |_| | |_| | |_| || |
    \____\___/ \__\_\\___/|___|
  --------------------------------
 |  Correlated Quantum Interface  |
  --------------------------------

Input Parameters
----------------

[mean_field.qe]
name = 'mf'
nbnd = 20
outdir = '../qe/OUT'
prefix = 'pwscf'

[wavefunction.mf]
mean_field = 'mf'
output = 'wfn.h5'

-- End of Input Parameters --

  Brillouin zone symmetry info
  ----------------------------
  Q-points in the irreducible zone = 1
  Symmetries applied to Q-points   = 1
  Time-reversal k-point pairs      = 0

  Quantum ESPRESSO reader
  -----------------------
  Number of spins                = 1
  Number of polarizations        = 1
  Number of bands                = 20
  Monkhorst-Pack mesh            = (1,1,1)
  K-points                       = 1 total, 1 in the IBZ
  Number of electrons            = 8.0
  Electron density energy cutoff = 100.000 a.u. | FFT mesh = (24,24,24)
  Wavefunction energy cutoff     = 11.994 a.u. | FFT mesh = (11,11,11), Number of PWs = 537

*************************************************
               Adding  Wavefunction              
*************************************************
 Adding default wavefunction (assuming MO basis)
 Total number of electrons in waveunction: nup:4, ndown:4:
 Number of determinants requested:1, found:1
 Number of occupied states per kpoint:
 determinant:0 spin up: [4]
 determinant:0 spin down: [4]
*************************************************
```

+++ {"id": "-QQPlGsMnzeJ"}

## Your Turn : Run SAFIRE

You now have all the ingredients for an AFQMC calculation.
Next, we'll run SAFIRE using the Hamiltonian and trial wavefunction that we wrote using CoQuí.
To learn more about the input file, see {doc}`../02_understanding_the_input_file/02_understanding_the_input_file`.

Your steps:

1. make an input file for SAFIRE to run using the Hamiltonian in `hamil.h5` and the trial wavefunction in `wfn.h5`. Use CLOSED walkers.
2. Run SAFIRE using that input file
3. use `scalar_data` to obtain the AFQMC energy

<details>
<summary>Hint:</summary>
  <p>This is the same calculation as in Tutorial 1: Hello Safire. Your energy should match the result there, `-0.729271 +/- 0.000532`, to within stochastic uncertainty.</p>
</details>

+++ {"id": "QybItZ2_98uJ"}

## Summary

In this tutorial we have learned,

1.  How to write a Hamiltonian to the SAFIRE HDF5 format, using CoQui, starting from a Quantum Espresso calculation 
2.  How to write a Trial wavefunction to the SAFIRE HDF5 format, using CoQui, starting from a Quantum Espresso calculation

```{code-cell} ipython3
:id: wF2XOkh7-D5X


```
