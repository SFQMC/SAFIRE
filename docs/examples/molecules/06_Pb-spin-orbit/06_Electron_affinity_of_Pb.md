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

+++ {"id": "bxazkxe2X9Wx"}

# Electron Affinity of Pb: Spin-Orbit Coupling (SOC) in Ab Initio AFQMC

At the end of this example, you will know how to include SOC in ab initio AFQMC calculations.

The electron affinity (EA) of heavy elements is strongly affected by SOC.
For $Pb$, the inclusion of SOC changes the EA by about 1 eV [1];
The neutral $Pb$ atom has a $6s^26p^2$ electron configuration with $^3P_0$ experimentally observed ground state
leading to rather large SOC effects.
The $Pb^-$ ion has a spin-quartet $6s^26p^3$ electron configuration with zero orbital angular momentum leading to
much SOC effects than the neutral atom.
Accounting for SOC is, of course, critical to achieve quantitative accuracy.

In contrast to real space methods in which non-local terms in the Hamiltonian, such as SOC, must be sampled,
AFQMC is able to exactly include SOC because it is formulated in second quantization for a general orbital
basis. This makes it an ideal tool for the study of systems and observables, where SOC effects are important.
must be sampled - making it an ideal tool for the study of systems and observables
where SOC effects are important.

## References

[1] B. Eskridge, H. Krakauer, H. Shi, S. Zhang; Ab initio calculations in atoms, molecules, and solids, treating spin–orbit coupling and electron interaction on an equal footing. J. Chem. Phys. 7 January 2022; 156 (1): 014107. https://doi.org/10.1063/5.0075900   
[2] K. A. Peterson, D. Figgen, E. Goll, H. Stoll, and M. Dolg, J. Chem. Phys. 119, 11113 (2003). https://doi.org/10.1063/1.1622924   
[3] K. A. Peterson, B. C. Shepler, D. Figgen, and H. Stoll, J. Phys. Chem. A 110, 13877 (2006). https://doi.org/10.1021/jp065887l    
[4] B. Metz, H. Stoll, and M. Dolg, J. Chem. Phys. 113, 2563 (2000). https://doi.org/10.1063/1.1305880    
[5] K. A. Peterson, J. Chem. Phys. 119, 11099 (2003). https://doi.org/10.1063/1.1622923

+++ {"id": "PjMcWau9i2Xa"}

## Set Up Input for SAFIRE

We we will use the relativistic Stuttgart effective core potentials (ECPs), in order to
capture both spin-orbit coupling (SOC) and scalar relativistic effects, as
well as the corresponding augmented cGTO basis sets [2-5].

### Run the Code Block Below to Load the ECP and Basis Definitions

```{code-cell} ipython3
:id: XSXP8e2IYxFb
:outputId: 6c4e562d-b891-4890-fe3c-45c7898cfbfd

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

ecp_soc = {
    'Pb' : '''
    Pb nelec 60
    Pb ul
    2       1.0000000              0.0000000
    Pb S
    2      12.2963030            281.2854990
    2       8.6326340             62.5202170
    Pb P
    2      10.2417900             72.2768970      -144.553795
    2       8.9241760            144.5910830       144.591083
    2       6.5813420              4.7586930        -9.517385
    2       6.2554030              9.9406210         9.940621
    Pb D
    2       7.7543360             35.8485070       -35.848507
    2       7.7202810             53.7243420        35.816228
    2       4.9702640             10.1152560       -10.115256
    2       4.5637890             14.8337310         9.889154
    Pb F
    2       3.8875120             12.2098920        -8.139928
    2       3.8119630             16.1902910         8.095145
    Pb G
    2       5.6915770             -9.0966650         4.548332
    2       5.7155670            -11.5319960        -4.612798
    '''
}

aug_cc_pvdz_pp = r"""
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (9s,7p,7d) -> [5s,4p,3d]
Pb    S
     37.4583000              0.0248240              0.0000000              0.0067850              0.0000000
     23.3170000             -0.1810620              0.0000000             -0.0601870              0.0000000
     14.5759000              0.5007060              0.0000000              0.1889710              0.0000000
      6.4337600             -0.9797370              0.0000000             -0.4248380              0.0000000
      1.6671900              0.9666300              0.0000000              0.6220350              0.0000000
      0.8077320              0.4364870              0.0000000              0.3160880              0.0000000
      0.2270840              0.0136860              1.0000000             -0.7594740              0.0000000
      0.0842520             -0.0020210              0.0000000             -0.5151310              1.0000000
Pb    S
      0.0278000              1.0000000
Pb    P
     10.6186000              0.1701740             -0.0479540              0.0000000
      7.1878100             -0.4241790              0.1245560              0.0000000
      1.8943900              0.6934430             -0.2597760              0.0000000
      0.8167090              0.4737730             -0.1414800              0.0000000
      0.2036880              0.0290470              0.5317690              0.0000000
      0.0649730             -0.0043860              0.6087590              1.0000000
Pb    P
      0.0197000              1.0000000
Pb    D
     13.7266000              0.0107870              0.0000000
      6.9898300             -0.0666500              0.0000000
      2.1735200              0.3342590              0.0000000
      1.0281200              0.4826840              0.0000000
      0.4558570              0.3004140              0.0000000
      0.1822000              0.0632580              1.0000000
Pb    D
      0.0609000              1.0000000
END
"""
```

+++ {"id": "WFTquom2i2Xb"}

## Generate Inputs for the Neutral Atom

We will begin by generating the Hamiltonian and trial wavefunction for the neutral Pb atom.

### Run PySCF for the Neutral Atom to Generate a Basis and SOC-GHF Trial Wavefunction

```{code-cell} ipython3
:id: FZrF4OuWi2Xc
:outputId: 658445dd-675a-4c26-8533-a309538529e1

from pyscf import gto, scf

## Run SOC-GHF using PySCF
mol = gto.M(
    atom="Pb 0. 0. 0.",
    basis=aug_cc_pvdz_pp,
    ecp=ecp_soc,
    charge=0,
    spin=2,
    verbose=4
)

# 1. Compute ROHF orbitals to use as a basis. NOTE: SOC will be ignored!
rohf = scf.ROHF(mol)
rohf.kernel()

# perform stability analysis - see PySCF example:
# https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/17-stability.py
mo1 = rohf.stability()[0]
dm1 = rohf.make_rdm1(mo1, rohf.mo_occ)
rohf = rohf.run(dm1)
rohf.stability()
rohf.kernel()

print(f"ROHF electronic energy: {rohf.energy_elec()}")

# 2.a. compute a "No SOC" trial wavefunction
ghf = scf.GHF(mol=mol)
ghf.kernel()

mo1 = ghf.stability()
dm1 = ghf.make_rdm1(mo1, ghf.mo_occ)
ghf = ghf.run(dm1)
ghf.stability()
ghf.kernel()

print(f"GHF electronic energy: {ghf.energy_elec()}")

# 2.b. compute a "SOC" trial wavefunction
ghf_soc = scf.GHF(mol=mol)
ghf_soc.with_soc = True
ghf_soc.kernel()

mo1 = ghf_soc.stability()
dm1 = ghf_soc.make_rdm1(mo1, ghf_soc.mo_occ)
ghf_soc = ghf_soc.run(dm1)
ghf_soc.stability()
ghf_soc.kernel()

print(f"SOC-GHF electronic energy: {ghf_soc.energy_elec()}")
```

+++ {"id": "H-m_KAsri2Xc"}

### Generate and Write Hamiltonian and Trial Wavefunction for Neutral Atom

We generate the Hamiltonian and trial wavefunction in the usual way. The spin-orbit
coupling enters through the SCF object the Hamiltonian is built from: the SOC-GHF
object, set up with `with_soc = True`, has the ECP spin-orbit term in its one-body
Hamiltonian, which makes the Hamiltonian noncollinear. The ROHF object supplies the
orbital basis, for the Hamiltonian and the trial wavefunction alike.

For the initial wavefunction, we use the ground state of the non-interacting Hamiltonian
as this was found in [1] to lead to significantly faster equilibration than using
the SOC-GHF or ROHF solution for the initial wavefunction.

```{code-cell} ipython3
:id: 8I_FH4jPi2Xc
:outputId: 308f3c0b-a95a-4fac-bb4f-10a6a5c45241

from safiretools import MolecularHamiltonian
from safiretools import Wavefunction

# inputs
chol_tol = 1e-5

# output
fout = 'neutral_afqmc_sf.h5'
fout_soc = 'neutral_afqmc_soc.h5'

#####################################
#                                   #
#  Write Hamiltonian in ROHF basis  #
#                                   #
#####################################

# The SOC-GHF object's hcore carries the ECP spin-orbit term, which makes the
#   Hamiltonian noncollinear
hamiltonian = MolecularHamiltonian.from_pyscf(
    ghf_soc,
    basis=rohf,
    chol_cut=chol_tol,
)

#####################################
#                                   #
#    Write GHF Wavefunction         #
#                                   #
#####################################

wavefunction = Wavefunction.from_pyscf(
    ghf_soc,
    basis=rohf
)
```

+++ {"id": "atWbeuX8i2Xd"}

### Run SAFIRE

```{code-cell} ipython3
:id: PsAy09S7i2Xd
:outputId: 1ef1d7f2-a846-4794-b2cd-1d91df0bf49d
:tags: [scroll-output]
:scrolled: true

from safiretools import write_jobfile
from safiretools import Results, vis

# Write JSON job file, which also writes the wavefunction and the Hamiltonian to HDF5
write_jobfile(
    "neutral_afqmc.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            "timestep": 0.005,
            "steps": 7000,
            "equilibration_steps": 1600,
            "measure_interval": 5,
            "population_control_interval": 5,
            "walker_ortho_interval": 10,
            "num_walkers": 1280
        }
    }
)

!mpirun -n 16 $AFQMC_EXEC neutral_afqmc.json

results = Results("neutral_afqmc.results.h5")
vis.plot_timeseries(results)
E, dE = results.average("Energy")
```

+++ {"id": "QnNlY3KVi2Xd"}

## Repeat for Charged $Pb^-$ Ion

### Run PySCF

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 8rI2HhjiZRWO
outputId: 4d9b87d0-91f6-4828-9d9a-67d86d9730a4
---
## Run SOC-GHF using PySCF
mol = gto.M(
    atom="Pb 0. 0. 0.",
    basis=aug_cc_pvdz_pp,
    ecp=ecp_soc,
    charge=-1,
    spin=3,
    verbose=4
)

# 1. Compute ROHF orbitals to use as a basis.
rohf = scf.ROHF(mol)
rohf.kernel()

# perform stability analysis - see PySCF example:
# https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/17-stability.py
mo1 = rohf.stability()[0]
dm1 = rohf.make_rdm1(mo1, rohf.mo_occ)
rohf = rohf.run(dm1)
rohf.stability()
rohf.kernel()

print(f"ROHF electronic energy: {rohf.energy_elec()}")

# 2.a. compute a "No SOC" trial wavefunction
ghf = scf.GHF(mol=mol)
ghf.kernel()

mo1 = ghf.stability()
dm1 = ghf.make_rdm1(mo1, ghf.mo_occ)
ghf = ghf.run(dm1)
ghf.stability()
ghf.kernel()

print(f"GHF electronic energy: {ghf.energy_elec()}")

# 2.b. compute a "SOC" trial wavefunction
ghf_soc = scf.GHF(mol=mol)
ghf_soc.with_soc = True
ghf_soc.kernel()

mo1 = ghf_soc.stability()
dm1 = ghf_soc.make_rdm1(mo1, ghf_soc.mo_occ)
ghf_soc = ghf_soc.run(dm1)
ghf_soc.stability()
ghf_soc.kernel()

print(f"SOC-GHF electronic energy: {ghf_soc.energy_elec()}")
```

+++ {"id": "ivaijztJi2Xe"}

### Generate and Write Hamiltonian and Trial Wavefunction

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: gAqSvXG3aB8I
outputId: 8a825787-6626-44d1-97c4-ee738d109252
---
# inputs
chol_tol = 1e-5

# output
fout = 'charged_afqmc_sf.h5'
fout_soc = 'charged_afqmc_soc.h5'

#####################################
#                                   #
#  Write Hamiltonian in ROHF basis  #
#                                   #
#####################################

# The SOC-GHF object's hcore carries the ECP spin-orbit term, which makes the
#   Hamiltonian noncollinear
hamiltonian = MolecularHamiltonian.from_pyscf(
    ghf_soc,
    basis=rohf,
    chol_cut=chol_tol,
)

#####################################
#                                   #
#    Write GHF Wavefunction         #
#                                   #
#####################################

wavefunction = Wavefunction.from_pyscf(
    ghf_soc,
    basis=rohf
)
```

+++ {"id": "6RUWlygXi2Xe"}

### Run SAFIRE

```{code-cell} ipython3
:id: JWAr64xIi2Xe
:outputId: 0f831053-5571-4d5f-88ca-1a7c8f7edd61
:tags: [scroll-output]
:scrolled: true

# Write JSON job file
write_jobfile(
    "charged_afqmc.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            "timestep": 0.005,
            "steps": 7000,
            "equilibration_steps": 1600,
            "measure_interval": 5,
            "population_control_interval": 5,
            "walker_ortho_interval": 10,
            "num_walkers": 1120
        }
    }
)

!mpirun -n 16 $AFQMC_EXEC charged_afqmc.json

results = Results("charged_afqmc.results.h5")
vis.plot_timeseries(results)
E_charged, dE_charged = results.average("Energy")
```

+++ {"id": "bM8pXy_ui2Xe"}

## Compute the Final Result

Run the codeblock below to get the final result.
It should be 0.48(3) eV for the aug-cc-pVDZ basis set.

```{code-cell} ipython3
:id: IM3c3WQni2Xe
:outputId: 4dee21c0-ce75-4556-d9ff-8aba47c8cb8a

import numpy as np

eV = 27.211407953

EA = (E - E_charged).real
dEA = np.sqrt(dE.real**2 + dE_charged.real**2)

print(f"The electron affinity is {EA*eV} +/- {dEA*eV} eV")
```

You should reproduce the result above if you ran with the same parameters and settings as we did.
Specifically ensure that the seed and number of MPI processes is the same.
