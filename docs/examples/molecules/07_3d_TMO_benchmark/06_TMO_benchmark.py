# reference data from Phys. Rev. X 10, 011041 (see "total_energy.csv" from the supporting materials )
TestSet = dict(
    VO = dict(
        atom=f"V 0. 0. {1.591/2} \nO 0. 0. {-1.591/2}",
        spin=3,
        ncas=12,
        nelec_cas = 9,
        E_ref_hf = -86.400211965699242,
        E_ref_afqmc = -87.0818,
        dE_ref_afqmc = 0.001
    ),
    TiO = dict(
        atom=f"Ti 0. 0. {1.623/2} \nO 0. 0. {-1.623/2}",
        spin=2, 
        ncas=9,
        nelec_cas = 8,
        E_ref_hf = -73.26973605789902,
        E_ref_afqmc = -73.9019,
        dE_ref_afqmc = 0.0009
    ),
    CrO = dict(
        atom=f"Cr 0. 0. {1.621/2} \nO 0. 0. {-1.621/2}", 
        spin=4,
        ncas=9,
        nelec_cas = 10,
        E_ref_hf = -101.8381354486648,
        E_ref_afqmc = -102.5533,
        dE_ref_afqmc = 	0.001
    ),
    MnO = dict(
        atom=f"Mn 0. 0. {1.648/2} \nO 0. 0. {-1.648/2}",
        spin=5,
        ncas=12,
        nelec_cas = 17,
        E_ref_hf = -119.1332394993054,
        E_ref_afqmc = -119.8515,
        dE_ref_afqmc = 0.0014
    ),
    FeO = dict(
        atom=f"Fe 0. 0. {1.616/2} \nO 0. 0. {-1.616/2}", 
        spin=4,
        ncas=12, 
        nelec_cas = 12,
        E_ref_hf = -138.66009692442393,
        E_ref_afqmc = -139.4296,
        dE_ref_afqmc = 0.001
    ),
    CuO = dict(
        atom=f"Cu 0. 0. {1.725/2} \nO 0. 0. {-1.725/2}",
        spin=1,
        ncas=12,
        nelec_cas = 15,
        E_ref_hf = -212.238829673217,
        E_ref_afqmc = -213.1271,
        dE_ref_afqmc = 0.0011 
    ),
    ScO = dict(
        atom=f"Sc 0. 0. {1.668/2} \nO 0. 0. {-1.668/2}",
        spin=1,
        ncas=9,
        nelec_cas = 7,
        E_ref_hf = -61.829342751826275,
        E_ref_afqmc = -62.4192,
        dE_ref_afqmc = 0.0013
    )
)

from pathlib import Path
import json
import time

import h5py as h5
import numpy as np
from pyscf import gto,scf,mcscf
import jax
import jax.numpy as jnp

import afqmctools
import autohf

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_cas_wfn
from afqmctools.inputs.from_hdf import write_json
from afqmctools.hamiltonian.io import write_to_hdf5

from tutorial_utils import get_scratch_dir

home = Path.home()
scratch_dir = get_scratch_dir("06_3d_tm_oxides",home / ".scratch")
ecp_dir = Path("./files").resolve()

# load the ECP and basis
with open((ecp_dir / "trail.json"),'r') as f:
    ecp_basis_data = json.loads(f.read())

def setup_benchmark(key:str, case:dict):
    """
    Note: each calculation will take some time; we recommend running
    each AFQMC calculation manually offline; however, you can 
    set run_afqmc=True to run them here. In that case, each AFQMC calculation
    will run one after another, taking significantly more wall time to finish
    than submitting each as a separate job.
    """
    
    local_scratch_dir = scratch_dir / key
    local_scratch_dir.mkdir(exist_ok=True)
    
    atom = case["atom"]
    spin = case["spin"]
    ncas = case["ncas"]
    nelec_cas = case["nelec_cas"]

    TM = key[:-1]
    
    mol = gto.M(
        atom = atom,
        basis = {TM : ecp_basis_data[TM]["vdz"], 'O' : ecp_basis_data['O']["vdz"]},
        ecp = {TM : ecp_basis_data[TM]["ecp"], 'O' : ecp_basis_data['O']["ecp"]},
        spin = spin,
        verbose = 4
    )
    
    # set up Hamiltonian for AutoHF
    onebody = mol.intor("int1e_kin") + mol.intor("int1e_nuc") + mol.intor("ECPscalar")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")
    nuclear_repulsion = mol.energy_nuc()

    X = afqmctools.utils.linalg.get_ortho_ao_mol(overlap,LINDEP_CUTOFF=1e-8)
    
    norb = onebody.shape[0]
    print(f"{norb=}")

    chol_tol = 1e-6
    cholesky_operators = afqmctools.utils.linalg.modified_cholesky_direct(
        eri.reshape(norb**2, norb**2), tol=chol_tol, verbose=False
    ).reshape(-1, norb, norb)

    onebody = X.T @ onebody @ X
    cholesky_operators = 1j * np.einsum("ij,njk,kl->nil", X.T, cholesky_operators, X)

    onebody += 0.5 * np.einsum("nij,njk", cholesky_operators, cholesky_operators).real

    # double along spin dimension
    cholesky_operators = np.stack([cholesky_operators, cholesky_operators], axis=1)
    onebody = np.stack([onebody, onebody], axis=0)


    T = np.zeros((norb, norb))

    H = autohf.AutoHFHamiltonian(T=(T, T), U=0)

    nelec = mol.nelec
    Nup = nelec[0]
    Ndown = nelec[1]

    # Custom ROHF ansatz
    @jax.jit
    def orbitalFunc(alpha):
        r"""
        produces |\Phi_\sigma (alpha) \rangle from \Phi_{\mu p} \equiv \alpha_{\mu p}
        using the first Nup / Ndown electrons for the spin-up / spin-down sectors.
        """
        phi_up = alpha[:,:Nup]
        phi_down = alpha[:,:Ndown]
        return (phi_up, phi_down)

    @jax.jit
    def rdmFunc(alpha):
        r"""takes (M,max(Nup,Ndown)) orbitals
        returns <\psi|c^dag c | \psi>/<\psi|\psi>
        with shape (2,M,M) where first axis is spin
        """
        phi_up,phi_down = orbitalFunc(alpha)
        rdm_up = phi_up.conj() @ jnp.linalg.solve(phi_up.T @ phi_up.conj(), phi_up.T)
        rdm_down = phi_down.conj() @ jnp.linalg.solve(phi_down.T @ phi_down.conj(), phi_down.T)
        return (rdm_up, rdm_down)                                                             


    hf_settings = {
        "verbose": False,
        "steps": 1200,
        "opt_method": "lbfgs",
        "ansatz": "CUSTOM",
        "batch_size": 20,
        "gpu": True,
        "nelec": nelec,
        "noncollinear": False,
        "state0_scale": 0.01,
        "seed": 12345,
        "measure_spin": True
    }

    @jax.jit
    def molecular_energy(state, rdms):
        E = (
            autohf.terms.cholesky.hf_onebody(rdms, onebody)
            + autohf.terms.cholesky.hf_cholesky(rdms, cholesky_operators)
            + nuclear_repulsion
        )
        return E

    rng = np.random.default_rng(42)

    N = onebody.shape[1]
    Ne = max(Nup,Ndown)
    
    state0_ref = rng.normal(scale=0.3, size=(N, Ne))
    state0 = state0_ref + rng.normal(scale=0.05, size=(hf_settings["batch_size"], N, Ne))

    ts_autohf = time.time()
    data, data_functions = autohf.lattice_hf(
        H, 
        settings=hf_settings, 
        custom_energy_term=molecular_energy,
        state2orbitals=orbitalFunc,
        state2rdm=rdmFunc,
        state0=state0,
        state0_ref=state0_ref,
        jaxoptargs=dict(tol=1e-8), #keyword settings passed to jaxopt
    )
    tf_autohf = time.time()
    print(f"AutoHF took {tf_autohf-ts_autohf} seconds for {key}")

    rounding_level = -np.log10(chol_tol).astype(int)

    # now gran the SD, and orthonormalize the orbitals
    makeRDMs = data_functions["makeRDMs"]

    # get the best state's Slater determinant
    auto_hf_result = makeRDMs(data["state"])

    guess_ao_uhf = (
        X @ auto_hf_result[0] @ X.T,
        X @ auto_hf_result[1] @ X.T
    )

    # run HF to generate a starting point
    # run CASSCF to get an orbital basis AND a CAS wavefunction
    casscf_chkfile = local_scratch_dir / 'rhf_chkfile.h5'

    # quick run
    rhf = scf.ROHF(mol)
    rhf.chkfile = casscf_chkfile
    rhf.max_cycle = -1
    rhf.init_guess = guess_ao_uhf
    E = rhf.kernel(guess_ao_uhf)
    
    print("==== some sanity checks ====\n")
    print("PySCF ROHF energy:", np.round(E, rounding_level))
    print("AutoHF ROHF energy:", np.round(data["E"], rounding_level))
    print(f"trace of guess_ao (AO basis) {np.trace(guess_ao_uhf[0] @ overlap) + np.trace(guess_ao_uhf[1] @ overlap)}")
    print("\n===========================\n")

    assert np.isclose(E, data["E"]), "PySCF energy does not match AutoHF energy"
    assert np.isclose(E, case["E_ref_hf"], atol=1e-6), "PySCF energy does not match reference HF energy"

    # run CASSCF to get a basis / trial wavefunction
    mc = mcscf.CASSCF(rhf, ncas, nelec_cas).run()

    # save some extra data to the checkpoint file
    with h5.File(casscf_chkfile, 'a') as fp:
        write_to_hdf5(fp,'mcscf/ci', data=mc.ci)
        write_to_hdf5(fp,'mcscf/ncore', data=mc.ncore)
        write_to_hdf5(fp,'mcscf/ncas', data=mc.ncas)
    
    # write the CAS wavefunction to a file
    write_cas_wfn(
        mol=mol,
        cas_chkfile=casscf_chkfile,
        tol_trunc=0.001, # from the PRX
        outname=local_scratch_dir / 'afqmc.h5',
        max_det=2000
    )
    
    basis_scf_data = load_from_pyscf_chk_mol(
        chkfile = casscf_chkfile,
        base = 'mcscf'
    )

    # write Hamiltonian
    write_hamil_mol(
        basis_scf_data,
        hamil_file = local_scratch_dir / 'afqmc.h5',
        chol_cut = 1e-6, # from the PRX
        verbose=True
    )

    execute_options = {
        "timestep": 0.005,
        "steps": 7000,
        "measure_interval_multiplier": 1,
        "population_control_interval" : 10,
        "walker_ortho_interval" : 10 ,
        "n_walkers_per_mpi_task": 50,
        "seed" : 42
    }    
    write_json(
        local_scratch_dir/ "afqmc.json",
        fwfn0=local_scratch_dir / 'afqmc.h5',
        exec_opts=execute_options
    )

# run the setup
for molecule,data in TestSet.items():
    setup_benchmark(molecule,data)
