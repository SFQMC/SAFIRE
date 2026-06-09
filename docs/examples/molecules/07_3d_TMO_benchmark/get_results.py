from pathlib import Path
from warnings import warn

import numpy as np

from stats.scalar_dat import analyze_scalar_data
from tutorial_utils import get_scratch_dir


# For you TODO: set a scratch directory for the files that will be generated
home = Path.home()
scratch_dir = get_scratch_dir("3d_tm_oxides",home / ".scratch")
ecp_dir = Path("./files").resolve()


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


def get_afqmc_results(key:str, case:dict, nequil:int=15, compare_to_ref:bool=True):
    """
    Process and save the AFQMC result for the benchmark case
    corresponding to 'key' into the 'case' dictionary 
    under keywords "E_afqmc" and "dE_afqmc"
    """
    local_scratch_dir = scratch_dir / key
    result_file = local_scratch_dir/"qmc.s000.scalar.dat"

    # skip if local scratch dir does not exist - save None for E, dE
    if not result_file.is_file():
        case["E_afqmc"] = None
        case["dE_afqmc"] = None
        warn(f"No AFQMC result found for {key}. Skipping analysis.")
        return
    
    print(f"Analyzing AFQMC results for {key}")
    settings = dict(
        fname = result_file,
        xaxis = "time",
        nequil = nequil,
        trace = True
    )

    E,dE = analyze_scalar_data(settings)

    case["E_afqmc"] = E
    case["dE_afqmc"] = dE
    
    if compare_to_ref:
        E_ref = case["E_ref_afqmc"]
        dE_ref = case["dE_ref_afqmc"]
        print(f"          E_afqmc = {E} +/- {dE}")
        print(f"reference E_afqmc = {E_ref} +/- {dE_ref}")
        print(f" (difference) = {E - E_ref} +/- (joint stoch. uncertatinty) {np.sqrt(dE_ref**2 + dE**2)}")



if __name__ == '__main__':
    for molecule,data in TestSet.items():
        get_afqmc_results(molecule,data,nequil=8)

    # loop through the cases and generate a results table
    print("======== Benchmark Results ========")
    print("| Molecule | E AFQMC | Ref. A AFQMC | abs diff | match? |\n|---|---|---|---|---|")
    for molecule,data in TestSet.items():
        E = data["E_afqmc"]
        dE = data["dE_afqmc"]
        E_ref = data["E_ref_afqmc"]
        dE_ref = data["dE_ref_afqmc"]

        if E is None or dE is None:
            print(f"| {molecule} | - | {E_ref:6.4f} +/- {dE_ref:4.4f} | - | N/A")
        else:
            abs_diff = np.abs(E - E_ref)
            joint_err = np.sqrt(dE**2 + dE_ref**2)
        
            if abs_diff < joint_err:
                matches = "1 sigma"
            elif abs_diff < 2*joint_err:
                matches = "2 sigma"
            else:
                matches = "No"
    
            print(f"| {molecule} | {E:6.4f} +/- {dE:4.4f} | {E_ref:6.4f} +/- {dE_ref:4.4f} | {abs_diff} +/- {joint_err} | {matches} |")
