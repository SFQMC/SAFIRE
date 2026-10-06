from safiretools import write_jobfile

write_jobfile(
    "afqmc.json",
    {
        "seed": 42,
        # these will be written to the "execute" block
        #    in the .json input file using the same keywards
        #     as in the .json format
        # Here are all of the possible inputs for the "execute" block
        #  *except* for the "estimators" key which has additional internal options.
        "execute": {
            "wavefunction": {
                "filename": "afqmc_wfn.h5"
            },
            "hamiltonian": {
                "filename": "afqmc_ham.h5"
            },
            "timestep": 0.01,
            "steps": 10000,
            "population_control_interval": 10,  # in units of steps
            "measure_interval": 10,             # in units of steps
            "walker_ortho_interval": 10,        # in units of steps
            "num_walkers": 200,
            "estimators": {
                "energy": {}
            }
        }
    }
)

print("Done")
