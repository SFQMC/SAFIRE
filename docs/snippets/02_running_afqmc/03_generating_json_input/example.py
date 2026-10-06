from afqmctools.inputs.from_hdf import write_json

# these will be written to the "execute" block
#    in the .json input file using the same keywards
#     as in the .json format
# Here are all of the possible inputs for the "execute" block 
#  *except* for the "estimator" key which has additional internal options.
afqmc_execution_options = {
    "timestep": 0.01,
    "steps": 10000,
    "population_control_interval" : 10,  # in units of steps
    "measure_interval": 10,              # in units of steps
    "walker_ortho_interval" : 10 ,       # in units of steps
    "num_walkers": 200,
    "seed" : 42,
    "estimator": {
        "name": "energy",
        "overwrite": True,
        "print_components": True
    }
}

write_json(
    "afqmc.json", 
    fwfn0="afqmc_wfn.h5",
    fham0="afqmc_ham.h5",
    exec_opts=afqmc_execution_options
)

print("Done")
