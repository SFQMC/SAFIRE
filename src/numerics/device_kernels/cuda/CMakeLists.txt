target_sources(safire_lib
  PRIVATE
    argmax_min.cu 
    copy_select.cu
    copy_cast.cu
    complex_tools.cu
    construct_fields.cu
    determinants.cu
    accumulate.cu
    split_singular_vals.cu
    add_diagonal.cu
    add_scalar.cu
    phmsd_determinants.cu
    phmsd_energy.cu
    apply.cu
    stochastic_reductions.cu
    probe.cu
#    symmetry_tools.cu
#    potentials.cu
#    kpoint_tools.cu
)

