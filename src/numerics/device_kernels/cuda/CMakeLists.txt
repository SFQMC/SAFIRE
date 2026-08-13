target_sources(safire_lib
  PRIVATE
    argmax_min.cu 
    copy_select.cu
    copy.cu
    complex_tools.cu
    construct_fields.cu
    determinants.cu
    accumulate.cu
    split_singular_vals.cu
    add_scalar.cu
    phmsd_determinants.cu
    phmsd_energy.cu
    phmsd_vmf.cu
    apply.cu
    probe.cu
)

