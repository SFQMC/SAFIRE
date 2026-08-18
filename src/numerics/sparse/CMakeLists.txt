# We want to make this a stand-alone library, but it relies on safire internals
# add_library(sparse_lib)

if(ENABLE_DEVICE)
  # add choices when other backends are available
  target_sources(
    safire_lib
    PRIVATE
      detail/CUDA/cusparse_interface.cpp
  )

  target_link_libraries(
    safire_lib
    PRIVATE
      CUDA::cusparse
  )
ENDIF()

