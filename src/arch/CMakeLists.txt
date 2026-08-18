target_sources(safire_lib
  PRIVATE
    arch.cpp
)

IF(ENABLE_CUDA)
  target_sources(safire_lib
    PRIVATE
      CUDA/cuda_init.cpp
  )
ENDIF()
