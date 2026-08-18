add_subdirectory(sparse)

IF(ENABLE_DEVICE)
  add_subdirectory(device_kernels)
ENDIF()

IF (BUILD_UNIT_TESTS)
#  add_subdirectory(fft/tests)
  add_subdirectory(shared_array/tests)
  add_subdirectory(sparse/tests)

  add_subdirectory(tests)
ENDIF()

