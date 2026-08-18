#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(fft_src
   )

if(ENABLE_FFTW)
  set(fft_src 
       ${fft_src}
       fftw.cpp 
     )
endif()

if(ENABLE_CUDA)
  set(fft_src 
       ${fft_src}
       cufft.cpp 
     )
endif()


add_library(fft_lib ${fft_src})
if(ENABLE_FFTW)
  target_link_libraries(fft_lib PUBLIC FFTW::Double) 
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft) 
endif()
install(TARGETS fft_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

