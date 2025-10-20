

find_library(CUSTATEVEC_LIB
    NAMES custatevec libcustatevec.so.1
    HINTS
    ${CUSTATEVEC_ROOT}/lib64
    ${CUSTATEVEC_ROOT}/lib
    ${CUSTATEVEC_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUSTATEVEC_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR})

if(NOT CUSTATEVEC_LIB)
  message(FATAL_ERROR "\nUnable to find custatevec installation. Please ensure it is correctly installed and set and define CUSTATEVEC_ROOT if necessary (currently set to: ${CUSTATEVEC_ROOT}).")
endif()
message(STATUS "CUSTATEVEC_LIB: ${CUSTATEVEC_LIB}")

get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}")

target_sources(${LIBRARY_NAME} PRIVATE state_vector.cpp state_vector_impl.cu)
target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CUDAToolkit_INCLUDE_DIRS} 
      ${CUDAToolkit_INCLUDE_DIRS}/cccl
      ${CUSTATEVEC_ROOT}/include)

target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUSTATEVEC_LIB} CUDA::cudart_static CUDA::cublas)