# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check if CUDA and cuStateVec are available
if(NOT CUSTATEVEC_ROOT)
  message(STATUS "CUSTATEVEC_ROOT not set, skipping GPU state_vector simulator")
  return()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

message(STATUS "Building experimental GPU state_vector simulator with cuStateVec")

# Find cuStateVec library
find_library(CUSTATEVEC_LIB
    NAMES custatevec libcustatevec.so.1
    HINTS
    ${CUSTATEVEC_ROOT}/lib64
    ${CUSTATEVEC_ROOT}/lib
    ${CUSTATEVEC_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUSTATEVEC_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR})

if(NOT CUSTATEVEC_LIB)
  message(FATAL_ERROR "Unable to find custatevec library. CUSTATEVEC_ROOT=${CUSTATEVEC_ROOT}")
endif()

message(STATUS "CUSTATEVEC_LIB: ${CUSTATEVEC_LIB}")

# Create main simulator library
target_sources(${LIBRARY_NAME} PRIVATE
    state_vector.cpp
    state_vector_state.cu 
    state_vector_kernels.cu
)

target_include_directories(${LIBRARY_NAME} PRIVATE
    ${CUSTATEVEC_ROOT}/include
)

target_link_libraries(${LIBRARY_NAME}
    PRIVATE
    ${CUSTATEVEC_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cudart_static
)

target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Add to RPATH
get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}"
)


