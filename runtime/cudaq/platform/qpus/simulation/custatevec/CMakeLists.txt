# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
message (STATUS "CUSTATEVEC_ROOT and CUDA_FOUND - building custatevec NVQIR backend.")

target_sources(${LIBRARY_NAME} PRIVATE CuStateVecCircuitSimulator.cu)

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

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME)
  target_sources(${LIBRARY_NAME} PRIVATE ${SOURCE_FILENAME})

  target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CUDAToolkit_INCLUDE_DIRS} 
      ${CMAKE_SOURCE_DIR}/runtime
      ${CUSTATEVEC_ROOT}/include)

  target_link_libraries(${LIBRARY_NAME} 
                        PRIVATE
                          fmt::fmt-header-only
                          cudaq-common
                          ${CUSTATEVEC_LIB}
                          CUDA::cudart_static
                          CUDA::cublas)
endmacro()

nvqir_create_cusv_plugin(${LIBRARY_NAME} CuStateVecCircuitSimulator.cpp)
add_target_config(nvidia)
