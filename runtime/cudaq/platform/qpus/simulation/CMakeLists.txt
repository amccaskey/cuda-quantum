# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-qpu-simulator)
add_library(cudaq-qpu-simulator SHARED simulator_qpu.cpp)

add_subdirectory(qpp)
add_subdirectory(stim) 
if (CUDA_FOUND)
  add_subdirectory(custatevec) 
  add_subdirectory(cudensitymat)
endif()

target_include_directories(${LIBRARY_NAME} PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE 
    cudaq-operator
    cudaq-common 
    fmt::fmt-header-only 
)

# Add a post-build check to verify that no unwanted Stim symbols are exported.
# This is important to allow users to import both `cudaq` and `stim` into the same
# Python environment (even if they are using different versions of Stim).
add_custom_command(
  TARGET ${LIBRARY_NAME}
  POST_BUILD
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/stim/verify_linkage.sh $<TARGET_FILE:${LIBRARY_NAME}>
  VERBATIM
  COMMENT "Verifying that no stim namespace symbols are exported in ${LIBRARY_NAME}..."
)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (CUDA_FOUND) 
  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_ENABLE_CUDA)
  target_link_libraries(${LIBRARY_NAME} PRIVATE CUDA::cudart_static)
endif() 