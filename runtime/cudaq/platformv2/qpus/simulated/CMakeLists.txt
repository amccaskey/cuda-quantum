# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-qpu-simulated SHARED simulator_qpu.cpp simulator_qpu_config.cpp)

add_subdirectory(qpp)
add_subdirectory(stim) 
if (CUDA_FOUND)
  add_subdirectory(custatevec) 
  add_subdirectory(cudensitymat)
endif()

target_include_directories(cudaq-qpu-simulated PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(cudaq-qpu-simulated
  PRIVATE 
    cudaq-operator
    cudaq-common 
    fmt::fmt-header-only 
)

install(TARGETS cudaq-qpu-simulated DESTINATION lib)

if (CUDA_FOUND) 
  target_compile_definitions(cudaq-qpu-simulated PRIVATE CUDAQ_ENABLE_CUDA)
  target_link_libraries(cudaq-qpu-simulated PRIVATE CUDA::cudart_static)
endif() 

# Add a post-build check to verify that no unwanted Stim symbols are exported.
# This is important to allow users to import both `cudaq` and `stim` into the same
# Python environment (even if they are using different versions of Stim).
add_custom_command(
  TARGET cudaq-qpu-simulated
  POST_BUILD
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/stim/verify_linkage.sh $<TARGET_FILE:cudaq-qpu-simulated>
  VERBATIM
  COMMENT "Verifying that no stim namespace symbols are exported in cudaq-qpu-simulated..."
)
