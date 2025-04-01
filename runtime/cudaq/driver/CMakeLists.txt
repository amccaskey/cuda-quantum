# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(controller)

add_subdirectory(qpu)

set(LIBRARY_NAME cudaq-driver) 

add_library(${LIBRARY_NAME} 
  SHARED 
    driver.cpp 
    channels/shared_memory.cpp 
    channels/rpc_controller_channel.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

if (CUDA_FOUND) 
  target_sources(${LIBRARY_NAME} PRIVATE channels/cuda_channel.cpp)
  target_link_libraries(cudaq-driver 
    PUBLIC 
    PRIVATE 
      fmt::fmt-header-only LLVMSupport rpc CUDA::cudart_static
  )
else() 
  target_link_libraries(cudaq-driver 
    PUBLIC 
    PRIVATE 
      fmt::fmt-header-only LLVMSupport rpc
  )
endif()