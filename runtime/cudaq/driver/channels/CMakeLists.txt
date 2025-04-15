
# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-ethernet-channel) 

add_library(${LIBRARY_NAME} 
  SHARED 
    ethernet_device_channel.cpp 
    )

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
target_link_libraries(${LIBRARY_NAME}
    PRIVATE 
      cudaq-common
      rpc 
      fmt::fmt-header-only 
  )

add_subdirectory(daemons)

if (CUDA_FOUND)
  set(LIBRARY_NAME cudaq-cuda-channel) 
  add_library(${LIBRARY_NAME} 
    SHARED 
      cuda_channel.cu 
    )
  target_include_directories(${LIBRARY_NAME} PUBLIC 
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
  target_link_libraries(${LIBRARY_NAME}
      PRIVATE 
        cudaq-common
        fmt::fmt-header-only 
        CUDA::cudart_static
        CUDA::cuda_driver
    )
endif()