# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
add_subdirectory(compilers)

set(LIBRARY_NAME cudaq-driver-controller) 

add_library(${LIBRARY_NAME} 
  SHARED 
    # Base library
    controller.cpp
    controller_qir_impl.cpp
    
    # Channel Subtypes, Device and Controller
    channels/controller/rpc_controller_channel.cpp 
    channels/controller/shared_memory_channel.cpp
    channels/device/shared_memory.cpp 
    channels/device/rpc_channel.cpp
    
    # Controller Subtypes
    controllers/rpc_controller.cpp
    controllers/shared_memory_controller.cpp
    
    # Targets 
    targets/default_target.cpp
    )

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_compile_options(cudaq-driver-controller PRIVATE -fno-rtti)

if (CUDA_FOUND) 
  target_sources(${LIBRARY_NAME} PRIVATE channels/device/cuda_channel.cpp)

  target_link_libraries(${LIBRARY_NAME}
  PRIVATE 
    rpc 
    fmt::fmt-header-only 
    cudaq-common
    CUDA::cudart_static
    LLVMSupport
  )
else() 

  target_link_libraries(${LIBRARY_NAME}
    PRIVATE 
      rpc 
      fmt::fmt-header-only 
      cudaq-common
      LLVMSupport
    )
endif()

add_subdirectory(channels/device/daemons)