# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-driver-controller) 

add_library(${LIBRARY_NAME} 
  SHARED 
    controller.cpp
    channels/rpc_controller_channel.cpp 
    channels/shared_memory.cpp 
    controller_qir_impl.cpp
    controllers/rpc_controller.cpp
    compilers/default_compiler.cpp
    targets/default_target.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_compile_options(cudaq-driver-controller PRIVATE -fno-rtti)


if (CUDA_FOUND) 
  target_sources(${LIBRARY_NAME} PRIVATE channels/cuda_channel.cpp)


target_link_libraries(${LIBRARY_NAME}
PRIVATE 
  rpc 
  fmt::fmt-header-only 
  cudaq-common
  CUDA::cudart_static
  MLIRIR
  MLIRPass 
  MLIRExecutionEngine 
  CCDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  )
else() 
target_link_libraries(${LIBRARY_NAME}
  PRIVATE 
    rpc 
    fmt::fmt-header-only 
    cudaq-common
    MLIRIR
    MLIRPass 
    MLIRExecutionEngine 
    CCDialect
    QuakeDialect
    OptCodeGen
    OptTransforms
    )
  endif()