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
    controller_qir_impl.cpp
    controllers/rpc_controller.cpp
    compilers/default_compiler.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_compile_options(cudaq-driver-controller PRIVATE -fno-rtti)

target_link_libraries(cudaq-driver-controller 
  PUBLIC 
  PRIVATE 
    rpc 
    fmt::fmt-header-only 
    cudaq-driver
    cudaq-common
    MLIRIR
    MLIRPass 
    MLIRExecutionEngine 
    CCDialect
    QuakeDialect
    OptCodeGen
    OptTransforms
    )