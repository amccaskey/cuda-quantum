# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(cudaq-driver-qpu SHARED driver_qpu.cpp 
   ../../platform/common/QuantumExecutionQueue.cpp)

target_include_directories(cudaq-driver-qpu PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(cudaq-driver-qpu 
  PUBLIC 
    cudaq-spin 
    cudaq-common 
  PRIVATE 
    # cudaq-mlir-runtime 
    # MLIRTranslateLib
    fmt::fmt-header-only 
    cudaq 
    cudaq-platform-default
    )

install(TARGETS cudaq-rest-qpu DESTINATION lib)

