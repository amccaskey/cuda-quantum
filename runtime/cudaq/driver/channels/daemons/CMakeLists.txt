# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(TOOL_NAME cudaq-ethernet-channel-daemon)
add_executable(${TOOL_NAME} ethernet_daemon.cpp ${CMAKE_SOURCE_DIR}/runtime/cudaq/utils/matrix.cpp )
set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
# FIXME only a gcc thing
target_link_libraries(${TOOL_NAME} 
  PRIVATE 
    rpc 
    cudaq-common
    cudaq-mlir-runtime
    fmt::fmt-header-only 
    CUDAQTargetConfigUtil
    LLVMSupport
    )
