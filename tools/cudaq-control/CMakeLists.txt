# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(TOOL_NAME cudaq-controller)
add_executable(${TOOL_NAME} cudaq-control.cpp)
set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
target_link_libraries(${TOOL_NAME} 
  PRIVATE 
    rpc 
    cudaq-driver-controller 
    fmt::fmt-header-only 
    CUDAQTargetConfigUtil
    LLVMSupport)