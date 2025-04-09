# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(TOOL_NAME cudaq-rpcd)
add_executable(${TOOL_NAME} cudaq-rpcd.cpp ${CMAKE_SOURCE_DIR}/runtime/cudaq/utils/matrix.cpp )
set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
# FIXME only a gcc thing
target_link_options(${TOOL_NAME} PRIVATE -Wl,--no-as-needed)
target_link_libraries(${TOOL_NAME} 
  PRIVATE 
    rpc 
    cudaq-common
    cudaq-operator 
    cudaq-quake-compilers # Make this runtime load
    fmt::fmt-header-only 
    cudaq-driver-controller # quake compiler needs this for targets...
    CUDAQTargetConfigUtil
    LLVMSupport
    )
