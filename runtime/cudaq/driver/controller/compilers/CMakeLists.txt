# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-quake-compilers) 

add_library(${LIBRARY_NAME} 
  SHARED 
    default_compiler.cpp
)

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_compile_options(${LIBRARY_NAME} PRIVATE -fno-rtti)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE 
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