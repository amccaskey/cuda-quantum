# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-experimental)

add_library(${LIBRARY_NAME} SHARED config.cpp) 
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(policies)
add_subdirectory(qis)
add_subdirectory(qpus)

target_include_directories(${LIBRARY_NAME}
    PUBLIC
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/cudaq/experimental>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

target_link_libraries(${LIBRARY_NAME} PRIVATE fmt::fmt-header-only spdlog::spdlog)
