# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-rt)

add_library(cudaq-rt SHARED utils/matrix.cpp utils/logger.cpp)
add_subdirectory(operators)
add_subdirectory(policies)
add_subdirectory(qis)
add_subdirectory(platform)

target_include_directories(${LIBRARY_NAME}
    PUBLIC
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/rt>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

target_link_libraries(${LIBRARY_NAME} PRIVATE fmt::fmt-header-only spdlog::spdlog)
