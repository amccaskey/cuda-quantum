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
    callback_support.cpp 
    controller.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_link_libraries(cudaq-driver-controller 
  PUBLIC 
  PRIVATE 
    rpc fmt::fmt-header-only cudaq-driver cudaq-common
    )