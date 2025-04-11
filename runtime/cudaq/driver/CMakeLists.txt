# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-driver) 

add_library(${LIBRARY_NAME} 
  SHARED 
    driver.cpp
    driver_qpu.cpp
    target.cpp
    ../platform/common/QuantumExecutionQueue.cpp
    )

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

target_link_libraries(${LIBRARY_NAME} 
    PRIVATE 
      rpc
      cudaq-common
      fmt::fmt-header-only 
  )

add_subdirectory(channels)
add_subdirectory(dispatchers)