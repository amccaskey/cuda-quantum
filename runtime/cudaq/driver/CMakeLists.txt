# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(controller)

add_subdirectory(qpu)

set(LIBRARY_NAME cudaq-driver) 

add_library(${LIBRARY_NAME} 
  SHARED 
    driver.cpp 
    )

target_include_directories(${LIBRARY_NAME} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
  target_link_libraries(cudaq-driver 
    PUBLIC cudaq-driver-controller
    PRIVATE 
      fmt::fmt-header-only LLVMSupport rpc
  )
