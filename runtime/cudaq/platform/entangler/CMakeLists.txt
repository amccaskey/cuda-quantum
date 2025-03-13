# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-entangler)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_DEFAULTPLATFORM_SRC
  entangler_platform.cpp
  ../common/QuantumExecutionQueue.cpp
)

add_library(${LIBRARY_NAME} SHARED entangler_platform.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    pthread cudaq-em-default cudaq-spin cudaq-common
  PRIVATE
    fmt::fmt-header-only cudaq CUDAQTargetConfigUtil)

install(TARGETS ${LIBRARY_NAME}
  EXPORT ${LIBRARY_NAME}-targets DESTINATION lib)

# install(EXPORT ${LIBRARY_NAME}-targets
#         FILE CUDAQPlatformDefaultTargets.cmake
#         NAMESPACE cudaq::
#         DESTINATION lib/cmake/cudaq)
  
