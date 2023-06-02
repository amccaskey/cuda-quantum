# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-quera)

add_library(${LIBRARY_NAME} SHARED QueraPlatform.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC pthread cudaq-em-qir cudaq-spin cudaq-common PRIVATE fmt::fmt-header-only)

cudaq_library_set_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
# install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-platform-quera-targets DESTINATION lib)

# install(EXPORT cudaq-platform-quera-targets
#         FILE CUDAQPlatformDefaultTargets.cmake
#         NAMESPACE cudaq::
#         DESTINATION lib/cmake/cudaq)

add_target_config(quera)