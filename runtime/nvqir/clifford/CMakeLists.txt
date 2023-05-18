# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# include(FetchContent)
# FetchContent_Declare(stim
#         GIT_REPOSITORY https://github.com/quantumlib/stim.git
#         GIT_TAG v1.4.0)  
# FetchContent_GetProperties(stim)
# if(NOT stim_POPULATED)
#   FetchContent_Populate(stim)
#   add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
# endif()


add_library(nvqir-clifford SHARED StimCircuitSimulator.cpp)
target_include_directories(nvqir-clifford
                 PUBLIC . .. /root/.stim/include)

if(APPLE)
    set_target_properties(nvqir-clifford
        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${LLVM_BINARY_DIR}/lib")
    set_target_properties(nvqir-clifford
        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
    set_target_properties(nvqir-clifford
        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${LLVM_BINARY_DIR}/lib")
    set_target_properties(nvqir-clifford PROPERTIES LINK_FLAGS "-shared")
endif()

target_compile_options(nvqir-clifford PRIVATE -march=native)
target_link_directories(nvqir-clifford PUBLIC /root/.stim/lib)
target_link_libraries(nvqir-clifford PUBLIC stim
                fmt::fmt-header-only 
                cudaq-common)
  

install(TARGETS nvqir-clifford DESTINATION lib)
add_platform_config(clifford)
