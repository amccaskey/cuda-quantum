# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# add_executable(test_oqc OQCTester.cpp)
# target_link_options(test_oqc PRIVATE -Wl,--no-as-needed)
# target_compile_definitions(test_oqc PRIVATE -DNVQIR_BACKEND_NAME=oqc)
# target_include_directories(test_oqc PRIVATE ../..)
# target_link_libraries(test_oqc
#   PRIVATE fmt::fmt-header-only 
#   cudaq-common 
#   cudaq
#   cudaq-builder
#   cudaq-mlir-runtime
#   cudaq-rest-qpu
#   cudaq-operator
#   cudaq-platform-default 
#   gtest_main)


create_simulator_runtime_test(test_oqc
  EXTRA_SOURCES OQCTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  SKIP_GTEST_DISCOVER TRUE
  BACKEND_CONFIG "oqc,entry_url,http://localhost:62442,auth_token,fake_auth_token,device,qpu:uk:-1:1234567890"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

configure_file("OQCStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/oqc/OQCStartServerAndTest.sh" @ONLY)
add_test(NAME oqc-tests COMMAND bash OQCStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/oqc/)
