# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

create_simulator_runtime_test(test_iqm
  EXTRA_SOURCES IQMTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated 
    gmock_main
  HAS_MAIN_CPP TRUE
  BACKEND_CONFIG "iqm,qpu-architecture,Apollo,url,http://localhost:62443"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
  SKIP_GTEST_DISCOVER TRUE
)

configure_file("IQMStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/iqm/IQMStartServerAndTest.sh" @ONLY)
add_test(NAME iqm-tests COMMAND bash IQMStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/iqm/)
set_property(TEST iqm-tests PROPERTY SKIP_RETURN_CODE 77)
