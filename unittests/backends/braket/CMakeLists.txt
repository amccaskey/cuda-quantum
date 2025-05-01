# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

create_simulator_runtime_test(test_braket_emulate
  EXTRA_SOURCES BraketLocalEmulatorTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  # BACKEND CONFIG Requires commas since semicolons are special in CMake, 
  # create_simulator_runtime_test will replace them before passing them along 
  BACKEND_CONFIG "braket,emulate,true,url,http://localhost:62445,credentials,$ENV{HOME}/FakeCppBraket.config"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

# AJM Refactor - the braket tests don't work
# create_simulator_runtime_test(test_braket
#   EXTRA_SOURCES BraketTester.cpp 
#   DISABLE_INTEGRATION TRUE 
#   QPU_LIB cudaq-rest-qpu 
#   EXTRA_LIBS 
#     cudaq-mlir-runtime 
#     cudaq-qpu-simulated
#   HAS_MAIN_CPP TRUE
#   SKIP_GTEST_DISCOVER TRUE
#   BACKEND_CONFIG "braket,url,http://localhost:62445,credentials,$ENV{HOME}/FakeCppBraket.config"
#   TARGET_OPTIONS "--target-option qpu:remote_rest"
# )

# configure_file("BraketStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/braket/BraketStartServerAndTest.sh" @ONLY)
# add_test(NAME braket-tests COMMAND bash BraketStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/braket/)
