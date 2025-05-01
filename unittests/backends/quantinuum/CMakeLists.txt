# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

create_simulator_runtime_test(test_quantinuum_emulate
  EXTRA_SOURCES QuantinuumLocalEmulationTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  # BACKEND CONFIG Requires commas since semicolons are special in CMake, 
  # create_simulator_runtime_test will replace them before passing them along 
  BACKEND_CONFIG "quantinuum,emulate,true,credentials,FakeCreds.txt"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

configure_file(
  "QuantinuumStartServerAndTest.sh.in" 
  "${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/QuantinuumStartServerAndTest.sh" 
  @ONLY
)

create_simulator_runtime_test(test_quantinuum
  EXTRA_SOURCES QuantinuumTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  BACKEND_CONFIG "quantinuum,credentials,$ENV{HOME}/FakeCppQuantinuum.config,url,http://localhost:62440"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

add_test(NAME quantinuum-tests 
  COMMAND bash QuantinuumStartServerAndTest.sh 
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantinuum
)
