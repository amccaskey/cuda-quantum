# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

create_simulator_runtime_test(test_anyon_emulate
  EXTRA_SOURCES AnyonLocalEmulationTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  # BACKEND CONFIG Requires commas since semicolons are special in CMake, 
  # create_simulator_runtime_test will replace them before passing them along 
  BACKEND_CONFIG "anyon,emulate,true,credentials,$ENV{HOME}/FakeCppAnyon.config,machine,telegraph-8q"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

create_simulator_runtime_test(test_anyon
  EXTRA_SOURCES AnyonTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  BACKEND_CONFIG "anyon,credentials,$ENV{HOME}/FakeCppAnyon.config,machine,telegraph-8q,url,http://localhost:62446"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)

configure_file("AnyonStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/anyon/AnyonStartServerAndTest.sh" @ONLY)
add_test(NAME anyon-tests COMMAND bash AnyonStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/anyon/)
