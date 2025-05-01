# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

create_simulator_runtime_test(test_infleqtion
  EXTRA_SOURCES InfleqtionTester.cpp 
  DISABLE_INTEGRATION TRUE 
  QPU_LIB cudaq-rest-qpu 
  EXTRA_LIBS 
    cudaq-mlir-runtime 
    cudaq-qpu-simulated
  HAS_MAIN_CPP TRUE
  ENV_VARS "SUPERSTAQ_API_KEY=00000000000000000000000000000000"
  BACKEND_CONFIG "infleqtion,url,http://localhost:62447,token,0"
  TARGET_OPTIONS "--target-option qpu:remote_rest"
)


configure_file("InfleqtionStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/InfleqtionStartServerAndTest.sh" @ONLY)
add_test(NAME infleqtion-tests COMMAND bash InfleqtionStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/)
