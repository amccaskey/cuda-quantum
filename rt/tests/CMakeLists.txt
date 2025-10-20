# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
add_executable(basic_tester basic_tester.cpp)
target_link_libraries(basic_tester PRIVATE GTest::gtest_main cudaq-rt)
target_compile_options(basic_tester PRIVATE -DCUDAQ_TEST_COMPILER_CONFIG)
target_link_options(basic_tester PRIVATE -Wl,--export-dynamic-symbol=initialize_qpu)
gtest_discover_tests(basic_tester)

add_executable(basic_tester_no_config_no_export_symbol basic_tester.cpp)
target_link_libraries(basic_tester_no_config_no_export_symbol PRIVATE GTest::gtest_main cudaq-rt)
gtest_discover_tests(basic_tester_no_config_no_export_symbol)
    