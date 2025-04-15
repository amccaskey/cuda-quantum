# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(add_driver_test SHARED add_driver_test.cpp)

add_executable(shmem_driver_tester shmem_driver_tester.cpp)

target_link_libraries(shmem_driver_tester
  PRIVATE
    cudaq-platform-default
    cudaq-driver
    cudaq-builder
    add_driver_test
    gtest_main
)

gtest_discover_tests(shmem_driver_tester)


add_executable(shmem_ethernet_driver_tester shmem_host_ethernet_channel_tester.cpp)

target_link_libraries(shmem_ethernet_driver_tester
  PRIVATE
    cudaq-platform-default
    cudaq-driver
    cudaq-builder
    cudaq-ethernet-channel
    add_driver_test
    gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(shmem_ethernet_driver_tester PRIVATE -Wl,--no-as-needed)
endif()
gtest_discover_tests(shmem_ethernet_driver_tester)


add_executable(ethernet_host_shmem_channel_driver_tester ethernet_host_shmem_channel_tester.cpp)

target_link_libraries(ethernet_host_shmem_channel_driver_tester
  PRIVATE
    cudaq-platform-default
    cudaq-driver
    cudaq-builder
    cudaq-ethernet-dispatcher
    add_driver_test
    gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(ethernet_host_shmem_channel_driver_tester PRIVATE -Wl,--no-as-needed)
endif()
gtest_discover_tests(ethernet_host_shmem_channel_driver_tester)


add_executable(shmem_full_pass_ptrs_tester shmem_full_pass_ptrs_tester.cpp)

target_link_libraries(shmem_full_pass_ptrs_tester
  PRIVATE
    cudaq-platform-default
    cudaq-driver
    cudaq-builder
    add_driver_test
    gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(shmem_full_pass_ptrs_tester PRIVATE -Wl,--no-as-needed)
endif()
gtest_discover_tests(shmem_full_pass_ptrs_tester)


add_executable(ethernet_host_shmem_channel_pass_ptrs_tester ethernet_host_shmem_channel_pass_ptrs_tester.cpp)

target_link_libraries(ethernet_host_shmem_channel_pass_ptrs_tester
  PRIVATE
    cudaq-platform-default
    cudaq-driver
    cudaq-builder
    cudaq-ethernet-dispatcher
    add_driver_test
    gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(ethernet_host_shmem_channel_pass_ptrs_tester PRIVATE -Wl,--no-as-needed)
endif()
gtest_discover_tests(ethernet_host_shmem_channel_pass_ptrs_tester)

if (CUDA_FOUND) 
  add_library(test_cuda_kernels OBJECT test_cuda.cu) 
  set_property(TARGET test_cuda_kernels PROPERTY CUDA_FATBIN_COMPILATION ON)
  
  add_executable(shmem_cuda_channel_tester shmem_cuda_channel_tester.cpp)

  target_link_libraries(shmem_cuda_channel_tester
    PRIVATE
      cudaq-platform-default
      cudaq-driver
      cudaq-builder
      cudaq-cuda-channel
      add_driver_test
      gtest_main
  )
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(shmem_cuda_channel_tester PRIVATE -Wl,--no-as-needed)
  endif()
  gtest_discover_tests(shmem_cuda_channel_tester)

  add_executable(ethernet_cuda_channel_tester ethernet_host_cuda_channel_tester.cpp)

  target_link_libraries(ethernet_cuda_channel_tester
    PRIVATE
      cudaq-platform-default
      cudaq-driver
      cudaq-ethernet-dispatcher
      cudaq-builder
      cudaq-cuda-channel
      add_driver_test
      gtest_main
  )
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(ethernet_cuda_channel_tester PRIVATE -Wl,--no-as-needed)
  endif()
  gtest_discover_tests(ethernet_cuda_channel_tester)
endif() 