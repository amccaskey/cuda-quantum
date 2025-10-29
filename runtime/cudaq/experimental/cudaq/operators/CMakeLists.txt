# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_link_options(${LIBRARY_NAME} PRIVATE -Wl,--no-as-needed)

set(CUDAQ_OPS_SRC
  callback.cpp
  scalar_op.cpp
  spin_op.cpp
  boson_op.cpp
  fermion_op.cpp
  matrix_op.cpp
  product_op.cpp
  sum_op.cpp
  rydberg_hamiltonian.cpp
  evaluation.cpp
  handler.cpp
  helpers.cpp
  super_op.cpp
)

target_sources(${LIBRARY_NAME} PRIVATE ${CUDAQ_OPS_SRC})
target_compile_definitions(${LIBRARY_NAME} PRIVATE -DCUDAQ_INSTANTIATE_TEMPLATES)

