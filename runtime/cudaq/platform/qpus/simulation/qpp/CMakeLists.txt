# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(${LIBRARY_NAME} PRIVATE QppCircuitSimulator.cpp QppDMCircuitSimulator.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE libqpp)

add_target_config(qpp-cpu)
add_target_config(density-matrix-cpu)
