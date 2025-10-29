# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CPU-based simulators using QPP

# Add CPU simulator sources
target_sources(cudaq-experimental PRIVATE
    state_vector.cpp
    density_matrix.cpp
)

# Link against QPP (which already provides Eigen3 and OpenMP)
target_link_libraries(cudaq-experimental PRIVATE 
    libqpp
)

# Install public headers
install(FILES 
    state_vector.h
    density_matrix.h
    all.h
  DESTINATION include/cudaq/qpus/simulator/cpu
)

