# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
add_subdirectory(Optimizer)
add_subdirectory(Support)

# Install Frontend headers that are transitively included by Optimizer headers
# (e.g. InlinerInterface.h includes cudaq/Frontend/nvqpp/AttributeNames.h).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Frontend/
        DESTINATION include/cudaq/Frontend
        FILES_MATCHING PATTERN "*.h")
