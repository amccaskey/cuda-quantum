# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(CC)
add_subdirectory(Quake)

file (GLOB COMMON_TBLGEN Common/*.td)
install(FILES ${COMMON_TBLGEN} DESTINATION include/cudaq/Optimizer/Dialect/Common)