# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if ((CUDAQ_BUILD_TESTS OR NOT CUDAQ_DISABLE_TOOLS) AND NOT CUDAQ_ENABLE_QIR_ONLY)
  add_subdirectory(CircuitCheck)
endif()