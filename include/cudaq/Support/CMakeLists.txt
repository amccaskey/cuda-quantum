# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Install all public headers (needed by Optimizer dialect headers and
# downstream projects). Also install the generated Version.h from the
# build tree.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION include/cudaq/Support
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        DESTINATION include/cudaq/Support
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "CMakeFiles" EXCLUDE)
