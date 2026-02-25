# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Transforms)

# Install source headers (.h) and tablegen definitions (.td) from the source
# tree. These are needed by external projects that link against the MLIR
# dialect libraries (QuakeDialect, CCDialect, OptCodeGen, OptTransforms, etc.).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION include/cudaq/Optimizer
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.td"
          PATTERN "CMakeLists.txt" EXCLUDE)

# Install tablegen-generated headers (.inc) from the build tree. The source
# headers #include these for dialect/op/type declarations and definitions.
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        DESTINATION include/cudaq/Optimizer
        FILES_MATCHING
          PATTERN "*.inc"
          PATTERN "CMakeFiles" EXCLUDE)
