# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect(QIR qir)

# Attributes TableGen
set(LLVM_TARGET_DEFINITIONS QIRAttrs.td)
mlir_tablegen(QIRAttrs.h.inc  -gen-attrdef-decls -attrdefs-dialect=qir)
mlir_tablegen(QIRAttrs.cpp.inc -gen-attrdef-defs  -attrdefs-dialect=qir)
mlir_tablegen(QIREnums.h.inc  -gen-enum-decls)
mlir_tablegen(QIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(QIRAttrsIncGen)
add_dependencies(cudaq-headers QIRAttrsIncGen)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        DESTINATION include/cudaq/Optimizer/Dialect/QIR
        FILES_MATCHING
          PATTERN "*.h.inc"
          PATTERN "*.cpp.inc"
          PATTERN "CMakeFiles" EXCLUDE)

# Install source headers alongside the generated .inc files.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION include/cudaq/Optimizer/Dialect/QIR
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.td"
          PATTERN "CMakeFiles" EXCLUDE)
