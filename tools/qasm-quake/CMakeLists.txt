# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_llvm_executable(qasm-quake qasm-quake.cpp)
llvm_update_compile_flags(qasm-quake)

target_link_libraries(qasm-quake
  PRIVATE
  OpenQASMParser
  QuakeDialect
  MLIRMemRefDialect
)

mlir_check_all_link_libraries(qasm-quake)

install(TARGETS qasm-quake DESTINATION bin)
