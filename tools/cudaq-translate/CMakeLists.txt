# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(cudaq-translate cudaq-translate.cpp)

llvm_update_compile_flags(cudaq-translate)
target_link_libraries(cudaq-translate
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIROptLib
  MLIRExecutionEngine
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRConvertToEmitC
  MLIRFuncToEmitC
  MLIRFuncTransforms
  MLIRTensorAllExtensions
  MLIRMPIToLLVM
  MLIRFuncAllExtensions
  MLIROpenMPToLLVM
  ${dialect_libs}
  ${translation_libs}

  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
)

mlir_check_all_link_libraries(cudaq-translate)

install(TARGETS cudaq-translate DESTINATION bin)
