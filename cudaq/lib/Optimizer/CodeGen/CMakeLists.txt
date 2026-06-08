# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-braced-scalar-init")
endif()

add_cudaq_library(QIRToLLVM
  QIRToLLVM.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  QIRDialectIncGen
  QIROpsIncGen
  QIRTypesIncGen
  QIRAttrsIncGen
  OptCodeGenPassIncGen

  LINK_LIBS PUBLIC
  QIRDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRSCFToControlFlow
  MLIRTransforms
)

if(NOT CUDAQ_ENABLE_QIR_ONLY)
add_cudaq_library(OptCodeGen
  CCToLLVM.cpp
  CodeGenDialect.cpp
  CodeGenOps.cpp
  CodeGenTypes.cpp
  ConvertCCToLLVM.cpp
  ConvertToExecMgr.cpp
  ConvertToQIRProfile.cpp
  ConvertToQIR.cpp
  ConvertToQIRAPI.cpp
  DelayMeasurements.cpp
  OptUtils.cpp
  Passes.cpp
  Pipelines.cpp
  QirInsertArrayRecord.cpp
  QuakeToCodegen.cpp
  QuakeToExecMgr.cpp
  QuakeToLLVM.cpp
  RemoveMeasurements.cpp
  ReturnToOutputLog.cpp
  TranslateToIQMJson.cpp
  TranslateToOpenQASM.cpp
  WireSetsToProfileQIR.cpp

  DEPENDS
    CCDialect
    CodeGenDialectIncGen
    CodeGenOpsIncGen
    CodeGenTypesIncGen
    OptCodeGenPassIncGen
    OptTransformsPassIncGen
    QECDialect
    QIRDialect
    QuakeDialect
    cudaq-qir-verifier

  LINK_LIBS PUBLIC
    CCDialect
    OptimBuilder
    QECDialect
    QIRDialect
    QIRToLLVM
    QuakeDialect

    MLIRArithTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIROpenACCDialect
    MLIROpenMPDialect
    MLIRTransforms

    # Conversions
    MLIRArithToLLVM
    MLIRComplexToLibm
    MLIRComplexToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMathToFuncs
    MLIRMathToLLVM

    # Translation
    MLIRTargetLLVMIRExport
)
endif() # NOT CUDAQ_ENABLE_QIR_ONLY
