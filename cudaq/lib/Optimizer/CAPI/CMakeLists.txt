# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_public_c_api_library(CUDAQuantumMLIRCAPI
  Dialects.cpp

  DEPENDS
  CodeGenDialectIncGen
  QuakeDialectIncGen
  QECOpsIncGen

  LINK_LIBS PRIVATE
  OptCodeGen
  OptTransforms
  QuakeDialect
  CCDialect
  QECDialect
  MLIRRegisterAllDialects
)

set(_cudaq_mlir_capi_embedded_libs
  OptCodeGen
  OptTransforms
  OptimBuilder
  QuakeDialect
  CCDialect
  QECDialect
)

foreach(_cudaq_mlir_capi_embedded_lib IN LISTS _cudaq_mlir_capi_embedded_libs)
  if(TARGET obj.${_cudaq_mlir_capi_embedded_lib})
    set_property(TARGET CUDAQuantumMLIRCAPI APPEND PROPERTY
      MLIR_AGGREGATE_OBJECTS
      "$<TARGET_OBJECTS:obj.${_cudaq_mlir_capi_embedded_lib}>")
    add_dependencies(CUDAQuantumMLIRCAPI
      obj.${_cudaq_mlir_capi_embedded_lib})
  endif()
endforeach()
