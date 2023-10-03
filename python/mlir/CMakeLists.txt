# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_cudaq`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_cudaq.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
   QuakeExtension.cpp
  EMBED_CAPI_LINK_LIBS
  CUDAQuantumMLIRCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CUDAQuantumPythonCAPI
  INSTALL_COMPONENT CUDAQuantumPythonModules
  INSTALL_DESTINATION python_packages/cudaq/mlir_cudaq/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/cudaq/mlir_cudaq/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(CUDAQuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/cudaq/mlir_cudaq"
  INSTALL_PREFIX "python_packages/cudaq/mlir_cudaq"
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    CUDAQuantumPythonCAPI
  )