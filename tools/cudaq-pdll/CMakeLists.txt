# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

# MLIR is not installing this target for some reason?
add_library(MLIRPDLLParser STATIC IMPORTED)
set_target_properties(MLIRPDLLParser PROPERTIES
  IMPORTED_LOCATION "/opt/latest_mlir/lib/libMLIRPDLLParser.a"
  IMPORTED_SONAME "libMLIRPDLLParser.a")

# Order matters here
set(LIBS
  MLIRIR
  MLIRPDLLAST
  MLIRPDLLCodeGen
  MLIRPDLLODS
  MLIRPDLLParser
  MLIRSupport
  MLIRTableGen
  MLIRBytecodeWriter
  LLVMSupport
  LLVMTableGen
  )

add_tablegen(cudaq-pdll CUDAQ_PDLL
  DESTINATION bin
  EXPORT CUDAQ
  cudaq-pdll.cpp

  DEPENDS
  ${LIBS}
  )

set_target_properties(cudaq-pdll PROPERTIES FOLDER "Tablegenning")
target_link_libraries(cudaq-pdll PRIVATE ${LIBS})

mlir_check_all_link_libraries(cudaq-pdll)
