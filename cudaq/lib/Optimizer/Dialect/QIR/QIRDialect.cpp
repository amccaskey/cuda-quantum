/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/QIR/QIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Generated dialect logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/QIR/QIRDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated enum logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/QIR/QIREnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated type implementations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/QIR/QIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attr implementations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "cudaq/Optimizer/Dialect/QIR/QIRAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated op implementations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QIR/QIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void cudaq::qir::QIRDialect::initialize() {
  registerTypes();
  registerAttrs();
  addOperations<
#define GET_OP_LIST
#include "cudaq/Optimizer/Dialect/QIR/QIROps.cpp.inc"
      >();
}

void cudaq::qir::QIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq/Optimizer/Dialect/QIR/QIRTypes.cpp.inc"
      >();
}

void cudaq::qir::QIRDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq/Optimizer/Dialect/QIR/QIRAttrs.cpp.inc"
      >();
}
