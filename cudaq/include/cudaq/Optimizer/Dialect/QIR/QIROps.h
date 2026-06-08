/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/QIR/QIRAttrs.h"
#include "cudaq/Optimizer/Dialect/QIR/QIRDialect.h"
#include "cudaq/Optimizer/Dialect/QIR/QIRTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QIR/QIROps.h.inc"
