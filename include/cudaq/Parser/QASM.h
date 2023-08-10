/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace cudaq {

mlir::OwningOpRef<mlir::ModuleOp> parseQASMCode(mlir::MLIRContext *context,
                                            const llvm::StringRef &code);

mlir::OwningOpRef<mlir::ModuleOp> parseQASMFile(mlir::MLIRContext *context,
                                            const llvm::SourceMgr &SourceMgr);

} // namespace cudaq::qasm
