/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace cudaq {

mlir::LogicalResult translateToCUDAQPython(mlir::Operation *op,
                                        llvm::raw_ostream &os);

} // namespace cudaq
