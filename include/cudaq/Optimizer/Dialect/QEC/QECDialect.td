/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC
#define CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC

include "mlir/IR/OpBase.td"

def QECDialect : Dialect {
  let name = "qec";
  let summary = "QEC-specific operations for CUDA-Q.";
  let description = [{
    The `qec` dialect carries QEC-specific declarations that need to survive
    compilation and reach a simulator or a DEM generator. Today it holds the
    detector and logical-observable declarations emitted from the user-facing
    `cudaq::detector` / `cudaq::logical_observable` APIs.
  }];
  let cppNamespace = "::cudaq::qec";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC
