/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QIR_IR_QIR
#define CUDAQ_OPTIMIZER_DIALECT_QIR_IR_QIR

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def QIRDialect : Dialect {
  let name = "qir";
  let summary = "QIR MLIR dialect for NVIDIA CUDA-Q.";
  let description = [{
    The `qir` dialect provides a first-class MLIR representation of the
    Quantum Intermediate Representation (QIR) specification.  It models the
    three standard QIR execution profiles:

    * **Full** – dynamic qubit allocation, unrestricted classical control flow.
    * **Adaptive** – static qubit/result resources, mid-circuit feed-forward
      via `qir.read_result`.
    * **Base** – static qubit/result resources, measurements deferred to the
      end of the circuit, no classical control flow from measurement results.

    The dialect uses standard MLIR control-flow dialects (`scf`, `cf`, `func`)
    for branching; no custom control-flow ops are defined here.  The final
    `ConvertQIRToLLVM` pass lowers every `qir.*` op to `llvm.call` intrinsics
    matching the `__quantum__qis__*` / `__quantum__rt__*` QIR runtime ABI.
  }];
  let cppNamespace = "cudaq::qir";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttrs();
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QIR_IR_QIR
