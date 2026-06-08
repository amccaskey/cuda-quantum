/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QIR_TYPES
#define CUDAQ_OPTIMIZER_DIALECT_QIR_TYPES

include "cudaq/Optimizer/Dialect/QIR/QIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Base class for QIR types.
//===----------------------------------------------------------------------===//

class QIRType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QIRDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// QubitType
//===----------------------------------------------------------------------===//

def QubitType : QIRType<"Qubit", "qubit"> {
  let summary = "An opaque handle to a single QIR qubit.";
  let description = [{
    `!qir.qubit` is an opaque reference to a single qubit as specified by the
    QIR standard.  It lowers to an LLVM opaque pointer (`ptr`).

    In the Full profile, qubit values are produced by `qir.alloc_qubit` /
    `qir.array_get`.  In the Base and Adaptive profiles, qubit values come from
    `qir.static_qubit(i64)` with a compile-time-known index.
  }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def ArrayType : QIRType<"Array", "array"> {
  let summary = "An opaque handle to a QIR qubit array.";
  let description = [{
    `!qir.array` represents the QIR `Array*` type — a dynamically-sized
    collection of qubits.  Used exclusively in the Full profile.
    It lowers to an LLVM opaque pointer (`ptr`).
  }];
}

//===----------------------------------------------------------------------===//
// ResultType
//===----------------------------------------------------------------------===//

def ResultType : QIRType<"Result", "result"> {
  let summary = "An opaque handle to a QIR measurement result.";
  let description = [{
    `!qir.result` represents the QIR `Result*` type — the output of a
    measurement operation.  In the Base profile, result values are
    write-only (only `qir.record_*` ops consume them).  In the Adaptive
    profile, `qir.read_result` converts a result to an `i1` value for
    classical feed-forward.  It lowers to an LLVM opaque pointer (`ptr`).
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QIR_TYPES
