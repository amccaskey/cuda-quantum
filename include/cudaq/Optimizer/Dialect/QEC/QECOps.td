/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QEC_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.td"
include "cudaq/Optimizer/Dialect/QEC/QECDialect.td"
include "cudaq/Optimizer/Dialect/QEC/QECTypes.td"

class QECOp<string mnemonic, list<Trait> traits = []> :
    Op<QECDialect, mnemonic, traits>;

def qec_DetectorOp : QECOp<"detector", []> {
  let summary = "Declare a detector over measurement results.";
  let description = [{
    Declares a parity constraint over one or more measurement handles. Under
    noise-free execution the XOR of the referenced measurements is
    deterministic. This op has no side effects and can be erased by backends
    that do not support QEC declarations.
  }];
  let arguments = (ins Variadic<MeasureType>:$measurements);
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` functional-type(operands, results)
  }];
}

def qec_DetectorVecOp : QECOp<"detector_vec", []> {
  let summary = "Declare a detector over a vector of measurement results.";
  let description = [{
    Vector-form overload of `qec.detector`: the parity of all measurement
    handles in the vector forms a single detector.
  }];
  let arguments = (ins
    StdvecOf<[MeasureType]>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` functional-type(operands, results)
  }];
}

def qec_LogicalObservableOp : QECOp<"logical_observable", []> {
  let summary = "Declare a logical observable over measurement results.";
  let description = [{
    Declares a logical observable as a binary sum of measurement handles. The
    referenced measurements, XORed together, yield the outcome of measuring a
    logical operator for any logical state.
  }];
  let arguments = (ins
    StdvecOf<[MeasureType]>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
