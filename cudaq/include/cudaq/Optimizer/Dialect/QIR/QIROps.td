/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QIR_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QIR_OPS

include "cudaq/Optimizer/Dialect/QIR/QIRDialect.td"
include "cudaq/Optimizer/Dialect/QIR/QIRTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Base op class.
//===----------------------------------------------------------------------===//

class QIROp<string mnemonic, list<Trait> traits = []>
    : Op<QIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Qubit management — Full profile
//===----------------------------------------------------------------------===//

def AllocQubitOp : QIROp<"alloc_qubit",
    [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a single qubit (Full profile).";
  let results = (outs QubitType:$qubit);
  let assemblyFormat = "attr-dict `:` type($qubit)";
}

def AllocArrayOp : QIROp<"alloc_array",
    [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a qubit array of the given size (Full profile).";
  let arguments = (ins I64:$size);
  let results = (outs ArrayType:$array);
  let assemblyFormat = "$size attr-dict `:` type($array)";
}

def ReleaseQubitOp : QIROp<"release_qubit",
    [MemoryEffects<[MemFree]>]> {
  let summary = "Release a previously allocated qubit (Full profile).";
  let arguments = (ins QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict";
}

def ReleaseArrayOp : QIROp<"release_array",
    [MemoryEffects<[MemFree]>]> {
  let summary = "Release a previously allocated qubit array (Full profile).";
  let arguments = (ins ArrayType:$array);
  let assemblyFormat = "$array attr-dict";
}

def ArrayGetOp : QIROp<"array_get"> {
  let summary = "Get element at index from a qubit array (Full profile).";
  let arguments = (ins ArrayType:$array, I64:$index);
  let results = (outs QubitType:$qubit);
  let assemblyFormat = "$array `[` $index `]` attr-dict";
}

def ArraySliceOp : QIROp<"array_slice"> {
  let summary = "Slice a qubit array (Full profile).";
  let arguments = (ins ArrayType:$array, I64:$start, I64:$len);
  let results = (outs ArrayType:$slice);
  let assemblyFormat = "$array `[` $start `,` $len `]` attr-dict";
}

//===----------------------------------------------------------------------===//
// Static qubit / result addressing — Base and Adaptive profiles
//===----------------------------------------------------------------------===//

def StaticQubitOp : QIROp<"static_qubit"> {
  let summary = "Reference a statically-indexed qubit (Base/Adaptive profile).";
  let arguments = (ins I64Attr:$index);
  let results = (outs QubitType:$qubit);
  let assemblyFormat = "$index attr-dict";
}

def StaticResultOp : QIROp<"static_result"> {
  let summary = "Reference a statically-indexed result slot (Base/Adaptive profile).";
  let arguments = (ins I64Attr:$index);
  let results = (outs ResultType:$result);
  let assemblyFormat = "$index attr-dict";
}

//===----------------------------------------------------------------------===//
// Single-qubit gates
//===----------------------------------------------------------------------===//

class QIRGate1<string mnemonic>
    : QIROp<mnemonic, [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict";
}

def HOp    : QIRGate1<"h">;
def XOp    : QIRGate1<"x">;
def YOp    : QIRGate1<"y">;
def ZOp    : QIRGate1<"z">;
def SOp    : QIRGate1<"s">;
def SdgOp  : QIRGate1<"sdg">;
def TOp    : QIRGate1<"t">;
def TdgOp  : QIRGate1<"tdg">;

class QIRRotation<string mnemonic>
    : QIROp<mnemonic, [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins F64:$angle, QubitType:$qubit);
  let assemblyFormat = "$angle `,` $qubit attr-dict";
}

def RxOp : QIRRotation<"rx">;
def RyOp : QIRRotation<"ry">;
def RzOp : QIRRotation<"rz">;
def R1Op : QIRRotation<"r1">;

def U3Op : QIROp<"u3", [MemoryEffects<[MemWrite]>]> {
  let summary = "Generic single-qubit rotation U3(θ, φ, λ).";
  let arguments = (ins F64:$theta, F64:$phi, F64:$lam, QubitType:$qubit);
  let assemblyFormat = "$theta `,` $phi `,` $lam `,` $qubit attr-dict";
}

//===----------------------------------------------------------------------===//
// Two-qubit gates
//===----------------------------------------------------------------------===//

class QIRGate2<string mnemonic>
    : QIROp<mnemonic, [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins QubitType:$ctrl, QubitType:$target);
  let assemblyFormat = "$ctrl `,` $target attr-dict";
}

def CNOTOp : QIRGate2<"cnot">;
def CZOp   : QIRGate2<"cz">;
def SWAPOp : QIRGate2<"swap">;

def CCXOp : QIROp<"ccx", [MemoryEffects<[MemWrite]>]> {
  let summary = "Toffoli (CCX) gate.";
  let arguments = (ins QubitType:$ctrl0, QubitType:$ctrl1, QubitType:$target);
  let assemblyFormat = "$ctrl0 `,` $ctrl1 `,` $target attr-dict";
}

//===----------------------------------------------------------------------===//
// Measurement and reset
//===----------------------------------------------------------------------===//

def MeasureOp : QIROp<"measure",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Measure a qubit, returning an opaque result handle.";
  let arguments = (ins QubitType:$qubit);
  let results = (outs ResultType:$result);
  let assemblyFormat = "$qubit attr-dict";
}

def MeasureNamedOp : QIROp<"measure_named",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Measure a qubit into a pre-allocated result slot (Base/Adaptive).";
  let arguments = (ins QubitType:$qubit, ResultType:$slot);
  let assemblyFormat = "$qubit `->` $slot attr-dict";
}

def ReadResultOp : QIROp<"read_result"> {
  let summary = "Convert a result handle to an i1 value (Adaptive profile).";
  let arguments = (ins ResultType:$result);
  let results = (outs I1:$value);
  let assemblyFormat = "$result attr-dict";
}

def ResetOp : QIROp<"reset",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Reset a qubit to |0⟩.";
  let arguments = (ins QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict";
}

//===----------------------------------------------------------------------===//
// Output recording — Base/Adaptive profiles
//===----------------------------------------------------------------------===//

def RecordBoolOp : QIROp<"record_bool"> {
  let summary = "Record a boolean measurement result with an optional label.";
  let arguments = (ins ResultType:$result, OptionalAttr<StrAttr>:$label);
  let assemblyFormat = "$result (`,` $label^)? attr-dict";
}

def RecordIntOp : QIROp<"record_int"> {
  let summary = "Record an integer output with an optional label.";
  let arguments = (ins I64:$value, OptionalAttr<StrAttr>:$label);
  let assemblyFormat = "$value (`,` $label^)? attr-dict";
}

def RecordDoubleOp : QIROp<"record_double"> {
  let summary = "Record a double output with an optional label.";
  let arguments = (ins F64:$value, OptionalAttr<StrAttr>:$label);
  let assemblyFormat = "$value (`,` $label^)? attr-dict";
}

def RecordArrayOutputOp : QIROp<"record_array_output"> {
  let summary = "Record an entire qubit-array's result outputs.";
  let arguments = (ins ArrayType:$array);
  let assemblyFormat = "$array attr-dict";
}

//===----------------------------------------------------------------------===//
// QEC hooks — detector and observable metadata
//===----------------------------------------------------------------------===//

def DetectorOp : QIROp<"detector"> {
  let summary = "Declare a QEC detector (check that XOR of results is deterministic).";
  let arguments = (ins Variadic<ResultType>:$results);
  let assemblyFormat = "($results^)? attr-dict";
}

def ObservableOp : QIROp<"observable"> {
  let summary = "Declare a logical observable in terms of result parity.";
  let arguments = (ins Variadic<ResultType>:$results);
  let assemblyFormat = "($results^)? attr-dict";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QIR_OPS
