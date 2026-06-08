/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QIR_ATTRS
#define CUDAQ_OPTIMIZER_DIALECT_QIR_ATTRS

include "cudaq/Optimizer/Dialect/QIR/QIRDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// QIR Profile enum
//===----------------------------------------------------------------------===//

def QIRProfile_Full     : I32EnumAttrCase<"Full",     0, "full">;
def QIRProfile_Base     : I32EnumAttrCase<"Base",     1, "base">;
def QIRProfile_Adaptive : I32EnumAttrCase<"Adaptive", 2, "adaptive">;

def QIRProfileEnum : I32EnumAttr<"QIRProfile",
    "QIR execution profile",
    [QIRProfile_Full, QIRProfile_Base, QIRProfile_Adaptive]> {
  let cppNamespace = "cudaq::qir";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ProfileAttr
//===----------------------------------------------------------------------===//

def ProfileAttr : AttrDef<QIRDialect, "Profile"> {
  let mnemonic = "profile";
  let summary = "Marks a function's target QIR execution profile.";
  let parameters = (ins
    "QIRProfile":$profile
  );
  let assemblyFormat = "`<` $profile `>`";
}

//===----------------------------------------------------------------------===//
// RequiredQubitsAttr / RequiredResultsAttr (Base/Adaptive)
//===----------------------------------------------------------------------===//

def RequiredQubitsAttr : AttrDef<QIRDialect, "RequiredQubits"> {
  let mnemonic = "required_qubits";
  let summary = "Static qubit count annotation for Base/Adaptive profiles.";
  let parameters = (ins "uint64_t":$count);
  let assemblyFormat = "`<` $count `>`";
}

def RequiredResultsAttr : AttrDef<QIRDialect, "RequiredResults"> {
  let mnemonic = "required_results";
  let summary = "Static result count annotation for Base/Adaptive profiles.";
  let parameters = (ins "uint64_t":$count);
  let assemblyFormat = "`<` $count `>`";
}

//===----------------------------------------------------------------------===//
// QIRVersionAttr
//===----------------------------------------------------------------------===//

def QIRVersionAttr : AttrDef<QIRDialect, "QIRVersion"> {
  let mnemonic = "qir_version";
  let summary = "QIR spec version (major.minor).";
  let parameters = (ins "uint32_t":$major, "uint32_t":$minor);
  let assemblyFormat = "`<` $major `,` $minor `>`";
}

//===----------------------------------------------------------------------===//
// OutputNameAttr
//===----------------------------------------------------------------------===//

def OutputNameAttr : AttrDef<QIRDialect, "OutputName"> {
  let mnemonic = "output_name";
  let summary = "Label for a measurement output record.";
  let parameters = (ins "mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QIR_ATTRS
