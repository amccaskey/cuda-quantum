/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/CodeGenDialect.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/QEC/QECDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/CAPI/Support.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quake, quake, cudaq::quake::QuakeDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QEC, qec, cudaq::qec::QECDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CC, cc, cudaq::cc::CCDialect)

extern "C" void cudaqRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<cudaq::codegen::CodeGenDialect, cudaq::quake::QuakeDialect,
                  cudaq::qec::QECDialect, cudaq::cc::CCDialect>();
  mlir::registerAllDialects(registry);
  auto *mlirContext = unwrap(context);
  mlirContext->appendDialectRegistry(registry);
  mlirContext->loadAllAvailableDialects();
}

extern "C" void cudaqRegisterAllPassesAndPipelines() {
  cudaq::registerCudaqPassesAndPipelines();
}

static llvm::SmallVector<mlir::Type> unwrapTypeArray(intptr_t numTypes,
                                                     const MlirType *types) {
  llvm::SmallVector<mlir::Type> result;
  result.reserve(numTypes);
  for (intptr_t i = 0; i < numTypes; ++i)
    result.push_back(unwrap(types[i]));
  return result;
}

template <typename TypeT>
static MlirTypeID getTypeID() {
  return wrap(mlir::TypeID::get<TypeT>());
}

extern "C" MlirTypeID cudaqQuakeRefTypeGetTypeID() {
  return getTypeID<cudaq::quake::RefType>();
}
extern "C" bool cudaqTypeIsAQuakeRefType(MlirType type) {
  return mlir::isa<cudaq::quake::RefType>(unwrap(type));
}
extern "C" MlirType cudaqQuakeRefTypeGet(MlirContext context) {
  return wrap(cudaq::quake::RefType::get(unwrap(context)));
}

extern "C" MlirTypeID cudaqQuakeMeasureTypeGetTypeID() {
  return getTypeID<cudaq::quake::MeasureType>();
}
extern "C" bool cudaqTypeIsAQuakeMeasureType(MlirType type) {
  return mlir::isa<cudaq::quake::MeasureType>(unwrap(type));
}
extern "C" MlirType cudaqQuakeMeasureTypeGet(MlirContext context) {
  return wrap(cudaq::quake::MeasureType::get(unwrap(context)));
}

extern "C" MlirTypeID cudaqQuakeVeqTypeGetTypeID() {
  return getTypeID<cudaq::quake::VeqType>();
}
extern "C" bool cudaqTypeIsAQuakeVeqType(MlirType type) {
  return mlir::isa<cudaq::quake::VeqType>(unwrap(type));
}
extern "C" MlirType cudaqQuakeVeqTypeGet(MlirContext context, uint64_t size) {
  return wrap(cudaq::quake::VeqType::get(unwrap(context), size));
}
extern "C" bool cudaqQuakeVeqTypeHasSpecifiedSize(MlirType type) {
  return mlir::cast<cudaq::quake::VeqType>(unwrap(type)).hasSpecifiedSize();
}
extern "C" uint64_t cudaqQuakeVeqTypeGetSize(MlirType type) {
  return mlir::cast<cudaq::quake::VeqType>(unwrap(type)).getSize();
}

extern "C" MlirTypeID cudaqQuakeStruqTypeGetTypeID() {
  return getTypeID<cudaq::quake::StruqType>();
}
extern "C" bool cudaqTypeIsAQuakeStruqType(MlirType type) {
  return mlir::isa<cudaq::quake::StruqType>(unwrap(type));
}
extern "C" MlirType cudaqQuakeStruqTypeGet(MlirContext context,
                                            intptr_t numMembers,
                                            const MlirType *members) {
  auto memberTypes = unwrapTypeArray(numMembers, members);
  return wrap(cudaq::quake::StruqType::get(unwrap(context), memberTypes));
}
extern "C" MlirType cudaqQuakeStruqTypeGetNamed(MlirContext context,
                                                 MlirStringRef name,
                                                 intptr_t numMembers,
                                                 const MlirType *members) {
  auto memberTypes = unwrapTypeArray(numMembers, members);
  return wrap(cudaq::quake::StruqType::get(unwrap(context), unwrap(name),
                                           memberTypes));
}
extern "C" intptr_t cudaqQuakeStruqTypeGetNumMembers(MlirType type) {
  return mlir::cast<cudaq::quake::StruqType>(unwrap(type)).getMembers().size();
}
extern "C" MlirType cudaqQuakeStruqTypeGetMember(MlirType type, intptr_t pos) {
  return wrap(mlir::cast<cudaq::quake::StruqType>(unwrap(type)).getMembers()[pos]);
}
extern "C" MlirStringRef cudaqQuakeStruqTypeGetName(MlirType type) {
  return wrap(mlir::cast<cudaq::quake::StruqType>(unwrap(type)).getName().getValue());
}

extern "C" MlirTypeID cudaqQuakeStateTypeGetTypeID() {
  return getTypeID<cudaq::quake::StateType>();
}
extern "C" bool cudaqTypeIsAQuakeStateType(MlirType type) {
  return mlir::isa<cudaq::quake::StateType>(unwrap(type));
}
extern "C" MlirType cudaqQuakeStateTypeGet(MlirContext context) {
  return wrap(cudaq::quake::StateType::get(unwrap(context)));
}

extern "C" MlirTypeID cudaqCCCharspanTypeGetTypeID() {
  return getTypeID<cudaq::cc::CharspanType>();
}
extern "C" bool cudaqTypeIsACCCharspanType(MlirType type) {
  return mlir::isa<cudaq::cc::CharspanType>(unwrap(type));
}
extern "C" MlirType cudaqCCCharspanTypeGet(MlirContext context) {
  return wrap(cudaq::cc::CharspanType::get(unwrap(context)));
}

extern "C" MlirTypeID cudaqCCMeasureHandleTypeGetTypeID() {
  return getTypeID<cudaq::cc::MeasureHandleType>();
}
extern "C" bool cudaqTypeIsACCMeasureHandleType(MlirType type) {
  return mlir::isa<cudaq::cc::MeasureHandleType>(unwrap(type));
}
extern "C" MlirType cudaqCCMeasureHandleTypeGet(MlirContext context) {
  return wrap(cudaq::cc::MeasureHandleType::get(unwrap(context)));
}

extern "C" MlirTypeID cudaqCCPointerTypeGetTypeID() {
  return getTypeID<cudaq::cc::PointerType>();
}
extern "C" bool cudaqTypeIsACCPointerType(MlirType type) {
  return mlir::isa<cudaq::cc::PointerType>(unwrap(type));
}
extern "C" MlirType cudaqCCPointerTypeGet(MlirContext context,
                                           MlirType elementType) {
  return wrap(cudaq::cc::PointerType::get(unwrap(context), unwrap(elementType)));
}
extern "C" MlirType cudaqCCPointerTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<cudaq::cc::PointerType>(unwrap(type)).getElementType());
}

extern "C" MlirTypeID cudaqCCArrayTypeGetTypeID() {
  return getTypeID<cudaq::cc::ArrayType>();
}
extern "C" bool cudaqTypeIsACCArrayType(MlirType type) {
  return mlir::isa<cudaq::cc::ArrayType>(unwrap(type));
}
extern "C" MlirType cudaqCCArrayTypeGet(MlirContext context,
                                         MlirType elementType, int64_t size) {
  return wrap(cudaq::cc::ArrayType::get(unwrap(context), unwrap(elementType),
                                        size));
}
extern "C" MlirType cudaqCCArrayTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<cudaq::cc::ArrayType>(unwrap(type)).getElementType());
}

extern "C" MlirTypeID cudaqCCStructTypeGetTypeID() {
  return getTypeID<cudaq::cc::StructType>();
}
extern "C" bool cudaqTypeIsACCStructType(MlirType type) {
  return mlir::isa<cudaq::cc::StructType>(unwrap(type));
}
extern "C" MlirType cudaqCCStructTypeGet(MlirContext context,
                                          intptr_t numMembers,
                                          const MlirType *members) {
  auto memberTypes = unwrapTypeArray(numMembers, members);
  return wrap(cudaq::cc::StructType::get(unwrap(context), memberTypes));
}
extern "C" MlirType cudaqCCStructTypeGetNamed(MlirContext context,
                                               MlirStringRef name,
                                               intptr_t numMembers,
                                               const MlirType *members) {
  auto memberTypes = unwrapTypeArray(numMembers, members);
  return wrap(cudaq::cc::StructType::get(unwrap(context), unwrap(name),
                                         memberTypes));
}
extern "C" intptr_t cudaqCCStructTypeGetNumMembers(MlirType type) {
  return mlir::cast<cudaq::cc::StructType>(unwrap(type)).getMembers().size();
}
extern "C" MlirType cudaqCCStructTypeGetMember(MlirType type, intptr_t pos) {
  return wrap(mlir::cast<cudaq::cc::StructType>(unwrap(type)).getMembers()[pos]);
}
extern "C" MlirStringRef cudaqCCStructTypeGetName(MlirType type) {
  return wrap(mlir::cast<cudaq::cc::StructType>(unwrap(type)).getName().getValue());
}

extern "C" MlirTypeID cudaqCCCallableTypeGetTypeID() {
  return getTypeID<cudaq::cc::CallableType>();
}
extern "C" bool cudaqTypeIsACCCallableType(MlirType type) {
  return mlir::isa<cudaq::cc::CallableType>(unwrap(type));
}
extern "C" MlirType cudaqCCCallableTypeGet(MlirContext context,
                                            MlirType functionType) {
  return wrap(cudaq::cc::CallableType::get(
      unwrap(context), mlir::cast<mlir::FunctionType>(unwrap(functionType))));
}
extern "C" MlirType cudaqCCCallableTypeGetFunctionType(MlirType type) {
  return wrap(mlir::cast<cudaq::cc::CallableType>(unwrap(type)).getSignature());
}

extern "C" MlirTypeID cudaqCCStdvecTypeGetTypeID() {
  return getTypeID<cudaq::cc::StdvecType>();
}
extern "C" bool cudaqTypeIsACCStdvecType(MlirType type) {
  return mlir::isa<cudaq::cc::StdvecType>(unwrap(type));
}
extern "C" MlirType cudaqCCStdvecTypeGet(MlirContext context,
                                          MlirType elementType) {
  return wrap(cudaq::cc::StdvecType::get(unwrap(context), unwrap(elementType)));
}
extern "C" MlirType cudaqCCStdvecTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<cudaq::cc::StdvecType>(unwrap(type)).getElementType());
}
