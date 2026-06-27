/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quake, quake);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QEC, qec);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CC, cc);

// Register Quake, CC, and all upstream MLIR dialects into `context`.
MLIR_CAPI_EXPORTED void cudaqRegisterAllDialects(MlirContext context);

// Register CUDA-Q passes and pipelines into the active MLIR process registry.
MLIR_CAPI_EXPORTED void cudaqRegisterAllPassesAndPipelines(void);

MLIR_CAPI_EXPORTED MlirTypeID cudaqQuakeRefTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsAQuakeRefType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeRefTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID cudaqQuakeMeasureTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsAQuakeMeasureType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeMeasureTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID cudaqQuakeVeqTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsAQuakeVeqType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeVeqTypeGet(MlirContext context,
                                                  uint64_t size);
MLIR_CAPI_EXPORTED bool cudaqQuakeVeqTypeHasSpecifiedSize(MlirType type);
MLIR_CAPI_EXPORTED uint64_t cudaqQuakeVeqTypeGetSize(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqQuakeStruqTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsAQuakeStruqType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeStruqTypeGet(MlirContext context,
                                                   intptr_t numMembers,
                                                   const MlirType *members);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeStruqTypeGetNamed(
    MlirContext context, MlirStringRef name, intptr_t numMembers,
    const MlirType *members);
MLIR_CAPI_EXPORTED intptr_t cudaqQuakeStruqTypeGetNumMembers(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeStruqTypeGetMember(MlirType type,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED MlirStringRef cudaqQuakeStruqTypeGetName(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqQuakeStateTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsAQuakeStateType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqQuakeStateTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCCharspanTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCCharspanType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCCharspanTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCMeasureHandleTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCMeasureHandleType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCMeasureHandleTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCPointerTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCPointerType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCPointerTypeGet(MlirContext context,
                                                  MlirType elementType);
MLIR_CAPI_EXPORTED MlirType cudaqCCPointerTypeGetElementType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCArrayTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCArrayType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCArrayTypeGet(MlirContext context,
                                                MlirType elementType,
                                                int64_t size);
MLIR_CAPI_EXPORTED MlirType cudaqCCArrayTypeGetElementType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCStructTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCStructType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCStructTypeGet(MlirContext context,
                                                 intptr_t numMembers,
                                                 const MlirType *members);
MLIR_CAPI_EXPORTED MlirType cudaqCCStructTypeGetNamed(
    MlirContext context, MlirStringRef name, intptr_t numMembers,
    const MlirType *members);
MLIR_CAPI_EXPORTED intptr_t cudaqCCStructTypeGetNumMembers(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCStructTypeGetMember(MlirType type,
                                                       intptr_t pos);
MLIR_CAPI_EXPORTED MlirStringRef cudaqCCStructTypeGetName(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCCallableTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCCallableType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCCallableTypeGet(MlirContext context,
                                                   MlirType functionType);
MLIR_CAPI_EXPORTED MlirType cudaqCCCallableTypeGetFunctionType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID cudaqCCStdvecTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool cudaqTypeIsACCStdvecType(MlirType type);
MLIR_CAPI_EXPORTED MlirType cudaqCCStdvecTypeGet(MlirContext context,
                                                 MlirType elementType);
MLIR_CAPI_EXPORTED MlirType cudaqCCStdvecTypeGetElementType(MlirType type);

#ifdef __cplusplus
}
#endif
