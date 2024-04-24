/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Optimizer/Transforms/Extensions.h"
#include "cudaq/Optimizer/Dialect/Common/Traits.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"

using namespace mlir;

namespace cudaq {

void CudaqTransformExtensions::init() {

  // Define PDL Quake Constraints here...
  auto isHermitian = [](PatternRewriter &rewriter, PDLResultList &,
                        ArrayRef<PDLValue> pdlValues) {
    for (const PDLValue &pdlValue : pdlValues)
      if (Operation *op = pdlValue.dyn_cast<Operation *>())
        return success(op->hasTrait<cudaq::Hermitian>());

    return failure();
  };

  auto isQuakeOp = [](PatternRewriter &rewriter, PDLResultList &,
                      ArrayRef<PDLValue> pdlValues) {
    for (const PDLValue &pdlValue : pdlValues)
      if (Operation *op = pdlValue.dyn_cast<Operation *>())
        if (auto *dialect = op->getDialect())
          return success(dialect->getNamespace().equals("quake"));
    return failure();
  };

  auto isSameName = [](PatternRewriter &rewriter, PDLResultList &,
                       ArrayRef<PDLValue> pdlValues) {
    if (pdlValues.size() != 2)
      return failure();
    auto *op1 = pdlValues[0].dyn_cast<Operation *>();
    auto *op2 = pdlValues[1].dyn_cast<Operation *>();

    return success(
        op1->getName().stripDialect().equals(op2->getName().stripDialect()));
  };

  addDialectDataInitializer<mlir::transform::PDLMatchHooks>(
      [&](mlir::transform::PDLMatchHooks &hooks) {
        llvm::StringMap<mlir::PDLConstraintFunction> constraints;
        constraints.try_emplace("IsHermitian", isHermitian);
        constraints.try_emplace("IsQuakeOperation", isQuakeOp);
        constraints.try_emplace("IsSameName", isSameName);
        hooks.mergeInPDLMatchHooks(std::move(constraints));
      });
}
} // namespace cudaq