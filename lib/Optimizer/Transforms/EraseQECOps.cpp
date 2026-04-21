/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEQECOPS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-qec-ops"

using namespace mlir;

namespace {
class EraseDetectorPattern
    : public OpRewritePattern<cudaq::qec::DetectorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cudaq::qec::DetectorOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseDetectorVecPattern
    : public OpRewritePattern<cudaq::qec::DetectorVecOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cudaq::qec::DetectorVecOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseLogicalObservablePattern
    : public OpRewritePattern<cudaq::qec::LogicalObservableOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cudaq::qec::LogicalObservableOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseQECOpsPass
    : public cudaq::opt::impl::EraseQECOpsBase<EraseQECOpsPass> {
public:
  using EraseQECOpsBase::EraseQECOpsBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseDetectorPattern, EraseDetectorVecPattern,
                    EraseLogicalObservablePattern>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
