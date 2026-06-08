/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

/// \file QIRToLLVM.cpp
/// Lowers cudaq::qir dialect ops to LLVM-IR with __quantum__* runtime calls.
///
/// Each qir.* op becomes one or more llvm.call to the QIR runtime ABI.
/// The type mapping is:
///   !qir.qubit  -> !llvm.ptr   (opaque, LLVM 15+ opaque-pointer mode)
///   !qir.array  -> !llvm.ptr
///   !qir.result -> !llvm.ptr
///
/// Control flow (scf/cf/func) is lowered by the standard MLIR conversion
/// helpers that this pass loads as dependent dialects.

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/QIR/QIROps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CONVERTQIRTOLLVM
#define GEN_PASS_DEF_QIRANNOTATESTATICRESOURCES
#define GEN_PASS_DEF_QIRDELAYMEASUREMENTS
#define GEN_PASS_DEF_QIRVERIFYPROFILE
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;
using namespace cudaq::qir;

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

namespace {

/// Converts QIR opaque handle types to LLVM opaque pointers.
struct QIRTypeConverter : public LLVMTypeConverter {
  explicit QIRTypeConverter(MLIRContext *ctx, bool useOpaquePtr = true)
      : LLVMTypeConverter(ctx) {
    addConversion([useOpaquePtr](QubitType) -> Type {
      return LLVM::LLVMPointerType::get(
          useOpaquePtr ? nullptr : nullptr /* keep as opaque ptr */);
    });
    addConversion([useOpaquePtr](ArrayType) -> Type {
      return LLVM::LLVMPointerType::get(nullptr);
    });
    addConversion([useOpaquePtr](ResultType) -> Type {
      return LLVM::LLVMPointerType::get(nullptr);
    });
  }
};

//===----------------------------------------------------------------------===//
// Helper: get or insert a function declaration in the module
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp getOrInsertFn(PatternRewriter &rewriter, ModuleOp mod,
                                       StringRef name,
                                       LLVM::LLVMFunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  auto *ctx = mod.getContext();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                                   LLVM::Linkage::External);
}

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//

/// Returns the module that contains \p op.
static ModuleOp getParentModule(Operation *op) {
  return op->getParentOfType<ModuleOp>();
}

static Type ptrTy(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}
static Type voidTy(MLIRContext *ctx) {
  return LLVM::LLVMVoidType::get(ctx);
}
static Type i64Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 64);
}
static Type i1Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 1);
}
static Type f64Ty(MLIRContext *ctx) {
  return Float64Type::get(ctx);
}

/// Emit a void(ptr) call to \p rtFn on operand \p arg.
static LogicalResult emitVoidPtrCall(PatternRewriter &rewriter, Operation *op,
                                      StringRef rtFn, Value arg) {
  auto *ctx = op->getContext();
  auto mod = getParentModule(op);
  auto fnTy =
      LLVM::LLVMFunctionType::get(voidTy(ctx), {ptrTy(ctx)});
  auto fn = getOrInsertFn(rewriter, mod, rtFn, fnTy);
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, ValueRange{arg});
  return success();
}

/// Emit a ptr() call (no args) to \p rtFn, returning a ptr.
static Value emitNullaryPtrCall(PatternRewriter &rewriter, Location loc,
                                 Operation *op, StringRef rtFn) {
  auto *ctx = op->getContext();
  auto mod = getParentModule(op);
  auto fnTy = LLVM::LLVMFunctionType::get(ptrTy(ctx), {});
  auto fn = getOrInsertFn(rewriter, mod, rtFn, fnTy);
  auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
  return call.getResults()[0];
}

//----------------------------------------------------------------------
// Qubit management
//----------------------------------------------------------------------

struct AllocQubitLowering : OpConversionPattern<AllocQubitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(AllocQubitOp op, OpAdaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto ptr = emitNullaryPtrCall(rewriter, op.getLoc(), op,
                                  cudaq::opt::QIRQubitAllocate);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct ReleaseQubitLowering : OpConversionPattern<ReleaseQubitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ReleaseQubitOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    return emitVoidPtrCall(rewriter, op, cudaq::opt::QIRArrayQubitReleaseQubit,
                           adaptor.getQubit());
  }
};

struct AllocArrayLowering : OpConversionPattern<AllocArrayOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(AllocArrayOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(ptrTy(ctx), {i64Ty(ctx)});
    auto fn = getOrInsertFn(rewriter, mod,
                            cudaq::opt::QIRArrayQubitAllocateArray, fnTy);
    auto call =
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getSize());
    (void)call;
    return success();
  }
};

struct ReleaseArrayLowering : OpConversionPattern<ReleaseArrayOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ReleaseArrayOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    return emitVoidPtrCall(rewriter, op, cudaq::opt::QIRArrayQubitReleaseArray,
                           adaptor.getArray());
  }
};

struct ArrayGetLowering : OpConversionPattern<ArrayGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(ptrTy(ctx), {ptrTy(ctx), i64Ty(ctx)});
    auto fn = getOrInsertFn(rewriter, mod,
                            cudaq::opt::QIRArrayGetElementPtr1d, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn, ValueRange{adaptor.getArray(), adaptor.getIndex()});
    return success();
  }
};

struct StaticQubitLowering : OpConversionPattern<StaticQubitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(StaticQubitOp op, OpAdaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    // Lower to inttoptr i64 <index> to ptr — the standard QIR static qubit
    // encoding used by Base/Adaptive profiles.
    auto *ctx = op.getContext();
    auto idx = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty(ctx),
                                        rewriter.getI64IntegerAttr(op.getIndex()));
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, ptrTy(ctx), idx.getResult());
    return success();
  }
};

struct StaticResultLowering : OpConversionPattern<StaticResultOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(StaticResultOp op, OpAdaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto idx = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty(ctx),
                                        rewriter.getI64IntegerAttr(op.getIndex()));
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, ptrTy(ctx), idx.getResult());
    return success();
  }
};

//----------------------------------------------------------------------
// Single-qubit gates — void(ptr) pattern
//----------------------------------------------------------------------

template <typename OpTy>
struct Gate1Lowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  StringRef rtFn;
  Gate1Lowering(MLIRContext *ctx, StringRef fn)
      : OpConversionPattern<OpTy>(ctx), rtFn(fn) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    return emitVoidPtrCall(rewriter, op, rtFn, adaptor.getQubit());
  }
};

//----------------------------------------------------------------------
// Rotation gates — void(f64, ptr)
//----------------------------------------------------------------------

template <typename OpTy>
struct RotationLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  StringRef rtFn;
  RotationLowering(MLIRContext *ctx, StringRef fn)
      : OpConversionPattern<OpTy>(ctx), rtFn(fn) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy =
        LLVM::LLVMFunctionType::get(voidTy(ctx), {f64Ty(ctx), ptrTy(ctx)});
    auto fn = getOrInsertFn(rewriter, mod, rtFn, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn, ValueRange{adaptor.getAngle(), adaptor.getQubit()});
    return success();
  }
};

struct U3Lowering : OpConversionPattern<U3Op> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(U3Op op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(
        voidTy(ctx), {f64Ty(ctx), f64Ty(ctx), f64Ty(ctx), ptrTy(ctx)});
    auto fn =
        getOrInsertFn(rewriter, mod, "__quantum__qis__u3__body", fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn,
        ValueRange{adaptor.getTheta(), adaptor.getPhi(), adaptor.getLam(),
                   adaptor.getQubit()});
    return success();
  }
};

//----------------------------------------------------------------------
// Two-qubit gates — void(ptr, ptr)
//----------------------------------------------------------------------

template <typename OpTy>
struct Gate2Lowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  StringRef rtFn;
  Gate2Lowering(MLIRContext *ctx, StringRef fn)
      : OpConversionPattern<OpTy>(ctx), rtFn(fn) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy =
        LLVM::LLVMFunctionType::get(voidTy(ctx), {ptrTy(ctx), ptrTy(ctx)});
    auto fn = getOrInsertFn(rewriter, mod, rtFn, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn, ValueRange{adaptor.getCtrl(), adaptor.getTarget()});
    return success();
  }
};

struct CCXLowering : OpConversionPattern<CCXOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(CCXOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(
        voidTy(ctx), {ptrTy(ctx), ptrTy(ctx), ptrTy(ctx)});
    auto fn = getOrInsertFn(rewriter, mod, "__quantum__qis__ccx__body", fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn,
        ValueRange{adaptor.getCtrl0(), adaptor.getCtrl1(), adaptor.getTarget()});
    return success();
  }
};

//----------------------------------------------------------------------
// Measurement
//----------------------------------------------------------------------

struct MeasureLowering : OpConversionPattern<MeasureOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    // __quantum__qis__mz__body(qubit* %q, result* %r) — takes a pre-allocated
    // result slot; we synthesize one via inttoptr 0.
    // For the "returns a result" variant used in Full profile we use a wrapper
    // that allocates a fresh result slot.
    auto fnTy = LLVM::LLVMFunctionType::get(ptrTy(ctx), {ptrTy(ctx)});
    // We define a thin wrapper symbol; in practice users should use
    // measure_named for Base/Adaptive and MeasureOp only in Full profile.
    auto fn =
        getOrInsertFn(rewriter, mod, "__quantum__qis__mz__alloc", fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getQubit());
    return success();
  }
};

struct MeasureNamedLowering : OpConversionPattern<MeasureNamedOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MeasureNamedOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy(ctx),
                                            {ptrTy(ctx), ptrTy(ctx)});
    auto fn =
        getOrInsertFn(rewriter, mod, cudaq::opt::QIRMeasureBody, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fn, ValueRange{adaptor.getQubit(), adaptor.getSlot()});
    return success();
  }
};

struct ReadResultLowering : OpConversionPattern<ReadResultOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ReadResultOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    // QIR 1.0 uses __quantum__rt__read_result(result*) -> i1
    auto fnTy =
        LLVM::LLVMFunctionType::get(i1Ty(ctx), {ptrTy(ctx)});
    auto fn = getOrInsertFn(rewriter, mod,
                            cudaq::opt::qir1_0::ReadResult, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getResult());
    return success();
  }
};

struct ResetLowering : OpConversionPattern<ResetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    return emitVoidPtrCall(rewriter, op, cudaq::opt::QIRResetBody,
                           adaptor.getQubit());
  }
};

//----------------------------------------------------------------------
// Output recording
//----------------------------------------------------------------------

struct RecordBoolLowering : OpConversionPattern<RecordBoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(RecordBoolOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    return emitVoidPtrCall(rewriter, op, cudaq::opt::QIRRecordOutput,
                           adaptor.getResult());
  }
};

struct RecordIntLowering : OpConversionPattern<RecordIntOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(RecordIntOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy(ctx), {i64Ty(ctx)});
    auto fn =
        getOrInsertFn(rewriter, mod, cudaq::opt::QIRIntegerRecordOutput, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getValue());
    return success();
  }
};

struct RecordDoubleLowering : OpConversionPattern<RecordDoubleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(RecordDoubleOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy(ctx), {f64Ty(ctx)});
    auto fn = getOrInsertFn(rewriter, mod, cudaq::opt::QIRDoubleRecordOutput,
                            fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getValue());
    return success();
  }
};

//----------------------------------------------------------------------
// QEC hooks — no runtime call; lowered to no-op (metadata only)
//----------------------------------------------------------------------

struct DetectorLowering : OpConversionPattern<DetectorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(DetectorOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    SmallVector<Type> argTys(adaptor.getResults().size(), ptrTy(ctx));
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy(ctx), argTys, /*vararg=*/true);
    auto fn = getOrInsertFn(rewriter, mod, cudaq::opt::QIRDetector, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getResults());
    return success();
  }
};

struct ObservableLowering : OpConversionPattern<ObservableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ObservableOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto mod = getParentModule(op);
    SmallVector<Type> argTys(adaptor.getResults().size(), ptrTy(ctx));
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy(ctx), argTys, /*vararg=*/true);
    auto fn = getOrInsertFn(rewriter, mod,
                            cudaq::opt::QIRLogicalObservable, fnTy);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, adaptor.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementations
//===----------------------------------------------------------------------===//

struct ConvertQIRToLLVMPass
    : cudaq::opt::impl::ConvertQIRToLLVMBase<ConvertQIRToLLVMPass> {
  using ConvertQIRToLLVMBase::ConvertQIRToLLVMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();

    QIRTypeConverter typeConverter(ctx, opaquePointers);
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    // Declare illegal dialects / ops.
    target.addIllegalDialect<QIRDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    // QIR → LLVM patterns.
    patterns.add<AllocQubitLowering, ReleaseQubitLowering, AllocArrayLowering,
                 ReleaseArrayLowering, ArrayGetLowering, StaticQubitLowering,
                 StaticResultLowering>(ctx);
    patterns.add<MeasureLowering, MeasureNamedLowering, ReadResultLowering,
                 ResetLowering>(ctx);
    patterns.add<RecordBoolLowering, RecordIntLowering, RecordDoubleLowering>(ctx);
    patterns.add<DetectorLowering, ObservableLowering>(ctx);
    patterns.add<U3Lowering, CCXLowering>(ctx);

    // Single-qubit gate table.
    using namespace cudaq::opt;
    patterns.add<Gate1Lowering<HOp>>(ctx,    "__quantum__qis__h__body");
    patterns.add<Gate1Lowering<XOp>>(ctx,    "__quantum__qis__x__body");
    patterns.add<Gate1Lowering<YOp>>(ctx,    "__quantum__qis__y__body");
    patterns.add<Gate1Lowering<ZOp>>(ctx,    "__quantum__qis__z__body");
    patterns.add<Gate1Lowering<SOp>>(ctx,    "__quantum__qis__s__body");
    patterns.add<Gate1Lowering<SdgOp>>(ctx,  "__quantum__qis__s__adj");
    patterns.add<Gate1Lowering<TOp>>(ctx,    "__quantum__qis__t__body");
    patterns.add<Gate1Lowering<TdgOp>>(ctx,  "__quantum__qis__t__adj");

    patterns.add<RotationLowering<RxOp>>(ctx, "__quantum__qis__rx__body");
    patterns.add<RotationLowering<RyOp>>(ctx, "__quantum__qis__ry__body");
    patterns.add<RotationLowering<RzOp>>(ctx, "__quantum__qis__rz__body");
    patterns.add<RotationLowering<R1Op>>(ctx, "__quantum__qis__r1__body");

    // Two-qubit gates.
    patterns.add<Gate2Lowering<CNOTOp>>(ctx, QIRCnot);
    patterns.add<Gate2Lowering<CZOp>>(ctx,   QIRCZ);
    patterns.add<Gate2Lowering<SWAPOp>>(ctx, "__quantum__qis__swap__body");

    // SCF → CF → LLVM.
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalOp<func::FuncOp, func::ReturnOp, func::CallOp,
                      ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// QIRAnnotateStaticResources
//===----------------------------------------------------------------------===//

struct QIRAnnotateStaticResourcesPass
    : cudaq::opt::impl::QIRAnnotateStaticResourcesBase<
          QIRAnnotateStaticResourcesPass> {
  using QIRAnnotateStaticResourcesBase::QIRAnnotateStaticResourcesBase;

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    uint64_t numQubits = 0, numResults = 0;
    fn.walk([&](StaticQubitOp op) {
      numQubits = std::max(numQubits, op.getIndex() + 1);
    });
    fn.walk([&](StaticResultOp op) {
      numResults = std::max(numResults, op.getIndex() + 1);
    });
    auto *ctx = fn.getContext();
    fn->setAttr("qir.required_qubits",
                cudaq::qir::RequiredQubitsAttr::get(ctx, numQubits));
    fn->setAttr("qir.required_results",
                cudaq::qir::RequiredResultsAttr::get(ctx, numResults));
  }
};

//===----------------------------------------------------------------------===//
// QIRDelayMeasurements
//===----------------------------------------------------------------------===//

struct QIRDelayMeasurementsPass
    : cudaq::opt::impl::QIRDelayMeasurementsBase<QIRDelayMeasurementsPass> {
  using QIRDelayMeasurementsBase::QIRDelayMeasurementsBase;

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    // Collect all measure ops and move them to just before the return.
    SmallVector<Operation *> measures;
    fn.walk([&](Operation *op) {
      if (isa<MeasureOp, MeasureNamedOp>(op))
        measures.push_back(op);
    });
    if (measures.empty())
      return;
    // Find the terminator of the last block.
    Block &lastBlock = fn.getBody().back();
    Operation *term = lastBlock.getTerminator();
    for (auto *m : measures)
      m->moveBefore(term);
  }
};

//===----------------------------------------------------------------------===//
// QIRVerifyProfile
//===----------------------------------------------------------------------===//

struct QIRVerifyProfilePass
    : cudaq::opt::impl::QIRVerifyProfileBase<QIRVerifyProfilePass> {
  using QIRVerifyProfileBase::QIRVerifyProfileBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    // Walk function ops checking profile annotations.
    op->walk([&](func::FuncOp fn) {
      auto attr = fn->getAttrOfType<cudaq::qir::ProfileAttr>("qir.profile");
      if (!attr)
        return;
      auto profile = attr.getProfile();
      if (profile == QIRProfile::Full)
        return; // no restrictions
      fn.walk([&](Operation *inner) {
        // Base/Adaptive: dynamic allocation is illegal.
        if (isa<AllocQubitOp, AllocArrayOp>(inner)) {
          inner->emitError()
              << "dynamic qubit allocation is not allowed in "
              << (profile == QIRProfile::Base ? "base" : "adaptive")
              << " QIR profile";
          signalPassFailure();
        }
        // Base: read_result is illegal.
        if (profile == QIRProfile::Base && isa<ReadResultOp>(inner)) {
          inner->emitError()
              << "qir.read_result is not allowed in base QIR profile";
          signalPassFailure();
        }
      });
    });
  }
};

} // anonymous namespace
