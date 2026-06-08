/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

/// QuakeToQIRDialect pass — converts quake::* ops to cudaq::qir::* ops.
///
/// Only the Full QIR profile (dynamic allocation) is supported here.
/// Handles:
///   - quake.alloca (!quake.ref, !quake.veq<N>)
///   - quake.dealloc
///   - quake.extract_ref
///   - 1-qubit gates: h x y z s t (with adjoint → sdg/tdg)
///   - 1-qubit rotations: rx ry rz r1
///   - 2-qubit gates (1 ctrl): cnot (X+ctrl), cz (Z+ctrl), swap (2 targets)
///   - 3-qubit Toffoli: ccx (X + 2 ctrls)
///   - quake.mz (z-axis measurement)
///   - quake.return_wire / quake.sink (value-semantics terminators — erased)
///
/// Controlled gates with array (veq) controls, custom_unitary, exp_pauli, and
/// the base/adaptive static-addressing profiles are not handled by this pass
/// and must be lowered via the existing quake-to-qir-api pipeline.

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/QIR/QIROps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKETOQIRDIALECT
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

struct QuakeToQIRTypeConverter : public TypeConverter {
  QuakeToQIRTypeConverter() {
    // Identity for everything that doesn't need conversion.
    addConversion([](Type ty) { return ty; });
    // Quake quantum types → QIR types.
    addConversion([](cudaq::quake::RefType ty) -> Type {
      return cudaq::qir::QubitType::get(ty.getContext());
    });
    addConversion([](cudaq::quake::WireType ty) -> Type {
      return cudaq::qir::QubitType::get(ty.getContext());
    });
    addConversion([](cudaq::quake::ControlType ty) -> Type {
      return cudaq::qir::QubitType::get(ty.getContext());
    });
    addConversion([](cudaq::quake::VeqType ty) -> Type {
      return cudaq::qir::ArrayType::get(ty.getContext());
    });
    addConversion([](cudaq::quake::CableType ty) -> Type {
      return cudaq::qir::ArrayType::get(ty.getContext());
    });
    addConversion([](cudaq::quake::MeasureType ty) -> Type {
      return cudaq::qir::ResultType::get(ty.getContext());
    });
    // Function types need recursive conversion.
    addConversion([this](FunctionType ft) -> Type {
      SmallVector<Type> ins, outs;
      if (failed(convertTypes(ft.getInputs(), ins)) ||
          failed(convertTypes(ft.getResults(), outs)))
        return {};
      return FunctionType::get(ft.getContext(), ins, outs);
    });
  }
};

//===----------------------------------------------------------------------===//
// Helper: erase-or-forward for gate ops
// In reference semantics the gate op has no results (wires.empty()); in value
// semantics the op has one result per target that must be forwarded.
//===----------------------------------------------------------------------===//

// In reference semantics the gate op has no results (wires.empty()); in value
// semantics the op has one result per quantum operand (controls + targets in
// order). Since QIR gate ops are in-place, we forward each quantum input as
// the corresponding wire output.
static void eraseOrForward(ConversionPatternRewriter &rewriter, Operation *op,
                           ValueRange controls, ValueRange targets) {
  if (op->getResults().empty()) {
    rewriter.eraseOp(op);
  } else {
    SmallVector<Value> fwd(controls.begin(), controls.end());
    fwd.append(targets.begin(), targets.end());
    rewriter.replaceOp(op, fwd);
  }
}

//===----------------------------------------------------------------------===//
// Allocation / deallocation
//===----------------------------------------------------------------------===//

struct AllocaRefToQIROp : public OpConversionPattern<cudaq::quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::AllocaOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<cudaq::quake::RefType>(op.getType()))
      return failure();
    auto qubitTy = cudaq::qir::QubitType::get(op.getContext());
    rewriter.replaceOpWithNewOp<cudaq::qir::AllocQubitOp>(op, qubitTy);
    return success();
  }
};

struct AllocaVeqToQIROp : public OpConversionPattern<cudaq::quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(op.getType());
    if (!veqTy)
      return failure();
    auto loc = op.getLoc();
    auto arrayTy = cudaq::qir::ArrayType::get(op.getContext());
    Value size;
    if (veqTy.hasSpecifiedSize()) {
      size = arith::ConstantIntOp::create(rewriter, loc, veqTy.getSize(), 64);
    } else if (!adaptor.getOperands().empty()) {
      size = adaptor.getOperands().front();
      if (auto intTy = dyn_cast<IntegerType>(size.getType()))
        if (intTy.getWidth() != 64)
          size = arith::ExtSIOp::create(rewriter, loc, rewriter.getI64Type(),
                                        size)
                     .getResult();
    } else {
      return op.emitOpError("veq alloca has no size");
    }
    rewriter.replaceOpWithNewOp<cudaq::qir::AllocArrayOp>(op, arrayTy, size);
    return success();
  }
};

struct DeallocToQIROp : public OpConversionPattern<cudaq::quake::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperands().front();
    auto loc = op.getLoc();
    if (isa<cudaq::qir::QubitType>(operand.getType())) {
      cudaq::qir::ReleaseQubitOp::create(rewriter, loc, operand);
    } else if (isa<cudaq::qir::ArrayType>(operand.getType())) {
      cudaq::qir::ReleaseArrayOp::create(rewriter, loc, operand);
    } else {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Extract-ref → array_get
//===----------------------------------------------------------------------===//

struct ExtractRefToQIROp
    : public OpConversionPattern<cudaq::quake::ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto qubitTy = cudaq::qir::QubitType::get(op.getContext());
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    Value idx;
    if (adaptor.getIndex()) {
      idx = adaptor.getIndex();
      if (idx.getType() != i64Ty)
        idx = arith::ExtSIOp::create(rewriter, loc, i64Ty, idx).getResult();
    } else {
      idx = arith::ConstantIntOp::create(
          rewriter, loc, static_cast<int64_t>(op.getConstantIndex()), 64);
    }
    rewriter.replaceOpWithNewOp<cudaq::qir::ArrayGetOp>(op, qubitTy,
                                                        adaptor.getVeq(), idx);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 1-qubit gates (no controls)
//===----------------------------------------------------------------------===//

// Macro for simple involutory gates (H, X, Y, Z): adjoint == original.
#define QIR_GATE1_INVOLUTORY(QuakeGate, QIRGate)                               \
  struct QuakeGate##ToQIROp                                                     \
      : public OpConversionPattern<cudaq::quake::QuakeGate> {                   \
    using OpConversionPattern::OpConversionPattern;                             \
    LogicalResult matchAndRewrite(                                               \
        cudaq::quake::QuakeGate op, OpAdaptor adaptor,                          \
        ConversionPatternRewriter &rewriter) const override {                   \
      if (!op.getControls().empty())                                             \
        return failure();                                                        \
      auto targets = adaptor.getTargets();                                      \
      if (targets.size() != 1)                                                  \
        return failure();                                                        \
      cudaq::qir::QIRGate::create(rewriter, op.getLoc(), targets[0]);           \
      eraseOrForward(rewriter, op, {}, targets);                                \
      return success();                                                          \
    }                                                                           \
  }

// Macro for gates with a distinct adjoint (S→Sdg, T→Tdg).
#define QIR_GATE1_WITH_ADJ(QuakeGate, QIRGate, QIRGateAdj)                     \
  struct QuakeGate##ToQIROp                                                     \
      : public OpConversionPattern<cudaq::quake::QuakeGate> {                   \
    using OpConversionPattern::OpConversionPattern;                             \
    LogicalResult matchAndRewrite(                                               \
        cudaq::quake::QuakeGate op, OpAdaptor adaptor,                          \
        ConversionPatternRewriter &rewriter) const override {                   \
      if (!op.getControls().empty())                                             \
        return failure();                                                        \
      auto targets = adaptor.getTargets();                                      \
      if (targets.size() != 1)                                                  \
        return failure();                                                        \
      if (op.getIsAdj())                                                        \
        cudaq::qir::QIRGateAdj::create(rewriter, op.getLoc(), targets[0]);     \
      else                                                                      \
        cudaq::qir::QIRGate::create(rewriter, op.getLoc(), targets[0]);        \
      eraseOrForward(rewriter, op, {}, targets);                                \
      return success();                                                          \
    }                                                                           \
  }

QIR_GATE1_INVOLUTORY(HOp, HOp);
QIR_GATE1_INVOLUTORY(XOp, XOp);
QIR_GATE1_INVOLUTORY(YOp, YOp);
QIR_GATE1_INVOLUTORY(ZOp, ZOp);
QIR_GATE1_WITH_ADJ(SOp, SOp, SdgOp);
QIR_GATE1_WITH_ADJ(TOp, TOp, TdgOp);

#undef QIR_GATE1_INVOLUTORY
#undef QIR_GATE1_WITH_ADJ

//===----------------------------------------------------------------------===//
// 1-qubit rotations (1 float parameter, no controls)
//===----------------------------------------------------------------------===//

// For rotations: adjoint negates the angle parameter.
template <typename QuakeGate, typename QIRGate>
struct RotationToQIROp : public OpConversionPattern<QuakeGate> {
  using Base = OpConversionPattern<QuakeGate>;
  using Base::Base;
  LogicalResult
  matchAndRewrite(QuakeGate op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    auto targets = adaptor.getTargets();
    if (targets.size() != 1)
      return failure();
    auto params = adaptor.getParameters();
    if (params.size() != 1)
      return failure();

    auto loc = op.getLoc();
    auto f64Ty = rewriter.getF64Type();
    Value angle = params[0];
    // Normalize to f64.
    if (angle.getType() != f64Ty)
      angle = arith::ExtFOp::create(rewriter, loc, f64Ty, angle).getResult();
    // Adjoint negates the angle.
    if (op.getIsAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle).getResult();

    QIRGate::create(rewriter, loc, angle, targets[0]);
    eraseOrForward(rewriter, op, {}, targets);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 2-qubit gates
//===----------------------------------------------------------------------===//

// CNOT: quake.x [%ctrl] %target
struct XOpCNOTToQIROp : public OpConversionPattern<cudaq::quake::XOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::XOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctrls = adaptor.getControls();
    auto targets = adaptor.getTargets();
    if (ctrls.size() != 1 || targets.size() != 1)
      return failure();
    if (!isa<cudaq::qir::QubitType>(ctrls[0].getType()))
      return failure(); // array controls not handled
    cudaq::qir::CNOTOp::create(rewriter, op.getLoc(), ctrls[0], targets[0]);
    eraseOrForward(rewriter, op, ctrls, targets);
    return success();
  }
};

// CZ: quake.z [%ctrl] %target
struct ZOpCZToQIROp : public OpConversionPattern<cudaq::quake::ZOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::ZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctrls = adaptor.getControls();
    auto targets = adaptor.getTargets();
    if (ctrls.size() != 1 || targets.size() != 1)
      return failure();
    if (!isa<cudaq::qir::QubitType>(ctrls[0].getType()))
      return failure();
    cudaq::qir::CZOp::create(rewriter, op.getLoc(), ctrls[0], targets[0]);
    eraseOrForward(rewriter, op, ctrls, targets);
    return success();
  }
};

// Toffoli (CCX): quake.x [%c0, %c1] %target
struct XOpCCXToQIROp : public OpConversionPattern<cudaq::quake::XOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::XOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctrls = adaptor.getControls();
    auto targets = adaptor.getTargets();
    if (ctrls.size() != 2 || targets.size() != 1)
      return failure();
    if (!isa<cudaq::qir::QubitType>(ctrls[0].getType()) ||
        !isa<cudaq::qir::QubitType>(ctrls[1].getType()))
      return failure();
    cudaq::qir::CCXOp::create(rewriter, op.getLoc(), ctrls[0], ctrls[1],
                              targets[0]);
    eraseOrForward(rewriter, op, ctrls, targets);
    return success();
  }
};

// SWAP: quake.swap %q0, %q1 (no controls, 2 targets)
struct SwapToQIROp : public OpConversionPattern<cudaq::quake::SwapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::SwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    auto targets = adaptor.getTargets();
    if (targets.size() != 2)
      return failure();
    cudaq::qir::SWAPOp::create(rewriter, op.getLoc(), targets[0], targets[1]);
    eraseOrForward(rewriter, op, {}, targets);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

struct MzToQIROp : public OpConversionPattern<cudaq::quake::MzOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::MzOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto targets = adaptor.getTargets();
    if (targets.size() != 1)
      return op.emitOpError("expected exactly 1 target qubit");
    auto resultTy = cudaq::qir::ResultType::get(op.getContext());
    auto meas = cudaq::qir::MeasureOp::create(rewriter, loc, resultTy,
                                              targets[0]);
    // Build replacement values: {measOut, wire0, ...}
    SmallVector<Value> replVals = {meas.getResult()};
    // In value semantics there are also wire results; forward the target qubit.
    for (size_t i = 0, e = op.getWires().size(); i < e; ++i)
      replVals.push_back(targets[0]);
    rewriter.replaceOp(op, replVals);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Value-semantics terminators (erased — qubit lifetime ends)
//===----------------------------------------------------------------------===//

struct ReturnWireErase : public OpConversionPattern<cudaq::quake::ReturnWireOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::ReturnWireOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SinkErase : public OpConversionPattern<cudaq::quake::SinkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudaq::quake::SinkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct QuakeToQIRDialectPass
    : public cudaq::opt::impl::QuakeToQIRDialectBase<QuakeToQIRDialectPass> {
  using QuakeToQIRDialectBase::QuakeToQIRDialectBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    QuakeToQIRTypeConverter typeConverter;

    ConversionTarget target(*ctx);
    // Everything in cudaq::qir is our output — always legal.
    target.addLegalDialect<cudaq::qir::QIRDialect>();
    // Downstream dialects that must remain legal.
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // Quake ops we know how to lower are illegal; others cause failure.
    target.addIllegalOp<cudaq::quake::AllocaOp, cudaq::quake::DeallocOp,
                        cudaq::quake::ExtractRefOp>();
    target.addIllegalOp<cudaq::quake::HOp, cudaq::quake::XOp,
                        cudaq::quake::YOp, cudaq::quake::ZOp,
                        cudaq::quake::SOp, cudaq::quake::TOp,
                        cudaq::quake::RxOp, cudaq::quake::RyOp,
                        cudaq::quake::RzOp, cudaq::quake::R1Op,
                        cudaq::quake::SwapOp>();
    target.addIllegalOp<cudaq::quake::MzOp, cudaq::quake::ReturnWireOp,
                        cudaq::quake::SinkOp>();

    RewritePatternSet patterns(ctx);
    patterns.insert<
        // Allocation / deallocation
        AllocaRefToQIROp, AllocaVeqToQIROp, DeallocToQIROp,
        // Array access
        ExtractRefToQIROp,
        // 1-qubit gates (no-control and involutory-adjoint)
        HOpToQIROp, XOpToQIROp, YOpToQIROp, ZOpToQIROp,
        SOpToQIROp, TOpToQIROp,
        // 2-qubit controlled gates
        XOpCNOTToQIROp, ZOpCZToQIROp, XOpCCXToQIROp,
        // SWAP
        SwapToQIROp,
        // Measurement
        MzToQIROp,
        // Value-semantics terminators
        ReturnWireErase, SinkErase>(typeConverter, ctx);
    // 1-qubit rotations
    patterns.insert<RotationToQIROp<cudaq::quake::RxOp, cudaq::qir::RxOp>,
                    RotationToQIROp<cudaq::quake::RyOp, cudaq::qir::RyOp>,
                    RotationToQIROp<cudaq::quake::RzOp, cudaq::qir::RzOp>,
                    RotationToQIROp<cudaq::quake::R1Op, cudaq::qir::R1Op>>(
        typeConverter, ctx);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
