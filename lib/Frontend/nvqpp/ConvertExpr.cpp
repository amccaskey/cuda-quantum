/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Frontend/nvqpp/QisBuilder.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "lower-ast-expr"

using namespace mlir;

// Get the result type if \p ty is a function type or just return \p ty.
static Type getResultType(Type ty) {
  assert(ty && "Type cannot be null");
  if (auto funcTy = dyn_cast<FunctionType>(ty)) {
    assert(funcTy.getNumResults() == 1);
    return funcTy.getResult(0);
  }
  return ty;
}

// Convert a name, value pair into a symbol name.
static std::string getQubitSymbolTableName(StringRef qregName, Value idxVal) {
  std::string name;
  if (auto idxIntVal = idxVal.getDefiningOp<arith::ConstantIntOp>())
    return qregName.str() + "%" + std::to_string(idxIntVal.value());
  if (auto idxIdxVal = idxVal.getDefiningOp<arith::ConstantIndexOp>())
    return qregName.str() + "%" + std::to_string(idxIdxVal.value());

  // this is a general value, like a loop idx
  std::stringstream ss;
  ss << qregName.str() << "%" << idxVal.getAsOpaquePointer();
  return ss.str();
}

static clang::NamedDecl *getNamedDecl(clang::Expr *expr) {
  auto *call = cast<clang::DeclRefExpr>(expr);
  return call->getDecl()->getUnderlyingDecl();
}

static std::pair<SmallVector<Value>, SmallVector<Value>>
maybeUnpackOperands(OpBuilder &builder, Location loc, ValueRange operands) {
  if (operands.size() > 1)
    return std::make_pair(SmallVector<Value>{operands.take_back()},
                          SmallVector<Value>{operands.drop_back(1)});
  Value target = operands.back();
  if (target.getType().isa<quake::VeqType>()) {
    // Split the vector. Last one is target, front N-1 are controls.
    auto vecSize = builder.create<quake::VeqSizeOp>(
        loc, builder.getIntegerType(64), target);
    auto size = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                   vecSize);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto offset = builder.create<arith::SubIOp>(loc, size, one);
    // Get the last qubit in the veq: the target.
    Value qTarg = builder.create<quake::ExtractRefOp>(loc, target, offset);
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto last = builder.create<arith::SubIOp>(loc, offset, one);
    // The canonicalizer will compute a constant size, if possible.
    auto unsizedVeqTy = quake::VeqType::getUnsized(builder.getContext());
    // Get the subvector of all qubits excluding the last one: controls.
    Value ctrlSubveq =
        builder.create<quake::SubVeqOp>(loc, unsizedVeqTy, target, zero, last);
    return std::make_pair(SmallVector<Value>{qTarg},
                          SmallVector<Value>{ctrlSubveq});
  }
  return std::make_pair(SmallVector<Value>{target}, SmallVector<Value>{});
}

namespace {
// Type used to specialize the buildOp function. This extends the cases below by
// prefixing a single parameter value to the list of arguments for cases 1
// and 2. A Param does not have a case 3 defined.
class Param {};
} // namespace

static DenseBoolArrayAttr
negatedControlsAttribute(MLIRContext *ctx, ValueRange ctrls,
                         SmallVector<Value> &negations) {
  if (negations.empty())
    return {};
  SmallVector<bool> negatedControls(ctrls.size());
  for (auto v : llvm::enumerate(ctrls))
    negatedControls[v.index()] = std::find(negations.begin(), negations.end(),
                                           v.value()) != negations.end();
  auto boolVecAttr = DenseBoolArrayAttr::get(ctx, negatedControls);
  negations.clear();
  return {boolVecAttr};
}

// There are three basic overloads of the "single target" CUDA Quantum ops.
//
// 1. op(qubit...)
//    This form takes the last qubit as the target and all qubits to
//    the left as controls.
// 2. op(qurange, qubit)
//    Similar to above except the control qubits are packed in a
//    range container.
// 3. op(qurange)
//    This is not like the other 2. This is syntactic sugar for
//    invoking the op elementally across the entire range container.
//    There are no controls.
//
// In the future, it may be decided to add more overloads to this family (e.g.,
// adding controls to case 3).
template <typename A, typename P = void>
bool buildOp(OpBuilder &builder, Location loc, ValueRange operands,
             SmallVector<Value> &negations,
             llvm::function_ref<void()> reportNegateError,
             bool isAdjoint = false) {
  if constexpr (std::is_same_v<P, Param>) {
    assert(operands.size() >= 2 && "must be at least 2 operands");
    auto params = operands.take_front();
    auto [target, ctrls] =
        maybeUnpackOperands(builder, loc, operands.drop_front(1));
    for (auto v : target)
      if (std::find(negations.begin(), negations.end(), v) != negations.end())
        reportNegateError();
    auto negs =
        negatedControlsAttribute(builder.getContext(), ctrls, negations);
    builder.create<A>(loc, isAdjoint, params, ctrls, target, negs);
  } else {
    assert(operands.size() >= 1 && "must be at least 1 operand");
    if ((operands.size() == 1) && operands[0].getType().isa<quake::VeqType>()) {
      auto target = operands[0];
      if (!negations.empty())
        reportNegateError();
      Type indexTy = builder.getIndexType();
      auto size = builder.create<quake::VeqSizeOp>(
          loc, builder.getIntegerType(64), target);
      Value rank = builder.create<arith::IndexCastOp>(loc, indexTy, size);
      auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                             Block &block) {
        Value ref = builder.create<quake::ExtractRefOp>(loc, target,
                                                        block.getArgument(0));
        builder.create<A>(loc, ValueRange(), ref);
      };
      cudaq::opt::factory::createCountedLoop(builder, loc, rank, bodyBuilder);
    } else {
      auto [target, ctrls] = maybeUnpackOperands(builder, loc, operands);
      for (auto v : target)
        if (std::find(negations.begin(), negations.end(), v) != negations.end())
          reportNegateError();
      auto negs =
          negatedControlsAttribute(builder.getContext(), ctrls, negations);
      builder.create<A>(loc, isAdjoint, ValueRange(), ctrls, target, negs);
    }
  }
  return true;
}

static Value getConstantInt(OpBuilder &builder, Location loc,
                            const uint64_t value, const int bitwidth) {
  return builder.create<arith::ConstantIntOp>(loc, value,
                                              builder.getIntegerType(bitwidth));
}

static Value getConstantInt(OpBuilder &builder, Location loc,
                            const uint64_t value, Type intTy) {
  assert(intTy.isa<IntegerType>());
  return builder.create<arith::ConstantIntOp>(loc, value, intTy);
}

/// Is \p x the `operator[]` function?
static bool isSubscriptOperator(clang::CXXOperatorCallExpr *x) {
  return x->getOperator() == clang::OverloadedOperatorKind::OO_Subscript;
}

/// Is \p kindValue the `operator==` function?
static bool isCompareEqualOperator(clang::OverloadedOperatorKind kindValue) {
  return kindValue == clang::OverloadedOperatorKind::OO_EqualEqual;
}

/// Is \p x the `operator!` function?
static bool isExclaimOperator(clang::CXXOperatorCallExpr *x) {
  return x->getOperator() == clang::OverloadedOperatorKind::OO_Exclaim;
}

// Map the measured bit vector to an i32 representation.
static Value toIntegerImpl(OpBuilder &builder, Location loc, Value bitVec) {
  // TODO: Consider moving toIntegerImpl to an intrinsic.

  // Overall strategy in pseudo-C
  // auto toInteger = [](vector<i1> bits, int nBits) {
  //   int i = 0;
  //   for (int j = 0; j < nBits; j++) {
  //     // lsb
  //     k = nBits-j-1;
  //     i ^= (-bits[k] ^ i) & (1 << k);
  //   }
  //   return i;
  // };
  // should print 7
  // auto k = toInteger({1,1,1}, 3);
  // printf("%d\n", k);

  // get bitVec size
  Value bitVecSize = builder.create<cudaq::cc::StdvecSizeOp>(
      loc, builder.getI64Type(), bitVec);

  // Useful types and values
  auto i32Ty = builder.getI32Type();
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, i32Ty);
  Value negOne = builder.create<arith::ConstantIntOp>(loc, -1, i32Ty);

  // Create int i = 0;
  Value stackSlot = builder.create<cudaq::cc::AllocaOp>(loc, i32Ty);
  Value zeroInt = builder.create<arith::ConstantIntOp>(loc, 0, i32Ty);
  builder.create<cudaq::cc::StoreOp>(loc, zeroInt, stackSlot);

  // Create the for loop
  Value rank = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                  bitVecSize);
  cudaq::opt::factory::createCountedLoop(
      builder, loc, rank,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Region &,
          Block &block) {
        Value iv = block.getArgument(0);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        Value castedIndex = builder.create<arith::IndexCastOp>(loc, i32Ty, iv);

        // Truncate the i64 to a i32
        Value size = builder.create<arith::TruncIOp>(loc, i32Ty, bitVecSize);

        // Compute the idx in bit[idx]
        Value kIter = builder.create<arith::SubIOp>(loc, size, castedIndex);
        kIter = builder.create<arith::SubIOp>(loc, kIter, one);

        // 1 << k
        Value rightPart = builder.create<arith::ShLIOp>(loc, one, kIter);

        // bits[k]
        auto eleTy =
            cast<cudaq::cc::StdvecType>(bitVec.getType()).getElementType();
        auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
        auto vecPtr =
            builder.create<cudaq::cc::StdvecDataOp>(loc, elePtrTy, bitVec);
        auto eleAddr = builder.create<cudaq::cc::ComputePtrOp>(
            loc, elePtrTy, vecPtr, ValueRange{kIter});
        Value bitElement = builder.create<cudaq::cc::LoadOp>(loc, eleAddr);

        // -bits[k]
        bitElement = builder.create<arith::ExtUIOp>(loc, builder.getI32Type(),
                                                    bitElement);
        bitElement = builder.create<arith::MulIOp>(loc, negOne, bitElement);

        // -bits[k] ^ i
        Value integer = builder.create<cudaq::cc::LoadOp>(loc, stackSlot);
        Value leftPart =
            builder.create<arith::XOrIOp>(loc, bitElement, integer);

        // (-bits[k] & i ) & (1 << k)
        Value andVal = builder.create<arith::AndIOp>(loc, leftPart, rightPart);

        // i ^ andVal
        Value result = builder.create<arith::XOrIOp>(loc, integer, andVal);
        builder.create<cudaq::cc::StoreOp>(loc, result, stackSlot);
      });
  return builder.create<cudaq::cc::LoadOp>(loc, stackSlot);
}

// Perform the standard type coercions when the syntactic expression from the
// AST has arguments of different types.
static void castToSameType(OpBuilder builder, Location loc,
                           const clang::Type *lhsType, Value &lhs,
                           const clang::Type *rhsType, Value &rhs) {
  if (lhs.getType().getIntOrFloatBitWidth() ==
      rhs.getType().getIntOrFloatBitWidth())
    return;
  auto lhsTy = lhs.getType();
  auto rhsTy = rhs.getType();
  if (lhsTy.isa<IntegerType>() && rhsTy.isa<IntegerType>()) {
    if (lhsTy.getIntOrFloatBitWidth() < rhsTy.getIntOrFloatBitWidth()) {
      if (lhsType && lhsType->isUnsignedIntegerOrEnumerationType())
        lhs = builder.create<arith::ExtUIOp>(loc, rhs.getType(), lhs);
      else
        lhs = builder.create<arith::ExtSIOp>(loc, rhs.getType(), lhs);
      return;
    }
    if (rhsType && rhsType->isUnsignedIntegerOrEnumerationType())
      rhs = builder.create<arith::ExtUIOp>(loc, lhs.getType(), rhs);
    else
      rhs = builder.create<arith::ExtSIOp>(loc, lhs.getType(), rhs);
    return;
  }
  if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
    if (lhsTy.getIntOrFloatBitWidth() < rhsTy.getIntOrFloatBitWidth()) {
      lhs = builder.create<arith::ExtFOp>(loc, rhs.getType(), lhs);
      return;
    }
    rhs = builder.create<arith::ExtFOp>(loc, lhs.getType(), rhs);
    return;
  }
  if (lhsTy.isa<FloatType>() && rhsTy.isa<IntegerType>()) {
    if (rhsType && rhsType->isUnsignedIntegerOrEnumerationType())
      rhs = builder.create<arith::UIToFPOp>(loc, lhs.getType(), rhs);
    else
      rhs = builder.create<arith::SIToFPOp>(loc, lhs.getType(), rhs);
    return;
  }
  if (lhsTy.isa<IntegerType>() && rhsTy.isa<FloatType>()) {
    if (lhsType && lhsType->isUnsignedIntegerOrEnumerationType())
      lhs = builder.create<arith::UIToFPOp>(loc, rhs.getType(), lhs);
    else
      lhs = builder.create<arith::SIToFPOp>(loc, rhs.getType(), lhs);
    return;
  }
  TODO_loc(loc, "conversion of operands in binary expression");
}

/// Generalized kernel argument morphing. When traversing the AST, the calling
/// context's argument values that have already been created may be similar to
/// but not identical to the callee's signature types. This function deals with
/// adding the glue code to make the call strongly (exactly) type conforming.
static SmallVector<Value> convertKernelArgs(OpBuilder &builder, Location loc,
                                            std::size_t dropFrontNum,
                                            const SmallVector<Value> &args,
                                            ArrayRef<Type> kernelArgTys) {
  SmallVector<Value> result;
  assert(args.size() - dropFrontNum == kernelArgTys.size());
  for (auto i = dropFrontNum, end = args.size(); i < end; ++i) {
    auto v = args[i];
    auto vTy = v.getType();
    auto kTy = kernelArgTys[i - dropFrontNum];
    if (vTy == kTy) {
      result.push_back(v);
      continue;
    }
    if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(vTy))
      if (ptrTy.getElementType() == kTy) {
        // Promote pass-by-reference to pass-by-value.
        auto load = builder.create<cudaq::cc::LoadOp>(loc, v);
        result.push_back(load);
        continue;
      }
    if (auto vVecTy = dyn_cast<quake::VeqType>(vTy))
      if (auto kVecTy = dyn_cast<quake::VeqType>(kTy)) {
        // Both are Veq but the Veq are not identical. If the callee has a
        // dynamic size, we can relax the size from the calling context.
        if (vVecTy.hasSpecifiedSize() && !kVecTy.hasSpecifiedSize()) {
          auto relax = builder.create<quake::RelaxSizeOp>(loc, kVecTy, v);
          result.push_back(relax);
          continue;
        }
      }
    LLVM_DEBUG(llvm::dbgs() << "convert: " << v << "\nto:" << kTy << '\n');
    TODO_loc(loc, "argument type conversion");
  }
  return result;
}

static clang::CXXRecordDecl *
classDeclFromTemplateArgument(clang::FunctionDecl &func,
                              std::size_t argumentPosition,
                              clang::ASTContext &astContext) {
  if (auto *paramDecl = func.getParamDecl(argumentPosition))
    if (auto *defn = paramDecl->getDefinition(astContext)) {
      // Check `auto &&` case.
      if (auto *rvalueRefTy = dyn_cast<clang::RValueReferenceType>(
              defn->getType().getTypePtr()))
        if (auto *substTmpl = dyn_cast<clang::SubstTemplateTypeParmType>(
                rvalueRefTy->getPointeeType().getTypePtr())) {
          auto qualTy = substTmpl->getReplacementType();
          return qualTy.getTypePtr()->getAsCXXRecordDecl();
        }
      // Check `class-name &` case.
      if (auto *lvalueRefTy = dyn_cast<clang::LValueReferenceType>(
              defn->getType().getTypePtr()))
        return lvalueRefTy->getPointeeType().getTypePtr()->getAsCXXRecordDecl();
    }
  return nullptr;
}

/// Is this type name one of the `cudaq` types that map to a VeqType?
static bool isCudaQType(StringRef tn) {
  return tn.equals("qreg") || tn.equals("qspan") || tn.equals("qarray") ||
         tn.equals("qview") || tn.equals("qvector");
}

namespace cudaq::details {
/// Is \p x the `operator()` function?
static bool isCallOperator(clang::CXXOperatorCallExpr *x) {
  return cudaq::isCallOperator(x->getOperator());
}

FunctionType QuakeBridgeVisitor::peelPointerFromFunction(Type ty) {
  if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty))
    ty = ptrTy.getElementType();
  return cast<FunctionType>(ty);
}

bool QuakeBridgeVisitor::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *x) {
  // TODO: add array support
  reportClangError(x, mangler, "arrays in kernels");
  return false;
}

bool QuakeBridgeVisitor::VisitFloatingLiteral(clang::FloatingLiteral *x) {
  // Literals do not push a type on the type stack.
  auto loc = toLocation(x->getSourceRange());
  auto bltTy = cast<clang::BuiltinType>(x->getType().getTypePtr());
  auto fltTy = cast<FloatType>(builtinTypeToType(bltTy));
  auto fltVal = x->getValue();
  return pushValue(builder.create<arith::ConstantFloatOp>(loc, fltVal, fltTy));
}

bool QuakeBridgeVisitor::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto intTy =
      builtinTypeToType(cast<clang::BuiltinType>(x->getType().getTypePtr()));
  auto intVal = x->getValue();
  return pushValue(getConstantInt(builder, loc, intVal ? 1 : 0, intTy));
}

bool QuakeBridgeVisitor::VisitIntegerLiteral(clang::IntegerLiteral *x) {
  auto loc = toLocation(x->getSourceRange());
  auto intTy =
      builtinTypeToType(cast<clang::BuiltinType>(x->getType().getTypePtr()));
  auto intVal = x->getValue().getLimitedValue();
  return pushValue(getConstantInt(builder, loc, intVal, intTy));
}

bool QuakeBridgeVisitor::VisitUnaryOperator(clang::UnaryOperator *x) {
  auto loc = toLocation(x->getSourceRange());
  switch (x->getOpcode()) {
  case clang::UnaryOperatorKind::UO_PostInc: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto incremented = builder.create<arith::AddIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, incremented, var);
    return pushValue(loaded);
  }
  case clang::UnaryOperatorKind::UO_PreInc: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto incremented = builder.create<arith::AddIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, incremented, var);
    return pushValue(incremented);
  }
  case clang::UnaryOperatorKind::UO_PostDec: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto decremented = builder.create<arith::SubIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, decremented, var);
    return pushValue(loaded);
  }
  case clang::UnaryOperatorKind::UO_PreDec: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto decremented = builder.create<arith::SubIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, decremented, var);
    return pushValue(decremented);
  }
  case clang::UnaryOperatorKind::UO_LNot: {
    auto var = popValue();
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, var.getType());
    Value unaryNot =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, var, zero);
    return pushValue(unaryNot);
  }
  case clang::UnaryOperatorKind::UO_Minus: {
    auto subExpr = popValue();
    auto resTy = subExpr.getType();
    if (resTy.isa<IntegerType>())
      return pushValue(builder.create<arith::MulIOp>(
          loc, subExpr,
          getConstantInt(builder, loc, -1, resTy.getIntOrFloatBitWidth())));

    if (resTy.isa<FloatType>()) {
      llvm::APFloat d(-1.0);
      auto neg_one = builder.create<arith::ConstantFloatOp>(
          loc, d, cast<FloatType>(resTy));
      return pushValue(builder.create<arith::MulFOp>(loc, subExpr, neg_one));
    }
    TODO_x(loc, x, mangler, "unknown type for unary minus");
    return false;
  }
  case clang::UnaryOperatorKind::UO_Deref: {
    auto subExpr = popValue();
    assert(isa<cc::PointerType>(subExpr.getType()));
    return pushValue(builder.create<cc::LoadOp>(loc, subExpr));
  }
  case clang::UnaryOperatorKind::UO_Extension: {
    TODO_x(loc, x, mangler, "__extension__ operator");
    return false;
  }
  case clang::UnaryOperatorKind::UO_Coawait: {
    TODO_x(loc, x, mangler, "co_await operator");
    return false;
  }
  }
  TODO_x(loc, x, mangler, "unprocessed unary operator");
  return false;
}

Value QuakeBridgeVisitor::floatingPointCoercion(Location loc, Type toType,
                                                Value value) {
  auto fromType = value.getType();
  if (toType == fromType)
    return value;
  if (fromType.isa<IntegerType>() && toType.isa<IntegerType>()) {
    if (fromType.getIntOrFloatBitWidth() < toType.getIntOrFloatBitWidth())
      return builder.create<arith::ExtFOp>(loc, toType, value);
    if (fromType.getIntOrFloatBitWidth() > toType.getIntOrFloatBitWidth())
      return builder.create<arith::TruncFOp>(loc, toType, value);
    TODO_loc(loc, "floating point types are distinct and same size");
  }
  TODO_loc(loc, "Float conversion but not floating point types");
}

Value QuakeBridgeVisitor::integerCoercion(Location loc,
                                          const clang::QualType &clangTy,
                                          Type dstTy, Value srcVal) {
  auto fromTy = getResultType(srcVal.getType());
  if (dstTy == fromTy)
    return srcVal;

  if (fromTy.isa<IntegerType>() && dstTy.isa<IntegerType>()) {
    if (fromTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth()) {
      if (clangTy->isUnsignedIntegerOrEnumerationType())
        return builder.create<arith::ExtUIOp>(loc, dstTy, srcVal);
      return builder.create<arith::ExtSIOp>(loc, dstTy, srcVal);
    }
    if (fromTy.getIntOrFloatBitWidth() > dstTy.getIntOrFloatBitWidth())
      return builder.create<arith::TruncIOp>(loc, dstTy, srcVal);
    TODO_loc(loc, "Types are not the same but have the same length");
  }
  TODO_loc(loc, "Integer conversion but not integer types");
}

bool QuakeBridgeVisitor::TraverseImplicitCastExpr(clang::ImplicitCastExpr *x,
                                                  DataRecursionQueue *) {
  // RecursiveASTVisitor is tuned for dumping surface syntax so doesn't visit
  // the type. Override so that the casted to type is visited and pushed on the
  // stack.
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (!TraverseType(x->getType()))
    return false;
  assert(typeStack.size() == typeStackDepth + 1 && "must push a type");
  auto result = Base::TraverseImplicitCastExpr(x);
  if (result) {
    assert(typeStack.size() == typeStackDepth && "must be original depth");
  }
  return result;
}

bool QuakeBridgeVisitor::VisitImplicitCastExpr(clang::ImplicitCastExpr *x) {
  // The type to cast the expression into is pushed during the traversal of the
  // ImplicitCastExpr in non-error cases.
  auto castToTy = popType();
  if (x->getCastKind() == clang::CastKind::CK_FunctionToPointerDecay)
    return true; // NOP

  auto loc = toLocation(x);
  auto intToIntCast = [&](Location locSub, Value mlirVal) {
    clang::QualType srcTy = x->getSubExpr()->getType();
    return pushValue(integerCoercion(locSub, srcTy, castToTy, mlirVal));
  };

  switch (x->getCastKind()) {
  case clang::CastKind::CK_LValueToRValue: {
    auto subValue = loadLValue(popValue());
    return pushValue(subValue);
  }
  case clang::CastKind::CK_FloatingCast: {
    auto dstType = x->getType();
    auto val = x->getSubExpr();
    assert(val->getType()->isFloatingType() && dstType->isFloatingType());
    auto value = popValue();
    auto toType = cast<FloatType>(castToTy);
    auto fromType = cast<FloatType>(value.getType());
    assert(toType && fromType);
    if (toType == fromType)
      return pushValue(value);
    if (fromType.getIntOrFloatBitWidth() < toType.getIntOrFloatBitWidth())
      return pushValue(builder.create<arith::ExtFOp>(loc, toType, value));
    return pushValue(builder.create<arith::TruncFOp>(loc, toType, value));
  }
  case clang::CastKind::CK_ArrayToPointerDecay:
    if (dyn_cast_or_null<clang::StringLiteral>(x->getSubExpr()))
      return true;
    // TODO: array support
    TODO_x(loc, x, mangler, "arrays in kernels");
    return false;
  case clang::CastKind::CK_IntegralCast: {
    auto locSub = toLocation(x->getSubExpr());
    auto result = intToIntCast(locSub, popValue());
    assert(result && "integer conversion failed");
    return result;
  }
  case clang::CastKind::CK_NoOp:
    return true;
  case clang::CastKind::CK_FloatingToIntegral: {
    auto qualTy = x->getType();
    if (qualTy->isUnsignedIntegerOrEnumerationType())
      return pushValue(
          builder.create<arith::FPToUIOp>(loc, castToTy, popValue()));
    return pushValue(
        builder.create<arith::FPToSIOp>(loc, castToTy, popValue()));
  }
  case clang::CastKind::CK_IntegralToFloating: {
    if (x->getSubExpr()->getType()->isUnsignedIntegerOrEnumerationType())
      return pushValue(
          builder.create<arith::UIToFPOp>(loc, castToTy, popValue()));
    return pushValue(
        builder.create<arith::SIToFPOp>(loc, castToTy, popValue()));
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto last = popValue();
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, last.getType());
    return pushValue(builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, last, zero));
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    auto sub = popValue();
    // castToTy is the converion function signature.
    castToTy = popType();
    if (isa<IntegerType>(castToTy) && isa<IntegerType>(sub.getType())) {
      auto locSub = toLocation(x->getSubExpr());
      bool result = intToIntCast(locSub, sub);
      assert(result && "integer conversion failed");
      return result;
    }
    TODO_loc(loc, "unhandled user defined implicit conversion");
  }
  case clang::CastKind::CK_ConstructorConversion: {
    // Enable implicit conversion of surface types, which both map to VeqType.
    if (isa<quake::VeqType>(castToTy))
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1 &&
            isa<quake::VeqType>(peekValue().getType()))
          return true;
    // ... or which both map to RefType.
    if (isa<quake::RefType>(castToTy))
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1 &&
            isa<quake::RefType>(peekValue().getType()))
          return true;

    // Enable implicit conversion of lambda -> std::function, which are both
    // cc::CallableType.
    if (isa<cc::CallableType>(castToTy)) {
      // Enable implicit conversion of callable -> std::function.
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1)
          return true;
    }
    if (auto funcTy = peelPointerFromFunction(castToTy))
      if (auto fromTy = dyn_cast<cc::CallableType>(peekValue().getType())) {
        auto inputs = funcTy.getInputs();
        if (!inputs.empty() && inputs[0] == fromTy)
          return true;
      }

    TODO_loc(loc, "unhandled implicit cast expression: constructor conversion");
  }
  }

  // Handle the case where we have if ( vec[i] ), where vec == vector<i32>.
  // This leads to an ImplicitCastExpr (IntegralToBoolean) -> ImplicitCastExpr
  // (LvalueToRvalue)
  if (auto anotherCast = dyn_cast<clang::ImplicitCastExpr>(x->getSubExpr())) {
    if (!VisitImplicitCastExpr(anotherCast))
      return false;
    if (x->getCastKind() == clang::CastKind::CK_IntegralToBoolean) {
      auto last = popValue();
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, last.getType());
      return pushValue(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, last, zero));
    }
  }
  TODO_loc(loc, "unhandled implicit cast expression");
}

bool QuakeBridgeVisitor::TraverseBinaryOperator(clang::BinaryOperator *x,
                                                DataRecursionQueue *) {
  bool shortCircuitWhenTrue =
      x->getOpcode() == clang::BinaryOperatorKind::BO_LOr;

  // The && and || operators are semantically if statements. Traverse them
  // differently than other expressions since both sides of the expression are
  // not always evaluated.
  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_LAnd:
  case clang::BinaryOperatorKind::BO_LOr: {
    auto *lhs = x->getLHS();
    if (!TraverseStmt(lhs))
      return false;
    auto lhsVal = popValue();
    auto loc = toLocation(x->getSourceRange());
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, lhsVal.getType());
    Value cond = builder.create<arith::CmpIOp>(loc,
                                               shortCircuitWhenTrue
                                                   ? arith::CmpIPredicate::ne
                                                   : arith::CmpIPredicate::eq,
                                               lhsVal, zero);
    bool result = true;
    auto ifOp = builder.create<cc::IfOp>(
        loc, TypeRange{cond.getType()}, cond,
        [=](OpBuilder &builder, Location loc, Region &region) {
          // Short-circuit taken: return the result of the lhs and do not
          // evaluate the rhs at all.
          region.push_back(new Block{});
          auto &bodyBlock = region.front();
          OpBuilder::InsertionGuard guad(builder);
          builder.setInsertionPointToStart(&bodyBlock);
          builder.create<cc::ContinueOp>(loc, TypeRange{}, cond);
        },
        [&result, this, rhs = x->getRHS()](OpBuilder &builder, Location loc,
                                           Region &region) {
          // Short-circuit not taken: evaluate the rhs and return that value.
          region.push_back(new Block{});
          auto &bodyBlock = region.front();
          OpBuilder::InsertionGuard guad(builder);
          builder.setInsertionPointToStart(&bodyBlock);
          if (!TraverseStmt(rhs)) {
            result = false;
            return;
          }
          auto rhsVal = popValue();
          builder.create<cc::ContinueOp>(loc, TypeRange{}, rhsVal);
        });
    if (!result)
      return result;
    return pushValue(ifOp.getResult(0));
  }
  default:
    break;
  }
  return Base::TraverseBinaryOperator(x);
}

bool QuakeBridgeVisitor::VisitBinaryOperator(clang::BinaryOperator *x) {
  auto rhs = popValue();
  auto lhs = popValue();
  auto loc = toLocation(x->getSourceRange());
  auto maybeLoadValue = [&](Value v) -> Value {
    if (isa<cc::PointerType>(v.getType()))
      return builder.create<cc::LoadOp>(loc, v);
    return v;
  };

  if (x->isRelationalOp() ||
      x->getOpcode() == clang::BinaryOperatorKind::BO_EQ ||
      x->getOpcode() == clang::BinaryOperatorKind::BO_NE) {
    rhs = maybeLoadValue(rhs);
    lhs = maybeLoadValue(lhs);
    // Floating point comparison?
    if (lhs.getType().isa<FloatType>()) {
      arith::CmpFPredicate pred;
      switch (x->getOpcode()) {
      case clang::BinaryOperatorKind::BO_LT:
        pred = arith::CmpFPredicate::OLT;
        break;
      case clang::BinaryOperatorKind::BO_LE:
        pred = arith::CmpFPredicate::OLE;
        break;
      case clang::BinaryOperatorKind::BO_EQ:
        pred = arith::CmpFPredicate::OEQ;
        break;
      case clang::BinaryOperatorKind::BO_GT:
        pred = arith::CmpFPredicate::OGT;
        break;
      case clang::BinaryOperatorKind::BO_GE:
        pred = arith::CmpFPredicate::OGE;
        break;
      case clang::BinaryOperatorKind::BO_NE:
        pred = arith::CmpFPredicate::ONE;
        break;
      default:
        TODO_loc(loc, "floating-point comparison");
      }
      return pushValue(builder.create<arith::CmpFOp>(loc, pred, lhs, rhs));
    }
    arith::CmpIPredicate pred;
    auto lhsTy = x->getLHS()->getType();
    auto rhsTy = x->getRHS()->getType();
    // Favor unsigned comparisons via implicit conversion.
    bool isUnsigned = lhsTy->isUnsignedIntegerOrEnumerationType() ||
                      rhsTy->isUnsignedIntegerOrEnumerationType();
    switch (x->getOpcode()) {
    case clang::BinaryOperatorKind::BO_LT:
      pred = isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
      break;
    case clang::BinaryOperatorKind::BO_LE:
      pred = isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
      break;
    case clang::BinaryOperatorKind::BO_EQ:
      pred = arith::CmpIPredicate::eq;
      break;
    case clang::BinaryOperatorKind::BO_GT:
      pred = isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
      break;
    case clang::BinaryOperatorKind::BO_GE:
      pred = isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
      break;
    case clang::BinaryOperatorKind::BO_NE:
      pred = arith::CmpIPredicate::ne;
      break;
    default:
      TODO_loc(loc, "integer comparison");
    }
    return pushValue(builder.create<arith::CmpIOp>(loc, pred, lhs, rhs));
  }

  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Assign: {
    builder.create<cc::StoreOp>(loc, rhs, lhs);
    return pushValue(lhs);
  }
  case clang::BinaryOperatorKind::BO_AddAssign:
  case clang::BinaryOperatorKind::BO_SubAssign:
  case clang::BinaryOperatorKind::BO_MulAssign:
  case clang::BinaryOperatorKind::BO_DivAssign:
  case clang::BinaryOperatorKind::BO_ShlAssign:
  case clang::BinaryOperatorKind::BO_ShrAssign:
  case clang::BinaryOperatorKind::BO_OrAssign:
  case clang::BinaryOperatorKind::BO_XorAssign:
  case clang::BinaryOperatorKind::BO_AndAssign:

    return true; // see CompoundAssignOperator
  default:
    break;
  }
  rhs = maybeLoadValue(rhs);
  lhs = maybeLoadValue(lhs);
  castToSameType(builder, loc, x->getLHS()->getType().getTypePtrOrNull(), lhs,
                 x->getRHS()->getType().getTypePtrOrNull(), rhs);
  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Add: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::AddIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::AddFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }
  case clang::BinaryOperatorKind::BO_Rem: {
    if (x->getType()->isIntegerType()) {
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return pushValue(builder.create<arith::RemUIOp>(loc, lhs, rhs));
      return pushValue(builder.create<arith::RemSIOp>(loc, lhs, rhs));
    }
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::AddFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }
  case clang::BinaryOperatorKind::BO_Sub: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::SubIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::SubFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }

  case clang::BinaryOperatorKind::BO_Mul: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::MulIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::MulFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_mul binary op");
  }

  case clang::BinaryOperatorKind::BO_Div: {
    if (x->getType()->isIntegerType()) {
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return pushValue(builder.create<arith::DivUIOp>(loc, lhs, rhs));
      return pushValue(builder.create<arith::DivSIOp>(loc, lhs, rhs));
    }
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::DivFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_div binary op");
  }

  case clang::BinaryOperatorKind::BO_Shl:
    return pushValue(builder.create<arith::ShLIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Shr:
    if (x->getLHS()->getType()->isUnsignedIntegerOrEnumerationType())
      return pushValue(builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs));
    return pushValue(builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Or:
    return pushValue(builder.create<arith::OrIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Xor:
    return pushValue(builder.create<arith::XOrIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_And:
    return pushValue(builder.create<arith::AndIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_LAnd:
  case clang::BinaryOperatorKind::BO_LOr:
    emitFatalError(loc, "&& and || ops are handled elsewhere.");
  }
  TODO_loc(loc, "unknown binary kind operator");
}

std::string QuakeBridgeVisitor::genLoweredName(clang::FunctionDecl *x,
                                               FunctionType funcTy) {
  auto loc = toLocation(x->getSourceRange());
  std::string result = [&]() {
    for (auto &pair : functionsToEmit)
      if (x == pair.second)
        return generateCudaqKernelName(pair);
    return cxxMangledDeclName(x);
  }();
  // Add the called function to the module as needed.
  getOrAddFunc(loc, result, funcTy);
  return result;
}

bool QuakeBridgeVisitor::VisitConditionalOperator(
    clang::ConditionalOperator *x) {
  auto args = lastValues(3);
  auto loc = toLocation(x->getSourceRange());
  auto ty = args[1].getType();
  auto select =
      builder.create<arith::SelectOp>(loc, ty, args[0], args[1], args[2]);
  return pushValue(select);
}

bool QuakeBridgeVisitor::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto ty = peekValue().getType();

  // The following cases are λ expressions, quantum data, or a std::vector view.
  // In those cases, there is nothing to materialize, so we can just pass the
  // Value on the top of the stack.
  if (isa<cc::CallableType, quake::VeqType, quake::RefType, cc::StdvecType>(ty))
    return true;

  // If not one of the above special cases, then materialize the value to a
  // temporary memory location and push the address to the stack.

  // Is it already materialized in memory?
  if (isa<cc::PointerType>(ty))
    return true;

  // Materialize the value into a glvalue location in memory.
  auto materialize = builder.create<cc::AllocaOp>(loc, ty);
  builder.create<cc::StoreOp>(loc, popValue(), materialize);
  return pushValue(materialize);
}

bool QuakeBridgeVisitor::TraverseLambdaExpr(clang::LambdaExpr *x,
                                            DataRecursionQueue *) {
  auto loc = toLocation(x->getSourceRange());
  bool result = true;
  if (!x->explicit_captures().empty()) {
    // Lambda expression with explicit capture list is not supported yet.
    TODO_x(loc, x, mangler, "lambda expression with explicit captures");
  }
  if (!TraverseType(x->getType()))
    return false;
  auto callableTy = cast<cc::CallableType>(popType());
  auto lambdaInstance = builder.create<cc::CreateLambdaOp>(
      loc, callableTy, [&](OpBuilder &builder, Location loc) {
        // FIXME: the capture list, etc. should be visited in an appropriate
        // context here, not as part of lowering the body of the lambda.
        auto *entryBlock = builder.getInsertionBlock();
        SymbolTableScope argsScope(symbolTable);
        addArgumentSymbols(entryBlock, x->getCallOperator()->parameters());
        if (!TraverseStmt(x->getBody())) {
          result = false;
          return;
        }
        builder.create<cc::ReturnOp>(loc);
      });
  pushValue(lambdaInstance);
  return result;
}

bool QuakeBridgeVisitor::TraverseMemberExpr(clang::MemberExpr *x,
                                            DataRecursionQueue *) {
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  // FIXME!
  // Accessing data members is not currently supported. For function members, we
  // want to push the type of the function, since the visit to CallExpr requires
  // a type to have been pushed.
  if (auto *methodDecl = dyn_cast<clang::CXXMethodDecl>(x->getMemberDecl()))
    if (!TraverseType(methodDecl->getType()))
      return false;
  assert(typeStack.size() == typeStackDepth + 1);
  return Base::TraverseMemberExpr(x);
}

bool QuakeBridgeVisitor::VisitCallExpr(clang::CallExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  // The called function is reified as a Value in the IR.
  auto *callee = x->getCalleeDecl();
  auto *func = dyn_cast<clang::FunctionDecl>(callee);
  if (!func)
    TODO_loc(loc, "call doesn't have function decl");
  assert(valueStack.size() >= x->getNumArgs() + 1 &&
         "stack must contain all arguments plus the expression to call");
  StringRef funcName;
  if (auto *id = func->getIdentifier())
    funcName = id->getName();

  // Handle any std::pow(N,M)
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      funcName.equals("pow")) {
    auto funcArity = func->getNumParams();
    SmallVector<Value> args = lastValues(funcArity);
    auto powFun = popValue();

    // Get the values involved
    auto peelIntToFloat = [&](Value v) {
      if (auto op = v.getDefiningOp<arith::SIToFPOp>())
        return op.getOperand();
      return v;
    };
    Value base = peelIntToFloat(args[0]);
    Value power = peelIntToFloat(args[1]);
    Type baseType = base.getType();
    Type powerType = power.getType();

    // Create the power op based on the types of the arguments.
    if (isa<IntegerType>(powerType)) {
      if (isa<IntegerType>(baseType)) {
        auto calleeTy = peelPointerFromFunction(powFun.getType());
        auto resTy = calleeTy.getResult(0);
        castToSameType(builder, loc, x->getArg(0)->getType().getTypePtrOrNull(),
                       base, x->getArg(1)->getType().getTypePtrOrNull(), power);
        auto ipow = builder.create<math::IPowIOp>(loc, base, power);
        if (isa<FloatType>(resTy))
          return pushValue(builder.create<arith::SIToFPOp>(loc, resTy, ipow));
        assert(resTy == ipow.getType());
        return pushValue(ipow);
      }
      return pushValue(builder.create<math::FPowIOp>(loc, base, power));
    }
    return pushValue(builder.create<math::PowFOp>(loc, base, power));
  }

  // Dealing with our std::vector as a view data structures. If we have some θ
  // with the type `std::vector<double/float/int>`, and in the kernel, θ.size()
  // is called, we need to convert that to loading the size field of the pair.
  // For θ.empty(), the size is loaded and compared to zero.
  if (isInClassInNamespace(func, "vector", "std")) {
    // Get the size of the std::vector.
    auto svec = popValue();
    auto ext =
        builder.create<cc::StdvecSizeOp>(loc, builder.getI64Type(), svec);
    if (funcName.equals("size"))
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(ext);
        }
    if (funcName.equals("empty"))
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(builder.create<mlir::arith::CmpIOp>(
              ext->getLoc(), arith::CmpIPredicate(arith::CmpIPredicate::eq),
              ext.getResult(),
              getConstantInt(
                  builder, ext->getLoc(), 0,
                  ext.getResult().getType().getIntOrFloatBitWidth())));
        }
    TODO_loc(loc, "unhandled std::vector member function, " + funcName);
  }

  if (isInClassInNamespace(func, "_Bit_reference", "std")) {
    // Calling std::_Bit_reference::method().
    auto loadFromReference = [&](mlir::Value ref) -> Value {
      if (auto mrTy = dyn_cast<cc::PointerType>(ref.getType())) {
        assert(mrTy.getElementType() == builder.getI1Type());
        return builder.create<cc::LoadOp>(loc, ref);
      }
      assert(ref.getType() == builder.getI1Type());
      return ref;
    };
    if (isa<clang::CXXConversionDecl>(func)) {
      assert(isa<cc::PointerType>(peekValue().getType()));
      return pushValue(builder.create<cc::LoadOp>(loc, popValue()));
    }
    if (func->isOverloadedOperator() &&
        isCompareEqualOperator(func->getOverloadedOperator())) {
      auto rhs = loadFromReference(popValue());
      auto lhs = loadFromReference(popValue());
      popValue(); // The compare equal operator address.
      return pushValue(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lhs, rhs));
    }
    TODO_loc(loc, "unhandled std::vector<bool> member function, " + funcName);
  }

  if (isInClassInNamespace(func, "qreg", "cudaq") ||
      isInClassInNamespace(func, "qspan", "cudaq")) {
    // This handles conversion of qreg.size()
    if (funcName.equals("size"))
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto qregArg = popValue();
          auto qrSize = builder.create<quake::VeqSizeOp>(
              loc, builder.getI64Type(), qregArg);
          return pushValue(qrSize);
        }

    if (funcName.equals("front"))
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto zero = getConstantInt(builder, loc, 0, 64);
          if (actArgs.size() == 1) {
            auto qrSize = actArgs.front();
            auto one = getConstantInt(builder, loc, 1, 64);
            auto offset = builder.create<arith::SubIOp>(loc, qrSize, one);
            auto unsizedVecTy =
                quake::VeqType::getUnsized(builder.getContext());
            return pushValue(builder.create<quake::SubVeqOp>(
                loc, unsizedVecTy, qregArg, zero, offset));
          }
          assert(actArgs.size() == 0);
          popValue();
          return pushValue(
              builder.create<quake::ExtractRefOp>(loc, qregArg, zero));
        }

    if (funcName.equals("back"))
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto qrSize = builder.create<quake::VeqSizeOp>(
              loc, builder.getI64Type(), qregArg);
          auto one = getConstantInt(builder, loc, 1, 64);
          auto endOff = builder.create<arith::SubIOp>(loc, qrSize, one);
          if (actArgs.size() == 1) {
            auto startOff =
                builder.create<arith::SubIOp>(loc, qrSize, actArgs.front());
            auto unsizedVecTy =
                quake::VeqType::getUnsized(builder.getContext());
            popValue();
            return pushValue(builder.create<quake::SubVeqOp>(
                loc, unsizedVecTy, qregArg, startOff, endOff));
          }
          assert(actArgs.size() == 0);
          return pushValue(
              builder.create<quake::ExtractRefOp>(loc, qregArg, endOff));
        }

    if (funcName.equals("slice")) {
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto start = actArgs[0];
          auto count = actArgs[1];

          auto one = getConstantInt(builder, loc, 1, 64);
          Value offset = builder.create<arith::AddIOp>(loc, start, count);
          offset = builder.create<arith::SubIOp>(loc, offset, one);
          auto unsizedVecTy = quake::VeqType::getUnsized(builder.getContext());
          return pushValue(builder.create<quake::SubVeqOp>(
              loc, unsizedVecTy, qregArg, start, offset));
        }
    }

    TODO_loc(loc, "unhandled cudaq member function, " + funcName);
  }

  auto funcArity = func->getNumParams();
  SmallVector<Value> args = lastValues(funcArity);
  if (isa<clang::CXXMethodDecl>(func)) {
    [[maybe_unused]] auto thisPtrValue = popValue();
  }
  auto calleeOp = popValue();

  if (isInNamespace(func, "cudaq")) {
    // Check and see if this quantum operation is adjoint
    bool isAdjoint = false;
    auto *functionDecl = x->getCalleeDecl()->getAsFunction();
    if (auto *templateArgs = functionDecl->getTemplateSpecializationArgs())
      if (templateArgs->size() > 0) {
        auto gateModifierArg = templateArgs->asArray()[0];
        if (gateModifierArg.getKind() == clang::TemplateArgument::ArgKind::Type)
          if (auto *structTy =
                  gateModifierArg.getAsType()->getAsStructureType())
            if (auto structTypeAsRecord = structTy->getAsCXXRecordDecl())
              isAdjoint = structTypeAsRecord->getName() == "adj";
      }

    if (funcName.equals("mx") || funcName.equals("my") ||
        funcName.equals("mz")) {
      // Measurements always return a bool or a std::vector<bool>.
      Type resTy = builder.getI1Type();
      if ((args.size() > 1) ||
          (args.size() == 1 && args[0].getType().isa<quake::VeqType>()))
        resTy = cc::StdvecType::get(builder.getI1Type());
      if (funcName.equals("mx"))
        return pushValue(builder.create<quake::MxOp>(loc, resTy, args));
      if (funcName.equals("my"))
        return pushValue(builder.create<quake::MyOp>(loc, resTy, args));
      return pushValue(builder.create<quake::MzOp>(loc, resTy, args));
    }

    // Handle the quantum gate set.
    auto reportNegateError = [&]() {
      reportClangError(x, mangler, "target qubit cannot be negated");
    };
    if (funcName.equals("h") || funcName.equals("ch"))
      return buildOp<quake::HOp>(builder, loc, args, negations,
                                 reportNegateError);
    if (funcName.equals("x") || funcName.equals("cnot") ||
        funcName.equals("cx") || funcName.equals("ccx"))
      return buildOp<quake::XOp>(builder, loc, args, negations,
                                 reportNegateError);
    if (funcName.equals("y") || funcName.equals("cy"))
      return buildOp<quake::YOp>(builder, loc, args, negations,
                                 reportNegateError);
    if (funcName.equals("z") || funcName.equals("cz"))
      return buildOp<quake::ZOp>(builder, loc, args, negations,
                                 reportNegateError);
    if (funcName.equals("s") || funcName.equals("cs"))
      return buildOp<quake::SOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint);
    if (funcName.equals("t") || funcName.equals("ct"))
      return buildOp<quake::TOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint);

    if (funcName.equals("reset")) {
      if (!negations.empty())
        reportNegateError();
      return builder.create<quake::ResetOp>(loc, TypeRange{}, args[0]);
    }
    if (funcName.equals("swap")) {
      const auto size = args.size();
      assert(size >= 2);
      SmallVector<Value> targets(args.begin() + size - 2, args.end());
      for (auto v : targets)
        if (std::find(negations.begin(), negations.end(), v) != negations.end())
          reportNegateError();
      SmallVector<Value> ctrls(args.begin(), args.begin() + size - 2);
      auto negs =
          negatedControlsAttribute(builder.getContext(), ctrls, negations);
      auto swap = builder.create<quake::SwapOp>(loc, ctrls, targets);
      if (negs)
        swap->setAttr("negated_qubit_controls", negs);
      return true;
    }
    if (funcName.equals("p") || funcName.equals("r1"))
      return buildOp<quake::R1Op, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint);
    if (funcName.equals("rx"))
      return buildOp<quake::RxOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint);
    if (funcName.equals("ry"))
      return buildOp<quake::RyOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint);
    if (funcName.equals("rz"))
      return buildOp<quake::RzOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint);

    if (funcName.equals("control")) {
      // Expect the first argument to be an instance of a Callable. Need to
      // construct the name of the operator() call to make here.
      Value calleeValue = args[0];
      Value ctrlValues = args[1];
      SymbolRefAttr calleeSymbol;
      auto *ctx = builder.getContext();

      // Expand the negations inline around the quake.apply. This will result in
      // less duplication of code than threading the negated sense of the
      // control recursively through the callable.
      auto inlinedStartControlNegations = [&]() {
        if (!negations.empty()) {
          // Loop over the ctrlValues and negate (apply an XOp) those in the
          // negations list.
          if (auto concat = ctrlValues.getDefiningOp<quake::ConcatOp>()) {
            for (auto v : concat.getQbits())
              if (std::find(negations.begin(), negations.end(), v) !=
                  negations.end()) {
                if (isa<quake::VeqType>(v.getType())) {
                  reportClangError(
                      x, mangler, "cannot negate an entire register of qubits");
                } else {
                  SmallVector<Value> dummy;
                  buildOp<quake::XOp>(builder, loc, v, dummy, []() {});
                }
              }
          } else if (isa<quake::VeqType>(ctrlValues.getType())) {
            assert(negations.size() == 1 && negations[0] == ctrlValues);
            reportClangError(x, mangler,
                             "cannot negate an entire register of qubits");
          } else {
            assert(isa<quake::RefType>(ctrlValues.getType()));
            assert(negations.size() == 1 && negations[0] == ctrlValues);
            SmallVector<Value> dummy;
            buildOp<quake::XOp>(builder, loc, ctrlValues, dummy, []() {});
          }
        }
      };

      // Finish (uncompute) the inlined control negations. Generates the same
      // code pattern as the starting negations. Specifically, we invoke an XOp
      // on each negated control.
      auto inlinedFinishControlNegations = [&]() {
        inlinedStartControlNegations();
        negations.clear();
        return true;
      };
      auto callableObjectStructType = [&](Value v) {
        Type ty = v.getType();
        if (auto ptrTy = dyn_cast<cc::PointerType>(ty))
          ty = ptrTy.getElementType();
        return dyn_cast<cc::StructType>(ty);
      };

      if (auto ty = callableObjectStructType(calleeValue)) {
        auto *classDecl = classDeclFromTemplateArgument(*func, 0, *astContext);
        if (!classDecl) {
          // This shouldn't happen if the cudaq headers are used, but add a
          // check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "expected cudaq::control to be a specific template");
          de.Report(x->getBeginLoc(), id);
          return false;
        }
        auto *kernelCallOper = findCallOperator(classDecl);
        if (!kernelCallOper) {
          // This should be caught by the concepts used in the header file, but
          // add a check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "first argument to cudaq::control must be a callable");
          de.Report(x->getBeginLoc(), id);
          return false;
        }
        auto calleeName = generateCudaqKernelName(kernelCallOper);
        calleeSymbol = SymbolRefAttr::get(ctx, calleeName);
        auto kernelFunc = module.lookupSymbol<func::FuncOp>(calleeName);
        assert(kernelFunc && "kernel call operator must be present");
        auto kernelTy = kernelFunc.getFunctionType();
        auto kernelArgs =
            convertKernelArgs(builder, loc, 2, args, kernelTy.getInputs());
        inlinedStartControlNegations();
        builder.create<quake::ApplyOp>(loc, TypeRange{}, calleeSymbol,
                                       /*isAdjoint=*/false, ctrlValues,
                                       kernelArgs);
        return inlinedFinishControlNegations();
      }
      if (auto func =
              dyn_cast_or_null<func::ConstantOp>(calleeValue.getDefiningOp())) {
        auto funcTy = cast<FunctionType>(func.getType());
        auto callableSym = func.getValueAttr();
        inlinedStartControlNegations();
        auto kernelArgs =
            convertKernelArgs(builder, loc, 2, args, funcTy.getInputs());
        builder.create<quake::ApplyOp>(loc, funcTy.getResults(), callableSym,
                                       /*isAdjoint=*/false, ctrlValues,
                                       kernelArgs);
        return inlinedFinishControlNegations();
      }
      if (auto ty = dyn_cast<cc::CallableType>(calleeValue.getType())) {
        // In order to autogenerate the control form of the called kernel, we
        // have to be able to determine precisely which kernel is being called
        // at this point. If this is a local lambda expression, it is handled
        // elsewhere. If this is a lambda expression argument, then we have to
        // recover it or give a compilation error.
        auto *tyPtr = x->getArg(0)->getType().getTypePtr();
        auto *recTy = dyn_cast<clang::RecordType>(tyPtr);
        if (!recTy && isa<clang::AutoType>(tyPtr)) {
          recTy = dyn_cast_or_null<clang::RecordType>(
              cast<clang::AutoType>(tyPtr)->desugar().getTypePtr());
        }
        if (!recTy && isa<clang::SubstTemplateTypeParmType>(tyPtr)) {
          auto *ty = cast<clang::SubstTemplateTypeParmType>(tyPtr);
          recTy = dyn_cast_or_null<clang::RecordType>(
              ty->getReplacementType().getTypePtr());
        }
        if (!recTy) {
          TODO_loc(loc,
                   "control does not appear to be on a user-defined kernel");
        }
        auto *decl = recTy->getDecl();
        if (decl->isLambda()) {
          auto *lambdaClass = cast<clang::CXXRecordDecl>(decl);
          auto mangledName =
              generateCudaqKernelName(findCallOperator(lambdaClass));
          calleeSymbol = SymbolRefAttr::get(ctx, mangledName);
          auto funcTy = ty.getSignature();
          inlinedStartControlNegations();
          auto kernelArgs =
              convertKernelArgs(builder, loc, 2, args, funcTy.getInputs());
          builder.create<quake::ApplyOp>(loc, funcTy.getResults(), calleeSymbol,
                                         /*isAdjoint=*/false, ctrlValues,
                                         kernelArgs);
          return inlinedFinishControlNegations();
        }
        TODO_loc(loc, "value has !cc.lambda type but decl isn't a lambda");
      }
      TODO_loc(loc, "unexpected callable argument");
    }

    if (funcName.equals("adjoint")) {
      // Expect the following declaration from qubit_qis.h:
      //
      // template <typename QuantumKernel, typename... Args>
      //   requires isCallableVoidKernel<QuantumKernel, Args...>
      // void adjoint(QuantumKernel &&kernel, Args &&...args);
      //
      // The first argument must be an instance of a Callable and a quantum
      // kernel. Traverse the AST here to construct the name of the operator()
      // to be called.
      auto kernelValue = args[0];
      SymbolRefAttr kernelSymbol;
      auto kernelTy = kernelValue.getType();
      if (auto ptrTy = dyn_cast<cc::PointerType>(kernelTy))
        kernelTy = ptrTy.getElementType();
      if (auto ty = dyn_cast<cc::StructType>(kernelTy)) {
        auto *ctx = builder.getContext();
        auto *classDecl = classDeclFromTemplateArgument(*func, 0, *astContext);
        if (!classDecl) {
          // This shouldn't happen if the cudaq headers are used, but add a
          // check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "expected cudaq::adjoint to be a specific template");
          de.Report(x->getBeginLoc(), id);
          return {};
        }
        auto *kernelCallOper = findCallOperator(classDecl);
        if (!kernelCallOper) {
          // This should be caught by the concepts used in the header file, but
          // add a check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "first argument to cudaq::adjoint must be a callable");
          de.Report(x->getBeginLoc(), id);
          return {};
        }
        auto kernelName = generateCudaqKernelName(kernelCallOper);
        kernelSymbol = SymbolRefAttr::get(ctx, kernelName);
        auto kernFunc = module.lookupSymbol<func::FuncOp>(kernelName);
        assert(kernFunc && "kernel call operator must be present");
        auto kernTy = kernFunc.getFunctionType();
        auto kernArgs =
            convertKernelArgs(builder, loc, 1, args, kernTy.getInputs());
        return builder.create<quake::ApplyOp>(loc, TypeRange{}, kernelSymbol,
                                              /*isAdjoint=*/true, ValueRange{},
                                              kernArgs);
      }
      if (auto func =
              dyn_cast_or_null<func::ConstantOp>(kernelValue.getDefiningOp())) {
        auto kernSym = func.getValueAttr();
        auto funcTy = cast<FunctionType>(func.getType());
        auto kernArgs =
            convertKernelArgs(builder, loc, 1, args, funcTy.getInputs());
        return builder.create<quake::ApplyOp>(loc, funcTy.getResults(), kernSym,
                                              /*isAdjoint=*/true, ValueRange{},
                                              kernArgs);
      }
      if (auto ty = dyn_cast<cc::CallableType>(kernelTy)) {
        // In order to autogenerate the control form of the called kernel, we
        // have to be able to determine precisely which kernel is being called
        // at this point. If this is a local lambda expression, it is handled
        // elsewhere. If this is a lambda expression argument, then we have to
        // recover it or give a compilation error.
        auto *tyPtr = x->getArg(0)->getType().getTypePtr();
        auto *recTy = dyn_cast<clang::RecordType>(tyPtr);
        if (!recTy && isa<clang::AutoType>(tyPtr)) {
          recTy = dyn_cast_or_null<clang::RecordType>(
              cast<clang::AutoType>(tyPtr)->desugar().getTypePtr());
        }
        if (!recTy && isa<clang::SubstTemplateTypeParmType>(tyPtr)) {
          auto *ty = cast<clang::SubstTemplateTypeParmType>(tyPtr);
          recTy = dyn_cast_or_null<clang::RecordType>(
              ty->getReplacementType().getTypePtr());
        }
        if (!recTy) {
          TODO_loc(loc,
                   "adjoint does not appear to be on a user-defined kernel");
        }
        auto *decl = recTy->getDecl();
        if (decl->isLambda()) {
          auto *lambdaClass = cast<clang::CXXRecordDecl>(decl);
          auto mangledName =
              generateCudaqKernelName(findCallOperator(lambdaClass));
          auto kernelSymbol =
              SymbolRefAttr::get(builder.getContext(), mangledName);
          auto funcTy = ty.getSignature();
          auto kernelArgs =
              convertKernelArgs(builder, loc, 1, args, funcTy.getInputs());
          return builder.create<quake::ApplyOp>(
              loc, funcTy.getResults(), kernelSymbol,
              /*isAdjoint=*/true, ValueRange{}, kernelArgs);
        }
        TODO_loc(loc, "value has !cc.lambda type but decl isn't a lambda");
      }
      TODO_loc(loc, "adjoint does not appear to be on a user-defined kernel");
    }

    if (funcName.equals("compute_action")) {
      builder.create<quake::ComputeActionOp>(loc, /*is_dagger=*/false, args[0],
                                             args[1]);
      return true;
    }
    if (funcName.equals("compute_dag_action")) {
      builder.create<quake::ComputeActionOp>(loc, /*is_dagger=*/true, args[0],
                                             args[1]);
      return true;
    }

    if (funcName.equals("toInteger") || funcName.equals("to_integer"))
      return pushValue(toIntegerImpl(builder, loc, args[0]));

    if (funcName.equals("slice_vector")) {
      auto svecTy = dyn_cast<cc::StdvecType>(args[0].getType());
      auto eleTy = svecTy.getElementType();
      assert(svecTy && "first argument must be std::vector");
      Value offset = args[1];
      Type ptrTy;
      Value vecPtr;
      if (eleTy == builder.getI1Type()) {
        eleTy = cc::ArrayType::get(builder.getI8Type());
        ptrTy = cc::PointerType::get(eleTy);
        vecPtr = builder.create<cc::StdvecDataOp>(loc, ptrTy, args[0]);
        auto bits = svecTy.getElementType().getIntOrFloatBitWidth();
        assert(bits > 0);
        auto scale = builder.create<arith::ConstantIntOp>(loc, (bits + 7) / 8,
                                                          args[1].getType());
        offset = builder.create<arith::MulIOp>(loc, scale, args[1]);
      } else {
        ptrTy = cc::PointerType::get(eleTy);
        vecPtr = builder.create<cc::StdvecDataOp>(loc, ptrTy, args[0]);
      }
      auto ptr = builder.create<cc::ComputePtrOp>(loc, ptrTy, vecPtr,
                                                  ArrayRef<Value>{offset});
      return pushValue(
          builder.create<cc::StdvecInitOp>(loc, svecTy, ptr, args[2]));
    }

    TODO_loc(loc, "unknown function, " + funcName + ", in cudaq namespace");
  }

  // If we get here, and the CallExpr takes qubits or qreg and it must be
  // another kernel call.
  auto mlirFuncTy = cast<FunctionType>(calleeOp.getType());
  auto funcResults = mlirFuncTy.getResults();
  auto convertedArgs =
      convertKernelArgs(builder, loc, 0, args, mlirFuncTy.getInputs());
  auto call = builder.create<func::CallIndirectOp>(loc, funcResults, calleeOp,
                                                   convertedArgs);
  if (call.getNumResults() > 0)
    return pushValue(call.getResult(0));
  return true;
}

std::optional<std::string> QuakeBridgeVisitor::isInterceptedSubscriptOperator(
    clang::CXXOperatorCallExpr *x) {
  if (isSubscriptOperator(x)) {
    if (auto decl = dyn_cast<clang::CXXMethodDecl>(x->getCalleeDecl())) {
      auto typeName = decl->getParent()->getNameAsString();
      if (isInNamespace(decl, "cudaq")) {
        if (isCudaQType(typeName))
          return {typeName};
      } else if (isInNamespace(decl, "std")) {
        if (typeName == "vector")
          return {typeName};
      } else if (isInNamespace(decl, "std")) {
        if (typeName == "_Bit_reference")
          return {typeName};
      }
    }
  }
  return {};
}

bool QuakeBridgeVisitor::WalkUpFromCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *x) {
  // Is this an operator[] that we're converting?
  if (isInterceptedSubscriptOperator(x)) {
    // Yes, so skip walking the superclass, CallExpr.
    return VisitCXXOperatorCallExpr(x);
  }
  if (auto *func = dyn_cast_or_null<clang::FunctionDecl>(x->getCalleeDecl())) {
    if (isCallOperator(x) ||
        (isInClassInNamespace(func, "qudit", "cudaq") && isExclaimOperator(x)))
      return VisitCXXOperatorCallExpr(x);
  }

  // Otherwise, handle with default traversal.
  return WalkUpFromCallExpr(x) && VisitCXXOperatorCallExpr(x);
}

bool QuakeBridgeVisitor::VisitCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto replaceTOSValue = [&](Value v) {
    [[maybe_unused]] auto funcVal = popValue();
    assert(funcVal.getDefiningOp<func::ConstantOp>());
    return pushValue(v);
  };
  if (auto typeNameOpt = isInterceptedSubscriptOperator(x)) {
    auto &typeName = *typeNameOpt;
    if (isCudaQType(typeName)) {
      auto idx_var = popValue();
      auto qreg_var = popValue();

      // Get name of the qreg, e.g. qr, and use it to construct a name for the
      // element, which is intended to be qr%n when n is the index of the
      // accessed qubit.
      StringRef qregName = getNamedDecl(x->getArg(0))->getName();
      auto name = getQubitSymbolTableName(qregName, idx_var);
      char *varName = strdup(name.c_str());

      // If the name exists in the symbol table, return its stored value.
      if (symbolTable.count(name))
        return replaceTOSValue(symbolTable.lookup(name));

      // Otherwise create an operation to access the qubit, store that value in
      // the symbol table, and return the AddressQubit operation's resulting
      // value.
      auto address_qubit =
          builder.create<quake::ExtractRefOp>(loc, qreg_var, idx_var);

      symbolTable.insert(StringRef(varName), address_qubit);
      return replaceTOSValue(address_qubit);
    }
    if (typeName == "vector") {
      // Here we have something like vector<float> theta, and in the kernel, we
      // are accessing it like theta[i].
      auto indexVar = popValue();
      auto svec = popValue();
      assert(svec.getType().isa<cc::StdvecType>());
      auto eleTy = cast<cc::StdvecType>(svec.getType()).getElementType();
      auto elePtrTy = cc::PointerType::get(eleTy);
      auto vecPtr = builder.create<cc::StdvecDataOp>(loc, elePtrTy, svec);
      auto eleAddr = builder.create<cc::ComputePtrOp>(loc, elePtrTy, vecPtr,
                                                      ValueRange{indexVar});
      return replaceTOSValue(eleAddr);
    }
    if (typeName == "_Bit_reference") {
      // For vector<bool>, on the kernel side this is represented as a sequence
      // of byte-sized boolean values (true and false). On the host side, C++ is
      // likely going to pack the booleans as bits in words.
      auto indexVar = popValue();
      auto svec = popValue();
      assert(svec.getType().isa<cc::StdvecType>());
      auto elePtrTy = cc::PointerType::get(builder.getI8Type());
      auto vecPtr = builder.create<cc::StdvecDataOp>(loc, elePtrTy, svec);
      auto eleAddr = builder.create<cc::ComputePtrOp>(loc, elePtrTy, vecPtr,
                                                      ValueRange{indexVar});
      auto i1PtrTy = cc::PointerType::get(builder.getI1Type());
      auto i1Cast = builder.create<arith::TruncIOp>(loc, i1PtrTy, eleAddr);
      return replaceTOSValue(i1Cast);
    }
    TODO_loc(loc, "unhandled operator call for quake conversion");
  }

  if (auto *func = dyn_cast_or_null<clang::FunctionDecl>(x->getCalleeDecl())) {
    // Lower <any>::operator()(...)
    if (isCallOperator(x)) {
      auto funcArity = func->getNumParams();
      SmallVector<Value> args = lastValues(funcArity);
      auto tos = popValue();
      auto tosTy = tos.getType();
      auto ptrTy = dyn_cast<cc::PointerType>(tosTy);
      bool isEntryKernel = [&]() {
        // TODO: make this lambda a member function.
        if (auto fn = peekValue().getDefiningOp<func::ConstantOp>()) {
          auto name = fn.getValue().str();
          for (auto fdPair : functionsToEmit)
            if (getCudaqKernelName(fdPair.first) == name)
              return true;
        }
        return false;
      }();
      if (ptrTy || isEntryKernel) {
        // The call operator has an object in the call position, so we want to
        // replace it with an indirect call to the func::ConstantOp.
        assert((isEntryKernel || isa<cc::StructType>(ptrTy.getElementType())) &&
               "expected kernel as callable class");
        auto indirect = popValue();
        auto funcTy = cast<FunctionType>(indirect.getType());
        auto call = builder.create<func::CallIndirectOp>(
            loc, funcTy.getResults(), indirect, args);
        if (call.getResults().empty())
          return true;
        return pushValue(call.getResult(0));
      }
      auto callableTy = cast<cc::CallableType>(tosTy);
      auto callInd = builder.create<cc::CallCallableOp>(
          loc, callableTy.getSignature().getResults(), tos, args);
      if (callInd.getResults().empty()) {
        popValue();
        return true;
      }
      return replaceTOSValue(callInd.getResult(0));
    }

    // Lower cudaq::qudit<>::operator!()
    if (isInClassInNamespace(func, "qudit", "cudaq") && isExclaimOperator(x)) {
      auto qubit = popValue();
      negations.push_back(qubit);
      return replaceTOSValue(qubit);
    }
  }
  return true;
}

/// When traversing an expression such as `Kernel{}` or `Kernel()`, the object
/// may be passed to a function that needs a special callable object rather than
/// just the object. In order to make sure the call operator is already
/// declared, it is added here if needed to the module.
/// This method must only be called from a Traverse<Foo> method.
void QuakeBridgeVisitor::maybeAddCallOperationSignature(clang::Decl *x) {
  while (x) {
    if (auto *classDecl = dyn_cast<clang::CXXRecordDecl>(x)) {
      auto *callOperDecl = findCallOperator(classDecl);
      if (callOperDecl && isKernelEntryPoint(callOperDecl)) {
        auto loc = toLocation(callOperDecl);
        if (!TraverseType(callOperDecl->getType()))
          emitFatalError(loc, "expected type for call operator");
        auto kernelName = generateCudaqKernelName(callOperDecl);
        getOrAddFunc(loc, kernelName, peelPointerFromFunction(popType()));
      }
      return;
    }
    if (isa<clang::NamespaceDecl, clang::TranslationUnitDecl>(x))
      return;
    x = cast<clang::Decl>(x->getDeclContext()->getParent());
  }
}

bool QuakeBridgeVisitor::TraverseCXXTemporaryObjectExpr(
    clang::CXXTemporaryObjectExpr *x, DataRecursionQueue *) {
  if (auto *ctor = x->getConstructor())
    maybeAddCallOperationSignature(ctor);
  if (!TraverseType(x->getType()))
    return false;
  return WalkUpFromCXXTemporaryObjectExpr(x);
}

bool QuakeBridgeVisitor::VisitCXXTemporaryObjectExpr(
    clang::CXXTemporaryObjectExpr *x) {
  // We probably want a distinctive op here instead of just leaving a type.
  // Really this means allocating at least 1 byte and calling the default ctor.
  return true;
}

bool QuakeBridgeVisitor::TraverseInitListExpr(clang::InitListExpr *x,
                                              DataRecursionQueue *) {
  if (x->isSyntacticForm()) {
    // The syntactic form is the surface level syntax as typed by the user. This
    // isn't really all the helpful during the lowering process. We want to deal
    // with the semantic form. See below.
    auto loc = toLocation(x);
    if (x->getNumInits() != 0)
      TODO_loc(loc, "initializer list containing elements");
    return true;
  }

  if (auto *ty = x->getType().getTypePtr())
    if (auto *tyDecl = ty->getAsRecordDecl())
      maybeAddCallOperationSignature(tyDecl);
  if (!TraverseType(x->getType()))
    return false;
  for (auto *subStmt : x->children())
    if (!TraverseStmt(subStmt))
      return false;
  return WalkUpFromInitListExpr(x);
}

bool QuakeBridgeVisitor::VisitInitListExpr(clang::InitListExpr *x) {
  auto loc = toLocation(x);
  auto size = x->getNumInits();
  if (size == 0) {
    // TODO: Maybe check that this is an instance of a callable class, if it is
    // a CXXRecordType.
    auto ty = popType();
    return pushValue(builder.create<cc::AllocaOp>(loc, ty));
  }

  // List has 1 or more members.
  auto last = lastValues(size);
  bool allRef = [&]() {
    for (auto v : last)
      if (!isa<quake::RefType, quake::VeqType>(v.getType()))
        return false;
    return true;
  }();
  if (allRef) {
    if (size > 1) {
      auto veqTy = [&]() -> quake::VeqType {
        unsigned size = 0;
        for (auto v : last) {
          if (auto veqTy = dyn_cast<quake::VeqType>(v.getType())) {
            if (!veqTy.hasSpecifiedSize())
              return quake::VeqType::getUnsized(builder.getContext());
            size += veqTy.getSize();
          } else {
            ++size;
          }
        }
        return quake::VeqType::get(builder.getContext(), size);
      }();
      return pushValue(builder.create<quake::ConcatOp>(loc, veqTy, last));
    }
    // Pass initialization list with one member as a Ref.
    return pushValue(last[0]);
  }

  // Check if all values here are Integer or Float Type
  bool isAllIntOrAllFloat = [&]() {
    for (auto v : last)
      if (!v.getType().isIntOrFloat())
        return false;
    return true;
  }();

  // If these are integers or floats, then let's allocate
  // some memory and store them there.
  if (isAllIntOrAllFloat) {
    // Clear the types for the init expr
    typeStack.clear();

    // This is a initlist on ints or floats, get which one
    Type dataType = last.front().getType();

    // Add the array size value
    Value arrSize =
        getConstantInt(builder, loc, last.size(), builder.getI64Type());

    // Allocate the required memory chunk
    Value alloca = builder.create<cc::AllocaOp>(loc, dataType, arrSize);

    // Store the values in the allocated memory
    for (std::size_t i = 0; auto v : last) {
      Value ptr = builder.create<cc::ComputePtrOp>(
          loc, cc::PointerType::get(dataType), alloca,
          getConstantInt(builder, loc, i++, builder.getI64Type()));
      builder.create<cc::StoreOp>(loc, v, ptr);
    }

    return pushValue(alloca);
  }

  TODO_x(loc, x, mangler, "list initialization (not quantum ref)");
  return true;
}

bool QuakeBridgeVisitor::TraverseCXXConstructExpr(clang::CXXConstructExpr *x,
                                                  DataRecursionQueue *) {
  if (x->isElidable())
    return true;
  if (auto *ctor = x->getConstructor()) {
    auto ctorName = ctor->getNameAsString();
    // In the std::function constructor case, we want to traverse the type
    // returned by the constructor since it is a high-level type and we cannot
    // traverse any of the arguments from the visit method. This extra type will
    // be popped from the stack in the visit method.
    if (isInNamespace(ctor, "std") && ctorName == "function")
      if (!TraverseType(x->getType()))
        return false;
  }
  return Base::TraverseCXXConstructExpr(x);
}

bool QuakeBridgeVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *x) {
  auto loc = toLocation(x);
  if (auto *ctor = x->getConstructor()) {
    auto ctorName = ctor->getNameAsString();
    if (isInNamespace(ctor, "cudaq")) {
      if (ctorName == "qreg" || ctorName == "qspan") {
        // This is a qreg q(N);
        auto sizeVal = popValue();
        if (isa<quake::VeqType>(sizeVal.getType()))
          return pushValue(sizeVal);
        assert(isa<IntegerType>(sizeVal.getType()));
        return pushValue(builder.create<quake::AllocaOp>(
            loc, quake::VeqType::getUnsized(builder.getContext()), sizeVal));
      }
      if (ctorName == "qudit") {
        // This is a "cudaq::qudit/qubit q;"
        return pushValue(builder.create<quake::AllocaOp>(loc));
      }
    } else if (isInNamespace(ctor, "std")) {
      auto isVectorOfQubitRefs = [&]() {
        if (auto *ctor = x->getConstructor()) {
          if (isInNamespace(ctor, "std") &&
              ctor->getNameAsString() == "vector") {
            Value v = peekValue();
            return v && isa<quake::VeqType>(v.getType());
          }
        }
        return false;
      };
      if (isVectorOfQubitRefs()) {
        assert(isa<quake::VeqType>(peekValue().getType()));
        return true;
      }
      if (ctorName == "function") {
        // Are we converting a lambda expr to a std::function?
        auto backVal = peekValue();
        auto backTy = backVal.getType();
        auto ctorTy = popType();
        if (auto ptrTy = dyn_cast<cc::PointerType>(backTy))
          backTy = ptrTy.getElementType();
        if (backTy.isa<cc::CallableType>()) {
          // Skip this constructor (for now).
          return true;
        }
        if (auto stTy = dyn_cast<cc::StructType>(backTy)) {
          if (!stTy.getMembers().empty()) {
            // TODO: We don't support a callable class with data members yet.
            TODO_loc(loc, "callable class with data members");
          }
          // Constructor generated as degenerate reference to call operator.
          auto *fromTy = x->getArg(0)->getType().getTypePtr();
          // FIXME: May need to peel off more than one layer of sugar?
          if (auto *elabTy = dyn_cast<clang::ElaboratedType>(fromTy))
            fromTy = elabTy->desugar().getTypePtr();
          auto *fromDecl =
              dyn_cast_or_null<clang::RecordType>(fromTy)->getDecl();
          if (!fromDecl)
            TODO_loc(loc, "recovering record type for a callable");
          auto *objDecl = dyn_cast_or_null<clang::CXXRecordDecl>(fromDecl);
          if (!objDecl)
            TODO_loc(loc, "recovering C++ declaration for callable");
          auto *callOperDecl = findCallOperator(objDecl);
          if (!callOperDecl) {
            auto &de = mangler->getASTContext().getDiagnostics();
            auto id = de.getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "std::function initializer must be a callable");
            de.Report(x->getBeginLoc(), id);
            return true;
          }
          auto kernelCallTy = cast<cc::CallableType>(ctorTy);
          auto kernelName = generateCudaqKernelName(callOperDecl);
          popValue(); // replace value at TOS.
          return pushValue(builder.create<cc::CreateLambdaOp>(
              loc, kernelCallTy, [&](OpBuilder &builder, Location loc) {
                auto args = builder.getBlock()->getArguments();
                auto call = builder.create<func::CallOp>(
                    loc, kernelCallTy.getSignature().getResults(), kernelName,
                    args);
                builder.create<cc::ReturnOp>(loc, call.getResults());
              }));
        }
      }
      if (ctorName == "reference_wrapper") {
        // The `reference_wrapper` class is used to guide the `qudit&` through a
        // container class (like `std::vector`). It is a NOP at the Quake level.
        [[maybe_unused]] auto tosTy = peekValue().getType();
        assert((isa<quake::RefType, quake::VeqType>(tosTy)));
        return true;
      }

      if (ctorName == "vector") {
        // This is a std::vector constructor, first we'll check if it
        // is constructed from a constant initializer list, in that case
        // we'll have a AllocaOp at the top of the stack that allocates a
        // ptr<array<TxC>>, where C is constant / known
        if (auto alloca = dyn_cast<cc::AllocaOp>(peekValue().getDefiningOp())) {
          auto asPtrType =
              dyn_cast<cc::PointerType>(alloca.getAddress().getType());
          if (auto arrayTy =
                  dyn_cast<cc::ArrayType>(asPtrType.getElementType()))
            if (alloca.getNumOperands() > 0)
              return pushValue(builder.create<cc::StdvecInitOp>(
                  loc, cc::StdvecType::get(arrayTy.getElementType()), alloca,
                  alloca.getSeqSize()));
        }

        // Next check if its created from a size integer
        if (peekValue().getType().isIntOrFloat()) {
          auto arrSize = popValue();
          auto dataType = popType();

          // create stdvec init op without a buffer.
          // Allocate the required memory chunk
          Value alloca = builder.create<cc::AllocaOp>(loc, dataType, arrSize);

          // Create the stdvec_init op
          return pushValue(builder.create<cc::StdvecInitOp>(
              loc, cc::StdvecType::get(dataType), alloca, arrSize));
        }

        // Disallow any default vector construction bc we don't
        // want any .push_back
        if (ctor->isDefaultConstructor())
          reportClangError(ctor, mangler,
                           "Default std::vector<T> constructor within quantum "
                           "kernel is not allowed "
                           "(cannot resize the vector).");
      }
    }

    // TODO: remove this when we can handle ctors more generally.
    if (!ctor->isDefaultConstructor()) {
      LLVM_DEBUG(llvm::dbgs() << "unhandled ctor:\n"; x->dump());
      TODO_loc(loc, "C++ ctor (not-default)");
    }

    // A regular C++ class constructor lowers as:
    //
    // 1) A unique object must be created, so the type must have a minimum of
    //    one byte.
    // 2) Allocate a new object.
    // 3) Call the constructor passing the address of the allocation as
    // `this`.

    // FIXME: As this is now, the stack space isn't correctly sized. The
    // next line should be something like:
    //   auto ty = genType(x->getType());
    auto ty = cc::StructType::get(builder.getContext(),
                                  ArrayRef<Type>{builder.getIntegerType(8)});
    auto mem = builder.create<cc::AllocaOp>(loc, ty);
    // FIXME: Using Ctor_Complete for mangled name generation blindly here.
    // Is there a programmatic way of determining which enum to use from the
    // AST?
    auto mangledName =
        cxxMangledDeclName(clang::GlobalDecl{ctor, clang::Ctor_Complete});
    auto funcTy =
        FunctionType::get(builder.getContext(), TypeRange{mem.getType()}, {});
    auto func = getOrAddFunc(loc, mangledName, funcTy).first;
    // FIXME: The ctor may not be the default ctor. Get all the args.
    builder.create<func::CallOp>(loc, func, ValueRange{mem});
    return pushValue(mem);
  }
  TODO_loc(loc, "C++ ctor (NULL)");
}

bool QuakeBridgeVisitor::TraverseDeclRefExpr(clang::DeclRefExpr *x,
                                             DataRecursionQueue *) {
  auto *decl = x->getDecl();
  if (auto *funcDecl = dyn_cast<clang::FunctionDecl>(decl)) {
    if (inRecType)
      return true;
    return TraverseFunctionDecl(funcDecl);
  }
  return WalkUpFromDeclRefExpr(x);
}

bool QuakeBridgeVisitor::VisitDeclRefExpr(clang::DeclRefExpr *x) {
  auto *decl = x->getDecl();
  assert(!isa<clang::FunctionDecl>(decl) &&
         "FunctionDecl should not reach here");
  if (!symbolTable.count(decl->getName())) {
    // This is a catastrophic error. This symbol is unknown and probably came
    // from a context that is inaccessible from this kernel.
    auto &de = astContext->getDiagnostics();
    const auto id =
        de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                           "symbol is not accessible in this kernel");
    auto db = de.Report(x->getBeginLoc(), id);
    const auto range = x->getSourceRange();
    db.AddSourceRange(clang::CharSourceRange::getCharRange(range));
    raisedError = true;
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "decl ref: " << decl << '\n');
  pushValue(symbolTable.lookup(decl->getName()));
  return true;
}

bool QuakeBridgeVisitor::VisitStringLiteral(clang::StringLiteral *x) {
  TODO_x(toLocation(x->getSourceRange()), x, mangler, "string literal");
  return false;
}

} // namespace cudaq::details
