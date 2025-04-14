/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DISTRIBUTEDDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "distributed-device-call"

using namespace mlir;

namespace {

class UnicornDeviceCallPat : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    constexpr const char PassthroughAttr[] = "passthrough";
    constexpr const char UnicornAttr[] = "cuda-q-fun-id";
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);
    if (!devFunc) {
      LLVM_DEBUG(llvm::dbgs() << "cannot find the function " << devFuncName
                              << " in module\n");
      return failure();
    }

    llvm::MD5 hash;
    hash.update(devFuncName);
    llvm::MD5::MD5Result result;
    hash.final(result);
    auto callbackCode = result.low();

    bool needToAddIt = true;
    SmallVector<Attribute> funcIdAttr;
    if (auto passthruAttr = devFunc->getAttr(PassthroughAttr)) {
      auto arrayAttr = cast<ArrayAttr>(passthruAttr);
      funcIdAttr.append(arrayAttr.begin(), arrayAttr.end());
      for (auto a : arrayAttr) {
        if (auto strArrAttr = dyn_cast<ArrayAttr>(a)) {
          auto strAttr = dyn_cast<StringAttr>(strArrAttr[0]);
          if (!strAttr)
            continue;
          if (strAttr.getValue() == UnicornAttr) {
            needToAddIt = false;
            break;
          }
        }
      }
    }
    if (needToAddIt) {
      auto callbackCodeAsStr = std::to_string(callbackCode);
      funcIdAttr.push_back(rewriter.getStrArrayAttr(
          {UnicornAttr, rewriter.getStringAttr(callbackCodeAsStr)}));
      devFunc->setAttr(PassthroughAttr, rewriter.getArrayAttr(funcIdAttr));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        devcall, devFunc.getFunctionType().getResults(), devFuncName,
        devcall.getOperands());
    return success();
  }
};

class DistributedFuncPat : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    if (!func->hasAttr(cudaq::deviceCallAttrName))
      return failure();
    FunctionType devFuncTy = func.getFunctionType();
    auto module = func->getParentOfType<ModuleOp>();
    FunctionType newDevFuncTy = cudaq::opt::factory::toHostSideFuncType(
        devFuncTy, /*addThisPtr=*/false, module);
    if (func.getFunctionType() == newDevFuncTy)
      return failure();
    rewriter.updateRootInPlace(func,
                               [&]() { func.setFunctionType(newDevFuncTy); });
    return success();
  }
};

class ExtractDevPtrOpPat
    : public OpRewritePattern<cudaq::cc::ExtractDevicePtrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ExtractDevicePtrOp extractOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        extractOp, TypeRange{extractOp.getResult().getType()},
        cudaq::runtime::extractDevPtr, ValueRange{extractOp.getDevicePtr()});
    return success();
  }
};

class DistributedDeviceCallPat
    : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = devcall.getLoc();
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);
    if (!devFunc) {
      LLVM_DEBUG(llvm::dbgs() << "cannot find the function " << devFuncName
                              << " in module\n");
      return failure();
    }

    // Is there already a thunk for marshaling arguments for this devFuncName?
    std::string marshalName = "marshal." + devFuncName.str();
    auto [marshalFunc, alreadyAdded] = cudaq::opt::factory::getOrAddFunc(
        loc, marshalName, devFunc.getFunctionType(), module);
    rewriter.replaceOpWithNewOp<func::CallOp>(devcall, devcall.getResultTypes(),
                                              marshalName, devcall.getArgs());

    if (alreadyAdded) {
      // This may happen if another kernel called this same callback.
      LLVM_DEBUG(llvm::dbgs() << "marshal function " << marshalName
                              << " already in module\n");
      return failure();
    }

    // Create a constant with the name of the callback func as a C string.
    auto callbackNameObj = [&]() -> mlir::LLVM::GlobalOp {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      return rewriter.create<LLVM::GlobalOp>(
          loc, cudaq::opt::factory::getStringType(ctx, devFuncName.size() + 1),
          /*isConstant=*/true, LLVM::Linkage::External,
          devFuncName.str() + ".callbackName",
          rewriter.getStringAttr(devFuncName.str() + '\0'), /*alignment=*/0);
    }();

    // The device call dispatcher will then call the `unmarshalFunc`, which is
    // added here.
    std::string unmarshalName = "unmarshal." + devFuncName.str();
    auto unmarshalTy = cudaq::opt::marshal::getThunkType(ctx);
    auto [unmarshalFunc, alreadyAdded2] = cudaq::opt::factory::getOrAddFunc(
        loc, unmarshalName, unmarshalTy, module);
    if (alreadyAdded2)
      return devcall.emitOpError("unmarshal function must not be present");

    // Create a new struct type to pass arguments and results.
    auto structTy =
        cudaq::opt::factory::buildInvokeStructType(devFunc.getFunctionType());

    // The unmarshaling code is autogenerated into `unmarshalFunc`.
    genNewUnmarshalFunc(loc, unmarshalFunc, rewriter, devFunc, structTy);

    // Autogenerate the code to marshal the arguments here in the function
    // `marshalFunc`.
    genNewMarshalFunc(loc, marshalFunc, rewriter, callbackNameObj,
                      unmarshalFunc, structTy, devFunc, module);

    // Finally, create the registration code for this particular callback.
    genRegistrationHook(loc, devFuncName, rewriter, callbackNameObj, module);

    return success();
  }

  static void genNewMarshalFunc(Location loc, func::FuncOp marshalFunc,
                                PatternRewriter &rewriter,
                                LLVM::GlobalOp callbackNameObj,
                                func::FuncOp unmarshalFunc,
                                cudaq::cc::StructType bufferTy,
                                func::FuncOp devFunc, ModuleOp module) {
    auto i64Ty = rewriter.getI64Type();
    Block *entryBlock = marshalFunc.addEntryBlock();
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    SmallVector<Value> dispatchArgs;
    // Arg 1: the device id.
    Value devId = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    dispatchArgs.push_back(devId);

    // Arg 2: the callback name.
    Value callbackNameVal = rewriter.create<cudaq::cc::AddressOfOp>(
        loc, cudaq::cc::PointerType::get(callbackNameObj.getType()),
        callbackNameObj.getSymName());
    dispatchArgs.push_back(
        rewriter.create<cudaq::cc::CastOp>(loc, ptrTy, callbackNameVal));

    // Arg 3: pointer to the unmarshal func
    auto unmar = rewriter.create<func::ConstantOp>(
        loc, unmarshalFunc.getFunctionType(), unmarshalFunc.getName());
    Value unmarPtr = rewriter.create<cudaq::cc::FuncToPtrOp>(loc, ptrTy, unmar);
    dispatchArgs.push_back(unmarPtr);

    // Arg 4: pointer to the argument buffer
    auto devFuncTy = devFunc.getFunctionType();
    const bool hasDynamicSignature =
        cudaq::opt::marshal::isDynamicSignature(devFuncTy);
    auto sizeScratch = rewriter.create<cudaq::cc::AllocaOp>(loc, i64Ty);
    SmallVector<std::tuple<unsigned, Value, Type>> zippedArgs;
    for (auto v : llvm::enumerate(entryBlock->getArguments()))
      zippedArgs.emplace_back(v.index(), v.value(), v.value().getType());
    auto bufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return cudaq::opt::marshal::genSizeOfDynamicCallbackBuffer(
            loc, rewriter, module, bufferTy, zippedArgs, sizeScratch);
      return rewriter.create<cudaq::cc::SizeOfOp>(loc, i64Ty, bufferTy);
    }();
    auto i8Ty = rewriter.getI8Type();
    Value rawBuffer =
        rewriter.create<cudaq::cc::AllocaOp>(loc, i8Ty, bufferSize);
    Value typedBuffer = rewriter.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(bufferTy), rawBuffer);
    if (hasDynamicSignature) {
      auto addendumScratch = rewriter.create<cudaq::cc::AllocaOp>(loc, ptrTy);
      Value prefixSize =
          rewriter.create<cudaq::cc::SizeOfOp>(loc, i64Ty, bufferTy);
      Value addendumPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
          loc, ptrTy, rawBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
      cudaq::opt::marshal::populateCallbackBuffer(loc, rewriter, module,
                                                  typedBuffer, zippedArgs,
                                                  addendumPtr, addendumScratch);

    } else {
      cudaq::opt::marshal::populateCallbackBuffer(loc, rewriter, module,
                                                  typedBuffer, zippedArgs);
    }
    dispatchArgs.push_back(rewriter.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(rewriter.getI8Type()), rawBuffer));

    // Arg 5: argument buffer size
    dispatchArgs.push_back(bufferSize);

    // Arg 6: offset of the return value in the buffer
    Value returnOffset = cudaq::opt::marshal::genComputeReturnOffset(
        loc, rewriter, devFuncTy, bufferTy);
    dispatchArgs.push_back(returnOffset);

    auto spanTy = cudaq::cc::StructType::get(rewriter.getContext(),
                                             ArrayRef<Type>{ptrTy, i64Ty});
    auto callback = rewriter.create<func::CallOp>(
        loc, spanTy, cudaq::runtime::callDeviceCallback, dispatchArgs);
    auto resTys = marshalFunc.getFunctionType().getResults();
    if (resTys.empty()) {
      // This device call returns `void`, so we're done.
      rewriter.create<func::ReturnOp>(loc);
      return;
    }

    assert(resTys.size() == 1);
    Type resTy = resTys.front();
    if (cudaq::cc::isDynamicType(resTy)) {
      // XXX: need to unmarshal all the cases here
      (void)callback;
      marshalFunc.emitOpError("return type is not yet implemented");
      Value undef = rewriter.create<cudaq::cc::UndefOp>(loc, resTy);
      rewriter.create<func::ReturnOp>(loc, undef);
      return;
    }
    std::int32_t numInputs = devFuncTy.getNumInputs();
    auto outputPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(resTy), typedBuffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
    Value resVal = rewriter.create<cudaq::cc::LoadOp>(loc, outputPtr);
    rewriter.create<func::ReturnOp>(loc, resVal);
    marshalFunc.setPublic();
  }

  static void genNewUnmarshalFunc(Location loc, func::FuncOp unmarshalFunc,
                                  PatternRewriter &rewriter,
                                  func::FuncOp devFunc,
                                  cudaq::cc::StructType bufferTy) {
    Block *entryBlock = unmarshalFunc.addEntryBlock();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    auto *ctx = rewriter.getContext();

    // Unmarshal arguments from the buffer and call the device function.
    auto i64Ty = rewriter.getI64Type();
    Value bufferSize =
        rewriter.create<cudaq::cc::SizeOfOp>(loc, i64Ty, bufferTy);
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    auto ptrArrTy = cudaq::cc::PointerType::get(
        cudaq::cc::ArrayType::get(rewriter.getI8Type()));
    auto rawBuffer = rewriter.create<cudaq::cc::CastOp>(
        loc, ptrArrTy, entryBlock->getArgument(0));
    Value trailingData = rewriter.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTy, rawBuffer, ArrayRef<cudaq::cc::ComputePtrArg>{bufferSize});
    auto ptrBuffTy = cudaq::cc::PointerType::get(bufferTy);
    auto argsBuffer = rewriter.create<cudaq::cc::CastOp>(
        loc, ptrBuffTy, entryBlock->getArgument(0));
    FunctionType devFuncTy = devFunc.getFunctionType();
    SmallVector<Value> args;
    auto module = devFunc->getParentOfType<ModuleOp>();
    auto newDevFuncTy = cudaq::opt::factory::toHostSideFuncType(
        devFuncTy, /*addThisPtr=*/false, module);
    std::size_t offset = 0;
    SmallVector<Value> stringArgs;
    for (auto iter : llvm::enumerate(devFuncTy.getInputs())) {
      auto [a, t] = cudaq::opt::marshal::processCallbackInputValue(
          loc, rewriter, trailingData, argsBuffer, iter.value(), iter.index(),
          bufferTy);
      // Save any strings, as we need to deconstruct them after the call.
      if (isa<cudaq::cc::CharspanType>(iter.value()))
        stringArgs.push_back(a);
      trailingData = t;
      if (auto strTy = dyn_cast<cudaq::cc::StructType>(iter.value())) {
        if (cudaq::opt::factory::isX86_64(module) &&
            cudaq::opt::factory::structUsesTwoArguments(strTy)) {
          auto i = iter.index() + offset;
          SmallVector<Type> inputs{newDevFuncTy.getInputs().begin(),
                                   newDevFuncTy.getInputs().end()};
          SmallVector<Type> pair{inputs[i], inputs[i + 1]};
          ++offset;
          auto dubTy = cudaq::cc::PointerType::get(
              cudaq::cc::StructType::get(ctx, pair));
          auto dubPtr = rewriter.create<cudaq::cc::CastOp>(loc, dubTy, a);
          auto ptr0 = rewriter.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(inputs[i]), dubPtr,
              ArrayRef<cudaq::cc::ComputePtrArg>{0});
          args.push_back(rewriter.create<cudaq::cc::LoadOp>(loc, ptr0));
          auto ptr1 = rewriter.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(inputs[i + 1]), dubPtr,
              ArrayRef<cudaq::cc::ComputePtrArg>{1});
          args.push_back(rewriter.create<cudaq::cc::LoadOp>(loc, ptr1));
          continue;
        }
      }
      args.push_back(a);
    }

    auto callDevFunc = rewriter.create<func::CallOp>(
        loc, newDevFuncTy.getResults(), devFunc.getName(), args);

    // Deconstruct any strings.
    for (Value v : stringArgs) {
      Value cast = rewriter.create<cudaq::cc::CastOp>(loc, ptrTy, v);
      rewriter.create<func::CallOp>(loc, TypeRange{},
                                    cudaq::runtime::bindingDeconstructString,
                                    ValueRange{cast});
    }

    // If the device function has a return value, then store it to the result
    // space in the buffer.
    if (!devFuncTy.getResults().empty()) {
      auto resTy = devFuncTy.getResult(0);
      if (cudaq::cc::isDynamicType(resTy)) {
        unmarshalFunc.emitOpError("return type is not yet implemented");
        Value undef = rewriter.create<cudaq::cc::UndefOp>(loc, resTy);
        rewriter.create<func::ReturnOp>(loc, undef);
        return;
      }
      std::int32_t numInputs = devFuncTy.getNumInputs();
      auto outputPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(resTy), argsBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
      rewriter.create<cudaq::cc::StoreOp>(loc, callDevFunc.getResult(0),
                                          outputPtr);
    }

    auto zeroCall = rewriter.create<func::CallOp>(
        loc, unmarshalFunc.getFunctionType().getResult(0),
        "__nvqpp_zeroDynamicResult", ValueRange{});
    rewriter.create<func::ReturnOp>(loc, zeroCall.getResult(0));
    unmarshalFunc.setPublic();
  }

  static void genRegistrationHook(Location loc, StringRef callbackName,
                                  PatternRewriter &rewriter,
                                  LLVM::GlobalOp callbackNameObj,
                                  ModuleOp module) {
    auto *ctx = rewriter.getContext();
    auto ptrType = cudaq::cc::PointerType::get(rewriter.getI8Type());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto initFun = rewriter.create<LLVM::LLVMFuncOp>(
        loc, callbackName.str() + ".callbackRegFunc",
        LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx), {}));
    auto *initFunEntry = initFun.addEntryBlock();
    rewriter.setInsertionPointToStart(initFunEntry);
    auto callbackAddr = rewriter.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(callbackNameObj.getType()),
        callbackNameObj.getSymName());
    auto castCallbackRef =
        rewriter.create<cudaq::cc::CastOp>(loc, ptrType, callbackAddr);
    rewriter.create<func::CallOp>(loc, std::nullopt,
                                  cudaq::runtime::CudaqRegisterCallbackName,
                                  ValueRange{castCallbackRef});
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

    // The registration is called by a global ctor at init time.
    cudaq::opt::factory::createGlobalCtorCall(
        module, FlatSymbolRefAttr::get(ctx, initFun.getName()));
  }
};

/// This pass replaces cc.device_call ops with a pattern of calls to the CUDA-Q
/// runtime layer to execute device calls on a distributed memory model
/// integrated QPU. The distributed memory model is required if different
/// processing units in the aggregate QPU have distributed (not shared) memory
/// spaces.
class DistributedDeviceCallPass
    : public cudaq::opt::impl::DistributedDeviceCallBase<
          DistributedDeviceCallPass> {
public:
  using DistributedDeviceCallBase::DistributedDeviceCallBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleOp module = getOperation();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::extractDevPtr))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::CudaqRegisterCallbackName);
      return;
    }

    patterns.add<ExtractDevPtrOpPat>(ctx);

    if (useMagicUnicorn) {
      // For this solution, we replace the device_call operations with calls to
      // the identified symbol. The function identified via the symbol is merely
      // annotated with a special passthrough attribute of "cuda-q-fun-id" with
      // an integer value. The integer value is a 64 bit hash of the symbol
      // name. It is entirely up to the unicorn solution to figure out how to
      // bind the hash to the correct callback function.
      patterns.insert<UnicornDeviceCallPat>(ctx);
      if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
        signalPassFailure();
      return;
    }

    // If we're not using the magic unicorn solution, then use the generalized
    // approach. This consists of having the compiler generate the marshal and
    // unmarshal code and call through the runtime's hook function, which should
    // be specialized for whatever target configuration happens to be selected.
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::CudaqRegisterCallbackName))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::CudaqRegisterCallbackName);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::runtime::callDeviceCallback))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::callDeviceCallback);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_zeroDynamicResult"))) {
      module.emitError("could not load __nvqpp_zeroDynamicResult");
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, cudaq::llvmMemCopyIntrinsic))) {
      module.emitError(std::string("could not load ") +
                       cudaq::llvmMemCopyIntrinsic);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::bindingInitializeString))) {
      module.emitError(std::string("could not load ") +
                       cudaq::runtime::bindingInitializeString);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::bindingDeconstructString))) {
      module.emitError(std::string("could not load ") +
                       cudaq::runtime::bindingDeconstructString);
      return;
    }

    patterns.insert<DistributedDeviceCallPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet patterns2(ctx);
    patterns2.insert<DistributedFuncPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns2))))
      signalPassFailure();
  }
};
} // namespace
