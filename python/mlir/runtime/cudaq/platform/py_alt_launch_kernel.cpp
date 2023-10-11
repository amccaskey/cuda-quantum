/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

#include "utils/OpaqueArguments.h"

namespace py = pybind11;
using namespace mlir;

namespace cudaq {

void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs) {
  auto mod = unwrap(module);
  auto cloned = mod.clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);
  
  PassManager pm(context);
  pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
  pm.addPass(cudaq::opt::createGenerateKernelExecution());
  cudaq::opt::addPipelineToQIR<>(pm);
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  SmallVector<StringRef, 4> sharedLibs;
  opts.llvmModuleBuilder =
      [](Operation *module,
         llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    llvmContext.setOpaquePointers(false);
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return nullptr;
    }
    ExecutionEngine::setupTargetTriple(llvmModule.get());
    return llvmModule;
  };

  auto jitOrError = ExecutionEngine::create(cloned, opts);
  assert(!!jitOrError);
  auto uniqueJit = std::move(jitOrError.get());
  auto *jit = uniqueJit.release();

  auto expectedPtr = jit->lookup(name + ".argsCreator");
  if (!expectedPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get argsCreator function.");
  }
  auto argsCreator =
      reinterpret_cast<std::size_t (*)(void **, void **)>(*expectedPtr);
  void *rawArgs = nullptr;
  [[maybe_unused]] auto size = argsCreator(runtimeArgs.data(), &rawArgs);

  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("cudaq::builder failed to get thunk function");

  auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

  auto &platform = cudaq::get_platform();
  if (platform.is_remote() || platform.is_emulated()) {
    struct ArgWrapper {
      ModuleOp mod;
      void *rawArgs = nullptr;
    };
    auto *wrapper = new ArgWrapper{mod, rawArgs};
    cudaq::altLaunchKernel(name.c_str(), thunk,
                           reinterpret_cast<void *>(wrapper), size, 0);
    delete wrapper;
  } else
    cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs, size, 0);

  std::free(rawArgs);
  delete jit;
}

void bindAltLaunchKernel(py::module &mod) {

  mod.def(
      "pyAltLaunchKernel",
      [](const std::string &name, MlirModule module, py::args runtimeArgs) {
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs);
        pyAltLaunchKernel(name, module, args);
      },
      "");
}
} // namespace cudaq