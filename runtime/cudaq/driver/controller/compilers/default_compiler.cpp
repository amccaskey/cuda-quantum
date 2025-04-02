/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"
#include "cudaq/driver/controller/quake_compiler.h"

#include "common/ThunkInterface.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::quake_compiler)

namespace cudaq::driver {
void setTarget(target *);

static std::once_flag mlir_init_flag;

struct LoadedKernel {
  std::string thunkName = "";
  std::optional<std::size_t> numRequiredQubits = std::nullopt;
  std::unique_ptr<ExecutionEngine> jitEngine;
};

class default_compiler : public quake_compiler {
protected:
  std::unique_ptr<MLIRContext> context;
  std::map<handle, LoadedKernel> loadedKernels;
  std::string qirType = "qir-adaptive";

public:
  void initialize(const config::TargetConfig &config) override {
    std::call_once(mlir_init_flag, []() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      cudaq::registerAllPasses();
    });
    DialectRegistry registry;
    cudaq::opt::registerCodeGenDialect(registry);
    cudaq::registerAllDialects(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
    registerLLVMDialectTranslation(*context);

    // FIXME make this configurable
    m_target = target::get("default_target");
    m_target->initialize(config);
    setTarget(m_target.get());
  }

  std::size_t compile(const std::string &quake) override {
    printf("%s\n", quake.c_str());
    // Parse the Quake code into an MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(quake, context.get());

    if (!module) {
      cudaq::info("Failed to parse Quake code into MLIR module");
      return 0;
    }

    std::string thunkName;
    module->walk([&thunkName](mlir::func::FuncOp op) {
      if (op.getName().endswith(".thunk")) {
        thunkName = op.getName();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    mlir::PassManager pm(context.get());
    cudaq::opt::addPipelineConvertToQIR(pm, qirType);
    if (failed(pm.run(*module)))
      return -1;

    module->dump();

    // Potentially get the number of required qubits.
    std::optional<std::size_t> numRequiredQubits =
        [&]() -> std::optional<std::size_t> {
      std::string result;
      module->walk([&result](LLVM::LLVMFuncOp op) {
        auto passthroughAttr = op->getAttrOfType<ArrayAttr>("passthrough");
        if (!passthroughAttr)
          return WalkResult::advance();
        for (std::size_t i = 0; i < passthroughAttr.size(); i++) {
          auto keyValPair = dyn_cast<ArrayAttr>(passthroughAttr[i]);
          if (!keyValPair)
            continue;

          auto keyAttr = dyn_cast<StringAttr>(keyValPair[0]);
          if (!keyAttr.getValue().equals("requiredQubits"))
            continue;
          result = dyn_cast<StringAttr>(keyValPair[1]).getValue();
          return WalkResult::interrupt();
        }

        return WalkResult::interrupt();
      });

      try {
        return std::stoi(result);
      } catch (...) {
        // do nothing, just return nullopt
      }

      return std::nullopt;
    }();

    if (numRequiredQubits)
      cudaq::info("Need to allocate {} qubits.", *numRequiredQubits);

    // Generate a unique handle for the loaded kernel
    static handle nextHandle = 0;

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOpt::Default;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
      cudaq::info("Failed to create execution engine");
      return 0;
    }
    loadedKernels.insert(
        {nextHandle, LoadedKernel{thunkName, numRequiredQubits,
                                  std::move(maybeEngine.get())}});

    auto retHandle = nextHandle;
    cudaq::info("Kernel loaded successfully with handle: {}", nextHandle);
    nextHandle++;
    return retHandle;
  }

  void launch(std::size_t kernelHandle, void *thunkArgs) override {
    // Get the loaded kernel.
    auto &[thunkName, numRequiredQubits, execEngine] =
        loadedKernels.at(kernelHandle);

    // Need to tell the target to allocate numRequiredQubits
    if (numRequiredQubits)
      m_target->allocate(*numRequiredQubits);

    // Construct the args
    KernelThunkResultType res;
    bool flag = false;
    auto execEngineResult = mlir::ExecutionEngine::result(res);
    std::vector<void *> args{&thunkArgs, &flag, &execEngineResult};

    // Invoke the thunk function
    auto err = execEngine->invokePacked(thunkName, args);
    if (err) {
      std::string errorMsg;
      llvm::raw_string_ostream os(errorMsg);
      llvm::handleAllErrors(std::move(err),
                            [&](const llvm::ErrorInfoBase &ei) { ei.log(os); });
      throw std::runtime_error("Kernel execution failed: " + errorMsg);
    }

    if (numRequiredQubits)
      m_target->deallocate(*numRequiredQubits);
      
    // potential result is in the thunk args.
    return;
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(quake_compiler, default_compiler);
};

CUDAQ_REGISTER_EXTENSION_TYPE(default_compiler)

} // namespace cudaq::driver