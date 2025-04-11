/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/controller/channel.h"
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

#include <dlfcn.h>

using namespace mlir;

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::quake_compiler)

namespace cudaq::driver {

/// @brief Set the target on the QIR implementation
extern void setTarget(target *);

static std::once_flag mlir_init_flag;

/// @brief A Loaded Module tracks the kernel thunk function name,
/// the number of required qubits, the MLIR ExecutionEngine, and
/// invoked callback functions.
struct LoadedModule {
  std::string thunkName = "";
  std::optional<std::size_t> numRequiredQubits = std::nullopt;
  std::unique_ptr<ExecutionEngine> jitEngine;
  std::vector<callback> callbacks;
};

class default_compiler : public quake_compiler {
protected:
  std::unique_ptr<MLIRContext> context;
  std::map<handle, LoadedModule> loadedKernels;
  std::string qirType = "qir-adaptive";

  std::string getKernelThunkName(OwningOpRef<ModuleOp> &module) {
    std::string thunkName;
    module->walk([&thunkName](mlir::func::FuncOp op) {
      if (op.getName().endswith(".thunk")) {
        thunkName = op.getName();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return thunkName;
  }

  std::vector<callback> extractCallbacks(OwningOpRef<ModuleOp> &module) {
    std::vector<std::string> unmarshalFuncNames;
    module->walk([&unmarshalFuncNames](func::FuncOp op) {
      if (op.getName().startswith("unmarshal.")) {
        unmarshalFuncNames.push_back(op.getName().str());
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    std::vector<callback> callbacks;
    for (auto &unmarshalFuncName : unmarshalFuncNames) {
      auto funcOp = module->lookupSymbol<func::FuncOp>(unmarshalFuncName);
      if (!funcOp)
        continue;
      auto actualCalledFuncOp = module->lookupSymbol<func::FuncOp>(
          StringRef(unmarshalFuncName).drop_front(10).str());
      auto zeroDynRes =
          module->lookupSymbol<func::FuncOp>("__nvqpp_zeroDynamicResult");
      std::string funcCode;
      {
        llvm::raw_string_ostream strOut(funcCode);
        OpPrintingFlags opf;
        funcOp.print(strOut, opf);
        strOut << '\n';
        actualCalledFuncOp.print(strOut, opf);
        strOut << '\n';
        zeroDynRes.print(strOut, opf);
      }

      callbacks.emplace_back(StringRef(unmarshalFuncName).drop_front(10).str(),
                             funcCode);

      // Replace with just a private declaration
      auto newFunc = mlir::func::FuncOp::create(
          funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());
      newFunc.setPrivate();
      // 1. Add entry block with proper signature
      mlir::Block *entryBlock = newFunc.addEntryBlock();

      // 2. Create builder at entry block's end
      auto builder = mlir::OpBuilder::atBlockBegin(entryBlock);
      builder.setInsertionPointToEnd(entryBlock);

      auto callOp = builder.create<func::CallOp>(
          builder.getUnknownLoc(),
          SymbolRefAttr::get(context.get(), "__nvqpp_zeroDynamicResult"),
          funcOp.getFunctionType().getResults(), ValueRange{});

      builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                     callOp.getResults());
      module->insert(module->begin(), newFunc);
      funcOp->erase(); // Remove original implementation
    }

    return callbacks;
  }

  std::optional<std::size_t>
  getNumRequiredQubits(OwningOpRef<ModuleOp> &module) {
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
  }

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

  std::vector<callback> get_callbacks(handle moduleHandle) override {
    return loadedKernels.at(moduleHandle).callbacks;
  }

  std::size_t
  compile_unmarshaler(const std::string &mlirCode,
                      const std::vector<std::string> &symbolLocations) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(
            fmt::format("module {{ {} }}", mlirCode), context.get());

    if (!module) {
      cudaq::info("Failed to parse Quake code into MLIR module");
      return 0;
    }

    mlir::PassManager pm(context.get());
    pm.addPass(cudaq::opt::createCCToLLVM());
    if (failed(pm.run(*module)))
      return -1;

    std::string unmarshalFuncName;
    module->walk([&unmarshalFuncName](LLVM::LLVMFuncOp op) {
      if (op.getName().startswith("unmarshal.")) {
        unmarshalFuncName = op.getName().str();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    static handle nextHandle = 0;

    std::vector<llvm::StringRef> stringRefs;
    for (const auto &str : symbolLocations) {
      stringRefs.push_back(llvm::StringRef(str));
    }

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.sharedLibPaths = stringRefs;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOpt::Default;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
      cudaq::info("Failed to create execution engine");
      return 0;
    }

    loadedKernels.insert({nextHandle, LoadedModule{unmarshalFuncName,
                                                   std::nullopt,
                                                   std::move(maybeEngine.get()),
                                                   {}}});

    auto retHandle = nextHandle;
    cudaq::info("Unmarshal Callback loaded successfully with handle: {}",
                nextHandle);
    nextHandle++;
    return retHandle;
  }

  std::size_t compile(const std::string &quake) override {
    // Parse the Quake code into an MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(quake, context.get());

    if (!module) {
      cudaq::info("Failed to parse Quake code into MLIR module");
      return 0;
    }

    // Get the kernel thunk function name
    auto thunkName = getKernelThunkName(module);

    // Lower device_call operations
    mlir::PassManager pm(context.get());
    pm.addPass(cudaq::opt::createDistributedDeviceCall());
    // Run now, and do some work on it
    if (failed(pm.run(*module)))
      return -1;

    // Search the module for unmarshal.* functions,
    // store the FuncOp code for each. Removes the
    // body of the unmarshal op since we don't
    // execute it here and we don't have the symbols it needs
    auto callbacks = extractCallbacks(module);

    // Lower the kernel code to adaptive QIR
    cudaq::opt::addPipelineConvertToQIR(pm, qirType);
    if (failed(pm.run(*module)))
      return -1;

    // Potentially get the number of required qubits.
    std::optional<std::size_t> numRequiredQubits = getNumRequiredQubits(module);
    if (numRequiredQubits)
      cudaq::info("Need to allocate {} qubits.", *numRequiredQubits);

    // Generate a unique handle for the loaded kernel
    static handle nextHandle = 0;

    // Create the ExecutionEngine for JIT compilation
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOpt::Default;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
      cudaq::info("Failed to create execution engine");
      return 0;
    }

    // Kernel is loaded, add it to the map
    loadedKernels.insert(
        {nextHandle, LoadedModule{thunkName, numRequiredQubits,
                                  std::move(maybeEngine.get()), callbacks}});

    auto retHandle = nextHandle;
    cudaq::info("Kernel loaded successfully with handle: {}", nextHandle);
    nextHandle++;
    return retHandle;
  }

  void launch(std::size_t moduleHandle, void *thunkArgs) override {
    // Get the loaded kernel.
    auto &[thunkName, numRequiredQubits, execEngine, callbacks] =
        loadedKernels.at(moduleHandle);

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
