/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeMLIR.h"
#include "ThunkInterface.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Support/TargetConfig.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"

using namespace mlir;
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::mlir_compiler)

namespace cudaq {
static bool mlirLLVMInitialized = false;

static llvm::StringMap<cudaq::Translation> &getTranslationRegistry() {
  static llvm::StringMap<cudaq::Translation> translationBundle;
  return translationBundle;
}
cudaq::Translation &getTranslation(StringRef name) {
  auto &registry = getTranslationRegistry();
  if (!registry.count(name))
    throw std::runtime_error("Invalid IR Translation (" + name.str() + ").");
  return registry[name];
}

static void registerTranslation(StringRef name, StringRef description,
                                const TranslateFromMLIRFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(function, description);
}

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function) {
  registerTranslation(name, description, function);
}
} // namespace cudaq

#include "RuntimeMLIRCommonImpl.h"

namespace cudaq {

std::unique_ptr<MLIRContext> initializeMLIR() {
  if (!mlirLLVMInitialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    cudaq::registerAllPasses();
    registerToQIRTranslation();
    registerToOpenQASMTranslation();
    registerToIQMJsonTranslation();
    mlirLLVMInitialized = true;
  }

  DialectRegistry registry;
  cudaq::opt::registerCodeGenDialect(registry);
  cudaq::registerAllDialects(registry);
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  registerLLVMDialectTranslation(*context);
  return context;
}

std::optional<std::string> getEntryPointName(OwningOpRef<ModuleOp> &module) {
  std::string name;
  module->walk([&name](mlir::func::FuncOp op) {
    if (op.getName().endswith(".thunk")) {
      name = op.getName();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (!name.empty())
    return name;

  std::vector<std::string> unmarshalNames;
  module->walk([&unmarshalNames](func::FuncOp op) {
    if (op.getName().startswith("unmarshal.")) {
      unmarshalNames.push_back(op.getName().str());
    }
    return mlir::WalkResult::advance();
  });

  if (unmarshalNames.size() == 1)
    return unmarshalNames.front();

  return std::nullopt;
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
  }

  return callbacks;
}

void mlir_compiler::initialize(const config::TargetConfig &config) {
  // check the config for symbol locations we care about
  for (auto &device : config.Devices) {
    auto devLibs =
        device.Config.ExposedLibraries.value_or(std::vector<std::string>{});
    for (auto &d : devLibs)
      symbolLocations.push_back(d);
  }

  context = initializeMLIR();
}

std::size_t mlir_compiler::load(const std::string &mlir_code) {
  // Start a loaded module id counter.
  static std::size_t nextHandle = 0;

  // Load the ModuleOp
  LoadedModule loaded;
  loaded.m_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_code, context.get());
  if (!loaded.m_module) {
    cudaq::info("Failed to parse Quake code into MLIR module");
    return 0;
  }

  // Infer the entry point FuncOp name. At this point,
  // could be thunk or unmarshal function
  auto entryName = getEntryPointName(loaded.m_module);
  if (entryName)
    loaded.entryPointName = entryName.value();
  else {
    loaded.m_module->dump();
    throw std::runtime_error(
        "Could not infer the entry point name for this module.");
  }
  // Lower the device_call ops to marshal / unmarshal ops.
  mlir::PassManager pm(context.get());
  pm.addPass(cudaq::opt::createDistributedDeviceCall());
  // Run now, and do some work on it
  if (failed(pm.run(*loaded.m_module)))
    return -1;

  // Extract the callbacks.
  loaded.callbacks = extractCallbacks(loaded.m_module);

  // Store the loaded module
  loaded_modules.insert({nextHandle, std::move(loaded)});

  // Return the module handle.
  auto retHandle = nextHandle;
  cudaq::info("Kernel loaded successfully with handle: {}", nextHandle);
  nextHandle++;
  return retHandle;
}

void mlir_compiler::remove_callback(std::size_t moduleHandle,
                                    const std::string &funcName) {
  auto &loadedMod = loaded_modules.at(moduleHandle);

  std::string local = funcName;
  if (!StringRef(funcName).contains("unmarshal."))
    local = "unmarshal." + funcName;

  // Here we want to remove the unmarshal and the actual symbol
  auto funcOp = loadedMod.m_module->lookupSymbol<func::FuncOp>(local);

  // Replace with just a private declaration
  auto newFunc = mlir::func::FuncOp::create(funcOp.getLoc(), funcOp.getName(),
                                            funcOp.getFunctionType());
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

  builder.create<func::ReturnOp>(builder.getUnknownLoc(), callOp.getResults());
  loadedMod.m_module->insert(loadedMod.m_module->begin(), newFunc);
  funcOp->erase(); // Remove original implementation
}

void mlir_compiler::compile(std::size_t moduleHandle) {
  auto &loadedMod = loaded_modules.at(moduleHandle);
  // FIXME run subtype specific lowering
  // Lower the kernel code to adaptive QIR

  mlir::PassManager pm(context.get());
  lowerToLLVM(pm, loadedMod.m_module);

  std::vector<llvm::StringRef> stringRefs;
  for (const auto &str : symbolLocations) {
    if (!StringRef(str).endswith(".fatbin")) {
      cudaq::info("Adding Symbol Location {}", str);
      stringRefs.push_back(llvm::StringRef(str));
    }
  }

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.sharedLibPaths = stringRefs;
  engineOptions.jitCodeGenOptLevel = llvm::CodeGenOpt::Default;
  auto maybeEngine =
      mlir::ExecutionEngine::create(*(loadedMod.m_module), engineOptions);
  if (!maybeEngine) {
    cudaq::info("Failed to create execution engine");
  }

  loadedMod.jitEngine = std::move(maybeEngine.get());
}

void mlir_compiler::launch(std::size_t moduleHandle, void *argsBuffer) {

  // Get the loaded kernel.
  auto &[thunkName, m_mod, execEngine, callbacks] =
      loaded_modules.at(moduleHandle);

  // Construct the args
  KernelThunkResultType res;
  bool flag = false;
  auto execEngineResult = mlir::ExecutionEngine::result(res);
  std::vector<void *> args{&argsBuffer, &flag, &execEngineResult};

  // Invoke the thunk function
  auto err = execEngine->invokePacked(thunkName, args);
  if (err) {
    std::string errorMsg;
    // FIXME fill this
    throw std::runtime_error("Kernel execution failed: " + errorMsg);
  }

  // potential result is in the thunk args.
  return;
}

std::vector<callback> mlir_compiler::get_callbacks(std::size_t moduleHandle) {
  return loaded_modules.at(moduleHandle).callbacks;
}

class default_qir_compiler : public mlir_compiler {
protected:
  void lowerToLLVM(mlir::PassManager &pm,
                   mlir::OwningOpRef<mlir::ModuleOp> &op) override {
    cudaq::opt::addPipelineConvertToQIR(pm, "qir");
    if (failed(pm.run(*op)))
      throw std::runtime_error("error lowering code to QIR.");
  }
  CUDAQ_EXTENSION_CREATOR_FUNCTION(mlir_compiler, default_qir_compiler);
};
CUDAQ_REGISTER_EXTENSION_TYPE(default_qir_compiler)

} // namespace cudaq
