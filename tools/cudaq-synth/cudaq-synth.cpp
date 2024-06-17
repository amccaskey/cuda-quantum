/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Support/Version.h"
#include "cudaq/Todo.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"

#include <filesystem>
//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional,
                  llvm::cl::desc("<input quake mlir file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    kernelName("kernel-name",
               llvm::cl::desc("Specify the name of the kernel to synthesize"),
               llvm::cl::value_desc("name"), llvm::cl::init(""));

static llvm::cl::list<std::string> args("arg", llvm::cl::desc("The argument"));

static constexpr const char BOLD[] = "\033[1m";
static constexpr const char RED[] = "\033[91m";
static constexpr const char CLEAR[] = "\033[0m";

using namespace mlir;

namespace cudaq {
/// Typedef the KernelArgs Creator Function
typedef std::size_t (*Creator)(void **, void **);

/// Retrieve the kernel args creator function for the kernel name
Creator getArgsCreator(const std::string &);

std::pair<void *, std::size_t> mapToRawArgs(const std::string &kernelName,
                                            void **argPointers) {
  void *rawArgs = nullptr;
  auto argsCreator = getArgsCreator(kernelName);
  auto argsSize = argsCreator(argPointers, &rawArgs);
  return std::make_pair(rawArgs, argsSize);
}

template <typename T>
void simpleAllocateAndSet(auto &argData, auto &deleters,
                          const auto &casterFunc) {
  T *runtimeArg = new T();
  *runtimeArg = static_cast<T>(casterFunc());
  argData.push_back(runtimeArg);
  deleters.push_back([](void *ptr) { delete static_cast<T *>(ptr); });
}

void packRuntimeArgs(std::vector<std::string> &values,
                     std::vector<void *> &args,
                     std::vector<std::function<void(void *)>> &deleters,
                     ArrayRef<Type> argTypes) {

  for (std::size_t counter = 0; auto &kernelArgTy : argTypes) {
    llvm::TypeSwitch<mlir::Type, void>(kernelArgTy)
        .Case([&](IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1)
            return simpleAllocateAndSet<bool>(
                args, deleters, [&]() { return std::stoi(values[counter++]); });

          if (ty.getIntOrFloatBitWidth() == 32)
            return simpleAllocateAndSet<int>(
                args, deleters, [&]() { return std::stoi(values[counter++]); });

          if (ty.getIntOrFloatBitWidth() == 64)
            return simpleAllocateAndSet<std::size_t>(args, deleters, [&]() {
              return std::stoul(values[counter++]);
            });

          throw std::runtime_error("Failed to process input integer argument.");
        })
        .Case([&](Float64Type ty) {
          return simpleAllocateAndSet<double>(
              args, deleters, [&]() { return std::stod(values[counter++]); });
        })
        .Default([&](Type ty) {
          ty.dump();
          throw std::runtime_error("Failed to process input argument.");
        });
  }
}

/// @brief Return the function-like Operation of type T that meets the
/// given requirements on the function name.
template <typename T>
T getFunctionOp(auto &module, const std::vector<std::string> &mustContain) {
  T kernelFunc;
  module->walk([&](T function) {
    auto name = function.getName();
    bool allSatisfied = true;
    for (auto &contain : mustContain) {
      if (!name.contains(contain)) {
        allSatisfied = false;
        break;
      }
    }

    if (allSatisfied) {
      kernelFunc = function;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (!kernelFunc)
    throw std::runtime_error("Could not find function operation with name = " +
                             kernelName + " in current module.");

  return kernelFunc;
}

auto *jitAndInvokeKernelRegFunc(auto &module, const std::string &kernelName) {
  auto kernelRegFuncOpName = cudaq::getFunctionOp<LLVM::LLVMFuncOp>(
                                 module, {"kernelRegFunc", kernelName})
                                 .getName();

  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  SmallVector<StringRef, 4> sharedLibs;
  opts.sharedLibPaths = sharedLibs;
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

  auto jitOrError = ExecutionEngine::create(*module, opts);
  assert(!!jitOrError);

  auto uniqueJit = std::move(jitOrError.get());
  auto regFuncPtr = uniqueJit->lookup(kernelRegFuncOpName);
  if (!regFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelReg = reinterpret_cast<void (*)()>(*regFuncPtr);
  kernelReg();
  return uniqueJit.release();
}
} // namespace cudaq

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerTranslationCLOptions();
  registerAllPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "quake mlir to llvm ir compiler\n");

  DialectRegistry registry;
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  registerAllDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError())
    cudaq::emitFatalError(UnknownLoc::get(&context),
                          "Could not open input file: " + ec.message());

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  DiagnosticEngine &engine = context.getDiagEngine();
  engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
    llvm::errs() << BOLD << RED
                 << "[quake-translate] Dumping Module after error.\n"
                 << CLEAR;
    for (auto &n : diag.getNotes()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      n.print(os);
      os.flush();
      llvm::errs() << BOLD << RED << "[quake-translate] Reported Error: " << s
                   << "\n"
                   << CLEAR;
    }
    bool should_propagate_diagnostic = true;
    return failure(should_propagate_diagnostic);
  });

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  registerLLVMDialectTranslation(*module->getContext());
  auto originalModule = module->clone();

  /// @brief Search the given Module for the function with provided name.
  auto kernelFuncOp = cudaq::getFunctionOp<func::FuncOp>(
      module, {kernelName, cudaq::runtime::cudaqGenPrefixName});
  auto kName = kernelFuncOp.getName().str();
  auto argTypes = kernelFuncOp.getArgumentTypes();
  if (argTypes.size() != args.size())
    throw std::runtime_error("invalid number of runtime arguments provided.");
  std::vector<std::string> localArgs;
  for (auto &arg : args)
    localArgs.push_back(arg);

  // Pack the user-provided runtime arguments
  std::vector<void *> argData;
  std::vector<std::function<void(void *)>> deleters;
  cudaq::packRuntimeArgs(localArgs, argData, deleters, argTypes);

  {
    PassManager pm(&context);
    pm.addPass(cudaq::opt::createGenerateKernelExecution());
    pm.addPass(createCanonicalizerPass());
    cudaq::opt::addLowerToCCPipeline(pm);
    OpPassManager &optPM = pm.nest<func::FuncOp>();
    optPM.addPass(cudaq::opt::createLowerToCFGPass());
    pm.addPass(cudaq::opt::createCCToLLVM());
    if (failed(pm.run(*module)))
      return -1;
  }

  // Register the argsCreator function
  auto *keepJitAround = cudaq::jitAndInvokeKernelRegFunc(module, kernelName);
  auto [rawArgs, argsSize] = cudaq::mapToRawArgs(
      StringRef(kName).drop_front(cudaq::runtime::cudaqGenPrefixLength).str(),
      argData.data());

  // Run the Quake Synth pass.
  {
    PassManager pm(&context);
    pm.addPass(cudaq::opt::createQuakeSynthesizer(kName, rawArgs));
    if (failed(pm.run(originalModule)))
      return -1;
  }
  originalModule.dump();

  // Cleanup
  delete keepJitAround;
  for (std::size_t i = 0; i < argData.size(); i++)
    deleters[i](argData[i]);

  return 0;
}
