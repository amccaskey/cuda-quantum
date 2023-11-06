/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/BaseRemoteRESTQPU.h"
#include "common/RuntimeMLIRCommonImpl.h"

using namespace mlir;

namespace cudaq {

// We have to reproduce the TranslationRegistry here in this Translation Unit

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

// We cannot use the RemoteRESTQPU since we'll get LLVM / MLIR statically loaded
// twice. We've extracted most of RemoteRESTQPU into BaseRemoteRESTQPU and will
// implement some core functionality here in PyRemoteRESTQPU so we don't load
// twice
class PyRemoteRESTQPU : public cudaq::BaseRemoteRESTQPU {
protected:
  std::tuple<ModuleOp, MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    struct ArgsWrapper {
      ModuleOp mod;
      std::vector<std::string> callablNames;
      void *rawArgs = nullptr;
    };
    auto *wrapper = reinterpret_cast<ArgsWrapper *>(data);
    auto m_module = wrapper->mod;
    auto callableNames = wrapper->callablNames;

    auto *context = m_module->getContext();
    static bool initOnce = [&] {
      registerToQIRTranslation();
      registerToOpenQASMTranslation();
      registerToIQMJsonTranslation();
      registerLLVMDialectTranslation(*context);
      return true;
    }();
    (void)initOnce;

    // Here we have an opportunity to run any passes that are
    // specific to python before the rest of the RemoteRESTQPU workflow
    auto cloned = m_module.clone();
    PassManager pm(cloned.getContext());
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(callableNames));
    cudaq::opt::addAggressiveEarlyInlining(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        cudaq::opt::createUnwindLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
    pm.addPass(createInlinerPass());
    pm.addPass(cudaq::opt::createExpandMeasurementsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "Failure to synthesize callable block arguments in PyRemoteRESTQPU ");

    return std::make_tuple(cloned, context, wrapper->rawArgs);
  }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyRemoteRESTQPU, py_remote_rest)
