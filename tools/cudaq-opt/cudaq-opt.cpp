/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Common/InlinerInterface.h"
#include "cudaq/Optimizer/Dialect/Common/Traits.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/Version.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"

using namespace llvm;
using namespace mlir;

/// @brief Add a command line flag for loading plugins
static cl::list<std::string> CudaQPlugins(
    "load-cudaq-plugin",
    cl::desc("Load CUDA Quantum plugin by specifying its library"));

class CudaqTransformExtensions
    : public mlir::transform::TransformDialectExtension<
          CudaqTransformExtensions> {
public:
  using Base::Base;
  void init() {

    auto isHermitian = [](PatternRewriter &rewriter, PDLResultList &,
                          ArrayRef<PDLValue> pdlValues) {
      for (const PDLValue &pdlValue : pdlValues)
        if (Operation *op = pdlValue.dyn_cast<Operation *>())
          return success(op->hasTrait<cudaq::Hermitian>());

      return failure();
    };

    auto isQuakeOp = [](PatternRewriter &rewriter, PDLResultList &,
                        ArrayRef<PDLValue> pdlValues) {
      for (const PDLValue &pdlValue : pdlValues)
        if (Operation *op = pdlValue.dyn_cast<Operation *>())
          if (auto *dialect = op->getDialect())
            return success(dialect->getNamespace().equals("quake"));
      return failure();
    };

    auto isSameName = [](PatternRewriter &rewriter, PDLResultList &,
                         ArrayRef<PDLValue> pdlValues) {
      if (pdlValues.size() != 2)
        return failure();
      auto *op1 = pdlValues[0].dyn_cast<Operation *>();
      auto *op2 = pdlValues[1].dyn_cast<Operation *>();

      return success(
          op1->getName().stripDialect().equals(op2->getName().stripDialect()));
    };

    addDialectDataInitializer<mlir::transform::PDLMatchHooks>(
        [&](mlir::transform::PDLMatchHooks &hooks) {
          llvm::StringMap<mlir::PDLConstraintFunction> constraints;
          constraints.try_emplace("IsHermitian", isHermitian);
          constraints.try_emplace("IsQuakeOperation", isQuakeOp);
          constraints.try_emplace("IsSameName", isSameName);
          hooks.mergeInPDLMatchHooks(std::move(constraints));
        });
  }
};
int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);

  mlir::registerAllPasses();
  cudaq::opt::registerOptCodeGenPasses();
  cudaq::opt::registerOptTransformsPasses();
  cudaq::opt::registerAggressiveEarlyInlining();
  cudaq::opt::registerUnrollingPipeline();
  cudaq::opt::registerTargetPipelines();

  // See if we have been asked to load a pass plugin,
  // if so load it.
  std::vector<std::string> args(&argv[0], &argv[0] + argc);
  for (std::size_t i = 0; i < args.size(); i++) {
    if (args[i].find("-load-cudaq-plugin") != std::string::npos) {
      auto Plugin = cudaq::Plugin::Load(args[i + 1]);
      if (!Plugin) {
        errs() << "Failed to load passes from '" << args[i + 1]
               << "'. Request ignored.\n";
        return 1;
      }
      Plugin.get().registerExtensions();
      i++;
    }
  }

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  cudaq::registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.addExtensions<CudaqTransformExtensions>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nvq++ optimizer\n", registry));
}
