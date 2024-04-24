/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Characteristics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLOpsDialect.h.inc"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <filesystem>
#include <fstream>

using namespace mlir;

namespace {

const static std::string transformInterpTemplate = R"#(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
     transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
      }
    }
    transform.yield
  }
}
)#";

class ApplyQuantumCircuitPatternRewritesPass
    : public cudaq::opt::ApplyQuantumCircuitPatternRewritesBase<
          ApplyQuantumCircuitPatternRewritesPass> {
public:
  ApplyQuantumCircuitPatternRewritesPass() = default;
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<cudaq::cc::CCDialect>();
    registry.insert<quake::QuakeDialect>();
    registry.insert<pdl::PDLDialect>();
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto context = mod.getContext();

    // Goals:
    // 1. create new ModuleOp and add the input module op
    auto location = FileLineColLoc::get(context, "<tmp>", 1, 1);
    ImplicitLocOpBuilder builder(location, context);
    auto moduleOp = builder.create<ModuleOp>();
    moduleOp.push_back(mod);

    // 2. Load all PDLPatterns from `INSTALL/passes` directory.
    std::vector<OwningOpRef<Operation *>> pdlMods;
    std::map<std::string, pdl::PatternOp> pdlPatterns;
    auto parseAndAddPattern =
        [&](const std::string &fileName) -> LogicalResult {
      if (!std::filesystem::exists(fileName))
        return mod->emitOpError("Invalid circuit pattern rewrite file - " +
                                fileName);

      if (std::filesystem::path(fileName).extension() != ".pdl")
        return mod->emitOpError(
            "Invalid circuit pattern rewrite file - must be a .pdl file (got " +
            fileName + ").");

      std::ifstream t(fileName);
      std::string str((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());
      auto pdlMod = parseSourceString(str, context);
      pdlMod->walk([&](pdl::PatternOp op) {
        auto iter = pdlPatterns.find(op.getName().str());
        if (iter == pdlPatterns.end())
          pdlPatterns.insert({op.getName().str(), op});

        return WalkResult::advance();
      });
      pdlMods.emplace_back(std::move(pdlMod));
      return success();
    };

    // If user provides a specific pattern, run it, skip
    // any patterns in the pattern folder location
    if (!pattern.empty()) {
      if (failed(parseAndAddPattern(pattern.getValue()))) {
        signalPassFailure();
        return;
      }
    } else {
      // If we made it here, we have to have a patterns folder
      if (patterns.empty()) {
        mod.emitOpError("Could not find a circuit rewrite pattern to apply.");
        signalPassFailure();
        return;
      }

      if (!std::filesystem::exists(patterns.getValue())) {
        mod->emitOpError("Invalid circuit rewrite patterns directory - " +
                         patterns.getValue());
        signalPassFailure();
        return;
      }

      // Loop over available patterns
      for (auto &file :
           std::filesystem::directory_iterator(patterns.getValue()))
        if (failed(parseAndAddPattern(file.path().string()))) {
          signalPassFailure();
          return;
        }
    }

    // 3. add the available PDL patterns to the transformer module
    auto transformMod = parseSourceString(transformInterpTemplate, context);
    transformMod->walk([&](transform::WithPDLPatternsOp op) {
      Block &block = *op.getBody().begin();
      auto localBuilder = OpBuilder::atBlockBegin(&block);
      for (auto &[name, pdlOp] : pdlPatterns)
        localBuilder.insert(pdlOp.clone());

      return WalkResult::advance();
    });

    transformMod->walk([&](transform::SequenceOp op) {
      Block &block = *op.getBody().begin();
      auto localBuilder = OpBuilder::atBlockBegin(&block);
      auto arg = block.getArgument(0);
      for (auto &[name, pattern] : pdlPatterns) {
        localBuilder.create<transform::PDLMatchOp>(
            location, transform::AnyOpType::get(context), arg,
            FlatSymbolRefAttr::get(context, name));
      }

      return WalkResult::advance();
    });

    // Add the transform module to the main payload
    moduleOp.push_back(cast<ModuleOp>(*transformMod));

    // 4. Run the transform interpreter pass
    transform::TransformOptions options;
    ModuleOp transformModule =
        transform::detail::getPreloadedTransformModule(context);
    Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(
        moduleOp, transformModule,
        transform::TransformDialect::kTransformEntryPointSymbolName.str());
    if (failed(transform::applyTransformNamedSequence(
            moduleOp, transformEntryPoint, transformModule, options))) {
      return signalPassFailure();
    }

    // 5. Code is now optimized. Done.
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
cudaq::opt::createApplyQuantumCircuitPatternRewrites() {
  return std::make_unique<ApplyQuantumCircuitPatternRewritesPass>();
}
