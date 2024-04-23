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

class QuantumCircuitOptimizationsPass
    : public cudaq::opt::QuantumCircuitOptimizationsBase<
          QuantumCircuitOptimizationsPass> {
public:
  QuantumCircuitOptimizationsPass() = default;
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<cudaq::cc::CCDialect>();
    registry.insert<quake::QuakeDialect>();
    registry.insert<pdl::PDLDialect>();
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto context = mod.getContext();
    // mod.dump();

    // Goals:
    // 1. create new ModuleOp and add the input module op
    auto location = FileLineColLoc::get(context, "<tmp>", 1, 1);
    ImplicitLocOpBuilder builder(location, context);
    auto moduleOp = builder.create<ModuleOp>();
    moduleOp.push_back(mod);

    // 2. Load all PDLPatterns from `INSTALL/passes` directory.
    std::string test = R"#(module {
  pdl.pattern @RemoveSingleTargetCommutingPair : benefit(0) {
    %0 = operand
    %1 = types
    %2 = operation(%0 : !pdl.value)  -> (%1 : !pdl.range<type>)
    %3 = results of %2 
    %4 = types
    %5 = operation(%3 : !pdl.range<value>)  -> (%4 : !pdl.range<type>)
    apply_native_constraint "IsQuakeOperation"(%2 : !pdl.operation)
    apply_native_constraint "IsQuakeOperation"(%5 : !pdl.operation)
    apply_native_constraint "IsHermitian"(%2 : !pdl.operation)
    apply_native_constraint "IsSameName"(%2, %5 : !pdl.operation, !pdl.operation)
    rewrite %5 {
      replace %5 with(%0 : !pdl.value)
      erase %2
    }
  }
  pdl.pattern @RemoveCNOTCommutingPair : benefit(0) {
    %0 = operand
    %1 = operand
    %2 = types
    %3 = operation(%0, %1 : !pdl.value, !pdl.value)  -> (%2 : !pdl.range<type>)
    apply_native_constraint "IsQuakeOperation"(%3 : !pdl.operation)
    apply_native_constraint "IsHermitian"(%3 : !pdl.operation)
    %4 = result 0 of %3
    %5 = result 1 of %3
    %6 = types
    %7 = operation(%4, %5 : !pdl.value, !pdl.value)  -> (%6 : !pdl.range<type>)
    apply_native_constraint "IsQuakeOperation"(%7 : !pdl.operation)
    apply_native_constraint "IsHermitian"(%7 : !pdl.operation)
    apply_native_constraint "IsSameName"(%3, %7 : !pdl.operation, !pdl.operation)
    rewrite %7 {
      replace %7 with(%0, %1 : !pdl.value, !pdl.value)
      erase %3
    }
  }
})#";
    auto pdlMod = parseSourceString(test, context);
    std::vector<pdl::PatternOp> patterns;
    pdlMod->walk([&](pdl::PatternOp op) {
      patterns.push_back(op);
      // llvm::errs() << "Found pattern:\n";
      // op.dump();
      return WalkResult::advance();
    });

    // 3. add the available PDL patterns to the transformer module
    auto transformMod = parseSourceString(transformInterpTemplate, context);
    transformMod->walk([&](transform::WithPDLPatternsOp op) {
      Block &block = *op.getBody().begin();
      auto localBuilder = OpBuilder::atBlockBegin(&block);
      for (auto &pdlOp : patterns)
        localBuilder.insert(pdlOp.clone());

      return WalkResult::advance();
    });

    transformMod->walk([&](transform::SequenceOp op) {
      Block &block = *op.getBody().begin();
      auto localBuilder = OpBuilder::atBlockBegin(&block);
      auto arg = block.getArgument(0);
      for (auto &pattern : patterns) {
        // llvm::errs() << "TESTING HERE\n";
        localBuilder.create<transform::PDLMatchOp>(
            location, transform::AnyOpType::get(context), arg,
            FlatSymbolRefAttr::get(context, pattern.getName()));
      }

      return WalkResult::advance();
    });

    // transformMod->dump();
    moduleOp.push_back(cast<ModuleOp>(*transformMod));

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

    // moduleOp.dump();
    // 4. parse the transformer moduleop into the new moduleop
    // 5. Run the transform interpreter pass
    // 6. Extract the optimized code and replace on the original module op
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuantumCircuitOptimizations() {
  return std::make_unique<QuantumCircuitOptimizationsPass>();
}
