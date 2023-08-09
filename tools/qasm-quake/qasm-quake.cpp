/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Parser/OpenQASM.h"

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/FileUtilities.h"

int main(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return -1;
  }

  // Create a source manager and tell it about this buffer, which is what the
  // parser will pick up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::MLIRContext context;
  context.loadDialect<quake::QuakeDialect, mlir::func::FuncDialect,
                      mlir::memref::MemRefDialect>();

  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  if (auto module = cudaq::qasm::parseFile(&context, sourceMgr)) {
    module->print(llvm::outs());
    return 0;
  }
  llvm::errs() << "Failed to parse: " << inputFilename << '\n';
  return -1;
}
