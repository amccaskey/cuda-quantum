/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/extension_point.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <map>
#include <memory>

namespace mlir {
class MLIRContext;
class ExecutionEngine;
class ModuleOp;
} // namespace mlir

namespace llvm {
class Module;
}

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq {
/// @brief Function to lower MLIR to target
/// @param op MLIR operation
/// @param output Output stream
/// @param additionalPasses Additional passes to run at the end
/// @param printIR Print IR to `stderr`
/// @param printIntermediateMLIR Print IR in between each pass
/// @param printStats Print pass statistics
using TranslateFromMLIRFunction = std::function<mlir::LogicalResult(
    mlir::Operation *, llvm::raw_string_ostream &, const std::string &, bool,
    bool, bool)>;

/// @brief Initialize MLIR with CUDA-Q dialects and return the
/// MLIRContext.
std::unique_ptr<mlir::MLIRContext> initializeMLIR();
/// @brief Given an LLVM Module, set its target triple corresponding to the
/// current host machine.
bool setupTargetTriple(llvm::Module *);

/// @brief Run the LLVM PassManager.
void optimizeLLVM(llvm::Module *);

/// @brief Lower ModuleOp to a full QIR LLVMIR representation
/// and return an ExecutionEngine pointer for JIT function pointer
/// execution. Clients are responsible for deleting this pointer.
mlir::ExecutionEngine *createQIRJITEngine(mlir::ModuleOp &moduleOp,
                                          llvm::StringRef convertTo);

class Translation {
public:
  Translation() = default;
  Translation(TranslateFromMLIRFunction function, llvm::StringRef description)
      : function(std::move(function)), description(description) {}

  /// Return the description of this translation.
  llvm::StringRef getDescription() const { return description; }

  /// Invoke the translation function with the given input and output streams.
  mlir::LogicalResult operator()(mlir::Operation *op,
                                 llvm::raw_string_ostream &output,
                                 const std::string &additionalPasses,
                                 bool printIR, bool printIntermediateMLIR,
                                 bool printStats) const {
    return function(op, output, additionalPasses, printIR,
                    printIntermediateMLIR, printStats);
  }

private:
  /// The underlying translation function.
  TranslateFromMLIRFunction function;

  /// The description of the translation.
  llvm::StringRef description;
};

cudaq::Translation &getTranslation(llvm::StringRef name);

struct TranslateFromMLIRRegistration {
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const cudaq::TranslateFromMLIRFunction &function);
};

// A callback is a simple struct to hold the name
// of a classical callback in a kernel, and the
// MLIR FuncOp code for it.
struct callback {
  std::string callbackName;
  std::string unmarshalFuncOpCode;
};

// The quake_compiler is an extension point for compiling
// both Quake kernel code and required callback unmarshal
// functions to executable object code
class quake_compiler : public extension_point<quake_compiler> {
public:
  virtual ~quake_compiler() {}

  /// @brief Initialize the compiler, give it the target config
  virtual void initialize(const config::TargetConfig &,
                          const std::map<std::string, bool> extra = {}) = 0;

  /// @brief Compile the Quake code to executable code and
  /// return a handle to the compiled kernel
  virtual std::size_t
  compile(const std::string &quake,
          const std::vector<std::string> &symbolLocations) = 0;

  /// @brief Compile the MLIR code for the unmarshal function
  /// for a given classical callback, provide potential external
  /// shared library locations to locate callable symbols. Return
  /// a handle to the unmarshal functino
  virtual std::size_t
  compile_unmarshaler(const std::string &mlirCode,
                      const std::vector<std::string> &symbolLocations) = 0;

  /// @brief Return all callbacks required by the kernel/module
  /// at the given handle.
  virtual std::vector<callback> get_callbacks(std::size_t moduleHandle) = 0;

  /// @brief Launch the kernel thunk, results are posted to the thunkArgs
  /// pointer
  virtual void launch(std::size_t moduleHandle, void *thunkArgs) = 0;

  virtual std::optional<std::size_t>
  get_required_num_qubits(std::size_t hdl) = 0;
};

} // namespace cudaq
