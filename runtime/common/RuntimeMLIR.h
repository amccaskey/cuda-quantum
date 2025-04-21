/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/extension_point.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <map>
#include <memory>

namespace mlir {
class MLIRContext;
class PassManager;
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

class mlir_compiler : public extension_point<mlir_compiler> {
protected:
  /// @brief A Loaded Module tracks the kernel thunk function name,
  /// the number of required qubits, the MLIR ExecutionEngine, and
  /// invoked callback functions.
  struct LoadedModule {
    std::string entryPointName = "";
    mlir::OwningOpRef<mlir::ModuleOp> m_module;
    std::unique_ptr<mlir::ExecutionEngine> jitEngine;
    std::vector<callback> callbacks;
  };

  std::vector<std::string> symbolLocations;
  std::unordered_map<std::size_t, LoadedModule> loaded_modules;
  std::unique_ptr<mlir::MLIRContext> context;
  virtual void lowerToLLVM(mlir::PassManager &pm,
                           mlir::OwningOpRef<mlir::ModuleOp> &) = 0;

public:
  virtual ~mlir_compiler() {}

  /// @brief Initialize the compiler, give it the target config
  virtual void initialize(const config::TargetConfig &);

  std::size_t load(const std::string &mlir);
  void compile(std::size_t moduleHandle);
  /// @brief There may be scenarios where the callback is non-local (distributed
  /// system of devices). Enable one to remove callbacks to avoid
  /// symbol-not-found JIT errors.
  void remove_callback(std::size_t moduleHandle, const std::string &funcName);

  /// @brief Return all callbacks required by the kernel/module
  /// at the given handle.
  std::vector<callback> get_callbacks(std::size_t moduleHandle);

  /// @brief Launch the kernel thunk, results are posted to the thunkArgs
  /// pointer
  virtual void launch(std::size_t moduleHandle, void *argsBuffer);
};

} // namespace cudaq
