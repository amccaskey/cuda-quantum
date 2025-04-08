/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/driver/controller/target.h"
#include "cudaq/utils/extension_point.h"

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::driver {

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
protected:
  /// @brief The execution target.
  std::unique_ptr<target> m_target;

public:
  virtual ~quake_compiler() {}

  /// @brief Initialize the compiler, give it the target config
  virtual void initialize(const config::TargetConfig &) = 0;

  /// @brief Compile the Quake code to executable code and
  /// return a handle to the compiled kernel
  virtual handle compile(const std::string &quake) = 0;

  /// @brief Compile the MLIR code for the unmarshal function
  /// for a given classical callback, provide potential external
  /// shared library locations to locate callable symbols. Return
  /// a handle to the unmarshal functino
  virtual handle
  compile_unmarshaler(const std::string &mlirCode,
                      const std::vector<std::string> &symbolLocations) = 0;

  /// @brief Return all callbacks required by the kernel/module
  /// at the given handle.
  virtual std::vector<callback> get_callbacks(handle moduleHandle) = 0;

  /// @brief Launch the kernel thunk, results are posted to the thunkArgs
  /// pointer
  virtual void launch(handle moduleHandle, void *thunkArgs) = 0;
};

} // namespace cudaq::driver
