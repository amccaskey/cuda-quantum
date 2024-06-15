/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/BaseRemoteRESTQPU.h"

using namespace mlir;

namespace cudaq {
std::string get_quake_by_name(const std::string &);
} // namespace cudaq

namespace {

/// @brief The `RemoteRESTQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service. Specific
/// details about JSON payloads are abstracted via an abstract type called
/// ServerHelper, which is meant to be subtyped by each provided remote QPU
/// service. Moreover, this QPU handles launching kernels under a number of
/// Execution Contexts, including sampling and observation via synchronous or
/// asynchronous client invocations. This type should enable both QIR-based
/// backends as well as those that take OpenQASM2 as input.
class RemoteRESTQPU : public cudaq::BaseRemoteRESTQPU<RemoteRESTQPU> {
protected:
  std::tuple<ModuleOp, MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    auto contextPtr = cudaq::initializeMLIR();
    MLIRContext &context = *contextPtr.get();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, &context);
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    return std::make_tuple(m_module.release(), contextPtr.release(), data);
  }

  void cleanupContext(MLIRContext *context) override { delete context; }

public:
  /// @brief The constructor
  RemoteRESTQPU() : BaseRemoteRESTQPU() {}

  RemoteRESTQPU(RemoteRESTQPU &&) = delete;
  virtual ~RemoteRESTQPU() = default;

  static std::string GetRemoteRESTQPUName() { return "remote_rest"; }
  static std::unique_ptr<cudaq::QPU> create() {
    return std::make_unique<RemoteRESTQPU>();
  }
};

template <>
const bool cudaq::BaseRemoteRESTQPU<RemoteRESTQPU>::registered_ =
    cudaq::BaseRemoteRESTQPU<RemoteRESTQPU>::register_type();

} // namespace

// CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteRESTQPU, remote_rest)