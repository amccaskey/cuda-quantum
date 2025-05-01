/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteRESTQPU.h"

using namespace mlir;
extern "C" void __cudaq_deviceCodeHolderAdd(const char *, const char *);

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
class RemoteRESTQPU : public cudaq::BaseRemoteRESTQPU {
protected:
  bool isPython = false;
  std::tuple<ModuleOp, MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    auto contextPtr =
        isPython ? cudaq::initializeMLIRPython() : cudaq::initializeMLIR();
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
  RemoteRESTQPU(const cudaq::v2::platform_metadata &m) : BaseRemoteRESTQPU(m) {
    isPython = m.initial_config_str.find("is_python;true") != std::string::npos;
    setTargetBackend(m.initial_config_str);
    if (is_emulator()) {
      cudaq::info("RemoteRESTQPU is in emulation mode, setting the simulator.");
      emulatorQPU = cudaq::v2::qpu_handle::get("circuit_simulator", m);
    }
  }

  RemoteRESTQPU(RemoteRESTQPU &&) = delete;
  virtual ~RemoteRESTQPU() = default;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      RemoteRESTQPU, "remote_rest",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<RemoteRESTQPU>(m);
      })
};

CUDAQ_REGISTER_EXTENSION_TYPE(RemoteRESTQPU)

} // namespace
