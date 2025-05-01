/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "OrcaExecutor.h"
#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/platformv2/platform.h"
#include "cudaq/utils/cudaq_utils.h"
#include "orca_qpu.h"

namespace cudaq {

/// @brief The OrcaRemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service.
/// Moreover, this QPU handles launching kernels under the Execution Context
/// that includes sampling via synchronous client invocations.
class OrcaRemoteRESTQPU : public cudaq::v2::qpu<v2::mlir_launch_trait> {
protected:
  /// @brief The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Flag indicating whether we should emulate
  /// execution locally.
  bool emulate = false;

  /// @brief Pointer to the concrete Executor for this QPU
  std::unique_ptr<OrcaExecutor> executor;

  /// @brief Pointer to the concrete ServerHelper, provides
  /// specific JSON payloads and POST/GET URL paths.
  std::unique_ptr<ServerHelper> serverHelper;

  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Mapping of thread and execution context
  std::unordered_map<std::size_t, cudaq::ExecutionContext *> contexts;

private:
  /// @brief RestClient used for HTTP requests.
  RestClient client;

public:
  /// @brief The constructor
  OrcaRemoteRESTQPU(const v2::platform_metadata &m) : qpu(m) {
    std::filesystem::path cudaqLibPath{getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<OrcaExecutor>();
    setTargetBackend(m.initial_config_str);
  }

  OrcaRemoteRESTQPU(OrcaRemoteRESTQPU &&) = delete;

  /// @brief The destructor
  virtual ~OrcaRemoteRESTQPU() = default;

  /// @brief Return true if the current backend is a simulator
  bool is_simulator() const override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supports_conditional_feedback() const override { return false; }

  /// @brief Return true if the current backend supports explicit measurements
  bool supports_explicit_measurements() const override { return false; }

  /// @brief Return true if the current backend is remote
  virtual bool is_remote() const override { return !emulate; }

  /// @brief Store the execution context for launching kernel
  void set_execution_context(cudaq::ExecutionContext *context) override {
    cudaq::info("OrcaRemoteRESTQPU::set_execution_context QPU {}", qpu_uid);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts.emplace(tid, context);
    // cudaq::getExecutionManager()->set_execution_context(contexts[tid]);
  }

  /// @brief Overrides reset_execution_context
  void reset_execution_context() override {
    cudaq::info("OrcaRemoteRESTQPU::reset_execution_context QPU {}", qpu_uid);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts[tid] = nullptr;
    contexts.erase(tid);
  }

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file.
  void setTargetBackend(const std::string &backend);

  /// @brief Launch the kernel. Handle all pertinent modifications for the
  /// execution context.
  KernelThunkResultType
  launch_kernel(const std::string &kernelName, KernelThunkType kernelFunc,
                void *args, std::uint64_t voidStarSize,
                std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) override;
  void launch_kernel(const std::string &kernelName,
                     const std::vector<void *> &rawArgs) override {
    throw std::runtime_error("launch kernel on raw args not implemented");
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      OrcaRemoteRESTQPU, "orca",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<OrcaRemoteRESTQPU>(m);
      })
};
} // namespace cudaq
