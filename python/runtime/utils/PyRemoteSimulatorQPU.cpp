/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteSimulatorQPU.h"
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

namespace {

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU and PyNvcfSimulatorQPU.
static void launchVqeImpl(cudaq::ExecutionContext *executionContextPtr,
                          std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
                          const std::string &m_simName, const std::string &name,
                          const void *kernelArgs, cudaq::gradient *gradient,
                          const cudaq::spin_op &H, cudaq::optimizer &optimizer,
                          const int n_params, const std::size_t shots) {
  auto *wrapper = reinterpret_cast<const cudaq::ArgWrapper *>(kernelArgs);
  auto m_module = wrapper->mod;
  auto *mlirContext = m_module->getContext();

  if (executionContextPtr && executionContextPtr->name == "tracer")
    return;

  auto ctx = std::make_unique<cudaq::ExecutionContext>("observe", shots);
  ctx->kernelName = name;
  ctx->spin = cudaq::spin_op::canonicalize(H);
  if (shots > 0)
    ctx->shots = shots;

  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(
      *mlirContext, *executionContextPtr, /*serializedCodeContext=*/nullptr,
      gradient, &optimizer, n_params, m_simName, name, /*kernelFunc=*/nullptr,
      wrapper->rawArgs, /*argSize=*/0, &errorMsg);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch VQE. Error: " + errorMsg);
}

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU and PyNvcfSimulatorQPU.
static void
launchKernelImpl(cudaq::ExecutionContext *executionContextPtr,
                 std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
                 const std::string &m_simName, const std::string &name,
                 void (*kernelFunc)(void *), void *args,
                 std::uint64_t voidStarSize, std::uint64_t resultOffset,
                 const std::vector<void *> &rawArgs) {
  auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(args);
  auto m_module = wrapper->mod;
  auto callableNames = wrapper->callableNames;

  auto *mlirContext = m_module->getContext();

  // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
  // was set before launching the kernel. Use a static variable per thread to
  // set up a single-shot execution context for this case.
  static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                             /*shots=*/1);
  cudaq::ExecutionContext &executionContext =
      executionContextPtr ? *executionContextPtr : defaultContext;
  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(
      *mlirContext, executionContext, /*serializedCodeContext=*/nullptr,
      /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
      m_simName, name, kernelFunc, wrapper->rawArgs, voidStarSize, &errorMsg);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
}

static void launchKernelStreamlineImpl(
    cudaq::ExecutionContext *executionContextPtr,
    std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
    const std::string &m_simName, const std::string &name,
    const std::vector<void *> &rawArgs) {
  if (rawArgs.empty())
    throw std::runtime_error(
        "Streamlined kernel launch: arguments cannot "
        "be empty. The first argument should be a pointer to the MLIR "
        "ModuleOp.");

  auto *moduleOpPtr = reinterpret_cast<mlir::ModuleOp *>(rawArgs[0]);
  auto m_module = *moduleOpPtr;
  auto *mlirContext = m_module->getContext();

  // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
  // was set before launching the kernel. Use a static variable per thread to
  // set up a single-shot execution context for this case.
  static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                             /*shots=*/1);
  cudaq::ExecutionContext &executionContext =
      executionContextPtr ? *executionContextPtr : defaultContext;
  std::string errorMsg;
  auto actualArgs = rawArgs;
  // Remove the first argument (the MLIR ModuleOp) from the list of arguments.
  actualArgs.erase(actualArgs.begin());

  const bool requestOkay = m_client->sendRequest(
      *mlirContext, executionContext, /*serializedCodeContext=*/nullptr,
      /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
      m_simName, name, nullptr, nullptr, 0, &errorMsg, &actualArgs);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
}

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class PyRemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
public:
  PyRemoteSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseRemoteSimulatorQPU(m) {}

  bool is_emulator() const override { return true; }

  void launch_vqe(const std::string &name, const void *kernelArgs,
                  cudaq::gradient *gradient, const cudaq::spin_op &H,
                  cudaq::optimizer &optimizer, const int n_params,
                  const std::size_t shots) override {
    cudaq::info(
        "PyRemoteSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_uid, m_simName);
    ::launchVqeImpl(getExecutionContextForMyThread(), m_client, m_simName, name,
                    kernelArgs, gradient, H, optimizer, n_params, shots);
  }

  cudaq::KernelThunkResultType
  launch_kernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
                void *args, std::uint64_t voidStarSize,
                std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) override {
    cudaq::info("PyRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_uid, m_simName);
    ::launchKernelImpl(getExecutionContextForMyThread(), m_client, m_simName,
                       name, make_degenerate_kernel_type(kernelFunc), args,
                       voidStarSize, resultOffset, rawArgs);
    // TODO: Python should probably support return values too.
    return {};
  }

  void launch_kernel(const std::string &name,
                     const std::vector<void *> &rawArgs) override {
    cudaq::info("PyRemoteSimulatorQPU: Streamline launch kernel named '{}' "
                "remote QPU {} "
                "(simulator = {})",
                name, qpu_uid, m_simName);
    ::launchKernelStreamlineImpl(getExecutionContextForMyThread(), m_client,
                                 m_simName, name, rawArgs);
  }

  PyRemoteSimulatorQPU(PyRemoteSimulatorQPU &&) = delete;
  virtual ~PyRemoteSimulatorQPU() = default;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      PyRemoteSimulatorQPU, "PyRemoteSimulatorQPU",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<PyRemoteSimulatorQPU>(m);
      })
};

CUDAQ_REGISTER_EXTENSION_TYPE(PyRemoteSimulatorQPU)

/// Implementation of QPU subtype that submits simulation request to NVCF.
/// NOTE: This class duplicates the `isEmulated` and `launchKernel` methods from
/// `PyRemoteSimulatorQPU` class above; tried using multiple inheritance, but,
/// got errors from the functionality to register type.
class PyNvcfSimulatorQPU : public cudaq::BaseNvcfSimulatorQPU {
public:
  PyNvcfSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseNvcfSimulatorQPU(m) {}

  bool is_emulator() const override { return true; }

  void launch_vqe(const std::string &name, const void *kernelArgs,
                  cudaq::gradient *gradient, const cudaq::spin_op &H,
                  cudaq::optimizer &optimizer, const int n_params,
                  const std::size_t shots) override {
    cudaq::info(
        "PyNvcfSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_uid, m_simName);
    ::launchVqeImpl(getExecutionContextForMyThread(), m_client, m_simName, name,
                    kernelArgs, gradient, H, optimizer, n_params, shots);
  }

  cudaq::KernelThunkResultType
  launch_kernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
                void *args, std::uint64_t voidStarSize,
                std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) override {
    cudaq::info("PyNvcfSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_uid, m_simName);
    ::launchKernelImpl(getExecutionContextForMyThread(), m_client, m_simName,
                       name, make_degenerate_kernel_type(kernelFunc), args,
                       voidStarSize, resultOffset, rawArgs);
    // TODO: Python should probably support return values too.
    return {};
  }

  void launch_kernel(const std::string &name,
                     const std::vector<void *> &rawArgs) override {
    cudaq::info("PyNvcfSimulatorQPU: Streamline launch kernel named '{}' "
                "remote QPU {} "
                "(simulator = {})",
                name, qpu_uid, m_simName);
    ::launchKernelStreamlineImpl(getExecutionContextForMyThread(), m_client,
                                 m_simName, name, rawArgs);
  }

  PyNvcfSimulatorQPU(PyNvcfSimulatorQPU &&) = delete;
  virtual ~PyNvcfSimulatorQPU() = default;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      PyNvcfSimulatorQPU, "PyNvcfSimulatorQPU",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<PyNvcfSimulatorQPU>(m);
      })
};
CUDAQ_REGISTER_EXTENSION_TYPE(PyNvcfSimulatorQPU)

} // namespace
