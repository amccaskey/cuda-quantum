/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteSimulatorQPU.h"
#include "cudaq/Support/TargetConfig.h"

#include "helpers/MQPUUtils.h"

#include "common/FmtCore.h"

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
class RemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
private:
  bool isPython = false;
  cudaq::AutoLaunchRestServerProcess *ownedProcess = nullptr;

public:
  RemoteSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseRemoteSimulatorQPU(m) {
    isPython = m.initial_config_str.find("is_python;true") != std::string::npos;
    m_mlirContext =
        isPython ? cudaq::initializeMLIRPython() : cudaq::initializeMLIR();
  }
  bool is_emulator() const override { return true; }

  RemoteSimulatorQPU(RemoteSimulatorQPU &&) = delete;
  virtual ~RemoteSimulatorQPU() {
    if (ownedProcess)
      delete ownedProcess;
  }

  void launch_vqe(const std::string &name, const void *kernelArgs,
                  cudaq::gradient *gradient, const cudaq::spin_op &H,
                  cudaq::optimizer &optimizer, const int n_params,
                  const std::size_t shots) override {
    if (isPython) {
      cudaq::info(
          "PyRemoteSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
          "(simulator = {})",
          name, qpu_uid, m_simName);
      ::launchVqeImpl(getExecutionContextForMyThread(), m_client, m_simName,
                      name, kernelArgs, gradient, H, optimizer, n_params,
                      shots);
      return;
    }

    BaseRemoteSimulatorQPU::launch_vqe(name, kernelArgs, gradient, H, optimizer,
                                       n_params, shots);
  }

  cudaq::KernelThunkResultType
  launch_kernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
                void *args, std::uint64_t voidStarSize,
                std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) override {
    if (isPython) {
      cudaq::info(
          "PyRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
          "(simulator = {})",
          name, qpu_uid, m_simName);
      ::launchKernelImpl(getExecutionContextForMyThread(), m_client, m_simName,
                         name, make_degenerate_kernel_type(kernelFunc), args,
                         voidStarSize, resultOffset, rawArgs);
      // TODO: Python should probably support return values too.
      return {};
    }

    return BaseRemoteSimulatorQPU::launch_kernel(
        name, kernelFunc, args, voidStarSize, resultOffset, rawArgs);
  }

  void launch_kernel(const std::string &name,
                     const std::vector<void *> &rawArgs) override {
    if (isPython) {
      cudaq::info("PyRemoteSimulatorQPU: Streamline launch kernel named '{}' "
                  "remote QPU {} "
                  "(simulator = {})",
                  name, qpu_uid, m_simName);
      ::launchKernelStreamlineImpl(getExecutionContextForMyThread(), m_client,
                                   m_simName, name, rawArgs);
      return;
    }

    return BaseRemoteSimulatorQPU::launch_kernel(name, rawArgs);
  }

  void takeAutoLaunchOwnership(cudaq::AutoLaunchRestServerProcess *process) {
    ownedProcess = process;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      RemoteSimulatorQPU, "RemoteSimulatorQPU",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<RemoteSimulatorQPU>(m);
      })
};

CUDAQ_REGISTER_EXTENSION_TYPE(RemoteSimulatorQPU)

/// Implementation of QPU subtype that submits simulation request to NVCF.
class NvcfSimulatorQPU : public cudaq::BaseNvcfSimulatorQPU {
  bool isPython = false;

public:
  NvcfSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseNvcfSimulatorQPU(m) {
    isPython = m.initial_config_str.find("is_python;true") != std::string::npos;

    m_mlirContext =
        isPython ? cudaq::initializeMLIRPython() : cudaq::initializeMLIR();
  }

  NvcfSimulatorQPU(NvcfSimulatorQPU &&) = delete;

  void launch_vqe(const std::string &name, const void *kernelArgs,
                  cudaq::gradient *gradient, const cudaq::spin_op &H,
                  cudaq::optimizer &optimizer, const int n_params,
                  const std::size_t shots) override {
    if (isPython) {
      cudaq::info(
          "PyNvcfSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
          "(simulator = {})",
          name, qpu_uid, m_simName);
      ::launchVqeImpl(getExecutionContextForMyThread(), m_client, m_simName,
                      name, kernelArgs, gradient, H, optimizer, n_params,
                      shots);
      return;
    }

    return BaseNvcfSimulatorQPU::launch_vqe(name, kernelArgs, gradient, H,
                                            optimizer, n_params, shots);
  }

  cudaq::KernelThunkResultType
  launch_kernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
                void *args, std::uint64_t voidStarSize,
                std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) override {
    if (isPython) {
      cudaq::info("PyNvcfSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                  "(simulator = {})",
                  name, qpu_uid, m_simName);
      ::launchKernelImpl(getExecutionContextForMyThread(), m_client, m_simName,
                         name, make_degenerate_kernel_type(kernelFunc), args,
                         voidStarSize, resultOffset, rawArgs);
      // TODO: Python should probably support return values too.
      return {};
    }

    return BaseNvcfSimulatorQPU::launch_kernel(
        name, kernelFunc, args, voidStarSize, resultOffset, rawArgs);
  }

  void launch_kernel(const std::string &name,
                     const std::vector<void *> &rawArgs) override {
    if (isPython) {
      cudaq::info("PyNvcfSimulatorQPU: Streamline launch kernel named '{}' "
                  "remote QPU {} "
                  "(simulator = {})",
                  name, qpu_uid, m_simName);
      ::launchKernelStreamlineImpl(getExecutionContextForMyThread(), m_client,
                                   m_simName, name, rawArgs);
      return;
    }
    return BaseNvcfSimulatorQPU::launch_kernel(name, rawArgs);
  }

  virtual ~NvcfSimulatorQPU() = default;
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      NvcfSimulatorQPU, "NvcfSimulatorQPU",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<NvcfSimulatorQPU>(m);
      })
};

CUDAQ_REGISTER_EXTENSION_TYPE(NvcfSimulatorQPU)

class remote_sim_config : public cudaq::v2::config::platform_config {

private:
  int getCudaGetDeviceCount() {
#ifdef CUDAQ_ENABLE_CUDA
    int nDevices{0};
    const auto status = cudaGetDeviceCount(&nDevices);
    return status != cudaSuccess ? 0 : nDevices;
#else
    return 0;
#endif
  }

public:
  void configure_qpus(
      std::vector<std::unique_ptr<cudaq::v2::qpu_handle>> &platformQPUs,
      const cudaq::v2::platform_metadata &metadata) override {
    cudaq::info("Configuring the Remote Simulator QPU");
    const auto getOpt = [](const std::string &str,
                           const std::string &prefix) -> std::string {
      // Return the first key-value configuration option found in the format:
      // "<prefix>;<option>".
      // Note: This expects an exact match of the prefix and the option value is
      // the next one.
      auto splitParts = cudaq::split(str, ';');
      if (splitParts.empty())
        return "";
      for (std::size_t i = 0; i < splitParts.size() - 1; ++i) {
        if (splitParts[i] == prefix) {
          cudaq::debug(
              "Retrieved option '{}' for the key '{}' from input string '{}'",
              splitParts[i + 1], prefix, str);
          return splitParts[i + 1];
        }
      }
      return "";
    };
    auto description = metadata.initial_config_str;
    bool isPython =
        metadata.initial_config_str.find("is_python;true") != std::string::npos;

    auto backendConfig = metadata.target_config.BackendConfig;
    const auto qpuSubType =
        backendConfig.has_value() ? backendConfig.value().PlatformQpu : "";
    if (!qpuSubType.empty()) {
      const auto formatUrl = [](const std::string &url) -> std::string {
        auto formatted = url;
        // Default to http:// if none provided.
        if (!formatted.starts_with("http"))
          formatted = std::string("http://") + formatted;
        if (!formatted.empty() && formatted.back() != '/')
          formatted += '/';
        return formatted;
      };

      if (!cudaq::v2::qpu_handle::is_registered(qpuSubType))
        throw std::runtime_error(
            fmt::format("Unable to retrieve {} QPU implementation. Please "
                        "check your installation.",
                        qpuSubType));
      if (qpuSubType == "NvcfSimulatorQPU") {
        platformQPUs.clear();
        // threadToQpuId.clear();
        auto simName = getOpt(description, "backend");
        if (simName.empty())
          simName = "custatevec-fp32";
        std::string configStr =
            fmt::format("target;nvqc;simulator;{};is_python;{}", simName,
                        isPython ? "true" : "false");
        auto getOptAndSetConfig = [&](const std::string &key) {
          auto val = getOpt(description, key);
          if (!val.empty())
            configStr += fmt::format(";{};{}", key, val);
        };
        getOptAndSetConfig("api_key");
        getOptAndSetConfig("function_id");
        getOptAndSetConfig("version_id");

        auto numQpusStr = getOpt(description, "nqpus");
        int numQpus = numQpusStr.empty() ? 1 : std::stoi(numQpusStr);

        if (simName.find("nvidia-mqpu") != std::string::npos && numQpus > 1) {
          // If the backend simulator is an MQPU simulator (like nvidia-mqpu),
          // then use "nqpus" to determine the number of GPUs to request for the
          // backend. This allows us to seamlessly translate requests for MQPU
          // requests to the NVQC platform.
          configStr += fmt::format(";{};{}", "ngpus", numQpus);
          // Now change numQpus to 1 for the downstream code, which will make a
          // single NVQC QPU.
          numQpus = 1;
        } else {
          getOptAndSetConfig("ngpus");
        }

        if (numQpus < 1)
          throw std::invalid_argument("Number of QPUs must be greater than 0.");
        for (int qpuId = 0; qpuId < numQpus; ++qpuId) {
          // Populate the information and add the QPUs
          auto qpu = cudaq::v2::qpu_handle::get(
              "NvcfSimulatorQPU",
              {metadata.target_config, metadata.target_options, configStr});
          platformQPUs.emplace_back(std::move(qpu));
        }
        // platformNumQPUs = platformQPUs.size();
      } else if (qpuSubType == "orca") {
        auto urls = cudaq::split(getOpt(description, "url"), ',');
        platformQPUs.clear();
        // threadToQpuId.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          // Populate the information and add the QPUs
          platformQPUs.emplace_back(
              cudaq::v2::qpu_handle::get("orca", metadata));
        }
      } else {
        auto urls = cudaq::split(getOpt(description, "url"), ',');
        auto sims = cudaq::split(getOpt(description, "backend"), ',');
        // Default to qpp simulator if none provided.
        if (sims.empty())
          sims.emplace_back("qpp");
        // If no URL is provided, default to auto launching one server instance.
        const bool autoLaunch =
            description.find("auto_launch") != std::string::npos ||
            urls.empty();

        std::vector<std::unique_ptr<cudaq::AutoLaunchRestServerProcess>>
            m_remoteServers;
        if (autoLaunch) {
          urls.clear();
          const auto numInstanceStr = getOpt(description, "auto_launch");
          // Default to launching one instance if no other setting is available.
          const int numInstances =
              numInstanceStr.empty() ? 1 : std::stoi(numInstanceStr);
          cudaq::info("Auto launch {} REST servers", numInstances);
          for (int i = 0; i < numInstances; ++i) {
            m_remoteServers.emplace_back(
                std::make_unique<cudaq::AutoLaunchRestServerProcess>(i));
            urls.emplace_back(m_remoteServers.back()->getUrl());
          }
        }

        // List of simulator names must either be one or the same length as the
        // URL list. If one simulator name is provided, assuming that all the
        // URL should be using the same simulator.
        if (sims.size() > 1 && sims.size() != urls.size())
          throw std::runtime_error(fmt::format(
              "Invalid number of remote backend simulators provided: "
              "receiving {}, expecting {}.",
              sims.size(), urls.size()));
        // platformQPUs.clear();
        // threadToQpuId.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          const auto simName = sims.size() == 1 ? sims.front() : sims[qId];
          const std::string configStr = fmt::format(
              "url;{};simulator;{};is_python;{}", formatUrl(urls[qId]), simName,
              isPython ? "true" : "false");
          // Populate the information and add the QPUs
          auto qpu = cudaq::v2::qpu_handle::get(
              "RemoteSimulatorQPU",
              {metadata.target_config, metadata.target_options, configStr});

          if (autoLaunch)
            // We want to tie the lifetime of the auto launched qpud process
            // to the lifetime of the qpu.
            dynamic_cast<RemoteSimulatorQPU *>(qpu.get())
                ->takeAutoLaunchOwnership(m_remoteServers[qId].release());

          platformQPUs.emplace_back(std::move(qpu));
        }
        m_remoteServers.clear();
      }
    }
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(cudaq::v2::config::platform_config,
                                   remote_sim_config)
};
CUDAQ_REGISTER_EXTENSION_TYPE(remote_sim_config)
} // namespace
