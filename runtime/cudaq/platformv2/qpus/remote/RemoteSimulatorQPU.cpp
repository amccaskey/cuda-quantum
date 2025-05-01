/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRemoteSimulatorQPU.h"
#include "cudaq/Support/TargetConfig.h"

#include "helpers/MQPUUtils.h"

#include "common/FmtCore.h"

using namespace mlir;

namespace {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
private:
  cudaq::AutoLaunchRestServerProcess *ownedProcess = nullptr;

public:
  RemoteSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseRemoteSimulatorQPU(m) {
    m_mlirContext = cudaq::initializeMLIR();
  }

  RemoteSimulatorQPU(RemoteSimulatorQPU &&) = delete;
  virtual ~RemoteSimulatorQPU() {
    if (ownedProcess)
      delete ownedProcess;
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
public:
  NvcfSimulatorQPU(const cudaq::v2::platform_metadata &m)
      : BaseNvcfSimulatorQPU(m) {
    m_mlirContext = cudaq::initializeMLIR();
  }

  NvcfSimulatorQPU(NvcfSimulatorQPU &&) = delete;
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
            fmt::format("target;nvqc;simulator;{}", simName);
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
          const std::string configStr =
              fmt::format("url;{};simulator;{}", formatUrl(urls[qId]), simName);
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
