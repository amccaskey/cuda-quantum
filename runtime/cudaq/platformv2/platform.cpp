/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "platform.h"

#include "common/Logger.h"

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/utils/cudaq_utils.h"

#include "llvm/Support/Base64.h"

#include <atomic>
#include <filesystem>
#include <fstream>

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::v2::config::platform_config)

INSTANTIATE_REGISTRY(cudaq::v2::qpu_handle,
                     const cudaq::v2::platform_metadata &)

namespace cudaq::v2 {

static std::vector<std::unique_ptr<qpu_handle>> platform_qpus;
static qpu_handle *manual_override_qpu;
static std::unordered_map<std::size_t, std::size_t> current_qpu_idx_for_thread;
std::size_t qpu_handle::uid_counter = 0;

void override_current_qpu(qpu_handle *q) { manual_override_qpu = q; }
void reset_override_qpu() { manual_override_qpu = nullptr; }
namespace config {

std::vector<std::string> get_options(const std::string &encoded) {

  llvm::SmallVector<llvm::StringRef> args;
  std::string targetArgsString = encoded;
  if (targetArgsString.starts_with("base64_")) {
    if (targetArgsString.size() > 7) {
      auto targetArgsStr = targetArgsString.substr(7);
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      targetArgsString = decoded;
    } else {
      targetArgsString = "";
    }
  }
  llvm::StringRef(targetArgsString).split(args, ' ', -1, false);
  std::vector<std::string> targetArgv;
  for (const auto &arg : args) {
    std::string targetArgsStr = arg.str();
    if (targetArgsStr.starts_with("base64_")) {
      targetArgsStr.erase(0, 7); // erase "base64_"
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      cudaq::info("Decoded '{}' from '{}'", decoded, targetArgsStr);
      targetArgsStr = decoded;
    }
    targetArgv.emplace_back(targetArgsStr);
  }
  return targetArgv;
}
} // namespace config

void initialize(const std::string &targetConfigName,
                const std::string &encodedOptions) {

  cudaq::info("initializing the platform - {}, {}.", targetConfigName,
              encodedOptions);

  // Get the options (--target-option OPTION pairs)
  auto options = config::get_options(encodedOptions);

  // Get backend option config map
  std::map<std::string, std::string> configMap;
  auto mutableBackend = targetConfigName;
  if (mutableBackend.find(";") != std::string::npos) {
    auto keyVals = cudaq::split(mutableBackend, ';');
    mutableBackend = keyVals[0];
    for (std::size_t i = 1; i < keyVals.size(); i += 2)
      configMap.insert({keyVals[i], keyVals[i + 1]});
  }

  // Could be that the semicolon separate string
  // has option;fp64, find that here
  bool isFp64 = [&]() {
    auto iter = configMap.find("option");
    return iter != configMap.end() && iter->second == "fp64";
  }();

  if (isFp64) {
    options.push_back("--target-option");
    options.push_back("fp64");
  }

  // Could be that the semicolon separate string
  // has option;mqpu, find that here
  bool isMqpu = [&]() {
    auto iter = configMap.find("option");
    return iter != configMap.end() && iter->second == "mqpu";
  }();

  if (isMqpu) {
    options.push_back("--target-option");
    options.push_back("mqpu");
  }

  // Load the target YAML file
  cudaq::config::TargetConfig config;
  {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string fileName = mutableBackend + std::string(".yml");
    auto configFilePath = platformPath / fileName;
    cudaq::info("Config file path = {}", configFilePath.string());
    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;
  }

  // FIXME Now we want to analyze runtime arguments / user requests,
  // and the target config, in order to populate the platform_qpus.

  // some extension point that targets can contribute for configuration
  // of the platform_qpus vector?

  // User Override - short circuit to specific QPU if --target-option qpu:NAME
  // is provided
  for (std::size_t idx = 0; idx < options.size();) {
    const auto argsStr = options[idx + 1];
    if (llvm::StringRef(argsStr).contains("qpu:")) {
      auto qpuType = cudaq::split(argsStr, ':')[1];
      cudaq::info("User specified the qpu type manually - {}", qpuType);
      platform_qpus.emplace_back(
          qpu_handle::get(qpuType, {config, options, targetConfigName}));
      return;
    }
    idx += 2;
  }

  // The taget YAML may provide the concrete qpu type, if so use it
  if (auto backendConfig = config.BackendConfig; backendConfig) {
    auto configConcerete = backendConfig.value();
    if (configConcerete.PlatformConfig.empty()) {
      // If we have both config and qpu, fall back on the config.
      auto qpuName = configConcerete.PlatformQpu;
      if (!qpuName.empty()) {
        cudaq::info("Target {} specified the qpu type as {}", mutableBackend,
                    qpuName);
        platform_qpus.emplace_back(
            qpu_handle::get(qpuName, {config, options, targetConfigName}));
        return;
      }
    }
  }

  std::string concreteConfig = "simulator_config";
  if (auto backendConfig = config.BackendConfig; backendConfig)
    if (!backendConfig.value().PlatformConfig.empty())
      concreteConfig = backendConfig.value().PlatformConfig;

  cudaq::info("Platform falling back on platform_config - {}", concreteConfig);

  // Always fall back on the platform_config
  auto platformConfig = config::platform_config::get(concreteConfig);
  platformConfig->configure_qpus(platform_qpus,
                                 {config, options, targetConfigName});
}

void reset_platform(const std::string &cfg) {
  platform_qpus.clear();
  qpu_handle::reset_uid_counter();
  initialize(cfg, "");
}
qpu_handle &get_qpu(std::size_t idx) {
  if (idx >= platform_qpus.size())
    throw std::runtime_error(fmt::format(
        "get_qpu(idx) error - invalid qpu index (num_qpus={}, idx={})",
        platform_qpus.size(), idx));
  return *platform_qpus[idx].get();
}
qpu_handle &get_qpu() {
  if (manual_override_qpu)
    return *manual_override_qpu;

  auto currentThread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::size_t idx = 0;
  auto iter = current_qpu_idx_for_thread.find(currentThread);
  if (iter != current_qpu_idx_for_thread.end())
    idx = iter->second;

  return *platform_qpus[idx].get();
}

std::size_t get_num_qpus() { return platform_qpus.size(); }

// Set the active qpu to the one at given index
void set_qpu(std::size_t idx) {
  std::mutex m;
  std::lock_guard<std::mutex> l(m);
  auto currentThread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  if (auto iter = current_qpu_idx_for_thread.find(currentThread);
      iter != current_qpu_idx_for_thread.end()) {
    iter->second = idx;
    return;
  }

  current_qpu_idx_for_thread.insert({currentThread, idx});
}

void set_qpu(const std::string &qpuName) {}

} // namespace cudaq::v2

// namespace cudaq {
extern "C" {

cudaq::KernelThunkResultType altLaunchKernel(const char *kernelName,
                                             cudaq::KernelThunkType kernelFunc,
                                             void *kernelArgs,
                                             std::uint64_t argsSize,
                                             std::uint64_t resultOffset) {
  ScopedTraceWithContext("altLaunchKernel", kernelName, argsSize);
  auto &qpu = cudaq::v2::get_qpu();
  // We have to have a qpu with the mlir launch trait
  if (auto launcher = qpu.as<cudaq::v2::mlir_launch_trait>()) {
    std::string kernName = kernelName;
    return launcher->launch_kernel(kernName, kernelFunc, kernelArgs, argsSize,
                                   resultOffset, {});
  }

  throw std::runtime_error(
      "Cannot launch a kernel from MLIR altLaunchKernel on a QPU that does not "
      "implement the mlir_launch_trait.");
  return {};
}

cudaq::KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs) {
  std::size_t argsSize = rawArgs.size();
  ScopedTraceWithContext("streamlinedLaunchKernel", kernelName, argsSize);
  auto &qpu = cudaq::v2::get_qpu();
  std::string kernName = kernelName;
  if (auto launcher = qpu.as<cudaq::v2::mlir_launch_trait>()) {
    launcher->launch_kernel(kernName, rawArgs);
  }
  // NB: The streamlined launch will never return results. Use alt or hybrid if
  // the kernel returns results.
  return {};
}

cudaq::KernelThunkResultType
hybridLaunchKernel(const char *kernelName, cudaq::KernelThunkType kernel,
                   void *args, std::uint64_t argsSize,
                   std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext("hybridLaunchKernel", kernelName);
  auto &qpu = cudaq::v2::get_qpu();
  const std::string kernName = kernelName;
  if (qpu.is_remote()) {
    // This path should never call a kernel that returns results.
    qpu.as<cudaq::v2::mlir_launch_trait>()->launch_kernel(kernName, rawArgs);
    return {};
  }

  return qpu.as<cudaq::v2::mlir_launch_trait>()->launch_kernel(
      kernName, kernel, args, argsSize, resultOffset, rawArgs);
}
// }
}
