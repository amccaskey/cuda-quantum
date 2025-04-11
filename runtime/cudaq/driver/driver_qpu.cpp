/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/driver.h"
#include "cudaq/platform/qpu.h"

#include <filesystem>
#include <fstream>

namespace cudaq {
class DriverQPU : public QPU {
protected:
public:
  DriverQPU() = default;

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {

    // Get the code
    auto quake = cudaq::get_quake_by_name(name);

    // Target-specific compilation
    auto kernelHandle = driver::load_kernel(quake);

    // Allocate the arguments on the QPU channel
    auto argsDevPtr = driver::malloc(argsSize);
    driver::memcpy(argsDevPtr, args);

    // Launch the kernel
    auto res = driver::launch_kernel(kernelHandle, argsDevPtr);
    driver::free(argsDevPtr);

    if (res.error)
      throw std::runtime_error("Error was encountered in launch_kernel: " +
                               res.error.value());
    if (!res.data.empty())
      std::memcpy(args, res.data.data(), argsSize);
    return {};
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {}

  void resetExecutionContext() override {}

  void enqueue(QuantumTask &task) override {}

  void setTargetBackend(const std::string &backend) override {

    // here we will know the backend.
    // we can get the yaml, and then initialize the driver
    cudaq::info("[driver_qpu] backend string is {}", backend);
    std::map<std::string, std::string> configMap;
    auto mutableBackend = backend;
    if (mutableBackend.find(";") != std::string::npos) {
      auto keyVals = cudaq::split(mutableBackend, ';');
      mutableBackend = keyVals[0];
      for (std::size_t i = 1; i < keyVals.size(); i += 2)
        configMap.insert({keyVals[i], keyVals[i + 1]});
    }

    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string fileName = mutableBackend + std::string(".yml");

    /// Once we know the backend, we should search for the config file
    /// from there we can get the URL/PORT and the required MLIR pass pipeline.
    auto configFilePath = platformPath / fileName;
    cudaq::info("Config file path = {}", configFilePath.string());

    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;

    driver::initialize(config);
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::DriverQPU, driver_qpu)
