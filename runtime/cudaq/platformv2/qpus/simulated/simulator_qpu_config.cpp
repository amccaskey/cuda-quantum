/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../../config.h"
#include "../../qpu.h"

#include "cudaq/Support/TargetConfig.h"

#include "cudaq/utils/cudaq_utils.h"

#include "common/EigenDense.h"
#include "common/Logger.h"

#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

namespace cudaq::v2 {

class simulator_config : public config::platform_config {
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
  void configure_qpus(std::vector<std::unique_ptr<qpu_handle>> &platform_qpus,
                      const platform_metadata &metadata) override {
    const auto &options = metadata.target_options;
    const auto &cfg = metadata.target_config;

    // Did the user request MQPU?
    bool isMqpu = false;
    for (std::size_t idx = 0; idx < options.size();) {
      const auto argsStr = options[idx + 1];
      if (argsStr == "mqpu")
        isMqpu = true;
      idx += 2;
    }

    // If not, just add a single simulated qpu
    if (!isMqpu) {
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
      cudaq::info("Added 1 qpu to the platform {}", platform_qpus.size());
      return;
    }

    // If so, get the number of available GPUs.
    // If none print a warning
    auto numGpus = getCudaGetDeviceCount();
    if (numGpus == 0) {
      cudaq::warn("mqpu platform requested, but could not find any GPUs.");
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
      return;
    }

    cudaq::info("multi-qpu has been requested - numGPUs is {}", numGpus);

    // Add the available simulators for each GPU.
    for (int i = 0; i < numGpus; i++)
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(config::platform_config, simulator_config)
};
CUDAQ_REGISTER_EXTENSION_TYPE(simulator_config)

}