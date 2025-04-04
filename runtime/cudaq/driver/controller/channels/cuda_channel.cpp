/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/controller/channel.h"

#include <cuda_runtime.h>

#include <dlfcn.h>

namespace cudaq::driver {

/// @brief The
class cuda_channel : public channel {
protected:
  int cudaDevice = 0;

  template <typename Applicator>
  auto runOnCorrectDevice(const Applicator &applicator)
      -> std::invoke_result_t<Applicator> {
    int dev;
    cudaGetDevice(&dev);
    if (cudaDevice == dev)
      return applicator();

    cudaSetDevice(cudaDevice);
    if constexpr (std::is_void_v<std::invoke_result_t<Applicator>>) {
      applicator();
      cudaSetDevice(dev);
      return;
    } else {
      auto val = applicator();
      cudaSetDevice(dev);
      return val;
    }
  }

public:
  using channel::channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaDevice = config.Devices[assignedID].Config.CudaDevice.value_or(0);
    cudaq::info("cuda channel connected (GPU Dev = {}).", cudaDevice);
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    cudaq::info("cuda channel (device {}) allocating data of size {}.", devId,
                size);
    return runOnCorrectDevice([&]() -> device_ptr {
      void *ptr = nullptr;
      cudaMalloc(&ptr, size);
      cudaMemset(ptr, 0, size);
      return {ptr, size, devId};
    });
  }

  void free(device_ptr &d) override {
    cudaq::info("cuda channel freeing data.");
    runOnCorrectDevice([&]() { cudaFree(d.data); });
  }

  void free(std::size_t argsHandle) override {}

  void memcpy(device_ptr &arg, const void *src) override {
    cudaq::info("cuda channel copying data to GPU.");
    runOnCorrectDevice(
        [&]() { cudaMemcpy(arg.data, src, arg.size, cudaMemcpyHostToDevice); });
  }

  void memcpy(void *dst, device_ptr &src) override {
    cudaq::info("cuda channel copying data from GPU.");
    runOnCorrectDevice(
        [&]() { cudaMemcpy(dst, src.data, src.size, cudaMemcpyDeviceToHost); });
  }

  // launching kernels
  // https://www.perplexity.ai/search/i-have-the-symbol-name-for-my-lV9vIec5Rn.Z1EV7BDItAQ

  launch_result launch_callback(const std::string &funcName,
                                device_ptr& argsHandle)  override {
    // auto * h = dlopen("/workspaces/cuda-quantum/build/add.o", RTLD_GLOBAL);
    // auto * ptr = dlsym(h, funcName );
    return {};
  }
  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, cuda_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(cuda_channel)

} // namespace cudaq::driver
