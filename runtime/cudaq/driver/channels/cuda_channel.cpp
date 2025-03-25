/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"

#include <cuda_runtime.h>

namespace cudaq::driver {

/// @brief The
class cuda_channel : public channel {
public:
  using channel::channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("shared_memory channel connected.");
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    cudaq::info("cuda channel (device {}) allocating data of size {}.", devId,
                size);
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
    cudaMemset(ptr, 0, size);
    return {ptr, size, devId};
  }

  void free(device_ptr &d) override {
    cudaq::info("cuda channel freeing data.");
    cudaFree(d.data);
  }

  void free(std::size_t argsHandle) override {}

  void memcpy(device_ptr &arg, const void *src) override {
    cudaq::info("cuda channel copying data to GPU.");
    cudaMemcpy(arg.data, src, arg.size, cudaMemcpyHostToDevice);
  }

  void memcpy(void *dst, device_ptr &src) override {
    cudaq::info("cuda channel copying data from GPU.");
    cudaMemcpy(dst, src.data, src.size, cudaMemcpyDeviceToHost);
  }
  // memcpy a logical grouping of data, return a handle on that (remote) data
  std::size_t memcpy(std::vector<device_ptr> &args,
                     std::vector<const void *> srcs) override {
    return 0;
  }

  error_code launch_callback(const std::string &funcName,
                             std::size_t argsHandle) const override {
    return 0;
  }

  std::size_t register_compiled(const std::string &quake) const override {
    return 0;
  }

  error_code launch_kernel(std::size_t kernelHandle,
                           std::size_t argsHandle) const override {
    return 0;
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, cuda_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(cuda_channel)

} // namespace cudaq::driver
