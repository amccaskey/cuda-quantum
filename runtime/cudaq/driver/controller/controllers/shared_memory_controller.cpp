/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/driver/controller/controller.h"
namespace cudaq::driver {
class shared_memory_controller : public controller {
public:
  void initialize(int argc, char **argv) override {
    cudaq::info("Initialize Shared Memory Controller");
  }

  // memcpy from driver to host (hence the return)
  std::vector<char> memcpy_from(handle handle, std::size_t size) override {
    auto iter = allocated_device_ptrs.find(handle);
    if (iter == allocated_device_ptrs.end())
      throw std::runtime_error("Invalid memcpy handle: " +
                               std::to_string(handle));

    device_ptr &dest = iter->second;
    char *data = reinterpret_cast<char *>(memory_pool[handle]);
    cudaq::info("memcpy data with handle {} and size {} from {} to host.",
                handle, size,
                dest.deviceId == std::numeric_limits<std::size_t>::max()
                    ? "driver"
                    : "device " + std::to_string(dest.deviceId));

    if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
      std::vector<char> result(data, data + size);
      return result;
    }

    std::vector<char> result(size);
    communication_channels[dest.deviceId]->memcpy(result.data(), dest);
    return result;
  }

  bool should_stop() override { return false; }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(controller, shared_memory_controller);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory_controller)

} // namespace cudaq::driver
