/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "controller.h"

#include "cudaq/Support/TargetConfig.h"

#include "rpc/server.h"

namespace cudaq::driver {
std::vector<std::unique_ptr<channel>> communication_channels;

std::map<intptr_t, device_ptr> memory_pool;

void connect(const std::string &cfgStr) {
  // parse back to target yaml
  // Create the communication channels
  llvm::yaml::Input yin(cfgStr.c_str());
  config::TargetConfig config;
  yin >> config;

  // Configure the communication channels
  for (std::size_t id = 0; auto &device : config.Devices) {
    cudaq::info("controller adding classical connected device with name {}.",
                device.Name);
    communication_channels.emplace_back(channel::get(device.Config.Channel));
    communication_channels.back()->connect(id++, config);
  }
}

// Allocate memory on devId. Return a unique handle
std::size_t malloc(std::size_t size, std::size_t devId) {
  cudaq::info("controller malloc requested. size = {}, devId = {}", size,
              devId);
  device_ptr ptr;
  if (devId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("allocating data locally on controller.");
    // Malloc is on this memory space
    // allocate the data and return a handle
    ptr.data = std::malloc(size);
    ptr.size = size;
    ptr.deviceId = devId;
  } else {
    cudaq::info("forwarding malloc to device {}", devId);
    ptr = communication_channels[devId]->malloc(size, devId);
  }

  // Get a unique handle
  auto uniqueInt = reinterpret_cast<intptr_t>(ptr.data);

  // Store the data here
  memory_pool.insert({uniqueInt, ptr});

  cudaq::info("return unique handle to allocated data {}.", uniqueInt);
  // Return the handle
  return uniqueInt;
}

void free(std::size_t handle) {
  cudaq::info("controller deallocating device pointer with handle {}.", handle);
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("invalid data to free");

  if (iter->second.deviceId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("deallocating local controller data");
    std::free(iter->second.data);
    // FIXME delete the device_ptr
    memory_pool.erase(iter->first);
    return;
  }

  // FIXME validate the devId
  cudaq::info("forward deallocation to device {}", iter->second.deviceId);
  return communication_channels[iter->second.deviceId]->free(iter->second);
}

// memcpy from driver to host (hence the return)
std::vector<char> memcpy_from(std::size_t handle, std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid memcpy handle");

  device_ptr &dest = iter->second;
  std::vector<char> result(size);
  cudaq::info(
      "memcpy data with handle {} and size {} from {} to host.",
      handle, size,
      dest.deviceId == std::numeric_limits<std::size_t>::max()
          ? "driver"
          : "device " + std::to_string(dest.deviceId));

  if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(result.data(), dest.data, size);
  } else {
    communication_channels[dest.deviceId]->memcpy(result.data(), dest);
  }

  return result;
}

void memcpy_to(std::size_t handle, std::vector<char> &data, std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid memcpy handle");

  device_ptr &dest = iter->second;
  cudaq::info(
      "memcpy data with handle {} and size {} to {}.",
      handle, size,
      dest.deviceId == std::numeric_limits<std::size_t>::max()
          ? "driver"
          : "device " + std::to_string(dest.deviceId));
  if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
    // Local controller copy
    std::memcpy(dest.data, data.data(), size);
  } else {
    // Forward to device's communication channel
    communication_channels[dest.deviceId]->memcpy(dest, data.data());
  }
}

} // namespace cudaq::driver