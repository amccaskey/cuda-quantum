/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "controller.h"

#include "common/ThunkInterface.h"
#include "cudaq/Support/TargetConfig.h"

#include "rpc/server.h"

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::controller)

namespace cudaq::driver {

// The concrete controller.
std::unique_ptr<controller> m_controller;

void controller::connect(const std::string &cfgStr) {
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

  // FIXME add this to the config
  compiler = quake_compiler::get("default_compiler");
  compiler->initialize(config);
}

// Allocate memory on devId. Return a unique handle
std::size_t controller::malloc(std::size_t size, std::size_t devId) {
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

void controller::free(std::size_t handle) {
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
std::vector<char> controller::memcpy_from(std::size_t handle,
                                          std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid memcpy handle");

  device_ptr &dest = iter->second;
  std::vector<char> result(size);
  cudaq::info("memcpy data with handle {} and size {} from {} to host.", handle,
              size,
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

void controller::memcpy_to(std::size_t handle, std::vector<char> &data,
                           std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid memcpy handle");

  device_ptr &dest = iter->second;
  cudaq::info("memcpy data with handle {} and size {} to {}.", handle, size,
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

handle controller::load_kernel(const std::string &quake) {
  cudaq::info("Loading and JIT compiling the kernel!");
  return compiler->compile(quake);
}

// launch and return a handle to the result, -1 if void
std::vector<char> controller::launch_kernel(std::size_t kernelHandle,
                                            std::size_t argsHandle) {
  // Get the arguments from the memory pool
  auto iter = memory_pool.find(argsHandle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid args handle");
  auto *thunkArgs = iter->second.data;

  cudaq::info("Launching Kernel {}, args size {}", kernelHandle,
              iter->second.size);

  // Launch the kernel
  compiler->launch(kernelHandle, thunkArgs);

  // Return the result data
  std::vector<char> retRes(iter->second.size);
  std::memcpy(retRes.data(), thunkArgs, iter->second.size);
  return retRes;
}

launch_result controller::launch_callback(std::size_t devId,
                                          const std::string &funcName,
                                          std::size_t argsHandle) {
  // FIXME check devId is valid
  return communication_channels[devId]->launch_callback(funcName, argsHandle);
}

void initialize(const std::string &controllerType, int argc, char **argv) {
  m_controller = controller::get(controllerType);
  m_controller->initialize(argc, argv);
}

bool should_stop() { return m_controller->should_stop(); }
} // namespace cudaq::driver

extern "C" {

cudaq::KernelThunkResultType __nvqpp__callback_run(std::size_t deviceId,
                                                   const char *funcName,
                                                   void *args,
                                                   std::size_t argsSize) {
  using namespace cudaq::driver;

  // Tell the controller to allocate memory on deviceId
  auto argsHandle = m_controller->malloc(argsSize, deviceId);

  // Send the data to that device pointer across the channel
  std::vector<char> asVec(argsSize);
  std::memcpy(asVec.data(), args, argsSize);
  m_controller->memcpy_to(argsHandle, asVec, argsSize);

  // Launch the callback
  auto callbackResultHolder =
      m_controller->launch_callback(deviceId, funcName, argsHandle);

  // Get the result pointer and size
  auto *resPtr = callbackResultHolder.result.data;
  auto resSize = callbackResultHolder.result.size;

  // Return the result
  return {resPtr, resSize};
}
}