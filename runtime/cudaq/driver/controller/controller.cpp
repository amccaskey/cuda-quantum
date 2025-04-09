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
std::unique_ptr<controller, void (*)(controller *)>
    m_controller(nullptr, [](controller *) {});

void set_controller_caller_retains_ownership(controller *c) {
  m_controller = std::unique_ptr<controller, void (*)(controller *)>(
      c, [](controller *cc) {});
}

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
    communication_channels.emplace_back(
        device_channel::get(device.Config.Channel));
    communication_channels.back()->connect(id++, config);
  }

  // FIXME add this to the config
  compiler = quake_compiler::get("default_compiler");
  compiler->initialize(config);
}

// Allocate memory on devId. Return a unique handle
handle controller::malloc(std::size_t size, std::size_t devId) {
  cudaq::info("controller malloc requested. size = {}, devId = {}", size,
              devId);
  device_ptr ptr;
  if (devId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("allocating data locally on controller.");
    // Malloc is on this memory space
    // allocate the data and return a handle
    auto *raw = std::malloc(size);
    ptr.handle = reinterpret_cast<uintptr_t>(raw);
    ptr.size = size;
    ptr.deviceId = devId;

    // Store the data here
    memory_pool.insert({ptr.handle, raw});
  } else {
    cudaq::info("forwarding malloc to device {}", devId);
    ptr = communication_channels[devId]->malloc(size, devId);
  }

  allocated_device_ptrs.insert({ptr.handle, ptr});

  cudaq::info("return unique handle to allocated data {}.", ptr.handle);
  // Return the handle
  return ptr.handle;
}

void controller::free(handle handle) {
  cudaq::info("controller deallocating device pointer with handle {}.", handle);
  auto iter = allocated_device_ptrs.find(handle);
  if (iter == allocated_device_ptrs.end())
    throw std::runtime_error("invalid data to free");

  if (iter->second.deviceId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("deallocating local controller data");
    std::free(memory_pool.at(iter->first));
    // FIXME delete the device_ptr
    memory_pool.erase(iter->first);
    return;
  }

  // FIXME validate the devId
  cudaq::info("forward deallocation to device {}", iter->second.deviceId);
  return communication_channels[iter->second.deviceId]->free(iter->second);
}

// memcpy from driver to host (hence the return)
std::vector<char> controller::memcpy_from(handle handle, std::size_t size) {
  auto iter = allocated_device_ptrs.find(handle);
  if (iter == allocated_device_ptrs.end())
    throw std::runtime_error("Invalid memcpy handle: " +
                             std::to_string(handle));

  device_ptr &dest = iter->second;
  std::vector<char> result(size);
  cudaq::info("memcpy data with handle {} and size {} from {} to host.", handle,
              size,
              dest.deviceId == std::numeric_limits<std::size_t>::max()
                  ? "driver"
                  : "device " + std::to_string(dest.deviceId));

  if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(result.data(), memory_pool[handle], size);
  } else {
    communication_channels[dest.deviceId]->memcpy(result.data(), dest);
  }

  return result;
}

void controller::memcpy_to(handle handle, std::vector<char> &data,
                           std::size_t size) {
  auto iter = allocated_device_ptrs.find(handle);
  if (iter == allocated_device_ptrs.end())
    throw std::runtime_error("Invalid memcpy handle: " +
                             std::to_string(handle));

  device_ptr &dest = iter->second;
  cudaq::info("memcpy data with handle {} and size {} to {}.", handle, size,
              dest.deviceId == std::numeric_limits<std::size_t>::max()
                  ? "driver"
                  : "device " + std::to_string(dest.deviceId));
  if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
    // Local controller copy
    std::memcpy(memory_pool[handle], data.data(), size);
  } else {
    // Forward to device's communication channel
    communication_channels[dest.deviceId]->memcpy(dest, data.data());
  }
}

handle controller::load_kernel(const std::string &quake) {
  cudaq::info("Loading and JIT compiling the kernel!");
  return compiler->compile(quake);
}

std::vector<std::string> controller::get_callbacks(handle hdl) {
  cudaq::info("get_callbacks() requested from controller for kernel {}", hdl);

  auto cbs = compiler->get_callbacks(hdl);
  std::vector<std::string> ret;
  for (auto &c : cbs)
    ret.push_back(c.callbackName);
  return ret;
}

// launch and return a handle to the result, -1 if void
std::vector<char> controller::launch_kernel(handle kernelHandle,
                                            std::size_t argsHandle) {
  // Get the arguments from the memory pool
  auto iter = memory_pool.find(argsHandle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid args handle");
  auto *thunkArgs = iter->second;
  auto size = allocated_device_ptrs[argsHandle].size;

  cudaq::info("Launching Kernel {}, args size {}", kernelHandle, size);

  // Get this kernel's callbacks
  auto callbacks = compiler->get_callbacks(kernelHandle);

  // Make callback code available to devices
  for (auto &callback : callbacks)
    for (auto &channel : communication_channels)
      channel->load_callback(callback.callbackName,
                             callback.unmarshalFuncOpCode);

  // Launch the kernel
  compiler->launch(kernelHandle, thunkArgs);

  // Return the result data
  std::vector<char> retRes(size);
  std::memcpy(retRes.data(), thunkArgs, size);
  return retRes;
}

launch_result controller::launch_callback(std::size_t devId,
                                          const std::string &funcName,
                                          std::size_t argsHandle) {
  // FIXME check devId is valid
  auto iter = allocated_device_ptrs.find(argsHandle);
  if (iter == allocated_device_ptrs.end())
    throw std::runtime_error("Invalid args handle");

  return communication_channels[devId]->launch_callback(funcName, iter->second);
}

void initialize(const std::string &controllerType, int argc, char **argv) {
  m_controller = std::unique_ptr<controller, void (*)(controller *)>(
      controller::get(controllerType).release(),
      [](controller *c) { delete c; });
  m_controller->initialize(argc, argv);
}

bool should_stop() { return m_controller->should_stop(); }
} // namespace cudaq::driver

extern "C" {

// This is the launch proxy when calling a device function from a QPU
// kernel. The marshaling code will call this function. This function will then
// call the desired callback function on the host side. The argsBuffer uses the
// same pointer-free encoding as altLaunchKernel.
cudaq::KernelThunkResultType
__nvqpp__device_callback_run(std::uint64_t deviceId, const char *funcName,
                             void *unmarshalFunc, void *argsBuffer,
                             std::uint64_t argsBufferSize,
                             std::uint64_t returnOffset) {
  using namespace cudaq::driver;
  cudaq::info("classical callback func={} args_size={}", std::string(funcName),
              argsBufferSize);
  // Tell the controller to allocate memory on deviceId
  auto argsHandle = m_controller->malloc(argsBufferSize, deviceId);

  // Send the data to that device pointer across the channel
  std::vector<char> asVec(static_cast<char *>(argsBuffer),
                          static_cast<char *>(argsBuffer) + argsBufferSize);
  m_controller->memcpy_to(argsHandle, asVec, argsBufferSize);

  // Launch the callback
  auto [resPtr, error] =
      m_controller->launch_callback(deviceId, funcName, argsHandle);

  // handle maybe error
  std::memcpy(argsBuffer, resPtr.data(), argsBufferSize);
  return {};
}
void *__nvqpp__callback_get_raw_ptr(cudaq::device_ptr *p) {
  printf("we are here22 %lu %lu %lu\n", p->handle, p->deviceId, p->size);
  return nullptr;
}
}
