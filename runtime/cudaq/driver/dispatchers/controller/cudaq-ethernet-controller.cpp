/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "rpc/server.h"

#include "common/Logger.h"

#include "common/RuntimeMLIR.h"
#include "cudaq/Support/TargetConfig.h"

#include "llvm/Support/CommandLine.h"

#include "cudaq/driver/channel.h"
#include "cudaq/driver/device_ptr.h"
#include "cudaq/driver/target.h"

#include <thread>

static std::atomic<bool> _stopServer = false;
static llvm::cl::opt<int>
    port("rpc-port",
         llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(8070));

namespace cudaq::driver {
extern void setTarget(target *);

std::vector<std::unique_ptr<channel>> communication_channels;
std::unique_ptr<quake_compiler> compiler;
std::map<intptr_t, device_ptr> memory_pool;
std::vector<std::string> symbolLocations;
std::unique_ptr<target> backend;

// Convert a shared memory device_ptr handle to its pointer
void *to_ptr(const device_ptr &d) { return reinterpret_cast<void *>(d.handle); }
// Convert a shared memory pointer to its handle representation
std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }

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

    auto devLibs =
        device.Config.ExposedLibraries.value_or(std::vector<std::string>{});
    for (auto &d : devLibs)
      symbolLocations.push_back(d);
  }

  // FIXME add this to the config
  compiler = quake_compiler::get("default_compiler");
  compiler->initialize(config);

  backend = target::get("default_target");
  backend->initialize(config);

  setTarget(backend.get());
}

// Allocate memory on devId. Return a unique handle
handle malloc(std::size_t size, std::size_t devId) {
  cudaq::info("controller malloc requested. size = {}, devId = {}", size,
              devId);
  device_ptr ptr;
  if (devId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("allocating data locally on controller.");
    auto hdl = to_handle(std::malloc(size));
    memory_pool.insert({hdl, {hdl, size, devId}});
    return hdl;
  }

  cudaq::info("forwarding malloc to device {}", devId);
  ptr = communication_channels[devId]->malloc(size);
  memory_pool.insert({ptr.handle, ptr});

  cudaq::info("return unique handle to allocated data {}.", ptr.handle);
  // Return the handle
  return ptr.handle;
}

void free(handle handle) {
  cudaq::info("controller deallocating device pointer with handle {}.", handle);
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("invalid data to free");

  auto &devPtr = iter->second;
  if (devPtr.deviceId == std::numeric_limits<std::size_t>::max()) {
    cudaq::info("deallocating local controller data");
    std::free(to_ptr(devPtr)); 
    return;
  }

  // FIXME validate the devId
  cudaq::info("forward deallocation to device {}", devPtr.deviceId);
  return communication_channels[devPtr.deviceId]->free(devPtr);
}

// memcpy from driver to host (hence the return)
std::vector<char> recv(handle handle, std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
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
    std::memcpy(result.data(), to_ptr(dest), size);
    return result;
  }

  communication_channels[dest.deviceId]->recv(result.data(), dest);
  return result;
}

void send(handle handle, std::vector<char> &data, std::size_t size) {
  auto iter = memory_pool.find(handle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid memcpy handle: " +
                             std::to_string(handle));

  device_ptr &dest = iter->second;
  cudaq::info("memcpy data with handle {} and size {} to {}.", handle, size,
              dest.deviceId == std::numeric_limits<std::size_t>::max()
                  ? "driver"
                  : "device " + std::to_string(dest.deviceId));
  if (dest.deviceId == std::numeric_limits<std::size_t>::max()) {
    // Local controller copy
    std::memcpy(to_ptr(dest), data.data(), size);
    return;
  }

  // Forward to device's communication channel
  communication_channels[dest.deviceId]->send(dest, data.data());
}

handle load_kernel(const std::string &quake) {
  cudaq::info("Loading and JIT compiling the kernel!");
  return compiler->compile(quake, symbolLocations);
}

std::vector<std::string> get_callbacks(handle hdl) {
  cudaq::info("get_callbacks() requested from controller for kernel {}", hdl);

  auto cbs = compiler->get_callbacks(hdl);
  std::vector<std::string> ret;
  for (auto &c : cbs)
    ret.push_back(c.callbackName);
  return ret;
}

// launch and return a handle to the result, -1 if void
std::vector<char> launch_kernel(handle kernelHandle, std::size_t argsHandle) {
  // Get the arguments from the memory pool
  auto iter = memory_pool.find(argsHandle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid args handle");

  auto &devPtr = iter->second;
  auto *thunkArgs = to_ptr(devPtr);
  auto size = devPtr.size;

  cudaq::info("Launching Kernel {}, args size {}", kernelHandle, size);

  // Get this kernel's callbacks
  auto callbacks = compiler->get_callbacks(kernelHandle);

  // Make callback code available to devices
  for (auto &callback : callbacks)
    for (auto &channel : communication_channels)
      channel->load_callback(callback.callbackName,
                             callback.unmarshalFuncOpCode);

  // We assume adaptive profile...
  auto maybeNumQubits = compiler->get_required_num_qubits(kernelHandle);
  if (maybeNumQubits)
    backend->allocate(*maybeNumQubits);

  // Launch the kernel
  compiler->launch(kernelHandle, thunkArgs);

  if (maybeNumQubits)
    backend->deallocate(*maybeNumQubits);
  // Return the result data
  std::vector<char> retRes(size);
  std::memcpy(retRes.data(), thunkArgs, size);
  return retRes;
}

std::vector<char> launch_callback(std::size_t devId,
                                  const std::string &funcName,
                                  std::size_t argsHandle) {
  // FIXME check devId is valid
  auto iter = memory_pool.find(argsHandle);
  if (iter == memory_pool.end())
    throw std::runtime_error("Invalid args handle");

  auto [resPtr, err] =
      communication_channels[devId]->launch_callback(funcName, iter->second);
  return resPtr;
}

} // namespace cudaq::driver

extern "C" {

void *__nvqpp__device_extract_device_ptr(cudaq::device_ptr *devPtr) {
  using namespace cudaq::driver;

  // Here we know we are in shared memory only
  if (devPtr->deviceId == std::numeric_limits<std::size_t>::max())
    return to_ptr(*devPtr);

  auto &channel = communication_channels[devPtr->deviceId];
  // Should this only be valid for CUDA and Shmem channels?
  if (channel->runs_on_separate_process())
    throw std::runtime_error("error extracting callback pointer argument - "
                             "channel is separate process.");

  return channel->get_raw_pointer(*devPtr);
}

cudaq::KernelThunkResultType
__nvqpp__device_callback_run(std::uint64_t deviceId, const char *funcName,
                             void *unmarshalFunc, void *argsBuffer,
                             std::uint64_t argsBufferSize,
                             std::uint64_t returnOffset) {
  using namespace cudaq::driver;
  cudaq::info("classical callback with shmem (host-side) func={} args_size={}",
              std::string(funcName), argsBufferSize);

  // Get the correct channel
  auto &channel = communication_channels[deviceId];

  // Separate process channel, allocate the unmarshal args
  auto argPtr = channel->malloc(argsBufferSize);
  // Send the args
  channel->send(argPtr, argsBuffer);
  // Launch the callback
  channel->launch_callback(funcName, argPtr);
  // Update the local args pointer
  channel->recv(argsBuffer, argPtr);
  // Free the data
  channel->free(argPtr);
  return {};
}
}

#define RPCD_BIND_FUNCTION(NAME) srv->bind(#NAME, &cudaq::driver::NAME);

int main(int argc, char **argv) {
  cudaq::info("Initialize Ethernet Controller on Port {}", port);
  auto srv = std::make_unique<rpc::server>(port);

  RPCD_BIND_FUNCTION(connect)
  RPCD_BIND_FUNCTION(malloc)
  RPCD_BIND_FUNCTION(free)
  RPCD_BIND_FUNCTION(send)
  RPCD_BIND_FUNCTION(recv)
  RPCD_BIND_FUNCTION(load_kernel)
  RPCD_BIND_FUNCTION(launch_kernel)

  srv->bind("ping", []() { return true; });
  srv->bind("stopServer", [&]() { _stopServer = true; });
  srv->async_run();
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (_stopServer)
      break;
  }
}
