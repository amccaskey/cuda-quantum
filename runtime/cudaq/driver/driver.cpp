/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "driver.h"
#include "channel.h"
#include "target.h"

#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Support/TargetConfig.h"

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdarg.h>
#include <string.h>
#include <vector>

/// @file
/// @brief Implementation of CUDA Quantum driver functionality for managing
/// communication between host and QPU.
/// @details This file implements the driver API, which facilitates kernel
/// compilation, memory management,
///          and execution workflows for quantum devices. It supports shared
///          memory and separate process channels.

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::control_dispatcher)

namespace cudaq::driver {

namespace details {

// The communication channel from host to QPU control.
// Has to be the driver_channel subtype
static std::unique_ptr<control_dispatcher> host_qpu_channel;

} // namespace details

void initialize(const config::TargetConfig &cfg) {
  // setup the host to qpu control communication channel
  // can swap out the host->control channel
  auto ctrlChannelTy = cfg.HostControlChannel.value_or("shmem_iface");
  cudaq::info("Creating the dispatcher - {}", ctrlChannelTy);
  details::host_qpu_channel = control_dispatcher::get(ctrlChannelTy);
  details::host_qpu_channel->connect(cfg);
}

device_ptr malloc(std::size_t size) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return details::host_qpu_channel->malloc(size, host_qpu_channel_id);
}

device_ptr malloc(std::size_t size, std::size_t devId) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return details::host_qpu_channel->malloc(size, devId);
}

void free(device_ptr &d) { return details::host_qpu_channel->free(d); }

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  return details::host_qpu_channel->send(arg, src);
}

void memcpy(void *dest, device_ptr &src) {
  details::host_qpu_channel->recv(dest, src);
}

handle load_kernel(const std::string &quake) {
  return details::host_qpu_channel->load_kernel(quake);
}

launch_result launch_kernel(handle kernelHandle, device_ptr args) {
  return details::host_qpu_channel->launch_kernel(kernelHandle, args);
}

void shutdown() { details::host_qpu_channel->disconnect(); }

} // namespace cudaq::driver

// --- Shared Memory Intrinsics Implementations
extern "C" {

void *__nvqpp__device_extract_device_ptr(cudaq::driver::device_ptr *devPtr) {
  using namespace cudaq::driver;

  cudaq::info("Extracting the device pointer for {}, {}", devPtr->handle,
              devPtr->deviceId);
  // Here we know we are in shared memory only
  if (devPtr->deviceId == host_qpu_channel_id)
    return reinterpret_cast<void *>(devPtr->handle);

  auto &channel = shmem::communication_channels[devPtr->deviceId];
  // Should this only be valid for CUDA and Shmem channels?
  if (channel->runs_on_separate_process())
    throw std::runtime_error("error extracting callback pointer argument - "
                             "channel is separate process.");

  return channel->get_raw_pointer(*devPtr);
}
cudaq::KernelThunkResultType __nvqpp__device_callback_run(
    std::uint64_t deviceId, const char *funcName, void *unmarshalFunc,
    void *argsBuffer, std::uint64_t argsBufferSize, std::uint64_t returnOffset,
    std::uint64_t blockSize, std::uint64_t gridSize) {
  using namespace cudaq::driver;
  cudaq::info("classical callback with shmem (host-side) func={} args_size={}",
              std::string(funcName), argsBufferSize);

  // Get the correct channel
  auto &channel = shmem::communication_channels[deviceId];

  // Could be that our host+controller+channel are all on shared memory
  // in which case we can just run the unmarshal function pointer
  // If not, then we need to send the data and call the JIT
  // compiled unmarshal function
  if (channel->runs_on_separate_process()) {
    // Separate process channel, allocate the unmarshal args
    auto argPtr = channel->malloc(argsBufferSize);
    // Send the args
    channel->send(argPtr, argsBuffer);
    // Launch the callback
    channel->launch_callback(funcName, argPtr, {blockSize, gridSize});
    // Update the local args pointer
    channel->recv(argsBuffer, argPtr);
    // Free the data
    channel->free(argPtr);
    return {};
  }

  // This is a local shared memory callback, use the unmarshal function pointer.
  auto *castedFunc =
      reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(
          unmarshalFunc);

  // Load the callback (this is simple, just provide the channel with the
  // function pointer)
  channel->load_callback(funcName, castedFunc);

  // Launch the callback, result data stored to argsBuffer.
  channel->launch_callback(
      funcName,
      {cudaq::driver::shmem::to_handle(argsBuffer), argsBufferSize, deviceId},
      {blockSize, gridSize});

  return {};
}
}
