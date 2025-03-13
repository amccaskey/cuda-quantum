/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "channel.h"
#include "common/ExecutionContext.h"

#include <memory>
#include <numeric>
#include <stdarg.h>
#include <string.h>
#include <vector>

namespace cudaq::driver {

struct target_info {};

namespace details {

// The communication channel from host to QPU control is special,
// give it a unique ID none of the others will get
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

// The communication channel from host to QPU control
static std::unique_ptr<channel> host_qpu_channel;

// Spec device IDs are implicit in the vector index here
static std::vector<std::unique_ptr<channel>> communication_channels;

bool isValidDeviceId(std::size_t deviceId) {
  return deviceId < details::communication_channels.size();
}
} // namespace details

class device {};

enum class memcpy_kind {
  host_to_driver,
  host_to_device,
  driver_to_host,
  device_to_host
};

void initialize(const target_info &info) {

  // setup the host to qpu control communication channel
  details::host_qpu_channel = channel::get("shared_memory");
  details::host_qpu_channel->connect(details::host_qpu_channel_id);

  // initialize target-specified control to classical device channels.
}

// query info on QPU architecture
// get device, query device info

// move runtime arguments from host to control

// allocate space for data on the QPU control
data malloc(std::size_t size) {
  return details::host_qpu_channel->malloc(size);
}

data malloc(std::size_t deviceId, std::size_t size) {
  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error("Invalid device id requested in driver::malloc");

  return details::communication_channels[deviceId]->malloc(size);
}

void free(data &d) { return details::host_qpu_channel->free(d); }

void free(std::size_t deviceId, data &d) {
  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error("Invalid device id requested in driver::free");
  return details::communication_channels[deviceId]->free(d);
}

// Copy the given src data into the data element.
void memcpy(data &arg, const void *src, std::size_t size, memcpy_kind kind,
            std::size_t deviceId = details::host_qpu_channel_id) {
  if (deviceId == details::host_qpu_channel_id)
    if (kind == memcpy_kind::host_to_driver)
      return details::host_qpu_channel->memcpy(arg, src, size);
    else
      throw std::runtime_error(
          "memcpy_kind not yet supported in driver::memcpy");

  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error("Invalid device id requested in driver::memcpy");
  return details::communication_channels[deviceId]->memcpy(arg, src, size);
}


// compile? 

// compile the kernel quake code with given kernel name, return a handle 
// to that kernel for further usage (launch). 
std::size_t compile_kernel(const std::string& kernelName) {
  // compile kernel over the channel. need extension point for 
  // actual quake lowering
}

// launch a kernel under a specific execution context
// ?

void launch_kernel(std::size_t kernelHandle, const std::vector<data>& args) {}

// sample, observe, run?

} // namespace cudaq::driver

extern "C" {

std::size_t __nvqpp__callback_marshal(std::size_t deviceId,
                                      const char *argFmtStr, ...) {
  using namespace cudaq::driver;

  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error(
        "Invalid device id requested in __nvqpp__callback_marshal");

  std::size_t numArgs = 0;
  for (const char *p = argFmtStr; (p = strchr(p, '%')) != NULL; numArgs++, p++)
    ;

  va_list args;
  va_start(args, numArgs);

  // do something with these args
  std::vector<data> marshaledArgs;
  std::vector<const void *> srcs;
  std::vector<std::size_t> sizes;
  // FIXME Fill the args (malloc and memcpy)

  return details::communication_channels.at(deviceId)->memcpy(marshaledArgs,
                                                              srcs, sizes);
}

void __nvqpp__callback_run(std::size_t deviceId, const char *funcName,
                           std::size_t argsHandle) {
  using namespace cudaq::driver;

  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error(
        "Invalid device id requested in __nvqpp__callback_run");

  auto err = details::communication_channels.at(deviceId)->launch_callback(
      funcName, argsHandle);
  
  if (err != 0)
    throw std::runtime_error("driver error in launching callback function");
  
  return;
}

void __nvqpp__callback_result(std::size_t deviceId, const char *formatStr,
                              std::size_t argsHandle, void *result) {
  using namespace cudaq::driver;

  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error(
        "Invalid device id requested in __nvqpp__callback_result");
}

void __nvqpp__callback_free(std::size_t deviceId, std::size_t argsHandle) {
  using namespace cudaq::driver;

  if (!details::isValidDeviceId(deviceId))
    throw std::runtime_error(
        "Invalid device id requested in __nvqpp__callback_run");
}
}