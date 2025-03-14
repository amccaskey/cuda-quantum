/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "channel.h"
#include "cudaq/Support/TargetConfig.h"

#include <cstddef>
#include <numeric>

namespace cudaq::driver {

struct target_info {};

namespace details {

// The communication channel from host to QPU control is special,
// give it a unique ID none of the others will get
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

} // namespace details

enum class memcpy_kind {
  host_to_driver,
  host_to_device,
  driver_to_host,
  device_to_host
};

void initialize(const config::TargetConfig &config);

// FIXME what about 
// query info on QPU architecture
// get device, query device info
// do we need a - class device {};

device_ptr malloc(std::size_t size);
device_ptr malloc(std::size_t deviceId, std::size_t size);

void free(device_ptr &d);
void free(std::size_t deviceId, device_ptr &d);

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src, std::size_t size, memcpy_kind kind,
            std::size_t deviceId = details::host_qpu_channel_id);

std::size_t compile_kernel(const std::string &quake);

template <typename QuantumKernel>
std::size_t compile_kernel(QuantumKernel &&kernel) {
  // get quake code, run compilation on it
  std::string quake = "";
  return compile_kernel(quake);
}

void launch_kernel(std::size_t kernelHandle, const std::vector<device_ptr> &args);
void launch_kernel(std::size_t kernelHandle, std::size_t argHandle);

// need to be able to get the result 

// sample, observe, run?

} // namespace cudaq::driver

extern "C" {
std::size_t __nvqpp__callback_marshal(std::size_t deviceId,
                                      const char *argFmtStr, ...);
void __nvqpp__callback_run(std::size_t deviceId, const char *funcName,
                           std::size_t argsHandle);
void __nvqpp__callback_result(std::size_t deviceId, const char *formatStr,
                              std::size_t argsHandle, void *result);
void __nvqpp__callback_free(std::size_t deviceId, std::size_t argsHandle);
}