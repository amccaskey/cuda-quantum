/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "driver.h"

#include <memory>
#include <numeric>
#include <stdarg.h>
#include <string.h>
#include <vector>

// Driver API is host-side, but needs to potentially
// communicate actions in a separate process (the controller)

namespace cudaq::driver {

namespace details {

// The communication channel from host to QPU control.
// Has to be the driver_channel subtype
static std::unique_ptr<controller_channel> host_qpu_channel;

} // namespace details

void initialize(const config::TargetConfig &cfg) {
  // setup the host to qpu control communication channel
  // can swap out the host->control channel
  details::host_qpu_channel = controller_channel::get("rpc_controller_channel");
  details::host_qpu_channel->connect(host_qpu_channel_id, cfg);
}

device_ptr malloc(std::size_t size, std::size_t devId) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return details::host_qpu_channel->malloc(size, devId);
}

void free(device_ptr &d) { return details::host_qpu_channel->free(d); }

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  return details::host_qpu_channel->memcpy(arg, src);
}

void memcpy(void *dest, device_ptr &src) {
  details::host_qpu_channel->memcpy(dest, src);
}

handle load_kernel(const std::string &quake) {
  return details::host_qpu_channel->load_kernel(quake);
}

launch_result launch_kernel(handle kernelHandle, device_ptr args) {
  return details::host_qpu_channel->launch_kernel(kernelHandle, args);
}
// sample, observe, run?

} // namespace cudaq::driver
