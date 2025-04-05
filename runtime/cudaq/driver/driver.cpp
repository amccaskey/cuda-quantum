/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "driver.h"
#include "controller/channel.h"

#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
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

std::optional<std::string> trace_symbol(const char *name) {
  void *addr = dlsym(RTLD_DEFAULT, name);
  Dl_info info;
  if (dladdr(addr, &info))
    return info.dli_fname;

  return std::nullopt;
}

} // namespace details

void initialize(const config::TargetConfig &cfg) {
  // setup the host to qpu control communication channel
  // can swap out the host->control channel
  details::host_qpu_channel = controller_channel::get("rpc_controller_channel");
  details::host_qpu_channel->connect(host_qpu_channel_id, cfg);
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
  return details::host_qpu_channel->memcpy(arg, src);
}

void memcpy(void *dest, device_ptr &src) {
  details::host_qpu_channel->memcpy(dest, src);
}

handle load_kernel(const std::string &quake) {
  // Loading a kernel involves the following:
  // 1. Analyze the quake code and find all function
  //    symbols being called by device_call
  // 2. Extract the library path for those symbols.
  // 3. When compiling the code, lower device_calls to
  //    the marshal/unmarshal intrinsics. Save these in a map
  //
  // When a kernel is launched, we need to send any possible
  //    unmarshalers to the other end of the channel and JIT
  auto kernelHandle = details::host_qpu_channel->load_kernel(quake);

  // get the names of any callbacks in the kernel
  // FIXME could add this to the driver API for users
  auto callbackSymbols = details::host_qpu_channel->get_callbacks(kernelHandle);

  // For each callback, see if we can get the library path
  // where it resides
  std::vector<std::string> symbolLocations;
  for (auto &c : callbackSymbols)
    if (auto loc = details::trace_symbol(c.c_str()); loc.has_value()) {
      symbolLocations.push_back(loc.value());
      cudaq::info("driver found required callback symbol {} at {}.", c,
                  loc.value());
    }

  // Distribute those library locations.
  details::host_qpu_channel->distribute_symbol_locations(symbolLocations);

  // Return the kernel handle to the user.
  return kernelHandle;
}

launch_result launch_kernel(handle kernelHandle, device_ptr args) {
  return details::host_qpu_channel->launch_kernel(kernelHandle, args);
}

} // namespace cudaq::driver
