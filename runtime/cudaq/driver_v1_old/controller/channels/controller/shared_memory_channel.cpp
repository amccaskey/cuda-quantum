/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/Support/TargetConfig.h"

#include "cudaq/driver/controller/channel.h"
#include "cudaq/driver/controller/controller.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

namespace cudaq::driver {
void set_controller_caller_retains_ownership(controller *c);

class shared_memory_controller_channel : public controller_channel {
protected:
  std::unique_ptr<controller> m_controller;

public:
  using controller_channel::controller_channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {

    m_controller = controller::get("shared_memory_controller");
    std::string asStr;
    {
      llvm::raw_string_ostream os(asStr);
      llvm::yaml::Output yout(os);
      yout << const_cast<config::TargetConfig &>(config);
    }
    m_controller->connect(asStr);

    cudaq::info("shared memory controller channel connected.");
    set_controller_caller_retains_ownership(m_controller.get());
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    // need to create some sort of mapping
    return {m_controller->malloc(size, devId), size, devId};
  }

  void free(device_ptr &d) override { m_controller->free(d.handle); }

  // copy to QPU
  void memcpy(device_ptr &src, const void *dst) override {
    const char *dstC = reinterpret_cast<const char *>(dst);
    auto size = src.size;
    std::vector<char> buffer(dstC, dstC + size);
    cudaq::info("Shared Mem Channel calling memcpy_to with {}", src.handle);
    m_controller->memcpy_to(src.handle, buffer, size);
  }

  void memcpy(void *dst, device_ptr &src) override {
    // Get remote handle for destination pointer
    auto size = src.size;
    auto result = m_controller->memcpy_from(src.handle, size);
    std::memcpy(dst, result.data(), size);
  }

  handle load_kernel(const std::string &quake) const {
    return m_controller->load_kernel(quake);
  }

  std::vector<std::string> get_callbacks(handle kernelHandle) override {
    return m_controller->get_callbacks(kernelHandle);
  }

  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const {
    auto resultData =
        m_controller->launch_kernel(kernelHandle, argsHandle.handle);
    return {resultData};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(controller_channel,
                                   shared_memory_controller_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory_controller_channel)

} // namespace cudaq::driver
