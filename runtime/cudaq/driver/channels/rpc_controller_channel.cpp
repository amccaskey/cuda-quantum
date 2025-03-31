/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/Support/TargetConfig.h"

#include "cudaq/driver/channel.h"
#include "cudaq/utils/cudaq_utils.h"

#include "rpc/client.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

std::pair<pid_t, std::string> launchProcess(const char *command) {
  // Create temporary files for storing stdout and stderr
  char tempStdout[] = "/tmp/stdout_XXXXXX";
  char tempStderr[] = "/tmp/stderr_XXXXXX";

  int fdOut = mkstemp(tempStdout);
  int fdErr = mkstemp(tempStderr);

  if (fdOut == -1 || fdErr == -1) {
    throw std::runtime_error("Failed to create temporary files");
  }

  // Construct command to redirect both stdout and stderr to temporary files
  std::string argString = std::string(command) + " 1>" + tempStdout + " 2>" +
                          tempStderr + " & echo $!";

  // Launch the process
  FILE *pipe = popen(argString.c_str(), "r");
  if (!pipe) {
    close(fdOut);
    close(fdErr);
    unlink(tempStdout);
    unlink(tempStderr);
    throw std::runtime_error("Error launching process: " +
                             std::string(command));
  }

  // Read PID
  char buffer[128];
  std::string pidStr = "";
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != nullptr)
      pidStr += buffer;
  }
  pclose(pipe);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // Read any error output
  std::string errorOutput;
  FILE *errorFile = fopen(tempStderr, "r");
  if (errorFile) {
    while (fgets(buffer, 128, errorFile) != nullptr) {
      errorOutput += buffer;
    }
    fclose(errorFile);
  }

  // Clean up temporary files
  close(fdOut);
  close(fdErr);
  unlink(tempStdout);
  unlink(tempStderr);

  // Convert PID string to integer
  pid_t pid = 0;
  try {
    pid = std::stoi(pidStr);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to get process ID: " + errorOutput);
  }

  return std::make_pair(pid, errorOutput);
}

bool is_server_available() {
  try {
    rpc::client client("127.0.0.1", 8080);
    client.set_timeout(200); // 200ms timeout
    client.call("ping");
    return true;
  } catch (...) {
    return false;
  }
}

namespace cudaq::driver {

class rpc_controller_channel : public controller_channel {
protected:
  bool startedLocalServer = false;
  std::unordered_map<void *, std::size_t> fakePtrsToRemoteHandles;

public:
  using controller_channel::controller_channel;

  std::unique_ptr<rpc::client> client;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {

    // If there is no controller IP address in the config, start one
    // locally
    std::string ip = "127.0.0.1";
    auto port = config.ControllerPort.value_or(8080);

    if (!config.ControllerIP) {
      std::filesystem::path libPath{cudaq::getCUDAQLibraryPath()};
      auto cudaqLibPath = libPath.parent_path();
      auto cudaqControllerTool =
          cudaqLibPath.parent_path() / "bin" / "cudaq-controller";
      auto argString = cudaqControllerTool.string();

      auto [ret, msg] = launchProcess(argString.c_str());
      if (ret == -1)
        throw std::runtime_error("Could not launch controller");

      for (int i = 0; i < 5; ++i) {
        if (is_server_available())
          break;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    } else
      ip = config.ControllerIP.value();

    client = std::make_unique<rpc::client>(ip, port);
    // FIXME handle didn't connect
    std::string asStr;
    {
      llvm::raw_string_ostream os(asStr);
      llvm::yaml::Output yout(os);
      yout << const_cast<config::TargetConfig &>(config);
    }
    client->call("connect", asStr);
    cudaq::info("controller channel connected to {}:{}.", ip, port);
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    auto result = client->call("malloc", size, devId);
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<std::size_t> dist(1, 1e12);

    // Allocate memory for an integer
    auto *random_ptr = new std::size_t(dist(rng));
    fakePtrsToRemoteHandles.insert(
        {reinterpret_cast<void *>(random_ptr), result.as<std::size_t>()});
    // need to create some sort of mapping
    return {random_ptr, size, devId};
  }

  void free(device_ptr &d) override {
    client->call("free", fakePtrsToRemoteHandles[d.data]);
  }
  void free(std::size_t argsHandle) override {}

  // copy to QPU
  void memcpy(device_ptr &src, const void *dst) override {
    auto it = fakePtrsToRemoteHandles.find(src.data);
    if (it == fakePtrsToRemoteHandles.end())
      throw std::runtime_error("Invalid device pointer");

    auto size = src.size;
    std::vector<char> buffer(size);
    std::memcpy(buffer.data(), dst, size);
    client->call("memcpy_to", it->second, buffer, size);
  }

  void memcpy(void *dst, device_ptr &src) override {
    // Get remote handle for destination pointer
    auto it = fakePtrsToRemoteHandles.find(src.data);
    if (it == fakePtrsToRemoteHandles.end())
      throw std::runtime_error("Invalid device pointer");
    auto size = src.size;
    std::vector<char> result =
        client->call("memcpy_from", it->second, size).as<std::vector<char>>();
    std::memcpy(dst, result.data(), size);
  }

  handle load_kernel(const std::string &quake) const {
    auto kernelHandle = client->call("load_kernel", quake);
    // FIXME may want to do something with this here
    return kernelHandle.as<std::size_t>();
  }

  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const {
    auto it = fakePtrsToRemoteHandles.find(argsHandle.data);
    if (it == fakePtrsToRemoteHandles.end())
      throw std::runtime_error("Invalid device pointer");
    auto size = argsHandle.size;
    auto resultData = client->call("launch_kernel", kernelHandle, it->second)
                          .as<std::vector<char>>();

    device_ptr ptr;
    ptr.data = std::malloc(resultData.size());
    ptr.size = resultData.size();
    std::memcpy(ptr.data, resultData.data(), ptr.size);
    return {ptr, 0, ""};
  }

  ~rpc_controller_channel() {
    if (startedLocalServer) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));
      cudaq::info("shutdown controller channel.");
      client->call("stopServer");
    }
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(controller_channel, rpc_controller_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(rpc_controller_channel)

} // namespace cudaq::driver
