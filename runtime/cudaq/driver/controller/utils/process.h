/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define CUDAQ_RTTI_DISABLED
#include "cudaq/utils/cudaq_utils.h"
#undef CUDAQ_RTTI_DISABLED

#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

namespace cudaq::driver {
inline std::pair<pid_t, std::string> launchProcess(const char *command) {
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

inline std::filesystem::path getCUDAQControllerPath() {
  std::filesystem::path libPath{cudaq::getCUDAQLibraryPath()};
  auto cudaqLibPath = libPath.parent_path();
  return cudaqLibPath.parent_path() / "bin" / "cudaq-controller";
}

} // namespace cudaq::driver
