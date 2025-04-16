/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "common/ThunkInterface.h"
#include "cudaq/utils/extension_point.h"
#include "device_ptr.h"

namespace cudaq {
namespace config {
class TargetConfig;
}

namespace driver {

/// @brief Special identifier for host-QPU communication channel.
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

/// @brief Abstract base class for managing host-control dispatch operations.
/// This class provides an interface for connecting to a target device,
/// managing memory allocation, transferring data, and executing quantum
/// kernels.
class control_dispatcher : public extension_point<control_dispatcher> {
public:
  /// @brief Establish a connection to the target process or memory space.
  /// @param config Configuration details for the target connection.
  virtual void connect(const config::TargetConfig &config) = 0;

  /// @brief Disconnect from the target process or memory space.
  virtual void disconnect() = 0;

  /// @brief Allocate memory on a specific device.
  /// @param size The size of the memory to allocate in bytes.
  /// @param devId The ID of the target device where memory is allocated.
  /// @return A handle to the allocated device memory.
  virtual device_ptr malloc(std::size_t size, std::size_t devId) = 0;

  /// @brief Free previously allocated device memory.
  /// @param d A reference to the device pointer to be freed.
  virtual void free(device_ptr &d) = 0;

  /// @brief Transfer data from host to device memory.
  /// @param dest The destination device pointer on the target device.
  /// @param src Pointer to the source data in host memory.
  virtual void send(device_ptr &dest, const void *src) = 0;

  /// @brief Transfer data from device to host memory.
  /// @param dest Pointer to the destination buffer in host memory.
  /// @param src The source device pointer on the target device.
  virtual void recv(void *dest, device_ptr &src) = 0;

  /// @brief Load a quantum kernel for execution on the target device.
  /// This may involve target-specific just-in-time (JIT) compilation.
  /// @param quake The quantum kernel code in string format (e.g., MLIR or QIR).
  /// @return A handle representing the loaded kernel on the target device.
  virtual handle load_kernel(const std::string &quake) const = 0;

  /// @brief Launch a quantum kernel on the target device with specified
  /// arguments.
  /// @param kernelHandle Handle to the loaded quantum kernel.
  /// @param argsHandle Device pointer containing arguments for the kernel
  /// execution.
  /// @return Result of the kernel launch operation (e.g., success or error
  /// code).
  virtual launch_result launch_kernel(handle kernelHandle,
                                      device_ptr &argsHandle) const = 0;
};

/// @brief Abstract base class for managing communication channels.
/// This class provides an interface for connecting to devices, managing memory,
/// transferring data, and executing classical callbacks in conjunction with
/// quantum operations.
class channel : public extension_point<channel> {
protected:
  std::size_t device_id =
      0; ///< The ID of the associated channel's target device.

  /// Locations of libraries containing symbols required for classical
  /// callbacks.
  std::vector<std::string> symbol_locations;

public:
  channel() = default;

  virtual ~channel() = default;

  /// @brief Establish a connection to a specific channel on a target process or
  /// memory space.
  /// @param assignedID The ID assigned to this channel's target device.
  /// @param config Configuration details for connecting to the target process
  /// or space.
  virtual void connect(std::size_t assignedID,
                       const config::TargetConfig &config) = 0;

  /// @brief Disconnect from the associated channel's process or memory space.
  virtual void disconnect() {}

  /// @brief Check if this channel operates on a separate process space.
  /// @return True if it runs on a separate process; false otherwise.
  virtual bool runs_on_separate_process() { return false; }
  virtual bool requires_unmarshaller() { return true; }

  /// @brief Return the raw pointer corresponding to the provided
  /// device_pointer. This should only be valid for channels
  /// that do not run on a separate process (e.g. shmem and cuda).
  /// @param devPtr The CUDA-Q device pointer handle.
  /// @return Raw pointer
  virtual void *get_raw_pointer(device_ptr &devPtr) = 0;

  /// @brief Allocate memory on this channel's associated device.
  /// @param size The size of the memory to allocate in bytes.
  /// @return A handle to the allocated device memory.
  virtual device_ptr malloc(std::size_t size) = 0;

  /// @brief Free previously allocated memory on this channel's associated
  /// device.
  /// @param d A reference to the device pointer to be freed.
  virtual void free(device_ptr &d) = 0;

  /// @brief Transfer data from host to this channel's associated device memory.
  /// @param dest The destination device pointer on this channel's associated
  /// device.
  /// @param src Pointer to the source data in host memory.
  virtual void send(device_ptr &dest, const void *src) = 0;

  /// @brief Transfer data from this channel's associated device memory to host
  /// memory.
  /// @param dest Pointer to the destination buffer in host memory.
  /// @param src The source device pointer on this channel's associated device.
  virtual void recv(void *dest, device_ptr &src) = 0;

  /// @brief Load a callback function into this channel using its name and MLIR
  /// code representation.
  /// This method is used when callbacks are represented as MLIR FuncOp code
  /// strings.
  /// @param funcName Name of the callback function being loaded.
  /// @param unmarshallerCode MLIR FuncOp string representation of the callback
  /// logic.
  virtual void load_callback(const std::string &funcName,
                             const std::string &unmarshallerCode) = 0;

  /// @brief Load a callback function into this channel using its name and a
  /// shared memory unmarshaller function. This method is used when callbacks
  /// are represented as C++ function pointers for shared memory unmarshalling.
  /// @param funcName Name of the callback function being loaded.
  /// @param shmemUnmarshallerFunc Function pointer to the shared memory
  /// unmarshaller logic.
  virtual void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) = 0;

  /// @brief Launch a callback function on this channel with specified
  /// arguments. This method executes a classical callback in conjunction with
  /// quantum operations.
  /// @param funcName Name of the callback function to be executed.
  /// @param argsHandle Device pointer containing arguments for the callback
  /// execution.
  /// @return Result of the callback execution (e.g., success or error code).
  virtual launch_result
  launch_callback(const std::string &funcName, const device_ptr &argsHandle,
                  cuda_launch_parameters params = cuda_launch_parameters()) = 0;
};

namespace shmem {
/// @brief Convert a shared memory device pointer handle to its raw pointer.
/// @param d Device pointer handle.
/// @return Raw pointer representation of the device memory.
inline void *to_ptr(const device_ptr &d) {
  return reinterpret_cast<void *>(d.handle);
}

/// @brief Convert a raw pointer to its handle representation.
/// @param ptr Raw pointer.
/// @return Handle representation of the pointer.
inline std::size_t to_handle(void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr);
}

/// @brief Collection of communication channels between shared memory host and
/// devices.
inline std::vector<std::unique_ptr<channel>> communication_channels;

} // namespace shmem

} // namespace driver
} // namespace cudaq
