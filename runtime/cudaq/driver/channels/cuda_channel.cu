/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma nv_diag_suppress = unsigned_compare_with_zero
#pragma nv_diag_suppress = unrecognized_gcc_pragma

#include "common/Logger.h"

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <fstream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      const char *errName;                                                     \
      cuGetErrorName(result, &errName);                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s failed with error: %s\n",       \
              __FILE__, __LINE__, #call, errName);                             \
    }                                                                          \
  } while (0)

namespace cudaq::driver {

/// @brief CUDA-based implementation of the `channel` interface for device
/// communication.
///
/// This class manages the connection, memory allocation, kernel loading, and
/// data transfer to and from a CUDA device. It supports loading multiple CUDA
/// modules, managing device pointers, and launching device callbacks (kernels).
class cuda_channel : public channel {
private:
  /// CUDA device ordinal this channel is associated with.
  int cudaDevice = 0;

  /// Paths to CUDA fatbin (module) files to be loaded.
  std::vector<std::string> fatbinLocations;

  /// Map of device pointer handles to raw device memory pointers.
  std::map<std::size_t, void *> local_memory_pool;

  /// Array of loaded CUDA modules.
  CUmodule *loadedModules = nullptr;

  /// Map of loaded callback (kernel) function names to CUfunction pointers.
  std::map<std::string, CUfunction *> loadedCallbacks;

  /// CUDA context for this channel.
  CUcontext context;

  /// CUDA device handle.
  CUdevice device;

  /// @brief Convert a raw pointer to a handle (size_t).
  /// @param ptr Raw pointer.
  /// @return Handle as size_t.
  std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }

  /// @brief Run the given applicator on the correct CUDA device.
  ///
  /// Switches to the channel's CUDA device if necessary, runs the applicator,
  /// then restores the previous device.
  /// @tparam Applicator Callable type.
  /// @param applicator Function or lambda to execute.
  /// @return Result of the applicator.
  template <typename Applicator>
  auto runOnCorrectDevice(const Applicator &applicator)
      -> std::invoke_result_t<Applicator> {
    int dev;
    cudaGetDevice(&dev);
    if (cudaDevice == dev)
      return applicator();

    cudaSetDevice(cudaDevice);
    if constexpr (std::is_void_v<std::invoke_result_t<Applicator>>) {
      applicator();
      cudaSetDevice(dev);
      return;
    } else {
      auto val = applicator();
      cudaSetDevice(dev);
      return val;
    }
  }

public:
  using channel::channel;

  /// @brief Connect to the CUDA device and load modules as specified by the
  /// config.
  /// @param assignedID Device ID to assign.
  /// @param config Target configuration containing device/module info.
  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("Connecting to cuda channel");
    device_id = assignedID;
    fatbinLocations =
        config.Devices[assignedID].Config.ExposedLibraries.value_or(
            std::vector<std::string>{});

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&device, cudaDevice));
    CUDA_CHECK(cuCtxCreate(&context, 0, device));
    loadedModules = new CUmodule[fatbinLocations.size()];

    for (std::size_t i = 0; i < fatbinLocations.size(); i++) {
      cudaq::info("loading module for {}", fatbinLocations[i]);
      std::ifstream file(fatbinLocations[i], std::ios::binary);
      std::vector<char> fatbin_data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());

      // Load from memory buffer
      CUDA_CHECK(cuModuleLoadDataEx(&loadedModules[i], fatbin_data.data(), 0,
                                    nullptr, nullptr));
    }
  }

  /// @brief Get the raw device pointer associated with a device_ptr handle.
  /// @param devPtr Device pointer abstraction.
  /// @return Raw device memory pointer.
  void *get_raw_pointer(device_ptr &devPtr) override {
    return local_memory_pool.at(devPtr.handle);
  }

  /// @brief Indicates whether this channel requires an unmarshaller.
  /// @return Always returns false for CUDA channel.
  bool requires_unmarshaller() override { return false; }

  /// @brief Disconnects from the CUDA device and unloads modules.
  void disconnect() override {
    for (std::size_t i = 0; i < fatbinLocations.size(); i++)
      CUDA_CHECK(cuModuleUnload(loadedModules[i]));

    delete loadedModules;

    CUDA_CHECK(cuCtxDestroy(context));
  }

  /// @brief Allocate device memory.
  /// @param size Number of bytes to allocate.
  /// @return Device pointer abstraction.
  device_ptr malloc(std::size_t size) override {

    return runOnCorrectDevice([&]() -> device_ptr {
      void *ptr = nullptr;
      cudaMalloc(&ptr, size);
      cudaMemset(ptr, 0, size);
      device_ptr devPtr{to_handle(ptr), size, device_id};
      local_memory_pool.insert({devPtr.handle, ptr});
      cudaq::info(
          "cuda channel (device {}) allocating data of size {}, hdl {}.",
          device_id, size, devPtr.handle);
      return devPtr;
    });
    return {};
  }

  /// @brief Free previously allocated device memory.
  /// @param d Device pointer abstraction to free.
  void free(device_ptr &d) override {
    cudaq::info("cuda channel freeing data.");
    runOnCorrectDevice([&]() { cudaFree(local_memory_pool.at(d.handle)); });
  }

  /// @brief Copy data from host to device memory.
  /// @param src Device pointer abstraction (destination on device).
  /// @param dst Source pointer on host.
  void send(device_ptr &src, const void *dst) override {
    cudaq::info("cuda channel copying data to GPU.");
    runOnCorrectDevice([&]() {
      cudaMemcpy(local_memory_pool.at(src.handle), dst, src.size,
                 cudaMemcpyHostToDevice);
    });
  }

  /// @brief Copy data from device to host memory.
  /// @param dst Destination pointer on host.
  /// @param src Device pointer abstraction (source on device).
  void recv(void *dst, device_ptr &src) override {
    cudaq::info("cuda channel copying data from GPU {}.", src.handle);
    runOnCorrectDevice([&]() {
      cudaMemcpy(dst, local_memory_pool.at(src.handle), src.size,
                 cudaMemcpyDeviceToHost);
    });
  }

  /// @brief Load a device callback (kernel) by name from loaded modules.
  /// @param funcName Name of the kernel function.
  /// @param unmarshallerCode Unmarshaller code (unused for CUDA).
  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    cudaq::info("loading gpu callback with name {}", funcName);
    // loop over our modules and find the CUkernel instance we want.
    for (std::size_t i = 0; i < fatbinLocations.size(); i++) {
      auto &mod = loadedModules[i];
      CUfunction *function = new CUfunction;
      auto result = cuModuleGetFunction(function, mod, funcName.c_str());
      if (result != CUDA_SUCCESS) {
        continue;
      } else {
        cudaq::info("callback found in {} cubin file", fatbinLocations[i]);
        loadedCallbacks.insert({funcName, function});
        break;
      }
    }

    auto iter = loadedCallbacks.find(funcName);
    if (iter == loadedCallbacks.end())
      throw std::runtime_error("could not find callback with name " + funcName);

    return;
  }

  /// @brief Load a device callback with a shared memory unmarshaller function.
  /// @param funcName Name of the kernel function.
  /// @param shmemUnmarshallerFunc Pointer to unmarshaller function.
  void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) override {
    load_callback(funcName, "");
  }

  /// @brief Launch a previously loaded device callback (kernel).
  /// @param funcName Name of the kernel function.
  /// @param args Device pointer to kernel arguments.
  /// @param blockSize Optional block size for kernel launch.
  /// @param gridSize Optional grid size for kernel launch.
  /// @return Launch result (unused).
  launch_result launch_callback(const std::string &funcName,
                                const device_ptr &args,
                                cuda_launch_parameters params) override {
    cudaq::info("Launching gpu callback with name {} and args size {}, {} - "
                "block/grid = {}/{}",
                funcName, args.size, args.handle, params.blockDim[0],
                params.gridDim[0]);
    auto *cuFunc = loadedCallbacks.at(funcName);
    auto size = args.size;
    auto *rawArgs = reinterpret_cast<void *>(args.handle);
    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, rawArgs,
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    CUDA_CHECK(cuLaunchKernel(*cuFunc, params.blockDim[0], params.blockDim[1],
                              params.blockDim[2], params.gridDim[0],
                              params.gridDim[1], params.gridDim[2], 0, 0, NULL,
                              config));
    return {};
  }

  /// @brief Indicates whether this channel runs on a separate process.
  /// @return Always returns false for CUDA channel.
  bool runs_on_separate_process() override { return false; }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, cuda_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(cuda_channel)

} // namespace cudaq::driver
