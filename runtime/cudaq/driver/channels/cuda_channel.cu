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

namespace cudaq::driver {

class cuda_channel : public channel {
private:
  int cudaDevice = 0;
  std::vector<std::string> fatbinLocations;
  std::map<std::size_t, void *> local_memory_pool;
  CUmodule *loadedModules = nullptr;
  std::map<std::string, CUfunction *> loadedCallbacks;
  CUcontext context;
  CUdevice device;

  std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }
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

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("Connecting to cuda channel");
    device_id = assignedID;
    fatbinLocations =
        config.Devices[assignedID].Config.ExposedLibraries.value_or(
            std::vector<std::string>{});
    CUresult result;

    cuInit(0);
    result = cuDeviceGet(&device, cudaDevice);
    result = cuCtxCreate(&context, 0, device);
    loadedModules = new CUmodule[fatbinLocations.size()];

    for (std::size_t i = 0; i < fatbinLocations.size(); i++) {
      cudaq::info("loading module for {}", fatbinLocations[i]);
      std::ifstream file(fatbinLocations[i], std::ios::binary);
      std::vector<char> fatbin_data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());

      // Load from memory buffer
      result = cuModuleLoadDataEx(&loadedModules[i], fatbin_data.data(), 0,
                                  nullptr, nullptr);
      if (result != CUDA_SUCCESS) {
        const char *errN;
        cuGetErrorName(result, &errN);
        fprintf(stderr, "Failed to load CUDA module: %s, %s\n",
                fatbinLocations[i].c_str(), errN);
      }
    }
  }

  void *get_raw_pointer(device_ptr &devPtr) override {
    auto cudaPtr = local_memory_pool.at(devPtr.handle);
    int i = 0;
    cudaMemcpy(&i, cudaPtr, 4, cudaMemcpyDeviceToHost);
    printf("HERE WE ARE GETTING THE ARG: %d\n", i);
    return local_memory_pool.at(devPtr.handle);
  }
  bool requires_unmarshaller() override { return false; }

  void disconnect() override {
    for (std::size_t i = 0; i < fatbinLocations.size(); i++)
      cuModuleUnload(loadedModules[i]);

    delete loadedModules;

    cuCtxDestroy(context);
  }

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

  void free(device_ptr &d) override {
    cudaq::info("cuda channel freeing data.");
    runOnCorrectDevice([&]() { cudaFree(local_memory_pool.at(d.handle)); });
  }

  void send(device_ptr &src, const void *dst) override {
    cudaq::info("cuda channel copying data to GPU.");
    runOnCorrectDevice([&]() {
      cudaMemcpy(local_memory_pool.at(src.handle), dst, src.size,
                 cudaMemcpyHostToDevice);
    });
  }

  void recv(void *dst, device_ptr &src) override {
    cudaq::info("cuda channel copying data from GPU {}.", src.handle);
    runOnCorrectDevice([&]() {
      cudaMemcpy(dst, local_memory_pool.at(src.handle), src.size,
                 cudaMemcpyDeviceToHost);
    });
  }

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

  void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) override {
    load_callback(funcName, "");
  }

  launch_result launch_callback(const std::string &funcName,
                                const device_ptr &args) override {
    cudaq::info("Launching gpu callback with name {} and args size {}, {}",
                funcName, args.size, args.handle);
    auto *cuFunc = loadedCallbacks.at(funcName);
    auto size = args.size;
    auto *rawArgs = reinterpret_cast<void *>(
        args.handle); // local_memory_pool.at(args.handle);
    int i = 0;
    struct test {
      int *ii;
    };
    auto *casted = reinterpret_cast<test *>(rawArgs);
    cudaMemcpy(&i, casted->ii, 4, cudaMemcpyDeviceToHost);
    printf("HERE WE ARE GETTING THE ARG: %d\n", i);
    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, rawArgs,
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    auto status = cuLaunchKernel(*cuFunc, 1, 1, 1, 1, 1, 1, 0, 0, NULL, config);
    return {};
  }

  bool runs_on_separate_process() override { return false; }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, cuda_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(cuda_channel)

} // namespace cudaq::driver
