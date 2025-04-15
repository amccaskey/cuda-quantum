/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/Support/TargetConfig.h"
#include <gtest/gtest.h>

#include <cudaq/builder.h>

// Define a global test environment
class DriverTestEnvironment : public ::testing::Environment {
public:
  ~DriverTestEnvironment() override {}
  std::unique_ptr<std::thread> serverThread;

  // Override SetUp to initialize the driver
  void SetUp() override {
    serverThread = std::make_unique<std::thread>(
        []() { std::system("./bin/cudaq-ethernet-controller"); });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string configContents = R"#(
        description: "Custom target for DGX-Q."
        name: EthernetHost_CUDAChannel 
        control-dispatcher: ethernet_dispatcher
        devices:
          - name: CUDAChannel 
            config: 
              channel: cuda_channel
              exposed-libraries: ["unittests/driver/CMakeFiles/test_cuda_kernels.dir/test_cuda.fatbin"]
        )#";

    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;

    // Initialize the driver with the configuration
    cudaq::driver::initialize(config);
  }

  // Override TearDown if any cleanup is needed
  void TearDown() override {
    // Perform any necessary cleanup here (if applicable)
    cudaq::driver::shutdown();
    serverThread->join();
  }
};

TEST(DriverTester, checkLaunchKernel) {

  const std::string quake = R"#(
  module {
  func.func @__nvqpp__mlirgen__function_callGpu.callGpu(%arg0: !cc.struct<"device_ptr" {i64, i64, i64} [192,8]>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %0 = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64} [192,8]>
    cc.store %arg0, %0 : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    %1 = quake.alloca !quake.ref
    quake.h %1 : (!quake.ref) -> ()
    %2 = cc.extract_ptr %0 : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>) -> !cc.ptr<i8>
    %3 = cc.cast %2 : (!cc.ptr<i8>) -> !cc.ptr<i32>
    cc.device_call @incrementInt<1, 1> (%3) : (!cc.ptr<i32>) -> ()
    return
  }
  func.func private @incrementInt(!cc.ptr<i32>) attributes {"cudaq-devicecall"}
  func.func private @__nvqpp_zeroDynamicResult() -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %2 = cc.insert_value %1[0], %0 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %3 = cc.insert_value %2[1], %c0_i64 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %3 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  func.func @function_callGpu.callGpu.thunk(%arg0: !cc.ptr<i8>, %arg1: i1) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %0 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<!cc.struct<{!cc.struct<{i64, i64, i64}>}>>
    %1 = cc.sizeof !cc.struct<{!cc.struct<{i64, i64, i64}>}> : i64
    %2 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
    %3 = cc.compute_ptr %2[%1] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
    %4 = cc.compute_ptr %0[0] : (!cc.ptr<!cc.struct<{!cc.struct<{i64, i64, i64}>}>>) -> !cc.ptr<!cc.struct<{i64, i64, i64}>>
    %5 = cc.cast %4 : (!cc.ptr<!cc.struct<{i64, i64, i64}>>) -> !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    %6 = cc.load %5 : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    call @__nvqpp__mlirgen__function_callGpu.callGpu(%6) : (!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>) -> ()
    %7 = call @__nvqpp_zeroDynamicResult() : () -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %7 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  }
  )#";

  // This is the kernel
  // __qpu__ void callGpu(cudaq::device_ptr in) {
  //   cudaq::qubit q;
  //   h(q);
  //   cudaq::device_call<1,1>(incrementInt, in);
  // }

  // Load the Kernel
  printf("loading the kernel.\n");
  auto hdl = cudaq::driver::load_kernel(quake);

  // Manually setup the Thunk Args for the Kernel
  // Here our kernel takes a device_ptr as input,
  // so the ThunkArgs are the following struct
  struct ThunkArgs {
    cudaq::device_ptr devPtr;
  };

  printf("calling malloc_set on GPU 0\n");
  // Create the int device_ptr we want to pass as input
  // Allocate it on the 0th device, a cuda_channel GPU
  auto devPtr = cudaq::driver::malloc_set((int)2, 0);

  // Initialize the ThunkArgs
  ThunkArgs thunkArgsConcrete{devPtr};

  printf("malloc the thunk args on the controller\n");
  // Tell the controller to allocate data for the Thunk Args
  auto thunkArgsDevPtr = cudaq::driver::malloc(sizeof(ThunkArgs));

  printf("Memcpy the thunk args to the controller\n");
  // Set the Thunk Args data on the controller
  cudaq::driver::memcpy(thunkArgsDevPtr, &thunkArgsConcrete);

  // Launch the Kernel!
  printf("launch the kernel\n");
  cudaq::driver::launch_kernel(hdl, thunkArgsDevPtr);

  // Get the Thunk Args data back from the controller
  cudaq::driver::memcpy(&thunkArgsConcrete, thunkArgsDevPtr);

  // We expect 2+1=3
  EXPECT_EQ(cudaq::driver::memcpy<int>(thunkArgsConcrete.devPtr), 3);

  // Free the controller data
  cudaq::driver::free(thunkArgsDevPtr);
  cudaq::driver::free(devPtr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Register the global test environment
  ::testing::AddGlobalTestEnvironment(new DriverTestEnvironment);

  // Run all tests
  return RUN_ALL_TESTS();
}
