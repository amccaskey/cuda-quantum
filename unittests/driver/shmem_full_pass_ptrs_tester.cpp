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

  // Override SetUp to initialize the driver
  void SetUp() override {
    std::string configContents = R"#(
        description: "Custom target for DGX-Q."
        name: ShmemTestTarget 
        devices:
          - name: SharedChannel 
            config: 
              channel: shmem_channel
              exposed-libraries: ["unittests/driver/libadd_driver_test.so"]
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
  }
};

TEST(DriverTester, checkShmem) {
  // Putting this here, we need MLIR startup to work
  // for all cases, builder + driver, no builder + driver
  auto kernel = cudaq::make_kernel();

  {
    // Allocate an int = 22 on the controller
    auto dev0Data = cudaq::driver::malloc(4);
    int ii = 22;
    cudaq::driver::memcpy(dev0Data, &ii);
    // Copy the data from the controller back to host
    int local = cudaq::driver::memcpy<int>(dev0Data);

    // Check it
    printf("Local is %d\n", local);
    EXPECT_TRUE(local == 22);
    cudaq::driver::free(dev0Data);
  }

  {
    // Allocate an int = 23 on the controller
    auto dev0Data = cudaq::driver::malloc_set(23);
    // Copy the data from the controller back to host
    int local = cudaq::driver::memcpy<int>(dev0Data);

    // Check it
    printf("Local is %d\n", local);
    EXPECT_TRUE(local == 23);
    cudaq::driver::free(dev0Data);
  }

  {
    // Allocate an int = 22 on the device 0
    auto dev0Data = cudaq::driver::malloc(4, /**devId**/ 0);
    int ii = 22;
    cudaq::driver::memcpy(dev0Data, &ii);
    // Copy the data from the controller back to host
    int local = cudaq::driver::memcpy<int>(dev0Data);

    // Check it
    printf("Local is %d\n", local);
    EXPECT_TRUE(local == 22);
    cudaq::driver::free(dev0Data);
  }
}

TEST(DriverTester, checkLaunchKernel) {

  const std::string quake = R"#(
  module {
  func.func @__nvqpp__mlirgen__function_test.test(%arg0: !cc.struct<"device_ptr" {i64, i64, i64} [192,8]>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %0 = cc.alloca !cc.struct<"device_ptr" {i64, i64, i64} [192,8]>
    cc.store %arg0, %0 : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    %1 = quake.alloca !quake.ref
    quake.h %1 : (!quake.ref) -> ()
    %2 = cc.extract_ptr %0 : (!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>) -> !cc.ptr<i8>
    %3 = cc.cast %2 : (!cc.ptr<i8>) -> !cc.ptr<i32>
    %4 = cc.alloca !cc.ptr<i32>
    cc.store %3, %4 : !cc.ptr<!cc.ptr<i32>>
    %5 = cc.load %4 : !cc.ptr<!cc.ptr<i32>>
    cc.device_call @cuda_kernel(%5) : (!cc.ptr<i32>) -> ()
    return
  }
  func.func private @cuda_kernel(!cc.ptr<i32>) attributes {"cudaq-devicecall"}
  func.func private @__nvqpp_zeroDynamicResult() -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %2 = cc.insert_value %1[0], %0 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %3 = cc.insert_value %2[1], %c0_i64 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %3 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  func.func @function_test.test.thunk(%arg0: !cc.ptr<i8>, %arg1: i1) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %0 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    %1 = cc.load %0 : !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>>
    call @__nvqpp__mlirgen__function_test.test(%1) : (!cc.struct<"device_ptr" {i64, i64, i64} [192,8]>) -> ()
    %2 = call @__nvqpp_zeroDynamicResult() : () -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %2 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  }
  )#";

  // Load the Kernel
  auto hdl = cudaq::driver::load_kernel(quake);

  // Manually setup the Thunk Args for the Kernel
  struct ThunkArgs {
    cudaq::driver::device_ptr devPtr;
  };

  // Allocate data on the 0th device
  auto devPtr = cudaq::driver::malloc_set((int)2, 0);

  ThunkArgs thunkArgsConcrete{devPtr};

  // Tell the controller to allocate data for the Thunk Args
  auto thunkArgsDevPtr = cudaq::driver::malloc(sizeof(ThunkArgs));

  // Set the Thunk Args data on the controller
  cudaq::driver::memcpy(thunkArgsDevPtr, &thunkArgsConcrete);

  // Launch the Kernel!
  cudaq::driver::launch_kernel(hdl, thunkArgsDevPtr);

  // Get the Thunk Args data back from the controller
  cudaq::driver::memcpy(&thunkArgsConcrete, thunkArgsDevPtr);

  // We expect 2+4=6
  EXPECT_EQ(cudaq::driver::memcpy<int>(thunkArgsConcrete.devPtr), 6);

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
