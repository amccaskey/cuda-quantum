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
  func.func private @add(i32, i32) -> i32 attributes {"cudaq-devicecall"}
  func.func @__nvqpp__mlirgen__function_testKernel.testKernel(%arg0: i32, %arg1: i32) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %0 = cc.alloca i32
    cc.store %arg0, %0 : !cc.ptr<i32>
    %1 = cc.alloca i32
    cc.store %arg1, %1 : !cc.ptr<i32>
    %2 = quake.alloca !quake.ref
    quake.h %2 : (!quake.ref) -> ()
    %3 = cc.load %0 : !cc.ptr<i32>
    %4 = cc.load %1 : !cc.ptr<i32>
    %5 = cc.device_call @add(%3, %4) : (i32, i32) -> i32
    return %5 : i32
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %2 = cc.insert_value %1[0], %0 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %3 = cc.insert_value %2[1], %c0_i64 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %3 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  func.func @function_testKernel.testKernel.thunk(%arg0: !cc.ptr<i8>, %arg1: i1) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %0 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<!cc.struct<{i32, i32, i32}>>
    %1 = cc.sizeof !cc.struct<{i32, i32, i32}> : i64
    %2 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
    %3 = cc.compute_ptr %2[%1] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
    %4 = cc.compute_ptr %0[0] : (!cc.ptr<!cc.struct<{i32, i32, i32}>>) -> !cc.ptr<i32>
    %5 = cc.load %4 : !cc.ptr<i32>
    %6 = cc.compute_ptr %0[1] : (!cc.ptr<!cc.struct<{i32, i32, i32}>>) -> !cc.ptr<i32>
    %7 = cc.load %6 : !cc.ptr<i32>
    %8 = call @__nvqpp__mlirgen__function_testKernel.testKernel(%5, %7) : (i32, i32) -> i32
    %9 = cc.compute_ptr %0[2] : (!cc.ptr<!cc.struct<{i32, i32, i32}>>) -> !cc.ptr<i32>
    cc.store %8, %9 : !cc.ptr<i32>
    %10 = call @__nvqpp_zeroDynamicResult() : () -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %10 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  }
  )#";

  // Load the Kernel
  auto hdl = cudaq::driver::load_kernel(quake);

  // Manually setup the Thunk Args for the Kernel
  struct IntIntRetIntArgs {
    int i;
    int j;
    int k;
  };
  IntIntRetIntArgs thunkArgsConcrete{1, 2, 0};

  // Tell the controller to allocate data for the Thunk Args
  auto thunkArgsDevPtr = cudaq::driver::malloc(sizeof(IntIntRetIntArgs));

  // Set the Thunk Args data on the controller
  cudaq::driver::memcpy(thunkArgsDevPtr, &thunkArgsConcrete);

  // Launch the Kernel!
  cudaq::driver::launch_kernel(hdl, thunkArgsDevPtr);

  // Get the Thunk Args data back from the controller
  cudaq::driver::memcpy(&thunkArgsConcrete, thunkArgsDevPtr);

  // We expect 1+2 =3
  EXPECT_EQ(thunkArgsConcrete.k, 3);

  // Free the controller data
  cudaq::driver::free(thunkArgsDevPtr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Register the global test environment
  ::testing::AddGlobalTestEnvironment(new DriverTestEnvironment);

  // Run all tests
  return RUN_ALL_TESTS();
}
