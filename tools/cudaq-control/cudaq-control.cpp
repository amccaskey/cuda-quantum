/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/driver/controller/controller.h"
#include "llvm/Support/CommandLine.h"
#include <thread>

static llvm::cl::opt<std::string>
    controllerType("controller",
                   llvm::cl::desc("The name of the controller process protocol "
                                  "(default rpc_controller)."),
                   llvm::cl::init("rpc_controller"));

// FIXME Make the controller load compilers / targets as plugins dynamically

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "cudaq-controller process\n");
  cudaq::driver::initialize(controllerType, argc, argv);
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (cudaq::driver::should_stop())
      break;
  }
  return 0;
}