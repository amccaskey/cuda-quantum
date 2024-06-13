/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
// #include "cudaq/platform.h"

#include "cudaq/host_config.h"

extern "C" {
extern target_config __nvqpp__get_target_config();
}

bool cudaq::__nvqpp__MeasureResultBoolConversion(int result) {
  // auto &platform = get_platform();
  // auto *ctx = platform.get_exec_ctx();
  // if (ctx && ctx->name == "tracer")
  //   ctx->registerNames.push_back("");
  // return result == 1;
  return false;
}

namespace cudaq {
static std::unique_ptr<ExecutionManager> manager;
ExecutionManager *getExecutionManager() {
  if (manager)
    return manager.get();

  auto config = __nvqpp__get_target_config();
  if (ExecutionManager::is_registered(config.executionManager)) {
    manager = ExecutionManager::get(config.executionManager);
    return manager.get();
  }

  throw std::runtime_error("[execution_manager] invalid manager requested (" +
                           std::string(config.executionManager) + ")");
}

void tearDownBeforeMPIFinalize() {
  cudaq::getExecutionManager()->tearDownBeforeMPIFinalize();
}
void setRandomSeed(std::size_t seed) {
  cudaq::getExecutionManager()->setRandomSeed(seed);
}
} // namespace cudaq