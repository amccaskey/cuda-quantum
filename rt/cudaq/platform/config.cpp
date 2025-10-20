/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "config.h"
#include "cudaq/utils/logger.h"

namespace cudaq::config {
__attribute__((constructor)) void __cudaq__startup__config() {
  auto *sym = dlsym(RTLD_DEFAULT, "initialize_qpu");
  if (!sym) {
    info("no initialize_qpu() function available (dlerror = {})",
         std::string(dlerror()));
    return;
  }
  
  info("initialize_qpu() function detected, configuring default qpu.");
  auto *functor = reinterpret_cast<void (*)(heterogeneous_map &)>(sym);
  functor(get_qpu_config());
}
} // namespace cudaq::config
