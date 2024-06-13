/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/host_config.h"

extern "C" {
target_config __nvqpp__get_target_config() {
  return {CUDAQ_TARGET_SIMULATOR, CUDAQ_EXECUTION_MANAGER};
}
}