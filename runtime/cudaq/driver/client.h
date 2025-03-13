/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/MeasureCounts.h"
#include "common/ObserveResult.h"

namespace cudaq::driver {

// subtypes should model the communication channel
// e.g. shared_memory, pcie, rpc 
class client {
public:
  virtual sample_result sample() const = 0;
  virtual observe_result observe() const = 0; 
  virtual void run() const = 0; 
};
} // namespace cudaq::driver