/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "remote/quantinuum.h"
#include "simulation/gpu/state_vector.h"

// Maybe this is how mqpu could work?
namespace cudaq::simulation {
template <typename ConcreteSimQpu>
class mqpu {
  std::vector<ConcreteSimQpu> qpus;
};

} // namespace cudaq::simulation
