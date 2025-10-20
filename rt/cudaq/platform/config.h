/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "qpu.h"

#include "cudaq/utils/heterogeneous_map.h"

#include "cudaq/platform/qpus/all.h"

#include <dlfcn.h>

namespace cudaq::config {

heterogeneous_map &get_qpu_config() {
  static heterogeneous_map m_qpu_config;
  return m_qpu_config;
}

#ifdef CUDAQ_TARGET_GPU_STATEVECTOR
using default_qpu = simulator::gpu::state_vector;
#elif defined(CUDAQ_TARGET_QUANTINUUM)
using default_qpu = remote::quantinuum;
#else
using default_qpu = simulator::gpu::state_vector;
// No nvq++ flag passed, no auto-generated static initialize provided
#endif

} // namespace cudaq::config
