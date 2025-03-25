/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/driver/channel.h"

namespace cudaq::driver {

// --- Controller Server API ---
void connect(const std::string& cfg); 

// Allocate memory on devId. Return a unique handle
std::size_t malloc(std::size_t size, std::size_t devId); 
void free(std::size_t handle); 
void memcpy_to(std::size_t handle, std::vector<char>& data, std::size_t size);
std::vector<char> memcpy_from(std::size_t handle, std::size_t size);

} // namespace cudaq::driver