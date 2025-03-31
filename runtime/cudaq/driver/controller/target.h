/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/utils/extension_point.h"

namespace cudaq::driver {

enum class opcode { r1 };

class target : public extension_point<target> {
public:
  virtual void apply_opcode(opcode, const std::vector<double> &,
                            const std::vector<std::size_t> &) = 0;
};

} // namespace cudaq::driver