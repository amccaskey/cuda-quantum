/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/driver/controller/target.h"
#include "cudaq/utils/extension_point.h"

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::driver {

class quake_compiler : public extension_point<quake_compiler> {
protected:
  std::unique_ptr<target> m_target;

public:
  virtual void initialize(const config::TargetConfig &) = 0;
  virtual std::size_t compile(const std::string &quake) = 0;

  // Launch the kernel thunk, results are posted to the thunkArgs pointer
  virtual void launch(std::size_t kernelHandle, void *thunkArgs) = 0;
};

} // namespace cudaq::driver