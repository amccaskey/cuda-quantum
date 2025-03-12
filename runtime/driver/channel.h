/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <vector> 
#include <string> 

namespace cudaq::driver {

class argument {};

class channel {
public:
  virtual void connect() const = 0;
  virtual std::size_t marshal(const std::vector<argument> &arguments) const = 0;
  virtual void invoke_function(const std::string &symbolName,
                               std::size_t argumentIdentifier) const = 0;
  virtual void free_arguments(std::size_t identifier) const = 0;

};

} // namespace cudaq::driver