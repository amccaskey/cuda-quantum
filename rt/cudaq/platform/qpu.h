/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>

#include "cudaq/utils/heterogeneous_map.h"
#include "cudaq/utils/type_traits.h"

namespace cudaq {

template <typename Derived, typename... Traits>
class qpu : public Traits... {
protected:
  heterogeneous_map current_config;
  void configure_qpu(const heterogeneous_map &config) {
    current_config = config;
    return crtp_cast<Derived>(this)->configure(config);
  }

public:
  qpu() {}
  qpu(const heterogeneous_map &config) { configure_qpu(config); }
  std::string name() const { return crtp_cast<Derived>(this)->name(); }
  heterogeneous_map &get_configuration() { return current_config; }
};

} // namespace cudaq

