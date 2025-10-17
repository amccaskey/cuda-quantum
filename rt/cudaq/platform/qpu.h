/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>

#include "type_traits.h"

namespace cudaq {

template <typename Derived, typename... Traits>
class qpu : public Traits... {
public:
  std::string name() const { return crtp_cast<Derived>(this)->name(); }

};



} // namespace cudaq

#include "qpus/all.h"
