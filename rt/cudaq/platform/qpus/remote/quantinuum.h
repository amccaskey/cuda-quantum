/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/traits/basic.h"
#include "cudaq/platform/traits/sampling.h"

// Remove this
#include <stdio.h>

namespace cudaq::remote {
class quantinuum
    : public qpu<quantinuum, traits::remote, traits::sample_trait<quantinuum>> {
public:
  std::string name() const { return "remote::quantinuum"; }

  sample_result sample(std::size_t num_shots, const std::string &kernel_name,
                       const std::function<void()> &wrapped_kernel) {
    printf(" - sample called\n");
    return sample_result();
  }
};
} // namespace cudaq::remote
