/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quantum_platform.h"

namespace cudaq {


class EntanglerPlatform : public cudaq::quantum_platform {
public:
  EntanglerPlatform() {
   // construct the QPUs based on the target information
  }

  /// @brief Set the target backend. Here we have an opportunity
  /// to know the -qpu QPU target we are running on. This function will
  /// read in the qpu configuration file and search for the PLATFORM_QPU
  /// variable, and if found, will change from the DefaultQPU to the QPU subtype
  /// specified by that variable.
  void setTargetBackend(const std::string &backend) override {
   
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(cudaq::EntanglerPlatform, entangler)
