/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraBaseQPU.h"

namespace {

/// @brief The `QuEraRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class QuEraRemoteRESTQPU : public cudaq::QuEraBaseQPU {
public:
  QuEraRemoteRESTQPU(const cudaq::v2::platform_metadata &m) : QuEraBaseQPU(m) {}
  QuEraRemoteRESTQPU(QuEraRemoteRESTQPU &&) = delete;
  virtual ~QuEraRemoteRESTQPU() = default;
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      QuEraRemoteRESTQPU, "quera",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<QuEraRemoteRESTQPU>(m);
      })
};
} // namespace

CUDAQ_REGISTER_EXTENSION_TYPE(QuEraRemoteRESTQPU)
