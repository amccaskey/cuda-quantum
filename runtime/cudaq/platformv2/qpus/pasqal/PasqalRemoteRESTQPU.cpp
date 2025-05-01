/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalBaseQPU.h"

namespace {

/// @brief The `PasqalRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class PasqalRemoteRESTQPU : public cudaq::PasqalBaseQPU {
public:
  PasqalRemoteRESTQPU(const cudaq::v2::platform_metadata &m)
      : PasqalBaseQPU(m) {}
  PasqalRemoteRESTQPU(PasqalRemoteRESTQPU &&) = delete;
  virtual ~PasqalRemoteRESTQPU() = default;

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      PasqalRemoteRESTQPU, "pasqal",
      static std::unique_ptr<qpu> create(
          const cudaq::v2::platform_metadata &m) {
        return std::make_unique<PasqalRemoteRESTQPU>(m);
      })
};

CUDAQ_REGISTER_EXTENSION_TYPE(PasqalRemoteRESTQPU)
} // namespace
