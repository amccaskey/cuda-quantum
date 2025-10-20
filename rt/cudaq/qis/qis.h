/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "gates.h"
#include "qvector.h"

#include "cudaq/platform/traits/simulator.h"

namespace cudaq {
struct base {};
struct ctrl {};
struct adj {};

void h(cudaq::qubit &q) {
  m_kernel_api->q_applicator(gates::h().getGate(), {}, {q.id()},
                             traits::operation_metadata{"h"});
}

template <typename mod = base, typename... QubitArgs>
void x(QubitArgs &&...qubits) {
  std::vector<std::size_t> ids{qubits.id()...};
  std::size_t target = ids.back();
  if constexpr (std::is_same_v<mod, ctrl>) {
    std::vector<std::size_t> ctrls(ids.begin(), ids.end() - 1);
    return m_kernel_api->q_applicator(gates::x().getGate(), ctrls, {target},
                                      traits::operation_metadata("x"));
  }

  for (auto &t : ids)
    m_kernel_api->q_applicator(gates::x().getGate(), {}, {t},
                               traits::operation_metadata("x"));
  return;
}
} // namespace cudaq
