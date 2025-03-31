/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/driver/controller/target.h"
#include <cstddef>

using Qubit = std::size_t;

namespace cudaq::driver {
std::unique_ptr<target> m_target;
}

extern "C" {
void __quantum__qis__r1__body(double angle, Qubit *qubit) {
  using namespace cudaq::driver;
  //   m_target->apply_opcode(opcode::r1, {angle},
  //                          {reinterpret_cast<std::size_t>(qubit)});
}

void __quantum__qis__cnot__body(Qubit *, Qubit *) {}
std::size_t *__quantum__qis__mz__body(Qubit *q, Qubit *r) { return nullptr; }

void __quantum__rt__result_record_output(Qubit *q, int8_t *) {}
}