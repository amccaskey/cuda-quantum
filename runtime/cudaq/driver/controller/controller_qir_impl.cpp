/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/driver/controller/target.h"
#include <cstddef>
#include <map>

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define QIS_FUNCTION_NAME(GATENAME) CONCAT(__quantum__qis__, GATENAME)
#define QIS_FUNCTION_CTRL_NAME(GATENAME)                                       \
  CONCAT(CONCAT(__quantum__qis__, GATENAME), __ctl)
#define QIS_FUNCTION_BODY_NAME(GATENAME)                                       \
  CONCAT(CONCAT(__quantum__qis__, GATENAME), __body)

using Qubit = std::size_t;
using Result = std::size_t;
static const Result ResultZeroVal = false;
static const Result ResultOneVal = true;
inline Result *ResultZero = const_cast<Result *>(&ResultZeroVal);
inline Result *ResultOne = const_cast<Result *>(&ResultOneVal);

namespace cudaq::driver {
target *m_target;
static thread_local std::map<Qubit *, Result *> measQB2Res;
static thread_local std::map<Result *, Qubit *> measRes2QB;
static thread_local std::map<Result *, Result> measRes2Val;
void setTarget(target *t) { m_target = t; }
} // namespace cudaq::driver

extern "C" {
using namespace cudaq::driver;

void __quantum__qis__r1__body(double angle, Qubit *qubit) {
  m_target->apply_opcode("r1", {angle}, {reinterpret_cast<std::size_t>(qubit)});
}

#define ONE_QUBIT_TEMPLATE(NAME)                                               \
  void QIS_FUNCTION_BODY_NAME(NAME)(Qubit * qubit) {                           \
    m_target->apply_opcode(#NAME, {}, {reinterpret_cast<std::size_t>(qubit)}); \
  }

ONE_QUBIT_TEMPLATE(x)
ONE_QUBIT_TEMPLATE(y)
ONE_QUBIT_TEMPLATE(z)
ONE_QUBIT_TEMPLATE(h)
ONE_QUBIT_TEMPLATE(s)
ONE_QUBIT_TEMPLATE(t)

void __quantum__qis__cnot__body(Qubit *q, Qubit *r) {
  m_target->apply_opcode(
      "x", {},
      {reinterpret_cast<std::size_t>(q), reinterpret_cast<std::size_t>(r)});
}

Result *__quantum__qis__mz__body(Qubit *q, Result *r) {
  auto res = m_target->measure_z(reinterpret_cast<std::size_t>(q));
  measQB2Res[q] = r;
  measRes2QB[r] = q;
  auto qI = reinterpret_cast<std::size_t>(q);
  measRes2Val[r] = res;
  return res ? ResultOne : ResultZero;
}

bool __quantum__qis__read_result__body(Result *r) {
  auto iter = measRes2Val.find(r);
  if (iter != measRes2Val.end())
    return iter->second;
  return ResultZeroVal;
}

void __quantum__rt__result_record_output(Qubit *q, int8_t *) {}
}
