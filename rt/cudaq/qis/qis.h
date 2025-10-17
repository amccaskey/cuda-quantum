#pragma once

#include "gates.h"
#include "qudit.h"

#include "cudaq/platform/traits/simulator.h"

namespace cudaq {

void h(cudaq::qubit &q) {
  m_kernel_api->q_applicator(gates::h().getGate(), {}, {q.id()},
                             traits::operation_metadata{"h"});
}
} // namespace cudaq
