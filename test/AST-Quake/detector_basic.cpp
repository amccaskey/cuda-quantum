/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// clang-format on

#include <cudaq.h>

// Smallest possible detector kernel: two qubits, two measurements, two
// detectors. Verifies that:
//   * `mz` returns `!quake.measure` (deferred discrimination).
//   * `auto m = mz(q)` does NOT emit a cc.alloca / cc.store / cc.load cycle
//     — the handle flows as a pure SSA value.
//   * `cudaq::detector(...)` lowers to `qec.detector(!quake.measure ...)`.
__qpu__ void single_det() {
  cudaq::qubit q0, q1;
  h(q0);
  h(q1);
  auto m0 = mz(q0);
  auto m1 = mz(q1);
  cudaq::detector(m0);
  cudaq::detector(m0, m1);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_single_det.
// CHECK:           %[[Q0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[Q1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[Q0]]
// CHECK:           quake.h %[[Q1]]
// CHECK:           %[[M0:.*]] = quake.mz %[[Q0]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[M1:.*]] = quake.mz %[[Q1]] : (!quake.ref) -> !quake.measure
// CHECK-NOT:       cc.alloca !quake.measure
// CHECK-NOT:       cc.store {{.*}}!quake.measure
// CHECK-NOT:       cc.load {{.*}}!quake.measure
// CHECK:           qec.detector(%[[M0]]) : (!quake.measure) -> ()
// CHECK:           qec.detector(%[[M0]], %[[M1]]) : (!quake.measure, !quake.measure) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on
