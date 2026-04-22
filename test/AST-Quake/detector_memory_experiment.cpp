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

// Repetition-code memory experiment from detectors.md §example-memory:
// cross-round detectors compare the same stabilizer across consecutive
// rounds. `prev_s0` / `prev_s1` carry the previous-round outcomes across
// loop iterations, so they sit in memory (cc.alloca !quake.measure with
// store/load) rather than as SSA values — a deliberate trade-off since
// SSA rebinding does not survive loop back-edges without loop-carried
// values.
__qpu__ void memory_experiment(int nRounds) {
  cudaq::qvector data(3);
  cudaq::qubit anc0, anc1;
  cudaq::measure_result prev_s0, prev_s1;

  for (int r = 0; r < nRounds; r++) {
    cx(data[0], anc0);
    cx(data[1], anc0);
    cx(data[1], anc1);
    cx(data[2], anc1);

    auto s0 = mz(anc0);
    auto s1 = mz(anc1);
    reset(anc0);
    reset(anc1);

    if (r > 0) {
      cudaq::detector(prev_s0, s0);
      cudaq::detector(prev_s1, s1);
    }
    prev_s0 = s0;
    prev_s1 = s1;
  }
  auto readout = mz(data);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_memory_experiment.
// CHECK:           quake.alloca !quake.veq<3>
// CHECK:           quake.alloca !quake.ref
// CHECK:           quake.alloca !quake.ref

// Two default-constructed measure_result locals — one stack slot per handle.
// CHECK:           cc.alloca !quake.measure
// CHECK:           cc.alloca !quake.measure

// Inside the round loop, two mz calls returning opaque handles.
// CHECK:           %[[S0:.*]] = quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK:           %[[S1:.*]] = quake.mz {{.*}} : (!quake.ref) -> !quake.measure

// `if (r > 0)` emits two cross-round detectors whose prev operand is a
// cc.load from the prev_s? slot and whose curr operand is the fresh mz SSA.
// CHECK:           cc.if
// CHECK:             %[[P0:.*]] = cc.load {{.*}} : !cc.ptr<!quake.measure>
// CHECK:             qec.detector(%[[P0]], %[[S0]]) : (!quake.measure, !quake.measure) -> ()
// CHECK:             %[[P1:.*]] = cc.load {{.*}} : !cc.ptr<!quake.measure>
// CHECK:             qec.detector(%[[P1]], %[[S1]]) : (!quake.measure, !quake.measure) -> ()

// `prev_s0 = s0;` stores the new handle back into the slot for the next round.
// CHECK:           cc.store %[[S0]], {{.*}} : !cc.ptr<!quake.measure>
// CHECK:           cc.store %[[S1]], {{.*}} : !cc.ptr<!quake.measure>

// Final data-qubit readout.
// CHECK:           quake.mz {{.*}} : (!quake.veq<3>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// clang-format on
