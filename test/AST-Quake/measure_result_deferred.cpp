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

// Case 1: `return mz(q)` from a bool-returning kernel — `quake.discriminate`
// should appear exactly once, at the return boundary, and not inside an
// alloca/store/load round trip.
__qpu__ bool simple_mz() {
  cudaq::qubit q;
  h(q);
  return mz(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_simple_mz.
// CHECK:           %[[Q:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[Q]]
// CHECK:           %[[M:.*]] = quake.mz %[[Q]] : (!quake.ref) -> !quake.measure
// CHECK-NOT:       cc.alloca !quake.measure
// CHECK:           %[[B:.*]] = quake.discriminate %[[M]] : (!quake.measure) -> i1
// CHECK:           return %[[B]] : i1

// Case 2: `if (mz(q))` — `quake.discriminate` fires at the condition; no
// pointless stack slot around the rvalue handle.
__qpu__ void mz_if_conv() {
  cudaq::qubit q;
  h(q);
  if (mz(q))
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_mz_if_conv.
// CHECK:           %[[Q:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[Q]]
// CHECK:           %[[M:.*]] = quake.mz %[[Q]] : (!quake.ref) -> !quake.measure
// CHECK-NOT:       cc.alloca !quake.measure
// CHECK:           %[[B:.*]] = quake.discriminate %[[M]] : (!quake.measure) -> i1
// CHECK:           cc.if(%[[B]])

// Case 3: an `auto` local followed by a detector — the `!quake.measure`
// handle flows through as a pure SSA value without discrimination.
__qpu__ void auto_local_detector() {
  cudaq::qubit q;
  h(q);
  auto m = mz(q);
  cudaq::detector(m);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_auto_local_detector.
// CHECK:           %[[M:.*]] = quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK-NOT:       cc.alloca !quake.measure
// CHECK-NOT:       quake.discriminate
// CHECK:           qec.detector(%[[M]]) : (!quake.measure) -> ()
// CHECK:           return
