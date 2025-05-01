/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__super(
// CHECK-SAME:      %[[ARG0:.*]]: f64) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = cc.alloca f64
// CHECK:           cc.store %[[ARG0]], %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           quake.rx (%[[VAL_3]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = arith.divf %[[VAL_4]], %[[VAL_0]] : f64
// CHECK:           %[[VAL_6:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_7]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_9:.*]] = quake.discriminate %[[VAL_8]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_9]] : i1
// CHECK:         }

struct super {
  bool operator()(double inputPi) __qpu__ {
    cudaq::qubit q;
    rx(inputPi, q);
    ry(inputPi / 2.0, q);
    return mz(q);
  }
};
