/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector<int> initializer_list constructor support

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void testInt() {
  cudaq::qarray<3> q;
  std::vector<int> index{0, 1, 2};
  ry(M_PI_2, q[index[0]]);
  ry(M_PI_2, q[index[1]]);
  ry(M_PI_2, q[index[2]]);
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testInt._Z7testIntv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<i32 x 3>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.cast signed %[[VAL_11]] : (i32) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_12]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_14]]) %[[VAL_13]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_15:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_15]] : !cc.ptr<f64>
// CHECK:           %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = cc.cast signed %[[VAL_17]] : (i32) -> i64
// CHECK:           %[[VAL_19:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_18]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_20]]) %[[VAL_19]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_21:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_21]] : !cc.ptr<f64>
// CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_22]] : !cc.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = cc.cast signed %[[VAL_23]] : (i32) -> i64
// CHECK:           %[[VAL_25:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_24]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_21]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_26]]) %[[VAL_25]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
