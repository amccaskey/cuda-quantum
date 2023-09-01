/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector initializer_list constructor support

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void test() {
  cudaq::qubit q;
  std::vector<double> angle{M_PI_2, M_PI_4};
  ry(angle[0], q);
  ry(angle[1], q);
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test._Z4testv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.78539816339744828 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<f64 x 2>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][0] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_8]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_9]][1] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_11]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void test1() {
  cudaq::qubit q;
  std::vector<double> angle{M_PI_2};
  // clang-format off
  // CHECK:           %[[VAL_0:.*]] = arith.constant 1.5707963267948966 : f64
  // CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
  // CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.array<f64 x 1>
  // CHECK:           %[[VAL_3:.*]] = cc.compute_ptr %[[VAL_2]][0] : (!cc.ptr<!cc.array<f64 x 1>>) -> !cc.ptr<f64>
  // CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<f64>
  // clang-format on
  ry(angle[0], q);
}

__qpu__ void test3() {
  cudaq::qubit q;
  std::vector<double> angle{M_PI_2, M_PI_4, M_PI};
  // clang-format off
  // CHECK:           %[[VAL_0:.*]] = arith.constant 3.1415926535897931 : f64
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0.78539816339744828 : f64
  // CHECK:           %[[VAL_2:.*]] = arith.constant 1.5707963267948966 : f64
  // CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
  // CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<f64 x 3>
  // CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
  // CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<f64>
  // CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
  // CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<f64>
  // CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
  // CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
  // clang-format on

  ry(angle[0], q);
  ry(angle[1], q);
  ry(angle[2], q);
}
