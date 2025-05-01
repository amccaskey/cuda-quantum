/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector support

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
struct simple_double_rotation {
  auto operator()(std::vector<double> theta) __qpu__ {
    auto size = theta.size();
    bool empty = theta.empty();
    cudaq::qvector q(1);
    int test = q.size();
    rx(theta[0], q[0]);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL: func.func @__nvqpp__mlirgen__simple_double_rotation
// CHECK-SAME: (%[[VAL_0:.*]]: !cc.stdvec<f64>{{.*}}) attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_4:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_7:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<i1>
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_9:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
// CHECK-DAG:       %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
// CHECK-DAG:       %[[VAL_13:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           quake.rx (%[[VAL_12]]) %[[VAL_13]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_14:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct simple_float_rotation {
  auto operator()(std::vector<float> theta) __qpu__ {
    int size = theta.size();
    bool empty = theta.empty();
    cudaq::qvector q(1);
    rx(abs(theta[0]), q[0]);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL: func.func @__nvqpp__mlirgen__simple_float_rotation
// CHECK-SAME:      %[[ARG0:.*]]: !cc.stdvec<f32>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = cc.stdvec_size %[[ARG0]] : (!cc.stdvec<f32>) -> i64
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (i64) -> i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_size %[[ARG0]] : (!cc.stdvec<f32>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_6:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i1>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_data %[[ARG0]] : (!cc.stdvec<f32>) -> !cc.ptr<!cc.array<f32 x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<f32 x ?>>) -> !cc.ptr<f32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f32>
// CHECK:           %[[VAL_11:.*]] = math.absf %[[VAL_10]] : f32
// CHECK:           %[[VAL_12:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_11]], %[[VAL_12]] : !cc.ptr<f32>
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_7]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f32>
// CHECK:           quake.rx (%[[VAL_14]]) %[[VAL_13]] : (f32, !quake.ref) -> ()
// CHECK:           %[[VAL_15:.*]] = quake.mz %[[VAL_7]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct difficult_symphony {
  auto operator()(std::vector<float> theta) __qpu__ {
    float *firstData = theta.data();
    cudaq::qvector q(1);
    rx(firstData[0], q[0]);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__difficult_symphony(
// CHECK-SAME:      %[[ARG0:.*]]: !cc.stdvec<f32>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = cc.stdvec_data %[[ARG0]] : (!cc.stdvec<f32>) -> !cc.ptr<!cc.array<f32 x ?>>
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.ptr<!cc.array<f32 x ?>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.ptr<!cc.array<f32 x ?>>>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<!cc.ptr<!cc.array<f32 x ?>>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f32 x ?>>) -> !cc.ptr<f32>
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<f32>
// CHECK:           quake.rx (%[[VAL_6]]) %[[VAL_5]] : (f32, !quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

int main() {
  std::vector<double> vec_args = {0.63};

  auto counts = cudaq::sample(simple_double_rotation{}, vec_args);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", counts.expectation());

  std::vector<float> float_args = {0.64};

  auto float_counts = cudaq::sample(simple_float_rotation{}, float_args);
  float_counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : float_counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  auto bob_counts = cudaq::sample(difficult_symphony{}, float_args);
  bob_counts.dump();

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", float_counts.expectation());
  return 0;
}
