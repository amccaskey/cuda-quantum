# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../../python_packages/cudaq pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_iterate_list_init():
    
    @cudaq.kernel(jit=True)
    def kernel(x : float): 
        q = cudaq.qvector(4)
        for i in [0, 1, 2, 3]:
            x = x + i
            ry(x, q[i%4])
        
    print(kernel)
    kernel(1.2)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_6:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_6]] : !cc.ptr<f64>
# CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.array<i64 x 4>
# CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_8]][0] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_4]], %[[VAL_9]] : !cc.ptr<i64>
# CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_10]] : !cc.ptr<i64>
# CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_2]], %[[VAL_11]] : !cc.ptr<i64>
# CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           %[[VAL_13:.*]] = cc.stdvec_init %[[VAL_8]], %[[VAL_5]] : (!cc.ptr<!cc.array<i64 x 4>>, i64) -> !cc.stdvec<i64>
# CHECK:           %[[VAL_14:.*]] = cc.stdvec_size %[[VAL_13]] : (!cc.stdvec<i64>) -> i64
# CHECK:           %[[VAL_15:.*]] = cc.loop while ((%[[VAL_16:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_14]] : i64
# CHECK:             cc.condition %[[VAL_17]](%[[VAL_16]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_19:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:             %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_19]]{{\[}}%[[VAL_18]]] : (!cc.ptr<i64>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_21:.*]] = cc.load %[[VAL_20]] : !cc.ptr<i64>
# CHECK:             %[[VAL_22:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_23:.*]] = arith.sitofp %[[VAL_21]] : i64 to f64
# CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f64
# CHECK:             cc.store %[[VAL_24]], %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_25:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_26:.*]] = arith.remui %[[VAL_21]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_27:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_26]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_25]]) %[[VAL_27]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_28:.*]]: i64):
# CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_29]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }