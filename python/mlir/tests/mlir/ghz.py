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

def test_ghz():
    @cudaq.kernel(jit=True)
    def ghz(N:int):
        q = cudaq.qvector(N)
        h(q[0])
        for i in range(N-1):
            x.ctrl(q[i], q[i+1])

    print(ghz)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz(
# CHECK-SAME:                                     %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_6:.*]] = math.absi %[[VAL_5]] : i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.array<i64 x ?>{{\[}}%[[VAL_6]] : i64]
# CHECK:           cc.scope {
# CHECK:             %[[VAL_8:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_2]], %[[VAL_8]] : !cc.ptr<i64>
# CHECK:             cc.loop while {
# CHECK:               %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i64>
# CHECK:               %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_5]] : i64
# CHECK:               cc.condition %[[VAL_10]]
# CHECK:             } do {
# CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i64>
# CHECK:               %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               cc.store %[[VAL_11]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:               cc.continue
# CHECK:             } step {
# CHECK:               %[[VAL_13:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i64>
# CHECK:               %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : i64
# CHECK:               cc.store %[[VAL_14]], %[[VAL_8]] : !cc.ptr<i64>
# CHECK:             }
# CHECK:           }
# CHECK:           cc.scope {
# CHECK:             %[[VAL_15:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_2]], %[[VAL_15]] : !cc.ptr<i64>
# CHECK:             cc.loop while {
# CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i64>
# CHECK:               %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_6]] : i64
# CHECK:               cc.condition %[[VAL_17]]
# CHECK:             } do {
# CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i64>
# CHECK:               %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i64>
# CHECK:               %[[VAL_21:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_1]] : i64
# CHECK:               %[[VAL_23:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_22]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_21]]] %[[VAL_23]] : (!quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue
# CHECK:             } step {
# CHECK:               %[[VAL_24:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i64>
# CHECK:               %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
# CHECK:               cc.store %[[VAL_25]], %[[VAL_15]] : !cc.ptr<i64>
# CHECK:             }
# CHECK:           }
# CHECK:           return
# CHECK:         }