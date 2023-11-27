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

def test_qreg_iter():
    @cudaq.kernel(jit=True)
    def foo(N:int):
        q = cudaq.qvector(N)
        for r in q:
            x(r)
    print(foo)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__foo(
# CHECK-SAME:                                     %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_8]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }