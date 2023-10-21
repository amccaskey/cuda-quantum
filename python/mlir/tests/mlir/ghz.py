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
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64{{\[}}%[[VAL_6]] : i64]
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_11]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_15:.*]] = cc.loop while ((%[[VAL_16:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_17]](%[[VAL_16]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i64>
# CHECK:             %[[VAL_21:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_23:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_22]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_21]]] %[[VAL_23]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_25]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

    @cudaq.kernel(jit=True)
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits-1)):
            x.ctrl(qubit, qubits[i+1])
    print(simple)
    
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__simple(
# CHECK-SAME:                                        %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_7:.*]] = quake.subveq %[[VAL_4]], %[[VAL_3]], %[[VAL_6]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_8:.*]] = quake.veq_size %[[VAL_7]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.struct<{i64, !quake.ref}>{{\[}}%[[VAL_8]] : i64]
# CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_8]] : i64
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = cc.undef !cc.struct<{i64, !quake.ref}>
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, !quake.ref}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_17:.*]] = cc.insert_value %[[VAL_13]], %[[VAL_14]][0] : (!cc.struct<{i64, !quake.ref}>, i64) -> !cc.struct<{i64, !quake.ref}>
# CHECK:             %[[VAL_18:.*]] = cc.insert_value %[[VAL_15]], %[[VAL_17]][1] : (!cc.struct<{i64, !quake.ref}>, !quake.ref) -> !cc.struct<{i64, !quake.ref}>
# CHECK:             cc.store %[[VAL_18]], %[[VAL_16]] : !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             cc.continue %[[VAL_13]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_8]] : i64
# CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_24]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, !quake.ref}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_27:.*]] = cc.extract_value %[[VAL_26]][0] : (!cc.struct<{i64, !quake.ref}>) -> i64
# CHECK:             %[[VAL_28:.*]] = cc.extract_value %[[VAL_26]][1] : (!cc.struct<{i64, !quake.ref}>) -> !quake.ref
# CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_27]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_30:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_29]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_28]]] %[[VAL_30]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_31:.*]]: i64):
# CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_32]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }