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

def test_control_kernel():
    @cudaq.kernel(jit=True)
    def applyX(q:cudaq.qubit):
        x(q)
    
    @cudaq.kernel(jit=True, verbose=True)
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cudaq.control(applyX, [q[0]], q[1])
    
    print(bell)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__applyX(
# CHECK-SAME:                                        %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:     func.func @__nvqpp__mlirgen__bell() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.concat %[[VAL_1]] : (!quake.ref) -> !quake.veq<1>
# CHECK:           %[[VAL_3:.*]] = quake.relax_size %[[VAL_2]] : (!quake.veq<1>) -> !quake.veq<?>
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.apply @__nvqpp__mlirgen__applyX {{\[}}%[[VAL_3]]] %[[VAL_4]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }