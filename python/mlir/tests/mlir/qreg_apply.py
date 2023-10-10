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


def test_kernel_qreg():
    """
    Test the `cudaq.Kernel` on each non-parameterized single qubit gate.
    Each gate is applied to both qubits in a 2-qubit register.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    # Apply each gate to entire register.
    # Test both with and without keyword arguments.
    kernel.h(target=qreg)
    kernel.x(target=qreg)
    kernel.y(target=qreg)
    kernel.z(qreg)
    kernel.t(qreg)
    kernel.s(qreg)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.t %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.t %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.s %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.s %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
