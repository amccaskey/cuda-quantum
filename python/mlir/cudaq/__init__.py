# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .kernel.kernel_decorator import kernel
from .kernel.builder import make_kernel
from .kernel.qubit_qis import h, x, y, z, s, t, rx, ry, rz, r1, swap, mx, my, mz, adjoint, control, compute_action
from .runtime.sample import sample 
from .runtime.observe import observe
from .runtime.state import get_state
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

spin = cudaq_runtime.spin
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
optimizers = cudaq_runtime.optimizers
set_target = cudaq_runtime.set_target
reset_target = cudaq_runtime.reset_target 
SpinOperator = cudaq_runtime.SpinOperator

h = h()
x = x()
y = y()
z = z()
s = s()
t = t()
rx = rx()
ry = ry()
rz = rz()
r1 = r1()
swap = swap()
