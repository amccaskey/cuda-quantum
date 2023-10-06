# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .kernel.kernel_decorator import kernel
from .kernel.builder import make_kernel
from .kernel.qubit_qis import h, x, y, z, s, t, rx, ry, rz, r1, swap
from .runtime.sample import sample 
from .runtime.observe import observe
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

spin = cudaq_runtime.spin
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
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

