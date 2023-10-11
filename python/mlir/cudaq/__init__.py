# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .kernel.kernel_decorator import kernel
from .kernel.builder import make_kernel, QuakeValue
from .kernel.qubit_qis import h, x, y, z, s, t, rx, ry, rz, r1, swap, mx, my, mz, adjoint, control, compute_action
from .runtime.sample import sample 
from .runtime.observe import observe
from .runtime.state import get_state
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

# Primitive Types
spin = cudaq_runtime.spin
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
SpinOperator = cudaq_runtime.SpinOperator
Pauli = cudaq_runtime.Pauli

# Optimizers + Gradients
optimizers = cudaq_runtime.optimizers
gradients = cudaq_runtime.gradients

# Runtime Functions
set_target = cudaq_runtime.set_target
reset_target = cudaq_runtime.reset_target 
set_random_seed = cudaq_runtime.set_random_seed

# Noise Modeling
KrausChannel = cudaq_runtime.KrausChannel
NoiseModel = cudaq_runtime.NoiseModel
DepolarizationChannel = cudaq_runtime.DepolarizationChannel
AmplitudeDampingChannel = cudaq_runtime.AmplitudeDampingChannel
PhaseFlipChannel = cudaq_runtime.PhaseFlipChannel 
BitFlipChannel = cudaq_runtime.BitFlipChannel

# Functions
sample_async = cudaq_runtime.sample_async
observe_async = cudaq_runtime.observe_async 

# Quantum Instruction Set
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

from ._packages import *

if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ["CUDAQ_DYNLIBS"] = f"{custatevec_path}:{cutensornet_path}"
    except:
        print("Could not find a suitable cuQuantum Python package.")
        pass