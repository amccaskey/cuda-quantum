# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

def sample(kernel, *args, shots=1000):
    ctx = cudaq_runtime.ExecutionContext("sample", shots)
    cudaq_runtime.setExecutionContext(ctx)
    kernel(*args)
    res = ctx.result
    cudaq_runtime.resetExecutionContext()
    return res