# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from .utils import __isBroadcast, __createArgumentSet 

def __broadcastSample(kernel, *args, shots_count=0):
    argSet = __createArgumentSet(*args)
    N = len(argSet)
    results = []
    for i, a in enumerate(argSet):
        ctx = cudaq_runtime.ExecutionContext('sample', shots_count)
        ctx.totalIterations = N
        ctx.batchIteration = i
        cudaq_runtime.setExecutionContext(ctx)
        kernel(*a)
        res = ctx.result
        cudaq_runtime.resetExecutionContext()
        results.append(res)

    return results

def sample(kernel, *args, shots_count=1000, noise_model=None):
    if noise_model != None:
        cudaq_runtime.set_noise(noise_model)
    
    if __isBroadcast(kernel, *args):
        res = __broadcastSample(kernel, *args, shots_count=shots_count)
    else:
        ctx = cudaq_runtime.ExecutionContext("sample", shots_count)
        cudaq_runtime.setExecutionContext(ctx)
        kernel(*args)
        res = ctx.result
        cudaq_runtime.resetExecutionContext()

    if noise_model != None:
        cudaq_runtime.unset_noise()

    return res
