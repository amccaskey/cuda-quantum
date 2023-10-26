# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from ..kernel.kernel_builder import PyKernel
from .utils import __isBroadcast, __createArgumentSet 
from mlir_cudaq.dialects import quake, cc

def __broadcastObserve(kernel, spin_operator, *args, shots_count=0):
    argSet = __createArgumentSet(*args)
    N = len(argSet)
    results = []
    for i, a in enumerate(argSet):
        ctx = cudaq_runtime.ExecutionContext('observe', shots_count)
        ctx.totalIterations = N
        ctx.batchIteration = i
        ctx.setSpinOperator(spin_operator)
        cudaq_runtime.setExecutionContext(ctx)
        kernel(*a)
        res = ctx.result
        cudaq_runtime.resetExecutionContext()
        results.append(cudaq_runtime.ObserveResult(
            ctx.getExpectationValue(), spin_operator, res))

    return results


def observe(kernel, spin_operator, *args, shots_count=0, noise_model=None):
    if noise_model != None:
        cudaq_runtime.set_noise(noise_model)

    results = None
    if __isBroadcast(kernel, *args):
        results = __broadcastObserve(kernel, spin_operator, *args, shots_count=shots_count)
    else: 
        localOp = spin_operator
        localOp = cudaq_runtime.SpinOperator()
        if isinstance(spin_operator, list):
            for o in spin_operator:
                localOp += o
            localOp -= cudaq_runtime.SpinOperator()
        else:
            localOp = spin_operator

        ctx = cudaq_runtime.ExecutionContext('observe', shots_count)
        ctx.setSpinOperator(localOp)
        cudaq_runtime.setExecutionContext(ctx)
        kernel(*args)
        res = ctx.result
        cudaq_runtime.resetExecutionContext()

        observeResult = cudaq_runtime.ObserveResult(
            ctx.getExpectationValue(), localOp, res)
        if not isinstance(spin_operator, list):
            return observeResult

        results = []
        for op in spin_operator:
            results.append(cudaq_runtime.ObserveResult(
                observeResult.expectation(op), op, observeResult.counts(op)))

    if noise_model != None:
        cudaq_runtime.unset_noise()

    return results
