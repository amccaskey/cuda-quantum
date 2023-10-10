# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from ..kernel.builder import PyKernel
from mlir_cudaq.dialects import quake, cc


def __isBroadcast(kernel, *args):
    # kernel could be a PyKernel or PyKernelDecorator
    if isinstance(kernel, PyKernel):
        argTypes = kernel.mlirArgTypes
        if len(argTypes) == 0 or len(args) == 0:
            return False

        firstArg = args[0]
        firstArgTypeIsStdvec = cc.StdvecType.isinstance(argTypes[0])
        if isinstance(firstArg, list) and not firstArgTypeIsStdvec:
            return True

        if hasattr(firstArg, "shape"):
            shape = firstArg.shape
            if len(shape) == 1 and not firstArgTypeIsStdvec:
                return True

            if len(shape) == 2:
                return True

        return False


def __createArgumentSet(*args):
    nArgSets = len(args[0])
    argSet = []
    for j in range(nArgSets):
        currentArgs = [0 for i in range(len(args))]
        for i, arg in enumerate(args):

            if isinstance(arg, list):
                currentArgs[i] = arg[j]
            
            if hasattr(arg, "tolist"):
                shape = arg.shape 
                if len(shape) == 2:
                    currentArgs[i] = arg[j].tolist()
                else:
                    currentArgs[i] = arg.tolist()[j]
        
        argSet.append(tuple(currentArgs))
    return argSet 
