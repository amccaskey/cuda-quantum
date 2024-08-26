# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# // I can get the MLIR for adapt here. I can get the MLIR
# // for the input initial state. Now what I want to do is
# // merge these two and run the PySynthCallable pass,
# // then I'll have an overload for adapt that takes the full adapt circuit

from ..mlir.ir import Module
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

def mergeExternalMLIRWithKernel(kernel, moduleBStr : str): 
    '''
    Merge all FuncOps in the external MLIR Module represented by moduleBStr 
    into the MLIR representing the input CUDA-Q kernel. Return a new ModuleOp
    '''
    return cudaq_runtime.mergeExternalMLIR(kernel.module, moduleBStr)

def synthesizeCallableBlockArgument(moduleA : Module, funcName : str):
    '''
    Search for a FuncOp in moduleA that has a callable block argument and 
    replace with the function of the given name. Operate on the ModuleOp in place. 
    '''
    cudaq_runtime.synthPyCallable(moduleA, funcName)

def jitAndGetFunctionPointer(module, funcName):
    return cudaq_runtime.jitAndGetFunctionPointer(module, funcName)