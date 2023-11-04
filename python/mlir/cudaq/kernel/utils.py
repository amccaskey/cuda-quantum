# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from mlir_cudaq.dialects import quake, cc

from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
import numpy as np
from typing import Callable

qvector = cudaq_runtime.qvector
qubit = cudaq_runtime.qubit
qreg = qvector

nvqppPrefix = '__nvqpp__mlirgen__'

# Keep a global registry of all kernel FuncOps
# keyed on their name (without __nvqpp__mlirgen__ prefix)
globalKernelRegistry = {}

# Keep a global registry of all kernel Python ast modules
# keyed on their name (without __nvqpp__mlirgen__ prefix)
globalAstRegistry = {}

# Keep a global registry of all registered custom
# unitary operations.
globalRegisteredUnitaries = {}


# By default and to keep things easier,
# we only deal with int==i64 and float=f64
def mlirTypeFromPyType(argType, ctx, **kwargs): #argInstance=None, argTypeToCompareTo=None):

    if argType == int:
        return IntegerType.get_signless(64, ctx)
    if argType == float:
        return F64Type.get(ctx)
    if argType == bool:
        return IntegerType.get_signless(1, ctx)
    if argType == complex:
        return ComplexType.get(mlirTypeFromPyType(float, ctx))
    
    if argType in [list, np.ndarray]:
        if 'argInstance' not in kwargs:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
        
        argInstance = kwargs['argInstance']
        argTypeToCompareTo = kwargs['argTypeToCompareTo']

        if isinstance(argInstance[0], int):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(int,ctx))
        if isinstance(argInstance[0], float):
            # check if we are comparing to a complex...
            eleTy = cc.StdvecType.getElementType(argTypeToCompareTo)
            if ComplexType.isinstance(eleTy):
                raise RuntimeError("invalid runtime argument to kernel. list[complex] required, but list[float] provided.")
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
        if isinstance(argInstance[0], complex):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(complex, ctx))
        
    if argType == qvector or argType == qreg:
        return quake.VeqType.get(ctx)
    if argType == qubit:
        return quake.RefType.get(ctx)

    if 'argInstance' in kwargs:
        argInstance = kwargs['argInstance']
        if isinstance(argInstance, Callable):
            return cc.CallableType.get(ctx, argInstance.argTypes)

    raise RuntimeError(
        "can not handle conversion of python type {} to mlir type.".format(
            argType))

def mlirTypeToPyType(argType):

    if IntegerType.isinstance(argType):
        if IntegerType(argType).width == 1:
            return bool
        return int 
    
    if F64Type.isinstance(argType):
        return float 
    
    if ComplexType.isinstance(argType):
        return complex 
    
    if cc.StdvecType.isinstance(argType):
        eleTy = cc.StdvecType.getElementType(argType)
        if IntegerType.isinstance(argType):
            if IntegerType(argType).width == 1:
                return list[bool]
            return list[int] 
        if F64Type.isinstance(argType):
            return list[float] 
        if ComplexType.isinstance(argType):
            return list[complex] 

    raise RuntimeError("unhandled mlir-to-pytype {}".format(argType))    

    
    