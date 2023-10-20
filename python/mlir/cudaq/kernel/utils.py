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

# By default and to keep things easier,
# we only deal with int==i64 and float=f64
def mlirTypeFromPyType(argType, ctx, argInstance = None):
    if argType == int:
        return IntegerType.get_signless(64, ctx)
    if argType == float:
        return F64Type.get(ctx)
    if argType == list or argType == np.ndarray:
        return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
    if argType == qvector:
        return quake.VeqType.get(ctx)
    if isinstance(argInstance, Callable):
        return cc.CallableType.get(ctx, argInstance.argTypes)


    raise RuntimeError("can not handle conversion of python type {} to mlir type.".format(argType))
