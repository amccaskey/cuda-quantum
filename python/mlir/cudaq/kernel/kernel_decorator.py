# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import importlib
import inspect
from typing import Callable
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.execution_engine import *
from mlir_cudaq.dialects import quake, cc
from .ast_bridge import compile_to_quake
from .utils import mlirTypeFromPyType
from .analysis import MidCircuitMeasurementAnalyzer
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime


class PyKernelDecorator(object):

    def __init__(self, function, verbose=False, library_mode=True, jit=False, module=None, kernelName=None):
        self.kernelFunction = function
        self.module = None if module == None else module
        self.executionEngine = None
        self.verbose = verbose
        self.name = kernelName if kernelName != None else self.kernelFunction.__name__

        # Library Mode
        self.library_mode = library_mode
        if jit == True:
            self.library_mode = False

        if self.kernelFunction is not None:
            src = inspect.getsource(self.kernelFunction)
            leadingSpaces = len(src) - len(src.lstrip())
            self.funcSrc = '\n'.join(
                [line[leadingSpaces:] for line in src.split('\n')])
            self.astModule = ast.parse(self.funcSrc)
            if verbose and importlib.util.find_spec('astpretty') is not None:
                import astpretty
                astpretty.pprint(self.astModule.body[0])

            # Need to build up the arg types here
            self.signature = inspect.getfullargspec(
                self.kernelFunction).annotations

            # Run analyzers and attach metadata (only have 1 right now)
            analyzer = MidCircuitMeasurementAnalyzer()
            analyzer.visit(self.astModule)
            self.metadata = {
                'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}

            if not self.library_mode:
                # FIXME Run any Python AST Canonicalizers (e.g. list comprehension to for loop)
                self.module, self.argTypes = compile_to_quake(
                    self.astModule, verbose=self.verbose)

            return

    def __str__(self):
        if not self.module == None:
            return str(self.module)
        return "{cannot print this kernel}"

    def __call__(self, *args):

        # Library Mode, don't need Quake, just call the function
        if self.library_mode:
            self.kernelFunction(*args)
            return

        # validate the arg types
        processedArgs = []
        callableNames = []
        for i, arg in enumerate(args):
            mlirType = mlirTypeFromPyType(
                type(arg), self.module.context, argInstance=arg)
            if not cc.CallableType.isinstance(mlirType) and mlirType != self.argTypes[i]:
                raise RuntimeError("invalid runtime arg type ({} vs {})".format(
                    mlirType, self.argTypes[i]))
            if cc.CallableType.isinstance(mlirType):
                # Assume this is a PyKernelDecorator
                callableNames.append(arg.name)

            # Convert np arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                if arg.ndim != 1:
                    raise RuntimeError('CUDA Quantum kernels only support 1D numpy array arguments.')
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        cudaq_runtime.pyAltLaunchKernel(
            self.name, self.module, *processedArgs, callable_names=callableNames)


def kernel(function=None, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA Quantum language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA Quantum kernel and should be compile and executed on 
    an available quantum coprocessor.
    """
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper
