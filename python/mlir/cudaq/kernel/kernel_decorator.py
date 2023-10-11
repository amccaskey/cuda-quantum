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
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.execution_engine import *
from mlir_cudaq.dialects import quake, cc
from ..language.ast_bridge import compile_to_quake
from .quake_value import mlirTypeFromPyType
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

class PyKernelDecorator(object):

    def __init__(self, function, verbose=False, library_mode=True, jit=False):
        self.kernelFunction = function
        self.module = None
        self.executionEngine = None
        self.verbose = verbose
        self.name = self.kernelFunction.__name__ 

        # Library Mode
        self.library_mode = library_mode
        if jit == True:
            self.library_mode = False

        src = inspect.getsource(function)
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])
        self.astModule = ast.parse(self.funcSrc)
        if verbose and importlib.util.find_spec('astpretty') is not None:
            import astpretty
            astpretty.pprint(self.astModule.body[0])
        
        if not self.library_mode:
            # FIXME Run any Python AST Canonicalizers (e.g. list comprehension to for loop, 
            # range-based for loop to for loop, etc.)
            #
            # FIXME Update to return required FuncOps (other kernels) not present 
            # in this module
            self.module, self.argTypes = compile_to_quake(self.astModule, verbose=self.verbose)
            # Add the other FuncOps to this module (FIXME need global dict of PyKernelDecorators)
        else: 
            # Need to build up the arg types here
            self.signature = inspect.getfullargspec(function).annotations
            
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
        for i, arg in enumerate(args):
            mlirType = mlirTypeFromPyType(type(arg), self.module.context)
            if mlirType != self.argTypes[i]:
                raise RuntimeError("invalid runtime arg type ({} vs {})".format(
                    mlirType, self.argTypes[i]))

            # Convert np arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        cudaq_runtime.pyAltLaunchKernel(self.name, self.module, *processedArgs)

def kernel(function=None, **kwargs):
    """The `cudaq.kernel` represents the CUDA Quantum language function 
        attribute that programmers leverage to indicate the following function 
        is a CUDA Quantum kernel and should be compile and executed on 
        an available quantum coprocessor."""
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper
