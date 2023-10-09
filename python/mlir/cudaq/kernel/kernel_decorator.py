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
from ..language.ast_bridge import compile_to_quake

class PyKernelDecorator(object):

    def __init__(self, function, verbose=False, library_mode=True):
        self.kernelFunction = function
        self.mlirModule = None
        self.executionEngine = None
        self.verbose = verbose
       
        # Library Mode
        self.library_mode = library_mode

        # FIXME if target == remote, must have library_mode = False
        
        src = inspect.getsource(function)
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])
        self.module = ast.parse(self.funcSrc)
        if verbose and importlib.find_loader('astpretty') is not None:
            import astpretty
            astpretty.pprint(self.module.body[0])
        if not self.library_mode:
            # FIXME Run any Python AST Canonicalizers (e.g. list comprehension to for loop, 
            # range-based for loop to for loop, etc.)
            #
            # FIXME Update to return required FuncOps (other kernels) not present 
            # in this module
            self.mlirModule = compile_to_quake(self.module, verbose=self.verbose)
            # Add the other FuncOps to this module (FIXME need global dict of PyKernelDecorators)
           
            
            return

    def __str__(self):
        if not self.mlirModule == None:
            return str(self.mlirModule)
        return "{cannot print this kernel}"

    def __call__(self, *args):

        # Library Mode, don't need Quake, just call the function
        if self.library_mode:
            self.kernelFunction(*args)
            return
        
        # If the Target is remote, then pass the Quake code as 
        # is to platform.launchKernel()
        #
        # If the Target is a simulator, lower to QIR and 
        # just create the ExecutionEngine
        # Lower the code to QIR and Execute
        if self.executionEngine == None:
            pm = PassManager.parse(
                "builtin.module(canonicalize,cse,func.func(quake-add-deallocs),quake-to-qir)",
                context=self.mlirModule.context)
            pm.run(self.mlirModule)
            self.executionEngine = ExecutionEngine(self.mlirModule, 2, [
                '/workspaces/cuda-quantum/builds/gcc-12-debug/lib/libnvqir.so',
                '/workspaces/cuda-quantum/builds/gcc-12-debug/lib/libnvqir-qpp.so'
            ])
            self.executionEngine.invoke(self.kernelFunction.__name__)


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
