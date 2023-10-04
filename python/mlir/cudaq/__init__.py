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
from .language.ast_bridge import compile_to_quake

def test():
    print("TESTING")

class PyKernelDecorator(object):
    def __init__(self, function, verbose=False, library_mode=False):
        self.kernelFunction = function
        self.mlirModule = None

        # Library Mode
        self.library_mode = library_mode

        src = inspect.getsource(function)
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])
        self.module = ast.parse(self.funcSrc)
        if verbose and importlib.find_loader('astpretty') is not None:
            import astpretty
            astpretty.pprint(self.module.body[0])
        if not self.library_mode:
            self.mlirModule = compile_to_quake(self.module, verbose=verbose)
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
        
        # Lower the code to QIR and Execute
        # pm = PassManager.parse("builtin.module(canonicalize,cse)", context=bridge.ctx)
        # pm.run(bridge.module)


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

