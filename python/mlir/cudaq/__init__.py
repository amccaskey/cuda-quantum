# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import inspect 
from .language.ast_bridge import compile_to_quake

def test():
    print("TESTING")

class PyKernelDecorator(object):
    def __init__(self, function, jit=False, library_mode=False):
        self.kernelFunction = function
        self.mlirModule = None

        # JIT Quake
        self.jitQuake = jit
        # Library Mode
        self.library_mode = library_mode

        src = inspect.getsource(function)
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])
        self.module = ast.parse(self.funcSrc)
        import astpretty
        astpretty.pprint(self.module.body[0])
        if not self.library_mode:
            self.mlirModule = compile_to_quake(self.module)
            return
        
    def __str__(self):
        if not self.mlirModule == None:
            return str(self.mlirModule)
        
    def __call__(self, *args):
        
        # Library Mode, don't need Quake, just call the function
        self.kernelFunction(*args)

def kernel(function=None, jit=False, library_mode=True):
    """The `cudaq.kernel` represents the CUDA Quantum language function 
        attribute that programmers leverage to indicate the following function 
        is a CUDA Quantum kernel and should be compile and executed on 
        an available quantum coprocessor."""
    if function:
        return PyKernelDecorator(function)
    else:
        def wrapper(function):
            return PyKernelDecorator(function, jit, library_mode)
        return wrapper

