# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import ast
import inspect
import os, os.path
from ._packages import *

if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ["CUDAQ_DYNLIBS"] = f"{custatevec_path}:{cutensornet_path}"
    except:
        import importlib.util
        if not importlib.util.find_spec("cuda-quantum") is None:
            print("Could not find a suitable cuQuantum Python package.")
        pass

from .domains import chemistry
from .language.analysis import MidCircuitMeasurementAnalyzer
from .language.ast_bridge import compile_to_quake


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
        analyzer = MidCircuitMeasurementAnalyzer()
        analyzer.visit(self.module)
        self.metadata = {'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}
        if not self.library_mode:
            self.mlirModule = compile_to_quake(self.module)
            return
        
    def __str__(self):
        if not self.mlirModule == None:
            return str(self.mlirModule)
        
    def __call__(self, *args):
        target = get_target()
        # remoteTarget = target.is_remote() 
        # emulatedTarget = target.is_emulated()

        # First, is this a pure-device kernel? 
        # isPureDevice = False
        # for arg in args:
        #     if isinstance(arg, qvector) or isinstance(arg, qubit):
        #         isPureDevice = True 

        # FIXME Could be a classical function call from a 
        # cuda quantum kernel. Check here 

        # if isPureDevice:
        #     self.kernelFunction(*args)
        #     return

        # Remote QPU (needs Quake), Quake Requested via Library Mode
        # if remoteTarget or emulatedTarget or (self.jitQuake and self.library_mode):
        #     initRuntimeKernelExec()
        #     self.kernelFunction(*args)
        #     tearDownRuntimeKernelExec()
        #     return 

        # Library Mode == False, JIT Compile Python to Quake
        # FIXME This is to be done. 



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


from ._pycudaq import *

initKwargs = {}
if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]

if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]

initialize_cudaq(**initKwargs)
