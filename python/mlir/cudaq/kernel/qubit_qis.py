# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Library Mode QIS

# qubit, qvector, qview all defined in C++
# (better tracking of construction / destruction)

import inspect
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime


class h(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])


class x(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]], True)

class y(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]], True)


class z(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
  
    @staticmethod
    def adj(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]], True)


class s(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]], True)

class t(object):
    @staticmethod
    def __call__(target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]], True)


class rx(object):
    @staticmethod
    def __call__(parameter, target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], [], [target.id()])

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

class ry(object):
    @staticmethod
    def __call__(parameter, target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], [], [target.id()])

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

class rz(object):
    @staticmethod
    def __call__(parameter, target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], [], [target.id()])

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

class r1(object):
    @staticmethod
    def __call__(parameter, target):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], [], [target.id()])

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])
    
    @staticmethod
    def adj(parameter, *args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

class swap(object):
    @staticmethod
    def __call__(first, second):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [first.id(), second.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = [a.id() for a in args]
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-2], [qubitIds[-2], qubitIds[-1]])
        
    