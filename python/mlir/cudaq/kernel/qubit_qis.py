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
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit


def processQubitIds(opName, *args):
    qubitIds = []
    for a in args:
        if isinstance(a, qubit):
            qubitIds.append(a.id())
        elif isinstance(a, qvector) or isinstance(a, qview):
            [qubitIds.append(q.id()) for q in a]
        else:
            raise Exception(
                "invalid argument type passed to {}.__call__".format(opName))
    return qubitIds


class h(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class x(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class y(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class z(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class s(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class t(object):
    @staticmethod
    def __call__(*args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(target):
        if isinstance(target, qvector):
            for q in target:
                cudaq_runtime.applyQuantumOperation(
                    __class__.__name__, [], [], [q.id()], True)
            return
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [target.id()], True)


class rx(object):
    @staticmethod
    def __call__(parameter, *args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [parameter], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(parameter, target):
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], [], [target.id()])


class ry(object):
    @staticmethod
    def __call__(parameter, *args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [parameter], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(parameter, target):
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], [], [target.id()])


class rz(object):
    @staticmethod
    def __call__(parameter, *args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [parameter], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(parameter, target):
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], [], [target.id()])


class r1(object):
    @staticmethod
    def __call__(parameter, *args):
        [cudaq_runtime.applyQuantumOperation(__class__.__name__, [parameter], [], [
                                             q]) for q in processQubitIds(__class__.__name__, *args)]

    @staticmethod
    def ctrl(parameter, *args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [parameter], qubitIds[:len(qubitIds)-1], [qubitIds[-1]])

    @staticmethod
    def adj(parameter, target):
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [-parameter], [], [target.id()])


class swap(object):
    @staticmethod
    def __call__(first, second):
        print(__class__.__name__)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [first.id(), second.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], qubitIds[:len(qubitIds)-2], [qubitIds[-2], qubitIds[-1]])


def mz(*args):
    qubitIds = processQubitIds('mz', *args)
    [cudaq_runtime.measure(q) for q in qubitIds]


def my(*args):
    s.adj(*args)
    h()(*args)
    mz(*args)


def mx(*args):
    h()(*args)
    mz(*args)


def adjoint(kernel, *args):
    cudaq_runtime.startAdjointRegion()
    kernel(*args)
    cudaq_runtime.endAdjointRegion()


def control(kernel, controls, *args):
    cudaq_runtime.startCtrlRegion([c.id() for c in controls])
    kernel(*args)
    cudaq_runtime.endCtrlRegion(len(controls))


def compute_action(compute, action):
    compute()
    action()
    adjoint(compute)
