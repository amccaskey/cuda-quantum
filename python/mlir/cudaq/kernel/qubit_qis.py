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
    """Return the qubit unique ID integers for a general tuple of 
    kernel arguments, where all arguments are assumed to be qubit-like 
    (qvector, qview, qubit)."""
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
    """The Hadmard operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Pauli X operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Pauli Y operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Pauli Z operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The S operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The T operation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Rx rotation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Ry rotation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The Rz rotation. Can be controlled on any number of qubits via the `ctrl` method."""
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
    """The general phase rotation. Can be controlled on any number of qubits via the `ctrl` method."""

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
    """The swap operation. Can be controlled on any number of qubits via the `ctrl` method."""
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


def mz(*args, register_name = ''):
    """Measure the qubit along the z-axis."""
    qubitIds = processQubitIds('mz', *args)
    res = [cudaq_runtime.measure(q, register_name) for q in qubitIds]
    if len(res) == 1: return res[0] 
    else: return res


def my(*args, register_name = ''):
    """Measure the qubit along the y-axis."""
    s.adj(*args)
    h()(*args)
    return mz(*args, register_name)


def mx(*args, register_name = ''):
    """Measure the qubit along the x-axis."""
    h()(*args)
    return mz(*args, register_name)


def adjoint(kernel, *args):
    """Apply the adjoint of the given kernel at the provided runtime arguments."""
    cudaq_runtime.startAdjointRegion()
    kernel(*args)
    cudaq_runtime.endAdjointRegion()


def control(kernel, controls, *args):
    """Apply the general control version of the given kernel at the provided runtime arguments."""
    cudaq_runtime.startCtrlRegion([c.id() for c in controls])
    kernel(*args)
    cudaq_runtime.endCtrlRegion(len(controls))


def compute_action(compute, action):
    """Apply the U V U^dag given U and V unitaries."""
    compute()
    action()
    adjoint(compute)
