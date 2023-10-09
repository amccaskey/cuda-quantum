# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import inspect
import sys
import random
import string
import numpy as np
import ctypes

from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.execution_engine import *
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.dialects import builtin, func, arith

from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

qvector = cudaq_runtime.qvector

# Goal here is to reproduce the current kernel_builder in Python

# We need static initializers to run in the CAPI ExecutionEngine,
# so here we run a simple JIT compile at global scope
with Context():
    module = Module.parse(
        r"""
llvm.func @none() {
  llvm.return
}""")
    ExecutionEngine(module)


def mlirTypeFromPyType(argType, ctx):
    if argType == int:
        return IntegerType.get_signless(64)
    if argType == float:
        return F64Type.get()
    if argType == list:
        return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
    if argType == qvector:
        return quake.VeqType.get()

class QuakeValue(object):

    def __init__(self, mlirValue, pyKernel):
        self.mlirValue = mlirValue
        self.pyKernel = pyKernel
        self.ctx = self.pyKernel.ctx

    def __str__(self):
        return str(self.mlirValue)

    def __getitem__(self, idx):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            if cc.StdvecType.isinstance(self.mlirValue.type):
                eleTy = mlirTypeFromPyType(float, self.ctx)
                arrPtrTy = cc.PointerType.get(self.ctx,
                                              cc.ArrayType.get(self.ctx, eleTy))
                vecPtr = cc.StdvecDataOp(arrPtrTy, self.mlirValue).result
                elePtrTy = cc.PointerType.get(self.ctx, eleTy)
                eleAddr = None
                i64Ty = IntegerType.get_signless(64)
                if isinstance(idx, QuakeValue):
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [idx.mlirValue],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                elif isinstance(idx, int):
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [],
                        DenseI32ArrayAttr.get([idx], context=self.ctx))
                loaded = cc.LoadOp(eleAddr.result)
                return QuakeValue(loaded.result, self.pyKernel)

            if quake.VeqType.isinstance(self.mlirValue.type):
                processedIdx = None
                if isinstance(idx, QuakeValue):
                    processedIdx = idx.mlirValue
                elif isinstance(idx, int):
                    i64Ty = IntegerType.get_signless(64)
                    processedIdx = arith.ConstantOp(i64Ty,
                                                    IntegerAttr.get(i64Ty,
                                                                    idx)).result
                else:
                    raise Exception("invalid idx passed to QuakeValue.")
                op = quake.ExtractRefOp(quake.RefType.get(self.ctx),
                                        self.mlirValue,
                                        -1,
                                        index=processedIdx)
                return QuakeValue(op.result, self.pyKernel)

        raise Exception("invalid getitem")


class PyKernel(object):

    def __init__(self, argTypeList):
        self.ctx = Context()
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        self.executionEngine = None
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.funcName = '__nvqpp__mlirgen____nvqppBuilderKernel{}'.format(''.join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))
        self.funcNameEntryPoint = self.funcName + '_entryPointRewrite'
        attr = DictAttr.get({self.funcName: StringAttr.get(
            self.funcNameEntryPoint, context=self.ctx)}, context=self.ctx)
        self.module.operation.attributes.__setitem__(
            'quake.mangled_name_map', attr)

        with self.ctx, InsertionPoint(self.module.body), self.loc:
            self.mlirArgTypes = [
                mlirTypeFromPyType(argType, self.ctx) for argType in argTypeList
            ]

            self.funcOp = func.FuncOp(self.funcName, (self.mlirArgTypes, []),
                                      loc=self.loc)
            self.funcOp.attributes.__setitem__(
                'cudaq-entrypoint', UnitAttr.get())
            e = self.funcOp.add_entry_block()
            self.argsAsQuakeValues = [
                self.__createQuakeValue(b) for b in e.arguments
            ]

            with InsertionPoint(e):
                func.ReturnOp([])

            self.insertPoint = InsertionPoint.at_block_begin(e)

    def __createQuakeValue(self, value):
        return QuakeValue(value, self)

    def __str__(self):
        return str(self.module)

    def qalloc(self, size=None):
        with self.insertPoint, self.loc:
            if size == None:
                qubitTy = quake.RefType.get(self.ctx)
                return self.__createQuakeValue(quake.AllocaOp(qubitTy).result)
            else:
                if isinstance(size, QuakeValue):
                    veqTy = quake.VeqType.get(self.ctx)
                    sizeVal = size.mlirValue
                    return self.__createQuakeValue(
                        quake.AllocaOp(veqTy, size=sizeVal))
                else:
                    veqTy = quake.VeqType.get(self.ctx, size)
                    return self.__createQuakeValue(quake.AllocaOp(veqTy).result)

    def __applyQuantumOp(self, opName, parameters, controls, targets):
        opCtor = getattr(quake, '{}Op'.format(opName.title()))
        opCtor([], parameters, controls, targets)

    def h(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def ch(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def x(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def cx(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def y(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def cy(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def z(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def cz(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def s(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def cs(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def t(self, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [target.mlirValue])

    def ct(self, controls, target):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3][-1], [],
                                  [c.mlirValue for c in controls] if isinstance(
                                      controls, list) else [controls.mlirValue],
                                  [target.mlirValue])

    def sdg(self, target):
        with self.insertPoint, self.loc:
            quake.SOp([], [], [], [target.mlirValue], is_adj=UnitAttr.get())

    def tdg(self, controls, target):
        with self.insertPoint, self.loc:
            quake.TOp([], [], [], [target.mlirValue], is_adj=UnitAttr.get())

    def rx(self, parameter, target):
        with self.insertPoint, self.loc:
            if isinstance(parameter, QuakeValue):
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.mlirValue], [],
                                      [target.mlirValue])
            elif isinstance(parameter, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                paramVal = arith.ConstantOp(
                    fty, FloatAttr.get(fty, parameter)).result
                self.__applyQuantumOp(inspect.stack()[0][3], [paramVal], [],
                                      [target.mlirValue])

    def ry(self, parameter, target):
        with self.insertPoint, self.loc:
            if isinstance(parameter, QuakeValue):
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.mlirValue], [],
                                      [target.mlirValue])
            elif isinstance(parameter, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.result], [],
                                      [target.mlirValue])

    def rz(self, parameter, target):
        with self.insertPoint, self.loc:
            if isinstance(parameter, QuakeValue):
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.mlirValue], [],
                                      [target.mlirValue])
            elif isinstance(parameter, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.result], [],
                                      [target.mlirValue])

    def r1(self, parameter, target):
        with self.insertPoint, self.loc:
            if isinstance(parameter, QuakeValue):
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.mlirValue], [],
                                      [target.mlirValue])
            elif isinstance(parameter, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
                self.__applyQuantumOp(inspect.stack()[0][3], [parameter.result], [],
                                      [target.mlirValue])

    def exp_pauli(self, theta, qubits, pauliWord):
        # FIXME implement for qubits...
        with self.insertPoint, self.loc:
            thetaVal = theta
            if isinstance(theta, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                thetaVal = arith.ConstantOp(
                    fty, FloatAttr.get(fty, theta)).result

            retTy = cc.PointerType.get(self.ctx, cc.ArrayType.get(self.ctx,
                                                                  IntegerType.get_signless(8), int(len(pauliWord)+1)))
            slVal = cc.CreateStringLiteralOp(retTy, pauliWord)
            quake.ExpPauliOp(thetaVal, qubits.mlirValue, slVal)

    def swap(self, qubitA, qubitB):
        with self.insertPoint, self.loc:
            self.__applyQuantumOp(inspect.stack()[0][3], [], [],
                                  [qubitA.mlirValue, qubitB.mlirValue])

    def adjoint(self, otherKernel, *args):
        raise Exception("adjoint not yet implemented")

    def control(self, otherKernel, control, *args):
        raise Exception("control not yet implemented")

    def c_if(self, measurement, thenBlockCallable):
        raise Exception("c_if not implemented yet.")

    def for_loop(self, start, stop, bodyCallable):
        raise Exception("for_loop not yet implemented.")

    def __call__(self, *args):
        if len(args) != len(self.mlirArgTypes):
            raise Exception("invalid number of arguments passed to kernel {} (passed {} but requires {})".format(
                self.funcName, len(args), len(self.mlirArgTypes)))
        cudaq_runtime.pyAltLaunchKernel(
            self.funcName.removeprefix(
                '__nvqpp__mlirgen__'), self.module, *args)
        return


def make_kernel(*args):
    kernel = PyKernel([*args])
    if len([*args]) == 0:
        return kernel

    return kernel, *kernel.argsAsQuakeValues
