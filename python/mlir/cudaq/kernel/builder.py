# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from functools import partialmethod
import inspect
import sys
import random
import string
import numpy as np
import ctypes

from .quake_value import QuakeValue, mlirTypeFromPyType
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


def __generalOperation(opName, parameters, controls, target, isAdj=False, context=None):
    opCtor = getattr(quake, '{}Op'.format(opName.title()))

    if not quake.VeqType.isinstance(target.mlirValue.type):
        opCtor([], parameters, controls,
               [target.mlirValue], is_adj=isAdj)
        return

    # target is a VeqType
    size = quake.VeqType.getSize(target.mlirValue.type)
    if size:
        for i in range(size):
            extracted = quake.ExtractRefOp(
                quake.RefType.get(context), target.mlirValue, i)
            opCtor([], parameters, controls,
                   [extracted], is_adj=isAdj)
        return
    else:
        raise RuntimeError(
            'operation broadcasting on veq<?> not supported yet.')


def __singleTargetOperation(self, opName, target, isAdj=False):
    with self.insertPoint, self.loc:
        __generalOperation(opName, [], [], target,
                           isAdj=isAdj, context=self.ctx)


def __singleTargetControlOperation(self, opName, controls, target, isAdj=False):
    with self.insertPoint, self.loc:
        fwdControls = None
        if isinstance(controls, list):
            fwdControls = [c.mlirValue for c in controls]
        elif quake.RefType.isinstance(controls.mlirValue.type) or quake.VeqType.isinstance(controls.mlirValue.type):
            fwdControls = [controls.mlirValue]
        else:
            raise RuntimeError("invalid controls type for {}.", opName)

        __generalOperation(opName, [], fwdControls, target,
                           isAdj=isAdj, context=self.ctx)


def __singleTargetSingleParameterOperation(self, opName, parameter, target, isAdj=False):
    with self.insertPoint, self.loc:
        paramVal = None
        if isinstance(parameter, float):
            fty = mlirTypeFromPyType(float, self.ctx)
            paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
        else:
            paramVal = parameter.mlirValue
        __generalOperation(opName, [paramVal], [], target,
                           isAdj=isAdj, context=self.ctx)


def __singleTargetSingleParameterControlOperation(self, opName, parameter, controls, target, isAdj=False):
    with self.insertPoint, self.loc:
        fwdControls = None
        if isinstance(controls, list):
            fwdControls = [c.mlirValue for c in controls]
        elif quake.RefType.isinstance(controls.mlirValue.type) or quake.VeqType.isinstance(controls.mlirValue.type):
            fwdControls = [controls.mlirValue]
        else:
            raise RuntimeError("invalid controls type for {}.", opName)

        paramVal = parameter
        if isinstance(parameter, float):
            fty = mlirTypeFromPyType(float, self.ctx)
            paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))

        __generalOperation(opName, [paramVal], fwdControls, target,
                           isAdj=isAdj, context=self.ctx)


class PyKernel(object):

    def __init__(self, argTypeList):
        self.ctx = Context()
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        self.executionEngine = None
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.funcName = '__nvqpp__mlirgen____nvqppBuilderKernel_{}'.format(''.join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))
        self.name = self.funcName.removeprefix('__nvqpp__mlirgen__')
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
            self.arguments = [
                self.__createQuakeValue(b) for b in e.arguments
            ]
            self.argument_count = len(self.arguments)

            with InsertionPoint(e):
                func.ReturnOp([])

            self.insertPoint = InsertionPoint.at_block_begin(e)

    def __createQuakeValue(self, value):
        return QuakeValue(value, self)

    def dump(self):
        print(str(self.module))

    def raw_quake(self):
        return str(self.module)

    def __str__(self, canonicalize=False):
        pm = PassManager.parse(
            "builtin.module(canonicalize,cse)",
            context=self.ctx)
        cloned = cudaq_runtime.cloneModuleOp(self.module)
        pm.run(cloned)
        return str(cloned)

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
                        quake.AllocaOp(veqTy, size=sizeVal).result)
                else:
                    veqTy = quake.VeqType.get(self.ctx, size)
                    return self.__createQuakeValue(quake.AllocaOp(veqTy).result)

    def exp_pauli(self, theta, qubits, pauliWord):
        # FIXME implement for qubits...
        with self.insertPoint, self.loc:
            thetaVal = None
            if isinstance(theta, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                thetaVal = arith.ConstantOp(
                    fty, FloatAttr.get(fty, theta)).result
            else:
                thetaVal = theta.mlirValue

            retTy = cc.PointerType.get(self.ctx, cc.ArrayType.get(self.ctx,
                                                                  IntegerType.get_signless(8), int(len(pauliWord)+1)))
            slVal = cc.CreateStringLiteralOp(retTy, pauliWord)
            quake.ExpPauliOp(thetaVal, qubits.mlirValue, slVal)

    def swap(self, qubitA, qubitB):
        with self.insertPoint, self.loc:
            quake.SwapOp([], [], [], [qubitA.mlirValue, qubitB.mlirValue])

    def reset(self, target):
        with self.insertPoint, self.loc:
            if not quake.VeqType.isinstance(target.mlirValue.type):
                quake.ResetOp([], target.mlirValue)
                return

            # target is a VeqType
            size = quake.VeqType.getSize(target.mlirValue.type)
            if size:
                for i in range(size):
                    extracted = quake.ExtractRefOp(
                        quake.RefType.get(self.ctx), target.mlirValue, i).result
                    quake.ResetOp([], extracted)
                return
            else:
                raise RuntimeError(
                    'reset operation broadcasting on veq<?> not supported yet.')

    def mz(self, target, regName=None):
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
            res = quake.MzOp(retTy, [], [target.mlirValue]) if regName == None else quake.MzOp(
                retTy, [], target, StrAttr.get(regName, context=self.ctx))
            return self.__createQuakeValue(res.result)

    def mx(self, target, regName=None):
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
            res = quake.MxOp(retTy, [], [target.mlirValue]) if regName == None else quake.MzOp(
                retTy, [], target, StrAttr.get(regName, context=self.ctx))
            return self.__createQuakeValue(res.result)

    def my(self, target, regName=None):
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
            res = quake.MyOp(retTy, [], [target.mlirValue]) if regName == None else quake.MzOp(
                retTy, [], target, StrAttr.get(regName, context=self.ctx))
            return self.__createQuakeValue(res.result)

    def adjoint(self, otherKernel, *args):
        raise RuntimeError("adjoint not yet implemented")

    def control(self, otherKernel, control, *args):
        raise RuntimeError("control not yet implemented")

    def c_if(self, measurement, thenBlockCallable):
        raise RuntimeError("c_if not implemented yet.")

    def for_loop(self, start, stop, bodyCallable):
        raise RuntimeError("for_loop not yet implemented.")

    def __call__(self, *args):
        if len(args) != len(self.mlirArgTypes):
            raise RuntimeError("invalid number of arguments passed to kernel {} (passed {} but requires {})".format(
                self.funcName, len(args), len(self.mlirArgTypes)))
        # validate the arg types
        processedArgs = []
        for i, arg in enumerate(args):
            mlirType = mlirTypeFromPyType(type(arg), self.ctx)
            if mlirType != self.mlirArgTypes[i]:
                raise RuntimeError("invalid runtime arg type ({} vs {})".format(
                    mlirType, self.mlirArgTypes[i]))

            # Convert np arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        cudaq_runtime.pyAltLaunchKernel(self.name, self.module, *processedArgs)
        return


setattr(PyKernel, 'h', partialmethod(__singleTargetOperation, 'h'))
setattr(PyKernel, 'x', partialmethod(__singleTargetOperation, 'x'))
setattr(PyKernel, 'y', partialmethod(__singleTargetOperation, 'y'))
setattr(PyKernel, 'z', partialmethod(__singleTargetOperation, 'z'))
setattr(PyKernel, 's', partialmethod(__singleTargetOperation, 's'))
setattr(PyKernel, 't', partialmethod(__singleTargetOperation, 't'))
setattr(PyKernel, 'sdg', partialmethod(
    __singleTargetOperation, 's', isAdj=True))
setattr(PyKernel, 'tdg', partialmethod(
    __singleTargetOperation, 't', isAdj=True))

setattr(PyKernel, 'ch', partialmethod(__singleTargetControlOperation, 'h'))
setattr(PyKernel, 'cx', partialmethod(__singleTargetControlOperation, 'x'))
setattr(PyKernel, 'cy', partialmethod(__singleTargetControlOperation, 'y'))
setattr(PyKernel, 'cz', partialmethod(__singleTargetControlOperation, 'z'))
setattr(PyKernel, 'cs', partialmethod(__singleTargetControlOperation, 's'))
setattr(PyKernel, 'ct', partialmethod(__singleTargetControlOperation, 't'))

setattr(PyKernel, 'rx', partialmethod(
    __singleTargetSingleParameterOperation, 'rx'))
setattr(PyKernel, 'ry', partialmethod(
    __singleTargetSingleParameterOperation, 'ry'))
setattr(PyKernel, 'rz', partialmethod(
    __singleTargetSingleParameterOperation, 'rz'))
setattr(PyKernel, 'r1', partialmethod(
    __singleTargetSingleParameterOperation, 'r1'))

setattr(PyKernel, 'crx', partialmethod(
    __singleTargetSingleParameterControlOperation, 'rx'))
setattr(PyKernel, 'cry', partialmethod(
    __singleTargetSingleParameterControlOperation, 'ry'))
setattr(PyKernel, 'crz', partialmethod(
    __singleTargetSingleParameterControlOperation, 'rz'))
setattr(PyKernel, 'cr1', partialmethod(
    __singleTargetSingleParameterControlOperation, 'r1'))


def make_kernel(*args):
    kernel = PyKernel([*args])
    if len([*args]) == 0:
        return kernel

    return kernel, *kernel.arguments
