# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import inspect
import sys
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

def mlirTypeFromPyType(argType, ctx):
    if argType == int:
        return IntegerType.get_signless(64)
    if argType == float:
        return F64Type.get()
    if argType == list:
        return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
    if argType == qvector:
        return quake.VeqType.get()

def runtimeArgToCtypePointer(arg):
    if isinstance(arg, float):
        c_float_p = ctypes.c_double * 1
        return c_float_p(arg)
    if isinstance(arg, int):
        c_int_p = ctypes.c_int * 1
        return c_int_p(arg)
    
    raise Exception("{} runtime arg type is not implemented yet.".format(type(arg)))

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
        with self.ctx, InsertionPoint(self.module.body), self.loc:
            mlirArgTypes = [
                mlirTypeFromPyType(argType, self.ctx) for argType in argTypeList
            ]
            f = func.FuncOp('{}'.format('kernelBuilder'), (mlirArgTypes, []),
                            loc=self.loc)
            f.attributes.__setitem__('llvm.emit_c_interface', UnitAttr.get())
            e = f.add_entry_block()
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
                paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter)).result
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
        with self.insertPoint, self.loc:
            raise Exception("exp_pauli not yet implemented.")

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
        if self.executionEngine == None:
            # FIXME This should be retrieved from the target
            pm = PassManager.parse(
                "builtin.module(canonicalize,cse,func.func(quake-add-deallocs),quake-to-qir)",
                context=self.ctx)
            pm.run(self.module)
            self.executionEngine = ExecutionEngine(self.module)
        
        if len([*args]) > 0:
            cTypeArgs = [runtimeArgToCtypePointer(a) for a in args]
            self.executionEngine.invoke('kernelBuilder', *cTypeArgs)
        else:
            self.executionEngine.invoke('kernelBuilder')

def make_kernel(*args):
    kernel = PyKernel([*args])
    if len([*args]) == 0:
        return kernel

    return kernel, *kernel.argsAsQuakeValues
