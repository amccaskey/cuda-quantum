# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import sys
from collections import deque
import numpy as np
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.dialects import builtin, func, arith, math

nvqppPrefix = '__nvqpp__mlirgen__'

# Keep a global registry of all kernel FuncOps
# keyed on their name (without __nvqpp__mlirgen__ prefix)
globalKernelRegistry = {}

# Keep a global registry of all kernel Python ast modules
# keyed on their name (without __nvqpp__mlirgen__ prefix)
globalAstRegistry = {}


class PyASTBridge(ast.NodeVisitor):

    def __init__(self, **kwargs):
        self.valueStack = deque()
        self.ctx = Context()
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.symbolTable = {}
        self.increment = 0
        self.verbose = 'verbose' in kwargs and kwargs['verbose']

    def getVeqType(self, size=None):
        if size == None:
            return quake.VeqType.get(self.ctx)
        return quake.VeqType.get(self.ctx, size)

    def getRefType(self):
        return quake.RefType.get(self.ctx)

    def isQuantumType(self, ty):
        return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(ty)

    def getIntegerType(self, width=64):
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        return IntegerAttr.get(type, value)

    def getFloatType(self):
        return F64Type.get()

    def getFloatAttr(self, type, value):
        return FloatAttr.get(type, value)

    def getConstantFloat(self, value):
        ty = self.getFloatType()
        return arith.ConstantOp(ty, self.getFloatAttr(ty, value)).result

    def getConstantInt(self, value):
        ty = self.getIntegerType(64)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def pushValue(self, value):
        if self.verbose:
            print('{}push {}'.format(self.increment * ' ', value))
        self.increment += 2
        self.valueStack.append(value)

    def popValue(self):
        val = self.valueStack.pop()
        self.increment -= 2
        if self.verbose:
            print('{}pop {}'.format(self.increment * ' ', val))
        return val
    
    def mlirTypeFromAnnotation(self, annotation):
        if annotation == None:
            raise RuntimeError(
                'cudaq.kernel functions must have argument type annotations.')

        if hasattr(annotation, 'attr') and annotation.value.id == 'cudaq':
            if annotation.attr == 'qview' or annotation.attr == 'qvector':
                return self.getVeqType()
            if annotation.attr == 'qubit':
                return self.getRefType()

        if annotation.id == 'int':
            return self.getIntegerType(64)
        elif annotation.id == 'float':
            return F64Type.get()
        else:
            raise RuntimeError(
                '{} is not a supported type yet.'.format(annotation.id))

    def createInvariantForLoop(self, endVal, bodyBuilder, startVal=None, stepVal=None, isDecrementing=False):
        startVal = self.getConstantInt(0) if startVal == None else startVal
        stepVal = self.getConstantInt(1) if stepVal == None else stepVal

        iTy = self.getIntegerType()
        inputs = [startVal]
        resultTys = [iTy]

        loop = cc.LoopOp(resultTys, inputs, BoolAttr.get(False))

        whileBlock = Block.create_at_start(loop.whileRegion, [iTy])
        with InsertionPoint(whileBlock):
            condPred = IntegerAttr.get(
                iTy, 2) if not isDecrementing else IntegerAttr.get(iTy, 4)
            cc.ConditionOp(arith.CmpIOp(
                condPred, whileBlock.arguments[0], endVal).result, whileBlock.arguments)

        bodyBlock = Block.create_at_start(loop.bodyRegion, [iTy])
        with InsertionPoint(bodyBlock):
            bodyBuilder(bodyBlock.arguments[0])
            cc.ContinueOp(bodyBlock.arguments)

        stepBlock = Block.create_at_start(loop.stepRegion, [iTy])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], stepVal).result
            cc.ContinueOp([incr])

        loop.attributes.__setitem__('invariant', UnitAttr.get())
        return

    def generic_visit(self, node):
        for field, value in reversed(list(ast.iter_fields(node))):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_FunctionDef(self, node):
        with self.ctx, InsertionPoint(self.module.body), self.loc:

            # Get the arg types and arg names
            # this will throw an error if the types aren't annotated
            self.argTypes = [
                self.mlirTypeFromAnnotation(arg.annotation) for arg in node.args.args
            ]
            # Get the argument names
            argNames = [arg.arg for arg in node.args.args]

            self.name = node.name

            # the full function name in MLIR is __nvqpp__mlirgen__ + the function name
            fullName = nvqppPrefix + node.name

            # Create the FuncOp
            f = func.FuncOp(fullName, (self.argTypes, []),
                            loc=self.loc)

            # Set this kernel as an entry point if the arg types are classical only
            def isQuantumTy(ty): return quake.RefType.isinstance(
                ty) or quake.VeqType.isinstance(ty)
            areQuantumTypes = [isQuantumTy(ty) for ty in self.argTypes]
            if True not in areQuantumTypes:
                f.attributes.__setitem__('cudaq-entrypoint', UnitAttr.get())

            # Create the entry block
            entry = f.add_entry_block()

            # Add the block args to the symbol table
            blockArgs = entry.arguments
            for i, b in enumerate(blockArgs):
                self.symbolTable[argNames[i]] = b

            # Set the insertion point to the start of the entry block
            with InsertionPoint(entry):
                # Visit the function
                self.generic_visit(node)
                # Add the return operation
                ret = func.ReturnOp([])

            if True not in areQuantumTypes:
                attr = DictAttr.get({fullName: StringAttr.get(
                    fullName+'_entryPointRewrite', context=self.ctx)}, context=self.ctx)
                self.module.operation.attributes.__setitem__(
                    'quake.mangled_name_map', attr)

            globalKernelRegistry[node.name] = f
            self.symbolTable.clear()

    def visit_Assign(self, node):
        if self.verbose:
            print('[Visit Assign {}]'.format(ast.unparse(node)))
        self.generic_visit(node)

        rhsVal = self.popValue()
        if self.isQuantumType(rhsVal.type):
            self.symbolTable[node.targets[0].id] = rhsVal
            return

        # We should allocate and store
        alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, rhsVal.type),
                             TypeAttr.get(rhsVal.type)).result
        cc.StoreOp(rhsVal, alloca)
        self.symbolTable[node.targets[0].id] = alloca

    def visit_Attribute(self, node):
        if self.verbose:
            print('[Visit Attribute]')

        if node.value.id in self.symbolTable and node.attr == 'size' and quake.VeqType.isinstance(self.symbolTable[node.value.id].type):
            self.pushValue(quake.VeqSizeOp(self.getIntegerType(
                64), self.symbolTable[node.value.id]).result)
            return

        if node.value.id in ['np', 'numpy'] and node.attr == 'pi':
            self.pushValue(self.getConstantFloat(np.pi))
            return

    def visit_Call(self, node):
        if self.verbose:
            print("[Visit Call] {}".format(ast.unparse(node)))

        # do not walk the FunctionDef decorator_list args
        if isinstance(
                node.func, ast.Attribute
        ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
            return

        if isinstance(node.func, ast.Name):
            # Just visit the args, we know the name
            [self.visit(arg) for arg in node.args]
            if node.func.id == "range":
                iTy = self.getIntegerType(64)
                zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
                one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

                # If this is a range(start, stop, step) call, then we
                # need to know if the step value is incrementing or decrementing
                # If we don't know this, we cannot compile this range statement
                # (this is an issue with compiling a runtime interpreted language)
                isDecrementing = False
                if len(node.args) == 3:
                    # Find the step val and we need to
                    # know if its decrementing
                    # can be incrementing or decrementing
                    stepVal = self.popValue()
                    if isinstance(node.args[2], ast.UnaryOp):
                        if isinstance(node.args[2].op, ast.USub):
                            if isinstance(node.args[2].operand, ast.Constant):
                                # greater than bc USub node above
                                if node.args[2].operand.value > 0:
                                    isDecrementing = True
                            else:
                                raise RuntimeError(
                                    'CUDA Quantum requires step value on range() to be a constant.')

                    # exclusive end
                    endVal = self.popValue()

                    # inclusive start
                    startVal = self.popValue()

                elif len(node.args) == 2:
                    stepVal = one
                    endVal = self.popValue()
                    startVal = self.popValue()
                else:
                    stepVal = one
                    endVal = self.popValue()
                    startVal = zero

                # The total number of elements in the iterable
                # we are generating should be N == endVal - startVal
                totalSize = math.AbsIOp(arith.SubIOp(
                    endVal, startVal).result).result

                # Create an array of i64 of the totalSize
                arrTy = cc.ArrayType.get(self.ctx, iTy)
                iterable = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                       TypeAttr.get(iTy), seqSize=totalSize).result

                # array = [start, start +- step, start +- 2*step, start +- 3*step, ...]
                def bodyBuilder(iterVar):
                    tmp = arith.MulIOp(iterVar, stepVal).result
                    arrElementVal = arith.AddIOp(startVal, tmp).result
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(self.ctx, iTy), iterable, [
                            iterVar],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                    cc.StoreOp(arrElementVal, eleAddr)

                self.createInvariantForLoop(
                    endVal, bodyBuilder, startVal=startVal,
                    stepVal=stepVal, isDecrementing=isDecrementing)

                self.pushValue(iterable)
                self.pushValue(totalSize)
                return

            if node.func.id == "enumerate":
                # We have to have something "iterable" on the stack,
                # could be coming from range() or an iterable like qvector
                totalSize = None
                iterable = None
                iterEleTy = None
                extractFunctor = None
                if len(self.valueStack) == 1:
                    # qreg-like thing
                    iterable = self.popValue()
                    # Create a new iterable, alloca cc.struct<i64, T>
                    totalSize = None
                    if quake.VeqType.isinstance(iterable.type):
                        iterEleTy = self.getRefType()
                        totalSize = quake.VeqSizeOp(
                            self.getIntegerType(), iterable).result
                        extractFunctor = lambda idxVal : quake.ExtractRefOp(iterEleTy, iterable, -1, index=idxVal).result
                    else:
                        raise RuntimeError(
                            "could not infer enumerate tuple type. {}".format(iterable.type))
                else:
                    assert len(self.valueStack) == 2, 'Error in AST processing, should have 2 values on the stack for enumerate {}'.format(
                        ast.unparse(node))
                    totalSize = self.popValue()
                    iterable = self.popValue()
                    arrTy = cc.PointerType.getElementType(iterable.type)
                    iterEleTy = cc.ArrayType.getElementType(arrTy)
                    def localFunc(idxVal):
                        eleAddr = cc.ComputePtrOp(cc.PointerType.get(self.ctx, iterEleTy), iterable, [
                                          idxVal], DenseI32ArrayAttr.get([-2147483648], context=self.ctx)).result
                        return cc.LoadOp(eleAddr).result
                    extractFunctor = localFunc

                # Enumerate returns a iterable of tuple(i64, T) for type T
                # Allocate an array of struct<i64, T> == tuple (for us)
                structTy = cc.StructType.get(
                    self.ctx, [self.getIntegerType(), iterEleTy])
                arrTy = cc.ArrayType.get(self.ctx, structTy)
                enumIterable = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                           TypeAttr.get(structTy), seqSize=totalSize).result
                
                # Now we need to loop through enumIterable and set the elements
                def bodyBuilder(iterVar):
                    # Create the struct
                    element = cc.UndefOp(structTy)
                    # Get the element from the iterable
                    extracted = extractFunctor(iterVar)
                    # Get the pointer to the enumeration iterable so we can set it
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(self.ctx, structTy), enumIterable, [
                            iterVar],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                    # Set the index value
                    element = cc.InsertValueOp(
                        structTy, element, iterVar, DenseI64ArrayAttr.get([0], context=self.ctx)).result
                    # Set the extracted element value
                    element = cc.InsertValueOp(
                        structTy, element, extracted, DenseI64ArrayAttr.get([1], context=self.ctx)).result
                    cc.StoreOp(element, eleAddr)
                self.createInvariantForLoop(totalSize, bodyBuilder)
                self.pushValue(enumIterable)
                self.pushValue(totalSize)
                return

            if node.func.id in ["h", "x", "y", "z", "s", "t"]:
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubit = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                if quake.VeqType.isinstance(qubit.type):
                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(
                            self.getRefType(), qubit, -1, index=iterVal).result
                        opCtor([], [], [], [q])
                    veqSize = quake.VeqSizeOp(
                        self.getIntegerType(), qubit).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(qubit.type):
                    opCtor([], [], [], [qubit])
                    return
                else:
                    raise Exception(
                        'quantum operation on incorrect type {}.'.format(qubit.type))

            if node.func.id in ["rx", "ry", "rz", "r1"]:
                target = self.popValue()
                param = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                if quake.VeqType.isinstance(target.type):
                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(
                            self.getRefType(), target, -1, index=iterVal).result
                        opCtor([], [param], [], [q])
                    veqSize = quake.VeqSizeOp(
                        self.getIntegerType(), target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [param], [], [target])
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(target.type))

            if node.func.id == 'swap':
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubitB = self.popValue()
                qubitA = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubitA, qubitB])
                return

            if node.func.id in globalKernelRegistry:
                # If in globalKernelRegistry, it has to be in this Module
                otherKernel = SymbolTable(self.module.operation)['__nvqpp__mlirgen__'+node.func.id]
                values = [self.popValue() for _ in node.args]
                op = func.CallOp(otherKernel, values)

            else:
                raise RuntimeError(
                    "unhandled function call - {}".format(node.func.id))

        elif isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            if node.func.value.id == 'cudaq':
                if node.func.attr == 'qvector':
                    # Handle cudaq.qvector(N)
                    size = self.popValue()
                    if hasattr(size, "literal_value"):
                        ty = self.getVeqType(size.literal_value)
                        qubits = quake.AllocaOp(ty)
                    else:
                        ty = self.getVeqType()
                        qubits = quake.AllocaOp(ty, size=size)
                    self.pushValue(qubits.results[0])
                    return
                elif node.func.attr == "qubit":
                    self.pushValue(quake.AllocaOp(self.getRefType()).result)
                    return
                elif node.func.attr == 'adjoint':
                    # Handle cudaq.adjoint(kernel, ...)
                    otherFuncName = node.args[0].id
                    if otherFuncName not in globalKernelRegistry:
                        raise RuntimeError(
                            "{} is not a known quantum kernel (was it annotated?).".format(otherFuncName))
                    values = [self.popValue()
                              for _ in range(len(self.valueStack))]
                    if len(values) != len(globalKernelRegistry[otherFuncName].arguments):
                        raise RuntimeError(
                            "incorrect number of runtime arguments for cudaq.control({},..) call.".format(otherFuncName))
                    # controls = self.popValue()
                    quake.ApplyOp([], [], [], values, callee=FlatSymbolRefAttr.get(
                        nvqppPrefix+otherFuncName), is_adj=True)
                    return
                elif node.func.attr == 'control':
                    # Handle cudaq.control(kernel, ...)
                    otherFuncName = node.args[0].id
                    if otherFuncName not in globalKernelRegistry:
                        raise RuntimeError(
                            "{} is not a known quantum kernel (was it annotated?).".format(otherFuncName))
                    values = [self.popValue()
                              for _ in range(len(self.valueStack)-1)]
                    if len(values) != len(globalKernelRegistry[otherFuncName].arguments):
                        raise RuntimeError(
                            "incorrect number of runtime arguments for cudaq.control({},..) call.".format(otherFuncName))
                    controls = self.popValue()
                    quake.ApplyOp([], [], [controls], values, callee=FlatSymbolRefAttr.get(
                        nvqppPrefix+otherFuncName))
                    return

            if node.func.value.id in self.symbolTable:
                # Method call on one of our variables
                var = self.symbolTable[node.func.value.id]
                if quake.VeqType.isinstance(var.type):
                    # qreg or qview method call
                    if node.func.attr == 'back':
                        qrSize = quake.VeqSizeOp(
                            self.getIntegerType(), var).result
                        one = self.getConstantInt(1)
                        endOff = arith.SubIOp(qrSize, one)
                        if len(node.args):
                            # extract the subveq
                            startOff = arith.SubIOp(qrSize, self.popValue())
                            self.pushValue(quake.SubVeqOp(
                                self.getVeqType(), var, startOff, endOff).result)
                        else:
                            # extract the qubit...
                            self.pushValue(quake.ExtractRefOp(
                                self.getRefType(), var, -1, index=endOff).result)
                        return
                    if node.func.attr == 'front':
                        zero = self.getConstantInt(0)
                        if len(node.args):
                            # extract the subveq
                            qrSize = self.popValue()
                            one = self.getConstantInt(1)
                            offset = arith.SubIOp(qrSize, one)
                            self.pushValue(quake.SubVeqOp(
                                self.getVeqType(), var, zero, offset).result)
                        else:
                            # extract the qubit...
                            self.pushValue(quake.ExtractRefOp(
                                self.getRefType(), var, -1, index=zero).result)
                        return

            # We have a func name . ctrl
            if node.func.value.id in ['h', 'x', 'y', 'z', 's', 't'] and node.func.attr == 'ctrl':
                target = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                opCtor([], [], controls, [target])
                return

            if node.func.value.id in ['rx', 'ry', 'rz', 'r1'] and node.func.attr == 'ctrl':
                target = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                param = controls[-1]
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                opCtor([], [param], controls[:-1], [target])
                return
            
            # We have a func name . adj
            if node.func.value.id in ['h', 'x', 'y', 'z', 's', 't'] and node.func.attr == 'adj':
                target = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.value.id.title()))
                if quake.VeqType.isinstance(target.type):
                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(
                            self.getRefType(), target, -1, index=iterVal).result
                        opCtor([], [], [], [q], is_adj=True)
                    veqSize = quake.VeqSizeOp(
                        self.getIntegerType(), target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [], [], [target], is_adj=True)
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(target.type))
                
            if node.func.value.id in ['rx', 'ry', 'rz', 'r1'] and node.func.attr == 'adj':
                target = self.popValue()
                param = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.value.id.title()))
                if quake.VeqType.isinstance(target.type):
                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(
                            self.getRefType(), target, -1, index=iterVal).result
                        opCtor([], [param], [], [q], is_adj=True)
                    veqSize = quake.VeqSizeOp(
                        self.getIntegerType(), target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [param], [], [target], is_adj=True)
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(target.type))

    def visit_List(self, node):
        if self.verbose:
            print('[Visit List] {}', ast.unparse(node))
        self.generic_visit(node)

        valueTys = [quake.VeqType.isinstance(
            v.type) or quake.RefType.isinstance(v.type) for v in self.valueStack]
        if False not in valueTys:
            # this is a list of quantum types,
            # concat them into a veq
            self.pushValue(quake.ConcatOp(self.getVeqType(), [
                           self.popValue() for _ in valueTys]).result)
            return

    def visit_Constant(self, node):
        if self.verbose:
            print("[Visit Constant {}]".format(node.value))
        if isinstance(node.value, int):
            self.pushValue(self.getConstantInt(node.value))
            return
        elif isinstance(node.value, float):
            self.pushValue(self.getConstantFloat(node.value))
            return
        else:
            raise RuntimeError(
                "unhandled constant: {}".format(ast.unparse(node)))

    def visit_Subscript(self, node):
        if self.verbose:
            print("[Visit Subscript]")
        self.generic_visit(node)
        assert len(self.valueStack) > 1

        # get the last name, should be name of var being subscripted
        var = self.popValue()
        idx = self.popValue()
        if quake.VeqType.isinstance(var.type):
            qrefTy = self.getRefType()
            self.pushValue(
                quake.ExtractRefOp(qrefTy, var, -1, index=idx).result)
        else:
            raise RuntimeError(
                "unhandled subscript: {}".format(ast.unparse(node)))

    def visit_For(self, node):
        """Visit the For Stmt node. This node represents the typical 
        Python for statement, `for VAR in ITERABLE`. Supports range(...) 
        and qvector iterables."""

        if self.verbose:
            print('[Visit For]')

        # We have here a `for VAR in range(...)`, where range produces
        # an iterable sequence. We model this in visit_Call (range is a called function)
        # as a cc.array of integer elements. So we visit that node here
        self.visit(node.iter)
        assert len(self.valueStack) > 0 and len(self.valueStack) < 3

        totalSize = None
        iterable = None
        extractFunctor = None
        if len(self.valueStack) == 1:
            iterable = self.popValue()
            if quake.VeqType.isinstance(iterable.type):
                size = quake.VeqType.getSize(iterable.type)
                if size:
                    totalSize = self.getConstantInt(size)
                else:
                    totalSize = quake.VeqSizeOp(
                        self.getIntegerType(64), iterable).result

                def functor(iter, idx):
                    return [quake.ExtractRefOp(self.getRefType(), iter, -1, index=idx).result]
                extractFunctor = functor

            else:
                raise RuntimeError(
                    '{} iterable type not yet supported.'.format(iterable.type))

        else:
            # and now the iterable on the stack is a pointer to a cc.array
            totalSize = self.popValue()
            iterable = self.popValue()

            # Double check our types are right
            assert cc.PointerType.isinstance(iterable.type)
            arrayType = cc.PointerType.getElementType(iterable.type)
            assert cc.ArrayType.isinstance(arrayType)
            elementType = cc.ArrayType.getElementType(arrayType)

            def functor(iter, idx):
                eleAddr = cc.ComputePtrOp(cc.PointerType.get(self.ctx, elementType), iter, [
                                          idx], DenseI32ArrayAttr.get([-2147483648], context=self.ctx)).result
                loaded = cc.LoadOp(eleAddr).result
                if IntegerType.isinstance(elementType):
                    return [loaded]
                elif cc.StructType.isinstance(elementType):
                    # Get struct types
                    types = cc.StructType.getTypes(elementType)
                    ret = []
                    for i, ty in enumerate(types):
                        ret.append(cc.ExtractValueOp(
                            ty, loaded, DenseI64ArrayAttr.get([i], context=self.ctx)).result)
                    return ret

            extractFunctor = functor

        # Get the name of the variable, VAR in for VAR in range(...)
        varNames = []
        if isinstance(node.target, ast.Name):
            varNames.append(node.target.id)
        else:
            # has to be a ast.Tuple 
            for elt in node.target.elts:
                varNames.append(elt.id)
        
        # We'll need a zero and one value of integer type
        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

        def bodyBuilder(iterVar):
            values = extractFunctor(iterable, iterVar)
            for i, v in enumerate(values):
                self.symbolTable[varNames[i]] = v
            [self.visit(b) for b in node.body]

        self.createInvariantForLoop(totalSize, bodyBuilder)

    def visit_UnaryOp(self, node):
        if self.verbose:
            print("[Visit Unary = {}]".format(ast.unparse(node)))

        self.generic_visit(node)
        operand = self.popValue()
        if isinstance(node.op, ast.USub):
            if F64Type.isinstance(operand.type):
                self.pushValue(arith.NegFOp(operand).result)
            else:
                negOne = self.getConstantInt(-1)
                self.pushValue(arith.MulIOp(negOne, operand).result)
            return

        raise RuntimeError("unhandled UnaryOp: {}".format(ast.unparse(node)))

    def visit_BinOp(self, node):
        if self.verbose:
            print("[Visit BinaryOp = {}]".format(ast.unparse(node)))
        self.generic_visit(node)

        # Get the left and right parts of this expression
        left = self.popValue()
        right = self.popValue()

        # Basedon the op type and the leaf types, create the MLIR operator
        if isinstance(node.op, ast.Add):
            if IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.AddIOp(left, right).result)
                return
            else:
                raise RuntimeError(
                    "unhandled BinOp.Add types: {}".format(ast.unparse(node)))

        if isinstance(node.op, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.SubIOp(left, right).result)
                return
            else:
                raise RuntimeError(
                    "unhandled BinOp.Add types: {}".format(ast.unparse(node)))
        if isinstance(node.op, ast.FloorDiv):
            if IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.FloorDivSIOp(left, right).result)
                return
            else:
                raise RuntimeError(
                    "unhandled BinOp.FloorDiv types: {}".format(ast.unparse(node)))
        if isinstance(node.op, ast.Div):
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(arith.DivFOp(left, right).result)
            return
        if isinstance(node.op, ast.Pow):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(right.type):
                # math.ipowi does not lower to llvm as is
                # workaround, use math to funcs conversion
                self.pushValue(math.IPowIOp(left, right).result)
                return

            if F64Type.isinstance(left.type) and IntegerType.isinstance(right.type):
                self.pushValue(math.FPowIOp(left, right).result)
                return

            # now we know the types are different, default to float
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(math.PowFOp(left, right).result)
            return
        else:
            raise RuntimeError(
                "unhandled binary operator: {}".format(ast.unparse(node)))

    def visit_Name(self, node):
        if self.verbose:
            print("[Visit Name {}]".format(node.id))
        
        if node.id in self.symbolTable:
            value = self.symbolTable[node.id]
            if cc.PointerType.isinstance(value.type):
                loaded = cc.LoadOp(value).result
                self.pushValue(loaded)
            else:
                self.pushValue(self.symbolTable[node.id])
            return


class FindDepKernelsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.depKernels = {}
        pass

    def visit_Call(self, node):
        if hasattr(node, 'func'):
            if isinstance(node.func, ast.Name) and node.func.id in globalAstRegistry:
                print("adding dep kernel ", node.func.id)
                self.depKernels[node.func.id] = globalAstRegistry[node.func.id]
            elif isinstance(node.func, ast.Attribute):
                if node.func.value.id == 'cudaq' and node.func.attr == 'control':
                    self.depKernels[node.args[0].id] = globalAstRegistry[node.args[0].id]


def compile_to_quake(astModule, **kwargs):
    global globalAstRegistry
    verbose = 'verbose' in kwargs and kwargs['verbose']

    # First we need to find any dependent kernels, they have to be
    # built as part of this ModuleOp...
    vis = FindDepKernelsVisitor()
    vis.visit(astModule)
    depKernels = vis.depKernels

    bridge = PyASTBridge(verbose=verbose)
    # Add all dependent kernels to the MLIR Module
    [bridge.visit(ast) for _, ast in depKernels.items()]
    # Build the MLIR Module for this kernel
    bridge.visit(astModule)

    if verbose:
        print(bridge.module)
    pm = PassManager.parse("builtin.module(canonicalize,cse)",
                           context=bridge.ctx)
    pm.run(bridge.module)

    globalAstRegistry[bridge.name] = astModule
    return bridge.module, bridge.argTypes
