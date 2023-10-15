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

    def getIntegerType(self, width):
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

    def visit_Assign(self, node):
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
            print("[Visit Call]")

        # do not walk the FunctionDef decorator_list args
        if isinstance(
                node.func, ast.Attribute
        ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
            return

        if not isinstance(node.func, ast.Attribute) and node.func.id == 'range':
            # The range generator function should return an iterable of a
            # specific size (which may only be runtime known)
            [self.visit(a) for a in node.args]

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
                            if node.args[2].operand.value > 0: # greater than bc USub node above
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
            totalSize = math.AbsIOp(arith.SubIOp(endVal, startVal).result).result

            # Create an array of i64 of the totalSize
            arrTy = cc.ArrayType.get(self.ctx, iTy)
            iterable = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                   TypeAttr.get(arrTy), seqSize=totalSize).result

            # Save this size for later
            # Fill the array based on start, end, and step
            # array = [start, start +- step, start +- 2*step, start +- 3*step, ...]
            scope = cc.ScopeOp([])
            scopeBlock = Block.create_at_start(scope.initRegion, [])
            with InsertionPoint(scopeBlock):
                # loop counter
                alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, iTy),
                                     TypeAttr.get(iTy)).result
                cc.StoreOp(startVal, alloca)

                loop = cc.LoopOp([], [], BoolAttr.get(False))
                bodyBlock = Block.create_at_start(loop.bodyRegion, [])
                with InsertionPoint(bodyBlock):
                    # compute start +- loadedIterVar * step
                    loadedIterVar = cc.LoadOp(alloca).result
                    tmp = arith.MulIOp(loadedIterVar, stepVal).result
                    arrElementVal = arith.AddIOp(startVal, tmp).result
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(self.ctx, iTy), iterable, [
                            loadedIterVar],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                    cc.StoreOp(arrElementVal, eleAddr)
                    cc.ContinueOp([])
                whileBlock = Block.create_at_start(loop.whileRegion, [])

                with InsertionPoint(whileBlock):
                    loaded = cc.LoadOp(alloca)
                    c = arith.CmpIOp(IntegerAttr.get(iTy, 2) if not isDecrementing else IntegerAttr.get(iTy, 4),
                                     loaded, endVal).result
                    cc.ConditionOp(c, [])

                stepBlock = Block.create_at_start(loop.stepRegion, [])
                with InsertionPoint(stepBlock):
                    loaded = cc.LoadOp(alloca)
                    incr = getattr(arith, '{}IOp'.format(
                        'Add' if isDecrementing else 'Add'))(loaded, stepVal).result
                    cc.StoreOp(incr, alloca)
                    cc.ContinueOp([])
                cc.ContinueOp([])

                self.pushValue(iterable)
                self.pushValue(totalSize)
                return

        if isinstance(node.func, ast.Name):
            # Just visit the args, we know the name
            [self.visit(arg) for arg in node.args] 
            if node.func.id in ["h", "x", "y", "z", "s", "t"]:
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubit = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubit])
                return

            if node.func.id in ["rx", "ry", "rz", "r1"]:
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubit = self.popValue()
                param = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [param], [], [qubit])
                return

            if node.func.id == 'swap':
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubitB = self.popValue()
                qubitA = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubitA, qubitB])
                return

        elif isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            if node.func.value.id == 'cudaq' and node.func.attr == 'qvector':
                size = self.popValue()
                if hasattr(size, "literal_value"):
                    ty = self.getVeqType(size.literal_value)
                    qubits = quake.AllocaOp(ty)
                else:
                    ty = self.getVeqType()
                    qubits = quake.AllocaOp(ty, size=size)
                self.pushValue(qubits.results[0])
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

            if node.func.value.id == 'swap':
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubitB = self.popValue()
                qubitA = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], controls, [qubitA, qubitB])
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
                quake.ExtractRefOp(qrefTy, var, -1, index=idx))
        else:
            raise RuntimeError(
                "unhandled subscript: {}".format(ast.unparse(node)))

    def visit_For(self, node):
        """Visit the For Stmt node. This node represents the typical 
        Python for statement, `for VAR in ITERABLE`."""

        if node.iter.func.id != 'range':
            raise RuntimeError(
                "CUDA Quantum only supports `for VAR in range(UPPER):`")

        if self.verbose:
            print('[Visit For]')

        # We have here a `for VAR in range(...)`, where range produces
        # an iterable sequence. We model this in visit_Call (range is a called function)
        # as a cc.array of integer elements. So we visit that node here
        self.visit(node.iter)
        assert len(self.valueStack) == 2

        # and now the iterable on the stack is a pointer to a cc.array
        totalSize = self.popValue()
        iterable = self.popValue()

        # Double check our types are right
        assert cc.PointerType.isinstance(iterable.type)
        arrayType = cc.PointerType.getElementType(iterable.type)
        assert cc.ArrayType.isinstance(arrayType)
        elementType = cc.ArrayType.getElementType(arrayType)
        assert IntegerType.isinstance(elementType)

        # Get the name of the variable, VAR in for VAR in range(...)
        varName = node.target.id

        # We'll need a zero and one value of integer type
        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

        # Create a scope
        scope = cc.ScopeOp([])
        scopeBlock = Block.create_at_start(scope.initRegion, [])
        with InsertionPoint(scopeBlock):
            # Allocate the integer loop counter
            alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, iTy),
                                 TypeAttr.get(iTy)).result
            # Start it at zero
            cc.StoreOp(zero, alloca)

            # Define the loop with a while, body, and step block
            loop = cc.LoopOp([], [], BoolAttr.get(False))
            bodyBlock = Block.create_at_start(loop.bodyRegion, [])
            with InsertionPoint(bodyBlock):
                # in the body, we load the iteration variable, call it k
                loaded = cc.LoadOp(alloca)
                # We get a pointer to the kth element of the range iterable array
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(self.ctx, iTy), iterable, [
                        loaded],
                    DenseI32ArrayAttr.get([-2147483648], context=self.ctx))

                # Store the element in the symbol table for the body block
                self.symbolTable[varName] = cc.LoadOp(eleAddr).result

                # Construct the code for the body
                [self.visit(b) for b in node.body]
                cc.ContinueOp([])

            whileBlock = Block.create_at_start(loop.whileRegion, [])
            with InsertionPoint(whileBlock):
                # In this while block, we load the k variable,
                # and check that the value of it is less than
                # the total number of elements in the range, a variable we
                # store as __rangeTotalSize__
                loaded = cc.LoadOp(alloca)
                c = arith.CmpIOp(IntegerAttr.get(iTy, 2),
                                 loaded, totalSize).result
                cc.ConditionOp(c, [])

            stepBlock = Block.create_at_start(loop.stepRegion, [])
            with InsertionPoint(stepBlock):
                loaded = cc.LoadOp(alloca)
                incr = arith.AddIOp(loaded, one).result
                cc.StoreOp(incr, alloca)
                cc.ContinueOp([])
            cc.ContinueOp([])

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

def compile_to_quake(astModule, **kwargs):
    verbose = 'verbose' in kwargs and kwargs['verbose']
    bridge = PyASTBridge(verbose=verbose)
    bridge.visit(astModule)
    if verbose:
        print(bridge.module)
    pm = PassManager.parse("builtin.module(canonicalize,cse)",
                           context=bridge.ctx)
    pm.run(bridge.module)
    return bridge.module, bridge.argTypes
