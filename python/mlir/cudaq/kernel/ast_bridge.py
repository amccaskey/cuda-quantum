# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import sys
from typing import Callable
from collections import deque
import numpy as np
from .analysis import FindDepKernelsVisitor
from .utils import globalAstRegistry, globalKernelRegistry, nvqppPrefix
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.dialects import builtin, func, arith, math

# This file implements the CUDA Quantum Python AST to MLIR conversion.
# It provides a PyASTBridge class that implements the ast.NodeVisitor type
# to walk the Python AST for a cudaq.kernel annotated function and generate
# valid MLIR code using Quake, CC, Arith, and Math dialects.


class PyASTBridge(ast.NodeVisitor):
    """
    The PyASTBridge class implements the ast.NodeVisitor type to convert a 
    python function definition (annotated with cudaq.kernel) to an MLIR ModuleOp 
    containing a func.FuncOp representative of the original python function but leveraging 
    the Quake and CC dialects provided by CUDA Quantum. This class keeps track of a 
    MLIR Value stack that is pushed to and popped from during visitation of the 
    function AST nodes. We leverage the auto-generated MLIR Python bindings for the internal 
    C++ CUDA Quantum dialects to build up the MLIR code. 

    For kernels that call other kernels, we require that the `ModuleOp` contain the 
    kernel being called. This is enabled via the `FindDepKernelsVisitor` in the local 
    analysis module, and is handled by the below `compile_to_mlir` function. For 
    callable block arguments, we leverage runtime-known callable argument function names 
    and synthesize them away with an internal C++ MLIR pass. 
    """

    def __init__(self, **kwargs):
        """
        The constructor. Initializes the mlir.Value stack, the mlir.Context, and the 
        mlir.Module that we will be building upon. This class keeps track of a 
        symbol table, which maps variable names to constructed mlir.Values. 
        """
        self.valueStack = deque()
        self.ctx = Context()
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.symbolTable = {}
        self.increment = 0
        self.buildingEntryPoint = False
        self.verbose = 'verbose' in kwargs and kwargs['verbose']

    def getVeqType(self, size=None):
        """
        Return a quake.VeqType. Pass the size of the quake.veq if known. 
        """
        if size == None:
            return quake.VeqType.get(self.ctx)
        return quake.VeqType.get(self.ctx, size)

    def getRefType(self):
        """
        Return a quake.RefType.
        """
        return quake.RefType.get(self.ctx)

    def isQuantumType(self, ty):
        """
        Return True if the given type is quantum (is a VeqType or RefType). 
        Return False otherwise.
        """
        return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(ty)

    def isMeasureResultType(self, ty):
        """
        Return true if the given type is a qubit measurement result type (an i1 type).
        """
        return IntegerType.isinstance(ty) and ty == IntegerType.get_signless(1)

    def getIntegerType(self, width=64):
        """
        Return an MLIR IntegerType of the given bit width (defaults to 64 bits).
        """
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        """
        Return an MLIR Integer Attribute of the given IntegerType.
        """
        return IntegerAttr.get(type, value)

    def getFloatType(self):
        """
        Return an MLIR Float type (double precision).
        """
        return F64Type.get()

    def getFloatAttr(self, type, value):
        """
        Return an MLIR Float Attribute (double precision).
        """
        return FloatAttr.get(type, value)

    def getConstantFloat(self, value):
        """
        Create a constant float operation and return its mlir result Value.
        Takes as input the concrete float value. 
        """
        ty = self.getFloatType()
        return arith.ConstantOp(ty, self.getFloatAttr(ty, value)).result

    def getConstantInt(self, value, width=64):
        """
        Create a constant integer operation and return its mlir result Value.
        Takes as input the concrete integer value. Can specify the integer bit width.
        """
        ty = self.getIntegerType(width)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def pushValue(self, value):
        """
        Push an MLIR Value onto the stack for usage in a subsequent AST node visit method.
        """
        if self.verbose:
            print('{}push {}'.format(self.increment * ' ', value))
        self.increment += 2
        self.valueStack.append(value)

    def popValue(self):
        """
        Pop an MLIR Value from the stack. 
        """
        val = self.valueStack.pop()
        self.increment -= 2
        if self.verbose:
            print('{}pop {}'.format(self.increment * ' ', val))
        return val

    def mlirTypeFromAnnotation(self, annotation):
        """
        Return the MLIR Type corresponding to the given kernel function argument type annotation.
        Throws an exception if the programmer did not annotate function argument types. 
        """
        if annotation == None:
            raise RuntimeError(
                'cudaq.kernel functions must have argument type annotations.')

        if hasattr(annotation, 'attr'):
            if annotation.value.id == 'cudaq':
                if annotation.attr == 'qview' or annotation.attr == 'qvector':
                    return self.getVeqType()
                if annotation.attr == 'qubit':
                    return self.getRefType()

            if annotation.value.id in ['numpy', 'np']:
                if annotation.attr == 'ndarray':
                    return cc.StdvecType.get(self.ctx, F64Type.get())

        if isinstance(annotation,
                      ast.Subscript) and annotation.value.id == 'Callable':
            if not hasattr(annotation, 'slice'):
                raise RuntimeError(
                    'Callable type must have signature specified.')

            argTypes = [
                self.mlirTypeFromAnnotation(a)
                for a in annotation.slice.elts[0].elts
            ]
            return cc.CallableType.get(self.ctx, argTypes)

        if annotation.id == 'int':
            return self.getIntegerType(64)
        elif annotation.id == 'float':
            return F64Type.get()
        elif annotation.id == 'list':
            return cc.StdvecType.get(self.ctx, F64Type.get())
        else:
            raise RuntimeError('{} is not a supported type yet.'.format(
                annotation.id))

    def createInvariantForLoop(self,
                               endVal,
                               bodyBuilder,
                               startVal=None,
                               stepVal=None,
                               isDecrementing=False):
        """
        Create an invariant loop using the CC dialect. 
        """
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
            cc.ConditionOp(
                arith.CmpIOp(condPred, whileBlock.arguments[0], endVal).result,
                whileBlock.arguments)

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
        """
        Create an MLIR func.FuncOp for the given FunctionDef AST node. For the top-level
        FunctionDef, this will add the FuncOp to the ModuleOp body, annotate the FuncOp with 
        cudaq-entrypoint if it is an Entry Point CUDA Quantum kernel, and visit the rest of the 
        FunctionDef body. If this is an inner FunctionDef, this will treat the function as a CC 
        lambda function and add the cc.callable-typed value to the symbol table, keyed on the 
        FunctionDef name. 

        We keep track of the top-level function name as well as its internal MLIR name, prefixed 
        with the __nvqpp__mlirgen__ prefix. 
        """
        if self.buildingEntryPoint:
            # This is an inner function def, we will
            # treat it as a cc.callable (cc.create_lambda)
            if self.verbose:
                print("Visiting inner FunctionDef {}".format(node.name))

            arguments = node.args.args
            if len(arguments):
                raise RuntimeError(
                    "inner function defs cannot have arguments yet.")

            ty = cc.CallableType.get(self.ctx, [])
            createLambda = cc.CreateLambdaOp(ty)
            initRegion = createLambda.initRegion
            initBlock = Block.create_at_start(initRegion, [])
            with InsertionPoint(initBlock):
                [self.visit(n) for n in node.body]
                cc.ReturnOp([])
            self.symbolTable[node.name] = createLambda.result
            return

        with self.ctx, InsertionPoint(self.module.body), self.loc:

            # Get the arg types and arg names
            # this will throw an error if the types aren't annotated
            self.argTypes = [
                self.mlirTypeFromAnnotation(arg.annotation)
                for arg in node.args.args
            ]
            # Get the argument names
            argNames = [arg.arg for arg in node.args.args]

            self.name = node.name

            # the full function name in MLIR is __nvqpp__mlirgen__ + the function name
            fullName = nvqppPrefix + node.name

            # Create the FuncOp
            f = func.FuncOp(fullName, (self.argTypes, []), loc=self.loc)

            # Set this kernel as an entry point if the arg types are classical only
            def isQuantumTy(ty):
                return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(
                    ty)

            areQuantumTypes = [isQuantumTy(ty) for ty in self.argTypes]
            if True not in areQuantumTypes:
                f.attributes.__setitem__('cudaq-entrypoint', UnitAttr.get())

            # Create the entry block
            self.entry = f.add_entry_block()

            # Add the block args to the symbol table
            blockArgs = self.entry.arguments
            for i, b in enumerate(blockArgs):
                self.symbolTable[argNames[i]] = b

            # Set the insertion point to the start of the entry block
            with InsertionPoint(self.entry):
                self.buildingEntryPoint = True
                # Visit the function
                [self.visit(n) for n in node.body]
                # Add the return operation
                ret = func.ReturnOp([])
                self.buildingEntryPoint = False

            if True not in areQuantumTypes:
                attr = DictAttr.get(
                    {
                        fullName:
                            StringAttr.get(fullName + '_entryPointRewrite',
                                           context=self.ctx)
                    },
                    context=self.ctx)
                self.module.operation.attributes.__setitem__(
                    'quake.mangled_name_map', attr)

            globalKernelRegistry[node.name] = f
            self.symbolTable.clear()

    def visit_Lambda(self, node):
        """
        Map a lambda expression in a CUDA Quantum kernel to a CC Lambda (a Value of cc.callable type 
        using the cc.create_lambda operation). Note that we extend Python with a novel 
        syntax to specify a list of independent statements (Python lambdas must have a single statement) by 
        allowing programmers to return a Tuple where each element is an independent statement. 

        functor = lambda : (h(qubits), x(qubits), ry(np.pi, qubits))  # qubits captured from parent region

        equivalent to 

        def functor(qubits):
            h(qubits)
            x(qubits)
            ry(np.pi, qubits)

        """
        if self.verbose:
            print('[Visit Lambda {}]'.format(ast.unparse(node)))

        arguments = node.args.args
        if len(arguments):
            raise RuntimeError("lambdas cannot have arguments yet.")

        ty = cc.CallableType.get(self.ctx, [])
        createLambda = cc.CreateLambdaOp(ty)
        initBlock = Block.create_at_start(createLambda.initRegion, [])
        with InsertionPoint(initBlock):
            # Python lambdas can only have a single statement.
            # Here we will enhance our language by processing a single Tuple statement
            # as a set of statements for each element of the tuple
            if isinstance(node.body, ast.Tuple):
                [self.visit(element) for element in node.body.elts]
            else:
                self.visit(
                    node.body)  # only one statement in a python lambda :(
            cc.ReturnOp([])
        self.pushValue(createLambda.result)
        return

    def visit_Assign(self, node):
        if self.verbose:
            print('[Visit Assign {}]'.format(ast.unparse(node)))

        # Retain the variable name for potential children (like mz(q, registerName=...))
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.currentAssignVariableName = str(node.targets[0].id)
            self.generic_visit(node)
            self.currentAssignVariableName = None
        else:
            self.generic_visit(node)

        varNames = []
        varValues = []

        # Can assign a, b, c, = Tuple...
        # or single assign a = something
        if isinstance(node.targets[0], ast.Tuple):
            assert len(self.valueStack) == len(node.targets[0].elts)
            varValues = [
                self.popValue() for _ in range(len(node.targets[0].elts))
            ]
            varValues.reverse()
            varNames = [name.id for name in node.targets[0].elts]
        else:
            varValues = [self.popValue()]
            varNames = [node.targets[0].id]

        for i, value in enumerate(varValues):
            if self.isQuantumType(value.type) or self.isMeasureResultType(
                    value.type) or cc.CallableType.isinstance(value.type):
                self.symbolTable[varNames[i]] = value
            else:
                # We should allocate and store
                alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, value.type),
                                     TypeAttr.get(value.type)).result
                cc.StoreOp(value, alloca)
                self.symbolTable[node.targets[0].id] = alloca

    def visit_Attribute(self, node):
        if self.verbose:
            print('[Visit Attribute]')

        if node.value.id in self.symbolTable and node.attr == 'size' and quake.VeqType.isinstance(
                self.symbolTable[node.value.id].type):
            self.pushValue(
                quake.VeqSizeOp(self.getIntegerType(64),
                                self.symbolTable[node.value.id]).result)
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
                                    'CUDA Quantum requires step value on range() to be a constant.'
                                )

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
                totalSize = math.AbsIOp(arith.SubIOp(endVal,
                                                     startVal).result).result

                # Create an array of i64 of the totalSize
                arrTy = cc.ArrayType.get(self.ctx, iTy)
                iterable = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                       TypeAttr.get(iTy),
                                       seqSize=totalSize).result

                # array = [start, start +- step, start +- 2*step, start +- 3*step, ...]
                def bodyBuilder(iterVar):
                    tmp = arith.MulIOp(iterVar, stepVal).result
                    arrElementVal = arith.AddIOp(startVal, tmp).result
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(self.ctx, iTy), iterable, [iterVar],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                    cc.StoreOp(arrElementVal, eleAddr)

                self.createInvariantForLoop(endVal,
                                            bodyBuilder,
                                            startVal=startVal,
                                            stepVal=stepVal,
                                            isDecrementing=isDecrementing)

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
                        totalSize = quake.VeqSizeOp(self.getIntegerType(),
                                                    iterable).result

                        def extractFunctor(idxVal):
                            return quake.ExtractRefOp(iterEleTy,
                                                      iterable,
                                                      -1,
                                                      index=idxVal).result
                    elif cc.StdvecType.isinstance(iterable.type):
                        iterEleTy = self.getFloatType()
                        totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                                    iterable).result

                        def extractFunctor(idxVal):
                            elePtrTy = cc.PointerType.get(self.ctx, iterEleTy)
                            vecPtr = cc.StdvecDataOp(elePtrTy, iterable).result
                            eleAddr = cc.ComputePtrOp(
                                elePtrTy, vecPtr, [idxVal],
                                DenseI32ArrayAttr.get([-2147483648],
                                                      context=self.ctx)).result
                            return cc.LoadOp(eleAddr).result
                    else:
                        raise RuntimeError(
                            "could not infer enumerate tuple type. {}".format(
                                iterable.type))
                else:
                    assert len(
                        self.valueStack
                    ) == 2, 'Error in AST processing, should have 2 values on the stack for enumerate {}'.format(
                        ast.unparse(node))
                    totalSize = self.popValue()
                    iterable = self.popValue()
                    arrTy = cc.PointerType.getElementType(iterable.type)
                    iterEleTy = cc.ArrayType.getElementType(arrTy)

                    def localFunc(idxVal):
                        eleAddr = cc.ComputePtrOp(
                            cc.PointerType.get(self.ctx, iterEleTy), iterable,
                            [idxVal],
                            DenseI32ArrayAttr.get([-2147483648],
                                                  context=self.ctx)).result
                        return cc.LoadOp(eleAddr).result

                    extractFunctor = localFunc

                # Enumerate returns a iterable of tuple(i64, T) for type T
                # Allocate an array of struct<i64, T> == tuple (for us)
                structTy = cc.StructType.get(self.ctx,
                                             [self.getIntegerType(), iterEleTy])
                arrTy = cc.ArrayType.get(self.ctx, structTy)
                enumIterable = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                           TypeAttr.get(structTy),
                                           seqSize=totalSize).result

                # Now we need to loop through enumIterable and set the elements
                def bodyBuilder(iterVar):
                    # Create the struct
                    element = cc.UndefOp(structTy)
                    # Get the element from the iterable
                    extracted = extractFunctor(iterVar)
                    # Get the pointer to the enumeration iterable so we can set it
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(self.ctx, structTy), enumIterable,
                        [iterVar],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                    # Set the index value
                    element = cc.InsertValueOp(
                        structTy, element, iterVar,
                        DenseI64ArrayAttr.get([0], context=self.ctx)).result
                    # Set the extracted element value
                    element = cc.InsertValueOp(
                        structTy, element, extracted,
                        DenseI64ArrayAttr.get([1], context=self.ctx)).result
                    cc.StoreOp(element, eleAddr)

                self.createInvariantForLoop(totalSize, bodyBuilder)
                self.pushValue(enumIterable)
                self.pushValue(totalSize)
                return

            if node.func.id in ["h", "x", "y", "z", "s", "t"]:
                # Here we enable application of the op on all the
                # provided arguments, e.g. x(qubit), x(qvector), x(q, r), etc.
                numValues = len(self.valueStack)
                qubitTargets = [self.popValue() for _ in range(numValues)]
                qubitTargets.reverse()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                for qubit in qubitTargets:
                    if quake.VeqType.isinstance(qubit.type):

                        def bodyBuilder(iterVal):
                            q = quake.ExtractRefOp(self.getRefType(),
                                                   qubit,
                                                   -1,
                                                   index=iterVal).result
                            opCtor([], [], [], [q])

                        veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                                  qubit).result
                        self.createInvariantForLoop(veqSize, bodyBuilder)
                        # return
                    elif quake.RefType.isinstance(qubit.type):
                        opCtor([], [], [], [qubit])
                        # return
                    else:
                        raise Exception(
                            'quantum operation on incorrect type {}.'.format(
                                qubit.type))
                return

            if node.func.id in ["rx", "ry", "rz", "r1"]:
                target = self.popValue()
                param = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                if quake.VeqType.isinstance(target.type):

                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(self.getRefType(),
                                               target,
                                               -1,
                                               index=iterVal).result
                        opCtor([], [param], [], [q])

                    veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                              target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [param], [], [target])
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(
                            target.type))

            if node.func.id in ['mx', 'my', 'mz']:
                qubit = self.popValue()
                # FIXME Handle registerName
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                i1Ty = self.getIntegerType(1)
                resTy = i1Ty if quake.RefType.isinstance(
                    qubit.type) else cc.StdvecType.get(self.ctx, i1Ty)
                self.pushValue(
                    opCtor(resTy, [], [qubit],
                           registerName=self.currentAssignVariableName).result)
                return

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
                otherKernel = SymbolTable(
                    self.module.operation)['__nvqpp__mlirgen__' + node.func.id]
                values = [self.popValue() for _ in node.args]
                op = func.CallOp(otherKernel, values)

            elif node.func.id in self.symbolTable:
                val = self.symbolTable[node.func.id]
                if cc.CallableType.isinstance(val.type):
                    # callable = cc.InstantiateCallableOp(val.type, val, []).result
                    numVals = len(self.valueStack)
                    values = [self.popValue() for _ in range(numVals)]
                    # FIXME check value types match callable type signature
                    callable = cc.CallableFuncOp(
                        cc.CallableType.getFunctionType(val.type), val).result
                    func.CallIndirectOp([], callable, values)
                    return

            else:
                raise RuntimeError("unhandled function call - {}".format(
                    node.func.id))

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
                    if otherFuncName in self.symbolTable:
                        # This is a callable argument
                        values = [
                            self.popValue()
                            for _ in range(len(self.valueStack) - 2)
                        ]
                        a = quake.ApplyOp([], [self.popValue()], [], values)
                        return

                    if otherFuncName not in globalKernelRegistry:
                        raise RuntimeError(
                            "{} is not a known quantum kernel (was it annotated?)."
                            .format(otherFuncName))
                    values = [
                        self.popValue() for _ in range(len(self.valueStack))
                    ]
                    if len(values) != len(
                            globalKernelRegistry[otherFuncName].arguments):
                        raise RuntimeError(
                            "incorrect number of runtime arguments for cudaq.control({},..) call."
                            .format(otherFuncName))
                    # controls = self.popValue()
                    quake.ApplyOp([], [], [],
                                  values,
                                  callee=FlatSymbolRefAttr.get(nvqppPrefix +
                                                               otherFuncName),
                                  is_adj=True)
                    return

                elif node.func.attr == 'control':
                    # Handle cudaq.control(kernel, ...)
                    otherFuncName = node.args[0].id
                    if otherFuncName in self.symbolTable:
                        # This is a callable argument
                        values = [
                            self.popValue()
                            for _ in range(len(self.valueStack) - 2)
                        ]
                        controls = self.popValue()
                        a = quake.ApplyOp([], [self.popValue()], [controls],
                                          values)
                        return

                    if otherFuncName not in globalKernelRegistry:
                        raise RuntimeError(
                            "{} is not a known quantum kernel (was it annotated?)."
                            .format(otherFuncName))
                    values = [
                        self.popValue()
                        for _ in range(len(self.valueStack) - 1)
                    ]
                    values.reverse()
                    if len(values) != len(
                            globalKernelRegistry[otherFuncName].arguments):
                        raise RuntimeError(
                            "incorrect number of runtime arguments for cudaq.control({},..) call."
                            .format(otherFuncName))
                    controls = self.popValue()
                    quake.ApplyOp([], [], [controls],
                                  values,
                                  callee=FlatSymbolRefAttr.get(nvqppPrefix +
                                                               otherFuncName))
                    return
                elif node.func.attr == 'compute_action':
                    # There can only be 2 args here.
                    action = None
                    compute = None
                    actionArg = node.args[1]
                    if isinstance(actionArg, ast.Name):
                        actionName = actionArg.id
                        if actionName in self.symbolTable:
                            action = self.symbolTable[actionName]
                        else:
                            raise RuntimeError(
                                "could not find action lambda / function in the symbol table."
                            )
                    else:
                        action = self.popValue()

                    computeArg = node.args[0]
                    if isinstance(computeArg, ast.Name):
                        computeName = computeArg.id
                        if computeName in self.symbolTable:
                            compute = self.symbolTable[computeName]
                        else:
                            raise RuntimeError(
                                "could not find compute lambda / function in the symbol table."
                            )
                    else:
                        compute = self.popValue()

                    quake.ComputeActionOp(compute, action)
                    return

            if node.func.value.id in self.symbolTable:
                # Method call on one of our variables
                var = self.symbolTable[node.func.value.id]
                if quake.VeqType.isinstance(var.type):
                    # qreg or qview method call
                    if node.func.attr == 'back':
                        qrSize = quake.VeqSizeOp(self.getIntegerType(),
                                                 var).result
                        one = self.getConstantInt(1)
                        endOff = arith.SubIOp(qrSize, one)
                        if len(node.args):
                            # extract the subveq
                            startOff = arith.SubIOp(qrSize, self.popValue())
                            self.pushValue(
                                quake.SubVeqOp(self.getVeqType(), var, startOff,
                                               endOff).result)
                        else:
                            # extract the qubit...
                            self.pushValue(
                                quake.ExtractRefOp(self.getRefType(),
                                                   var,
                                                   -1,
                                                   index=endOff).result)
                        return
                    if node.func.attr == 'front':
                        zero = self.getConstantInt(0)
                        if len(node.args):
                            # extract the subveq
                            qrSize = self.popValue()
                            one = self.getConstantInt(1)
                            offset = arith.SubIOp(qrSize, one)
                            self.pushValue(
                                quake.SubVeqOp(self.getVeqType(), var, zero,
                                               offset).result)
                        else:
                            # extract the qubit...
                            self.pushValue(
                                quake.ExtractRefOp(self.getRefType(),
                                                   var,
                                                   -1,
                                                   index=zero).result)
                        return

            # We have a func name . ctrl
            if node.func.value.id in ['h', 'x', 'y', 'z', 's', 't'
                                     ] and node.func.attr == 'ctrl':
                target = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                opCtor([], [], controls, [target])
                return

            # We have a func name . ctrl
            if node.func.value.id == 'swap' and node.func.attr == 'ctrl':
                targetB = self.popValue()
                targetA = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                opCtor([], [], controls, [targetA, targetB])
                return

            if node.func.value.id in ['rx', 'ry', 'rz', 'r1'
                                     ] and node.func.attr == 'ctrl':
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
            if node.func.value.id in ['h', 'x', 'y', 'z', 's', 't'
                                     ] and node.func.attr == 'adj':
                target = self.popValue()
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                if quake.VeqType.isinstance(target.type):

                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(self.getRefType(),
                                               target,
                                               -1,
                                               index=iterVal).result
                        opCtor([], [], [], [q], is_adj=True)

                    veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                              target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [], [], [target], is_adj=True)
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(
                            target.type))

            if node.func.value.id in ['rx', 'ry', 'rz', 'r1'
                                     ] and node.func.attr == 'adj':
                target = self.popValue()
                param = self.popValue()
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                if quake.VeqType.isinstance(target.type):

                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(self.getRefType(),
                                               target,
                                               -1,
                                               index=iterVal).result
                        opCtor([], [param], [], [q], is_adj=True)

                    veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                              target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [param], [], [target], is_adj=True)
                    return
                else:
                    raise Exception(
                        'adj quantum operation on incorrect type {}.'.format(
                            target.type))

    def visit_List(self, node):
        if self.verbose:
            print('[Visit List] {}', ast.unparse(node))
        self.generic_visit(node)

        valueTys = [
            quake.VeqType.isinstance(v.type) or quake.RefType.isinstance(v.type)
            for v in self.valueStack
        ]
        if False not in valueTys:
            # this is a list of quantum types,
            # concat them into a veq
            if len(self.valueStack) == 1:
                self.pushValue(self.popValue())
            else:
                # FIXME, may need to reverse the list here, order matters
                self.pushValue(
                    quake.ConcatOp(self.getVeqType(),
                                   [self.popValue() for _ in valueTys]).result)
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
            raise RuntimeError("unhandled constant: {}".format(
                ast.unparse(node)))

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
        elif cc.StdvecType.isinstance(var.type):
            eleTy = F64Type.get()
            elePtrTy = cc.PointerType.get(self.ctx, eleTy)
            vecPtr = cc.StdvecDataOp(elePtrTy, var).result
            eleAddr = cc.ComputePtrOp(
                elePtrTy, vecPtr, [idx],
                DenseI32ArrayAttr.get([-2147483648], context=self.ctx)).result
            self.pushValue(cc.LoadOp(eleAddr).result)
            return
        else:
            raise RuntimeError("unhandled subscript: {}".format(
                ast.unparse(node)))

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
                    totalSize = quake.VeqSizeOp(self.getIntegerType(64),
                                                iterable).result

                def functor(iter, idx):
                    return [
                        quake.ExtractRefOp(self.getRefType(),
                                           iter,
                                           -1,
                                           index=idx).result
                    ]

                extractFunctor = functor

            else:
                raise RuntimeError('{} iterable type not yet supported.'.format(
                    iterable.type))

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
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(self.ctx, elementType), iter, [idx],
                    DenseI32ArrayAttr.get([-2147483648],
                                          context=self.ctx)).result
                loaded = cc.LoadOp(eleAddr).result
                if IntegerType.isinstance(elementType):
                    return [loaded]
                elif cc.StructType.isinstance(elementType):
                    # Get struct types
                    types = cc.StructType.getTypes(elementType)
                    ret = []
                    for i, ty in enumerate(types):
                        ret.append(
                            cc.ExtractValueOp(
                                ty, loaded,
                                DenseI64ArrayAttr.get([i],
                                                      context=self.ctx)).result)
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

    def visit_If(self, node):
        if self.verbose:
            print("[Visit If = {}]".format(ast.unparse(node)))

        self.visit(node.test)
        condition = self.popValue()
        if self.getIntegerType(1) != condition.type:
            # not equal to 0, then true
            condPred = IntegerAttr.get(self.getIntegerType(), 1)
            condition = arith.CmpIOp(condPred, condition,
                                     self.getConstantInt(0)).result

        ifOp = cc.IfOp([], condition)
        thenBlock = Block.create_at_start(ifOp.thenRegion, [])
        with InsertionPoint(thenBlock):
            [self.visit(b) for b in node.body]
            cc.ContinueOp([])

        if len(node.orelse) > 0:
            elseBlock = Block.create_at_start(ifOp.elseRegion, [])
            with InsertionPoint(elseBlock):
                [self.visit(b) for b in node.orelse]
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
                self.pushValue(arith.AddIOp(left, right).result)
                return
            else:
                raise RuntimeError("unhandled BinOp.Add types: {}".format(
                    ast.unparse(node)))

        if isinstance(node.op, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.SubIOp(left, right).result)
                return
            else:
                raise RuntimeError("unhandled BinOp.Add types: {}".format(
                    ast.unparse(node)))
        if isinstance(node.op, ast.FloorDiv):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.FloorDivSIOp(left, right).result)
                return
            else:
                raise RuntimeError("unhandled BinOp.FloorDiv types: {}".format(
                    ast.unparse(node)))
        if isinstance(node.op, ast.Div):
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(arith.DivFOp(left, right).result)
            return
        if isinstance(node.op, ast.Pow):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                # math.ipowi does not lower to llvm as is
                # workaround, use math to funcs conversion
                self.pushValue(math.IPowIOp(left, right).result)
                return

            if F64Type.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                self.pushValue(math.FPowIOp(left, right).result)
                return

            # now we know the types are different, default to float
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(math.PowFOp(left, right).result)
            return
        if isinstance(node.op, ast.Mult):
            if F64Type.isinstance(left.type):
                if not F64Type.isinstance(right.type):
                    right = arith.SIToFPOp(self.getFloatType(), right).result

            # FIXME more type checks
            self.pushValue(arith.MulFOp(left, right).result)
            return
        if isinstance(node.op, ast.Mod):
            if F64Type.isinstance(left.type):
                left = arith.FPToSIOp(self.getIntegerType(), left).result
            if F64Type.isinstance(right.type):
                right = arith.FPToSIOp(self.getIntegerType(), right).result

            self.pushValue(arith.RemUIOp(left, right).result)
            return
        else:
            raise RuntimeError("unhandled binary operator: {}, {}".format(
                ast.unparse(node), node.op))

    def visit_Name(self, node):
        if self.verbose:
            print("[Visit Name {}]".format(node.id))

        if node.id in self.symbolTable:
            value = self.symbolTable[node.id]
            if cc.PointerType.isinstance(value.type):
                loaded = cc.LoadOp(value).result
                self.pushValue(loaded)
            elif cc.CallableType.isinstance(value.type):
                return
            else:
                self.pushValue(self.symbolTable[node.id])
            return


def compile_to_quake(astModule, **kwargs):
    global globalAstRegistry
    verbose = 'verbose' in kwargs and kwargs['verbose']

    # Create the AST Bridge
    bridge = PyASTBridge(verbose=verbose)

    # First we need to find any dependent kernels, they have to be
    # built as part of this ModuleOp...
    vis = FindDepKernelsVisitor()
    vis.visit(astModule)
    depKernels = vis.depKernels

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
