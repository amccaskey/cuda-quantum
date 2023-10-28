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
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

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
        self.inForBodyStack = deque()
        self.inIfStmtBlockStack = deque()
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

    def pushForBodyStack(self, bodyBlockArgs):
        self.inForBodyStack.append(bodyBlockArgs)

    def popForBodyStack(self):
        self.inForBodyStack.pop()

    def pushIfStmtBlockStack(self):
        self.inIfStmtBlockStack.append(0)

    def popIfStmtBlockStack(self):
        self.inIfStmtBlockStack.pop()

    def isInForBody(self):
        return len(self.inForBodyStack) > 0

    def isInIfStmtBlock(self):
        return len(self.inIfStmtBlockStack) > 0

    def hasTerminator(self, block):
        if len(block.operations) > 0:
            return cudaq_runtime.isTerminator(
                block.operations[len(block.operations) - 1])
        return False

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

        if isinstance(annotation,
                      ast.Subscript) and annotation.value.id == 'list':
            if not hasattr(annotation, 'slice'):
                raise RuntimeError('list subscript missing slice node.')

            # expected that slice is a Name node
            listEleTy = self.mlirTypeFromAnnotation(annotation.slice)
            return cc.StdvecType.get(self.ctx, listEleTy)

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
            self.pushForBodyStack(bodyBlock.arguments)
            bodyBuilder(bodyBlock.arguments[0])
            if not self.hasTerminator(bodyBlock):
                cc.ContinueOp(bodyBlock.arguments)
            self.popForBodyStack()

        stepBlock = Block.create_at_start(loop.stepRegion, [iTy])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], stepVal).result
            cc.ContinueOp([incr])

        loop.attributes.__setitem__('invariant', UnitAttr.get())
        return

    def __applyQuantumOperation(self, opName, parameters, targets):
        opCtor = getattr(quake, '{}Op'.format(opName.title()))
        for quantumValue in targets:
            if quake.VeqType.isinstance(quantumValue.type):

                def bodyBuilder(iterVal):
                    q = quake.ExtractRefOp(self.getRefType(),
                                           quantumValue,
                                           -1,
                                           index=iterVal).result
                    opCtor([], parameters, [], [q])

                veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                          quantumValue).result
                self.createInvariantForLoop(veqSize, bodyBuilder)
            elif quake.RefType.isinstance(quantumValue.type):
                opCtor([], parameters, [], [quantumValue])
            else:
                raise Exception(
                    'quantum operation on incorrect type {}.'.format(
                        quantumValue.type))
        return

    def needsStackSlot(self, type):
        """
        Return true if this is a type that has been "passed by value" and 
        needs a stack slot created (i.e. a cc.alloca) for use throughout the 
        function. 
        """
        # FIXME add more as we need them
        return F64Type.isinstance(type) or IntegerType.isinstance(type)

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

            # Get the potential docstring
            self.docstring = ast.get_docstring(node)

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

            # Set the insertion point to the start of the entry block
            with InsertionPoint(self.entry):
                self.buildingEntryPoint = True
                # Add the block args to the symbol table,
                # create a stack slot for value arguments
                blockArgs = self.entry.arguments
                for i, b in enumerate(blockArgs):
                    if self.needsStackSlot(b.type):
                        stackSlot = cc.AllocaOp(
                            cc.PointerType.get(self.ctx, b.type),
                            TypeAttr.get(b.type)).result
                        cc.StoreOp(b, stackSlot)
                        self.symbolTable[argNames[i]] = stackSlot
                    else:
                        self.symbolTable[argNames[i]] = b

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
        """
        Map an assign operation in the AST to an equivalent variable value assignment 
        in the MLIR. This method will first see if this is a tuple assignment, enabling one 
        to assign multiple values in a single statement, 

        a, b, c = cudaq.qubit(), cudaq.qubit(), cudaq.qvector(2)

        For all assignments, the variable name will be used as a key for the symbol table, 
        mapping to the corresponding MLIR Value. For values of ref / veq, i1, or cc.callable, 
        the values will be stored directly in the table. For all aother values, the variable 
        will be allocated with a cc.alloca op, and the loaded value will be stored in the 
        symbol table.
        """
        if self.verbose:
            print('[Visit Assign {}]'.format(ast.unparse(node)))

        # Retain the variable name for potential children (like mz(q, registerName=...))
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.currentAssignVariableName = str(node.targets[0].id)
            self.visit(node.value)
            self.currentAssignVariableName = None
        else:
            self.visit(node.value)

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
            elif varNames[i] in self.symbolTable:
                cc.StoreOp(value, self.symbolTable[varNames[i]])
            else:
                # We should allocate and store
                alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, value.type),
                                     TypeAttr.get(value.type)).result
                cc.StoreOp(value, alloca)
                self.symbolTable[varNames[i]] = alloca

    def visit_Attribute(self, node):
        """
        Visit an attribute node and map to valid MLIR code. This method specifically 
        looks for attributes like method calls (e.g. veq size), or common attributes we'll 
        see from ubiquitous external modules like numpy.
        """
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
        """
        Map a Python Call operation to equivalent MLIR. This method will first check 
        for call operations that are ast.Name nodes in the tree (the name of a function to call). 
        It will handle the Python range(start, stop, step) function by creating an array of 
        integers to loop through via an invariant CC loop operation. Subsequent users of the 
        range() result can iterate through the elements of the returned cc.array. It will handle the 
        Python enumerate(iterable) function by constructing another invariant loop that builds up and 
        array of cc.struct<i64, T>, representing the counter and the element. 

        It will next handle any quantum operation (optionally with a rotation parameter). 
        Single target operations can be represented that take a single qubit reference,
        multiple single qubits, or a vector of qubits, where the latter two 
        will apply the operation to every qubit in the vector: 

        q, r = cudaq.qubit(), cudaq.qubit()
        qubits = cudaq.qubit(2)

        x(q) # apply x to q
        x(q, r) # apply x to q and r
        x(qubits) # for q in qubits: apply x 
        ry(np.pi, qubits)

        Valid single qubit operations are h, x, y, z, s, t, rx, ry, rz, r1. 

        Measurements mx, my, mz are mapped to corresponding quake operations and the return i1 
        value is added to the value stack. Measurements of single qubit reference and registers of 
        qubits are supported. 

        General calls to previously seen CUDA Quantum kernels are supported. By this we mean that 
        an kernel can not be invoked from a kernel unless it was defined before the current kernel.
        Kernels can also be reversed or controlled with cudaq.adjoint(kernel, ...) and cudaq.control(kernel, ...).

        Finally, general operation modifiers are supported, specifically OPERATION.adj and OPERATION.ctrl 
        for adjoint and control synthesis of the operation.  
        """
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
            if node.func.id == "len":
                listVal = self.popValue()
                assert cc.StdvecType.isinstance(listVal.type)
                self.pushValue(
                    cc.StdvecSizeOp(self.getIntegerType(), listVal).result)
                return

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
                    # qreg-like or stdvec=like thing thing
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
                        iterEleTy = cc.StdvecType.getElementType(iterable.type)
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
                self.__applyQuantumOperation(node.func.id, [], qubitTargets)
                return

            if node.func.id in ["rx", "ry", "rz", "r1"]:
                numValues = len(self.valueStack)
                qubitTargets = [self.popValue() for _ in range(numValues - 1)]
                qubitTargets.reverse()
                param = self.popValue()
                if IntegerType.isinstance(param.type):
                    param = arith.SIToFPOp(self.getFloatType(), param).result
                self.__applyQuantumOperation(node.func.id, [param],
                                             qubitTargets)
                return

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
                fType = otherKernel.type
                if len(fType.inputs) != len(node.args):
                    raise RuntimeError(
                        "invalid number of arguments passed to callable {} ({} vs required {})"
                        .format(node.func.id, len(node.args),
                                len(fType.inputs)))

                values = [self.popValue() for _ in node.args]
                values.reverse()
                func.CallOp(otherKernel, values)
                return

            elif node.func.id in self.symbolTable:
                val = self.symbolTable[node.func.id]
                if cc.CallableType.isinstance(val.type):
                    numVals = len(self.valueStack)
                    values = [self.popValue() for _ in range(numVals)]
                    # FIXME check value types match callable type signature
                    callableTy = cc.CallableType.getFunctionType(val.type)
                    # if len(callableTy.inputs) != len(values):
                    # raise RuntimeError("invalid number of arguments passed to callable {}".format(node.func.id))
                    callable = cc.CallableFuncOp(callableTy, val).result
                    func.CallIndirectOp([], callable, values)
                    return

            elif node.func.id == 'exp_pauli':
                pauliWord = self.popValue()
                qubits = self.popValue()
                theta = self.popValue()
                if IntegerType.isinstance(theta.type):
                    theta = arith.SIToFPOp(self.getFloatType(), theta).result
                quake.ExpPauliOp(theta, qubits, pauliWord)
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
                if IntegerType.isinstance(param.type):
                    param = arith.SIToFPOp(self.getFloatType(), param).result
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
                if IntegerType.isinstance(param.type):
                    param = arith.SIToFPOp(self.getFloatType(), param).result
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

    def visit_ListComp(self, node):
        """
        This method currently supports lowering simple list comprehensions 
        to the MLIR. By simple, we mean expressions like 
        `[expr(iter) for iter in iterable]`
        """

        if len(node.generators) > 1:
            raise RuntimeError(
                "currently only support single generators for list comprehension."
            )

        if not isinstance(node.generators[0].target, ast.Name):
            raise RuntimeError(
                "only support named targets in list comprehension")

        # now we know we have [expr(r) for r in iterable]
        # reuse what we do in visit_For()
        forNode = ast.For()
        forNode.iter = node.generators[0].iter
        forNode.target = node.generators[0].target
        forNode.body = [node.elt]
        self.visit_For(forNode)
        return

    def visit_List(self, node):
        """
        This method will visit the ast.List node and represent lists of 
        quantum typed values as a concatenated quake.ConcatOp producing a 
        single veq instances. 
        """
        if self.verbose:
            print('[Visit List] {}', ast.unparse(node))
        self.generic_visit(node)

        listElementValues = [self.popValue() for _ in range(len(node.elts))]
        listElementValues.reverse()
        valueTys = [
            quake.VeqType.isinstance(v.type) or quake.RefType.isinstance(v.type)
            for v in listElementValues
        ]
        if False not in valueTys:
            # this is a list of quantum types,
            # concat them into a veq
            if len(listElementValues) == 1:
                self.pushValue(listElementValues[0])
            else:
                # FIXME, may need to reverse the list here, order matters
                self.pushValue(
                    quake.ConcatOp(self.getVeqType(), listElementValues).result)
            return

        # not a list of quantum types
        firstTy = listElementValues[0].type
        for v in listElementValues:
            if firstTy != v.type:
                raise RuntimeError(
                    "non-homogenous list not allowed - must all be same type: {}"
                    .format([v.type for v in values]))

        arrSize = self.getConstantInt(len(node.elts))
        arrTy = cc.ArrayType.get(self.ctx, listElementValues[0].type)
        alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                             TypeAttr.get(listElementValues[0].type),
                             seqSize=arrSize).result

        for i, v in enumerate(listElementValues):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(self.ctx, listElementValues[0].type), alloca,
                [self.getConstantInt(i)],
                DenseI32ArrayAttr.get([-2147483648], context=self.ctx)).result
            cc.StoreOp(v, eleAddr)

        self.pushValue(
            cc.StdvecInitOp(
                cc.StdvecType.get(self.ctx, listElementValues[0].type), alloca,
                arrSize).result)

    def visit_Constant(self, node):
        """
        Convert constant values in the code to constant values in the MLIR. 
        """
        if self.verbose:
            print("[Visit Constant {}]".format(node.value))
        if isinstance(node.value, int):
            self.pushValue(self.getConstantInt(node.value))
            return
        elif isinstance(node.value, float):
            self.pushValue(self.getConstantFloat(node.value))
            return
        elif isinstance(node.value, str):
            # Do not process the function doc string
            if self.docstring != None:
                if node.value.strip() == self.docstring.strip():
                    return

            strLitTy = cc.PointerType.get(
                self.ctx,
                cc.ArrayType.get(self.ctx, self.getIntegerType(8),
                                 len(node.value) + 1))
            self.pushValue(
                cc.CreateStringLiteralOp(strLitTy,
                                         StringAttr.get(node.value)).result)
            return
        else:
            raise RuntimeError("unhandled constant: {}".format(
                ast.unparse(node)))

    def visit_Subscript(self, node):
        """
        Convert element extractions (__getitem__, operator[](idx), q[1:3]) to 
        corresponding extraction or slice code in the MLIR. This method handles 
        extraction for veq types and Stdvec types. 
        """
        if self.verbose:
            print("[Visit Subscript]")

        # handle complex slice, VAR[lower:upper]
        if isinstance(node.slice, ast.Slice):

            self.visit(node.value)
            var = self.popValue()

            lowerVal, upperVal, stepVal = (None, None, None)
            if node.slice.lower is not None:
                self.visit(node.slice.lower)
                lowerVal = self.popValue()
            else:
                lowerVal = self.getConstantInt(0)
            if node.slice.upper is not None:
                self.visit(node.slice.upper)
                upperVal = self.popValue()
            else:
                if quake.VeqType.isinstance(var.type):
                    upperVal = quake.VeqSizeOp(self.getIntegerType(64),
                                               var).result
                elif cc.StdvecType.isinstance(var.type):
                    upperVal = cc.StdvecSizeOp(self.getIntegerType(),
                                               var).result
                else:
                    raise RuntimeError(
                        "unhandled upper slice == None, can't handle type {}".
                        format(var.type))

            if node.slice.step is not None:
                raise RuntimeError("step value in slice is not supported.")

            if quake.VeqType.isinstance(var.type):
                # Upper bound is exclusive
                upperVal = arith.SubIOp(upperVal, self.getConstantInt(1)).result
                self.pushValue(
                    quake.SubVeqOp(self.getVeqType(), var, lowerVal,
                                   upperVal).result)
            elif cc.StdvecType.isinstance(var.type):
                eleTy = cc.StdvecType.getElementType(var.type)
                ptrTy = cc.PointerType.get(self.ctx, eleTy)
                nElementsVal = arith.SubIOp(upperVal, lowerVal).result
                # need to compute the distance between upperVal and lowerVal
                # then slice is stdvecdataOp + computeptr[lower] + stdvecinit[ptr,distance]
                vecPtr = cc.StdvecDataOp(ptrTy, var).result
                ptr = cc.ComputePtrOp(
                    ptrTy, vecPtr, [lowerVal],
                    DenseI32ArrayAttr.get([-2147483648],
                                          context=self.ctx)).result
                self.pushValue(
                    cc.StdvecInitOp(var.type, ptr, nElementsVal).result)
            else:
                raise RuntimeError(
                    "unhandled slice operation, cannot handle type {}".format(
                        var.type))

            return

        self.generic_visit(node)

        assert len(self.valueStack) > 1

        # get the last name, should be name of var being subscripted
        var = self.popValue()
        idx = self.popValue()

        # Support VAR[-1] -> last element, for veq VAR
        if quake.VeqType.isinstance(var.type) and isinstance(
                idx.owner.opview, arith.ConstantOp):
            if 'value' in idx.owner.attributes:
                try:
                    concreteIntAttr = IntegerAttr(idx.owner.attributes['value'])
                    idxConcrete = concreteIntAttr.value
                    if idxConcrete == -1:
                        qrSize = quake.VeqSizeOp(self.getIntegerType(),
                                                 var).result
                        one = self.getConstantInt(1)
                        endOff = arith.SubIOp(qrSize, one)
                        self.pushValue(
                            quake.ExtractRefOp(self.getRefType(),
                                               var,
                                               -1,
                                               index=endOff).result)
                        return
                except ValueError as e:
                    pass

        # Made it here, general VAR[idx], handle veq and stdvec
        if quake.VeqType.isinstance(var.type):
            qrefTy = self.getRefType()
            self.pushValue(
                quake.ExtractRefOp(qrefTy, var, -1, index=idx).result)
        elif cc.StdvecType.isinstance(var.type):
            eleTy = cc.StdvecType.getElementType(var.type)
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
        """
        Visit the For Stmt node. This node represents the typical 
        Python for statement, `for VAR in ITERABLE`. Currently supported 
        ITERABLEs are the veq type, the stdvec type, and the result of 
        range() and enumerate(). 
        """

        if self.verbose:
            print('[Visit For]')

        self.visit(node.iter)
        assert len(self.valueStack) > 0 and len(self.valueStack) < 3

        totalSize = None
        iterable = None
        extractFunctor = None

        # It could be that its the only value we have,
        # in which case we know we have for var in iterable,
        # but we could also have another value on the stack,
        # the total size of the iterable, produced by range() / enumerate()
        if len(self.valueStack) == 1:
            # Get the iterable from the stack
            iterable = self.popValue()
            # for single iterables, we currently handle Veq and Stdvec types
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
            elif cc.StdvecType.isinstance(iterable.type):
                iterEleTy = cc.StdvecType.getElementType(iterable.type)
                totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                            iterable).result

                def functor(iter, idxVal):
                    elePtrTy = cc.PointerType.get(self.ctx, iterEleTy)
                    vecPtr = cc.StdvecDataOp(elePtrTy, iter).result
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [idxVal],
                        DenseI32ArrayAttr.get([-2147483648],
                                              context=self.ctx)).result
                    return [cc.LoadOp(eleAddr).result]

                extractFunctor = functor

            else:
                raise RuntimeError('{} iterable type not yet supported.'.format(
                    iterable.type))

        else:
            # In this case, we are coming from range() or enumerate(),
            # and the iterable is a cc.array and the total size of the
            # array is on the stack, pop it here
            totalSize = self.popValue()
            # Get the iterable from the stack
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

        # Get the name of the variable, VAR in for VAR in range(...),
        # could be a tuple of names too
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
            # we set the extract functor above, use it here
            values = extractFunctor(iterable, iterVar)
            for i, v in enumerate(values):
                self.symbolTable[varNames[i]] = v
            [self.visit(b) for b in node.body]

        self.createInvariantForLoop(totalSize, bodyBuilder)

    def visit_While(self, node):
        """
        Convert Python while statements into the equivalent CC LoopOp. 
        """
        if self.verbose:
            print("[Visit While = {}]".format(ast.unparse(node)))

        loop = cc.LoopOp([], [], BoolAttr.get(False))
        whileBlock = Block.create_at_start(loop.whileRegion, [])
        with InsertionPoint(whileBlock):
            # BUG you cannot print MLIR values while building the cc LoopOp while region.
            # verify will get called, no terminator yet, CCOps.cpp:520
            v = self.verbose
            self.verbose = False
            self.visit(node.test)
            condition = self.popValue()
            if self.getIntegerType(1) != condition.type:
                # not equal to 0, then compare with 1
                condPred = IntegerAttr.get(self.getIntegerType(), 1)
                condition = arith.CmpIOp(condPred, condition,
                                         self.getConstantInt(0)).result
            cc.ConditionOp(condition, [])
            self.verbose = v

        bodyBlock = Block.create_at_start(loop.bodyRegion, [])
        with InsertionPoint(bodyBlock):
            self.pushForBodyStack([])
            [self.visit(b) for b in node.body]
            if not self.hasTerminator(bodyBlock):
                cc.ContinueOp([])
            self.popForBodyStack()

    def visit_BoolOp(self, node):
        """
        Convert boolean operations into equivalent MLIR operations using 
        the Arith Dialect.
        """
        [self.visit(v) for v in node.values]
        numValues = len(self.valueStack)
        values = [self.popValue() for _ in range(numValues)]
        assert len(values) > 1, "boolean operation must have more than 1 value."

        if isinstance(node.op, ast.And):
            res = arith.AndIOp(values[0], values[1]).result
            for v in values[2:]:
                res = arith.AndIOp(res, v).result
            self.pushValue(res)
            return

    def visit_Compare(self, node):
        """
        Visit while loop compare operations and translate to equivalent MLIR. 
        Note, Python lets you construct expressions with multiple comparators, 
        here we limit ourselves to just a single comparator. 
        """

        if len(node.ops) > 1:
            raise RuntimeError("only single comparators are supported.")

        iTy = self.getIntegerType()

        if isinstance(node.left, ast.Name):
            if node.left.id not in self.symbolTable:
                raise RuntimeError(
                    "{} was not initialized before use in compare expression.".
                    format(node.left.id))

        self.visit(node.left)
        left = self.popValue()
        self.visit(node.comparators[0])
        comparator = self.popValue()
        op = node.ops[0]

        if isinstance(op, ast.Gt):
            if IntegerType.isinstance(left.type):
                if F64Type.isinstance(comparator.type):
                    raise RuntimeError(
                        "invalid rhs for comparison (f64 type and not i64 type)."
                    )

                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 4), left,
                                 comparator).result)
            elif F64Type.isinstance(left.type):
                if IntegerType.isinstance(comparator.type):
                    comparator = arith.SIToFPOp(self.getFloatType(),
                                                comparator).result
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 2), left,
                                 comparator).result)
            return

        if isinstance(op, ast.GtE):
            self.pushValue(
                arith.CmpIOp(self.getIntegerAttr(iTy, 5), left,
                             comparator).result)
            return

        if isinstance(op, ast.Lt):
            self.pushValue(
                arith.CmpIOp(self.getIntegerAttr(iTy, 2), left,
                             comparator).result)
            return

        if isinstance(op, ast.LtE):
            self.pushValue(
                arith.CmpIOp(self.getIntegerAttr(iTy, 7), left,
                             comparator).result)
            return

        if isinstance(op, ast.NotEq):
            self.pushValue(
                arith.CmpIOp(self.getIntegerAttr(iTy, 1), left,
                             comparator).result)
            return

        if isinstance(op, ast.Eq):
            self.pushValue(
                arith.CmpIOp(self.getIntegerAttr(iTy, 0), left,
                             comparator).result)
            return

    def visit_AugAssign(self, node):
        """
        Visit augment-assign operations (e.g. +=). 
        """
        target = None
        if isinstance(node.target,
                      ast.Name) and node.target.id in self.symbolTable:
            target = self.symbolTable[node.target.id]
        else:
            raise RuntimeError(
                "unable to get augment-assign target variable from symbol table."
            )

        self.visit(node.value)
        value = self.popValue()

        loaded = cc.LoadOp(target).result
        if isinstance(node.op, ast.Sub):
            # i -= 1 -> i = i - 1
            if IntegerType.isinstance(loaded.type):
                res = arith.SubIOp(loaded, value).result
                cc.StoreOp(res, target)
                return
            else:
                raise RuntimeError("unhandled AugAssign.Sub types: {}".format(
                    ast.unparse(node)))

        if isinstance(node.op, ast.Add):
            # i += 1 -> i = i + 1
            if IntegerType.isinstance(loaded.type):
                res = arith.AddIOp(loaded, value).result
                cc.StoreOp(res, target)
                return
            else:
                raise RuntimeError("unhandled AugAssign.Add types: {}".format(
                    ast.unparse(node)))

        if isinstance(node.op, ast.Mult):
            # i *= 3 -> i = i * 3
            if IntegerType.isinstance(loaded.type):
                res = arith.MulIOp(loaded, value).result
                cc.StoreOp(res, target)
                return
            else:
                raise RuntimeError("unhandled AugAssign.Mult types: {}".format(
                    ast.unparse(node)))

        raise RuntimeError("unhandled aug-assign operation: {}".format(
            ast.unparse(node)))

    def visit_If(self, node):
        """
        Map a Python ast.If node to an if statement operation in the CC dialect. 
        """
        if self.verbose:
            print("[Visit If = {}]".format(ast.unparse(node)))

        self.visit(node.test)
        condition = self.popValue()

        if self.getIntegerType(1) != condition.type:
            # not equal to 0, then compare with 1
            condPred = IntegerAttr.get(self.getIntegerType(), 1)
            condition = arith.CmpIOp(condPred, condition,
                                     self.getConstantInt(0)).result

        ifOp = cc.IfOp([], condition)
        thenBlock = Block.create_at_start(ifOp.thenRegion, [])
        with InsertionPoint(thenBlock):
            self.pushIfStmtBlockStack()
            [self.visit(b) for b in node.body]
            if not self.hasTerminator(thenBlock):
                cc.ContinueOp([])
            self.popIfStmtBlockStack()

        if len(node.orelse) > 0:
            elseBlock = Block.create_at_start(ifOp.elseRegion, [])
            with InsertionPoint(elseBlock):
                self.pushIfStmtBlockStack()
                [self.visit(b) for b in node.orelse]
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp([])
                self.popIfStmtBlockStack()

    def visit_UnaryOp(self, node):
        """
        Map unary operations in the Python AST to equivalents in MLIR.
        """
        if self.verbose:
            print("[Visit Unary = {}]".format(ast.unparse(node)))

        self.generic_visit(node)
        operand = self.popValue()
        if isinstance(node.op, ast.USub):
            # Make our lives easier for -1 used in variable subscript extraction
            if isinstance(node.operand,
                          ast.Constant) and node.operand.value == 1:
                self.pushValue(self.getConstantInt(-1))
                return

            if F64Type.isinstance(operand.type):
                self.pushValue(arith.NegFOp(operand).result)
            else:
                negOne = self.getConstantInt(-1)
                self.pushValue(arith.MulIOp(negOne, operand).result)
            return

        raise RuntimeError("unhandled UnaryOp: {}".format(ast.unparse(node)))

    def visit_Break(self, node):
        if self.verbose:
            print("[Visit Break]")

        if not self.isInForBody():
            raise RuntimeError("break statement outside of for loop body.")

        if self.isInIfStmtBlock():
            inArgs = [b for b in self.inForBodyStack[0]]
            cc.UnwindBreakOp(inArgs)
        else:
            cc.BreakOp([])

        return

    def visit_Continue(self, node):
        if self.verbose:
            print("[Visit Continue]")

        if not self.isInForBody():
            raise RuntimeError("continue statement outside of for loop body.")

        if self.isInIfStmtBlock():
            inArgs = [b for b in self.inForBodyStack[0]]
            cc.UnwindContinueOp(inArgs)
        else:
            cc.ContinueOp([])

    def visit_BinOp(self, node):
        """
        Visit binary operation nodes in the AST and map them to equivalents in the 
        MLIR. This method handles arithmetic operations between values. 
        """

        if self.verbose:
            print("[Visit BinaryOp = {}]".format(ast.unparse(node)))

        # Get the left and right parts of this expression
        self.visit(node.left)
        left = self.popValue()
        self.visit(node.right)
        right = self.popValue()

        # Basedon the op type and the leaf types, create the MLIR operator
        if isinstance(node.op, ast.Add):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AddIOp(left, right).result)
                return
            elif F64Type.isinstance(left.type):
                if IntegerType.isinstance(right.type):
                    right = arith.SIToFPOp(left.type, right).result
                self.pushValue(arith.AddFOp(left, right).result)
                return
            else:
                raise RuntimeError("unhandled BinOp.Add types: {}".format(
                    ast.unparse(node)))

        if isinstance(node.op, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.SubIOp(left, right).result)
                return
            else:
                raise RuntimeError("unhandled BinOp.Sub types: {}".format(
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
        """
        Visit ast.Name nodes and extract the correct value from the symbol table.
        """
        if self.verbose:
            print("[Visit Name {}]".format(node.id))

        if node.id in self.symbolTable:
            value = self.symbolTable[node.id]
            if cc.PointerType.isinstance(value.type):
                loaded = cc.LoadOp(value).result
                self.pushValue(loaded)
            # FIXME need to handle this one better
            elif cc.CallableType.isinstance(
                    value.type) and not BlockArgument.isinstance(value):
                return
            else:
                self.pushValue(self.symbolTable[node.id])
            return


def compile_to_mlir(astModule, **kwargs):
    """
    Compile the given Python AST Module for the CUDA Quantum 
    kernel FunctionDef to an MLIR ModuleOp. 
    Return both the ModuleOp and the list of function 
    argument types as MLIR Types. 

    This function will first check to see if there are any dependent 
    kernels that are required by this function. If so, those kernels 
    will also be compiled into the ModuleOp. The AST will be stored 
    later for future potential dependent kernel lookups. 
    """

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

    # Canonicalize the code
    pm = PassManager.parse("builtin.module(canonicalize,cse)",
                           context=bridge.ctx)
    pm.run(bridge.module)

    globalAstRegistry[bridge.name] = astModule

    return bridge.module, bridge.argTypes
