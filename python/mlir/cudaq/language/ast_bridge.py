# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
from collections import deque
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.dialects import builtin, func, arith


class TypedValue(object):

    def __init__(self, type, value):
        self.type = type
        self.value = value


class PyASTBridge(ast.NodeVisitor):

    def __init__(self):
        self.valueStack = deque()
        self.ctx = Context()
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.symbolTable = {}

    def getVeqType(self, size=None):
        if size == None:
            return quake.VeqType.get(self.ctx)
        return quake.VeqType.get(self.ctx, size)

    def getRefType(self):
        return quake.RefType.get(self.ctx)

    def getIntegerType(self, width):
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        return IntegerAttr.get(type, value)

    def pushValue(self, value):
        self.valueStack.push(value)

    def popValue(self):
        return self.valueStack.pop()

    def typeFromStr(self, typeStr):
        return self.getIntegerType(32)
    
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
            # FIXME throw an error if the types aren't annotated
            argTypes = [self.typeFromStr(arg.annotation.id) for arg in node.args.args]
            argNames = [arg.arg for arg in node.args.args]
            
            # Create the function and the entry block
            f = func.FuncOp('{}'.format(node.name), (argTypes, []), loc=self.loc)
            e = f.add_entry_block()
            
            # Add the block args to the symbol table
            blockArgs = e.arguments 
            for i, b in enumerate(blockArgs):
                self.symbolTable[argNames[i]] = TypedValue(argTypes[i], b)
            
            # Set the insertion point to the start of the entry block
            with InsertionPoint(e):
                self.generic_visit(node)
                ret = func.ReturnOp([])

    def visit_Assign(self, node):
        self.generic_visit(node)
        self.symbolTable[node.targets[0].id] = self.valueStack.pop()

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ["h"]:
                # should have 1 value on the stack if 
                # this is a vanilla hadamard
                qubit = self.valueStack.pop()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubit.value])

        elif isinstance(node.func, ast.Attribute):
            if node.func.value.id == 'cudaq' and node.func.attr == 'qvector':
                size = self.valueStack.pop()
                if hasattr(size.value, "literal_value") :
                    ty = self.getVeqType(size.value.literal_value)
                    qubits = quake.AllocaOp(ty)
                else:
                    ty = self.getVeqType()
                    qubits = quake.AllocaOp(ty, size=size.value)
                self.valueStack.append(TypedValue(ty, qubits.results[0]))
                return

            # We have a func name . ctrl
            if node.func.value.id in ['x'] and node.func.attr == 'ctrl':
                target = self.valueStack.pop()
                controls = [self.valueStack.pop() for i in range(len(self.valueStack))]
                controls = [c.value for c in controls]
                opCtor = getattr(quake, '{}Op'.format(node.func.value.id.title()))
                opCtor([],[],controls, [target.value])

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            i64Ty = self.getIntegerType(64)
            self.valueStack.append(
                TypedValue(
                    i64Ty,
                    arith.ConstantOp(i64Ty,
                                     self.getIntegerAttr(i64Ty, node.value))))
            return
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        assert len(self.valueStack) > 1
        
        # get the last name, should be name of var being subscripted
        var = self.valueStack.pop()
        idx = self.valueStack.pop()
        if quake.VeqType.isinstance(var.type):
            qrefTy = self.getRefType()
            self.valueStack.append(
                TypedValue(
                    qrefTy,
                    quake.ExtractRefOp(qrefTy, var.value, idx.value.value)))

    def visit_BinOp(self, node):
        self.generic_visit(node)
        print(type(node.op).__name__)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id in self.symbolTable:
            self.valueStack.append(self.symbolTable[node.id])

def compile_to_quake(astModule):
    # TODO
    bridge = PyASTBridge()
    bridge.visit(astModule)
    pm = PassManager.parse("builtin.module(canonicalize,cse)", context=bridge.ctx)
    pm.run(bridge.module)
    return bridge.module
