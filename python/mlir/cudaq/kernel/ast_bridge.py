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
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.dialects import builtin, func, arith

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

    def getIntegerType(self, width):
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        return IntegerAttr.get(type, value)

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

    def typeFromStr(self, typeStr):
        if typeStr == 'int':
            return self.getIntegerType(64)
        elif typeStr == 'float':
            return F64Type.get()
        else:
            raise Exception('{} is not a supported type yet.'.format(typeStr))

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
            self.argTypes = [
                self.typeFromStr(arg.annotation.id) for arg in node.args.args
            ]
            argNames = [arg.arg for arg in node.args.args]

            fullName = nvqppPrefix + node.name
            # Create the function and the entry block
            f = func.FuncOp(fullName, (self.argTypes, []),
                            loc=self.loc)
            f.attributes.__setitem__('cudaq-entrypoint', UnitAttr.get())
            e = f.add_entry_block()

            # Add the block args to the symbol table
            blockArgs = e.arguments
            for i, b in enumerate(blockArgs):
                self.symbolTable[argNames[i]] = b

            # Set the insertion point to the start of the entry block
            with InsertionPoint(e):
                self.generic_visit(node)
                ret = func.ReturnOp([])

            attr = DictAttr.get({fullName: StringAttr.get(
                fullName+'_entryPointRewrite', context=self.ctx)}, context=self.ctx)
            self.module.operation.attributes.__setitem__(
                'quake.mangled_name_map', attr)

    def visit_Assign(self, node):
        self.generic_visit(node)
        self.symbolTable[node.targets[0].id] = self.popValue()

    def visit_Call(self, node):
        if self.verbose:
            print("[Visit Call]")

        # do not walk the FunctionDef decorator_list args
        if isinstance(
                node.func, ast.Attribute
        ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
            return

        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ["h", "x", "y", "z", "s", "t"]:
                # should have 1 value on the stack if
                # this is a vanilla hadamard
                qubit = self.popValue()
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubit])

        elif isinstance(node.func, ast.Attribute):
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
                # controls = [c for c in controls]
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                opCtor([], [], controls, [target])

    def visit_Constant(self, node):
        if self.verbose:
            print("[Visit Constant {}]".format(node.value))
        if isinstance(node.value, int):
            i64Ty = self.getIntegerType(64)
            self.pushValue(
                arith.ConstantOp(i64Ty,
                                 self.getIntegerAttr(i64Ty,
                                                     node.value)).result)
            return
        else:
            raise Exception("unhandled constant: {}".format(ast.unparse(node)))

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
            raise Exception(
                "unhandled subscript: {}".format(ast.unparse(node)))

    def visit_For(self, node):
        if node.iter.func.id != 'range' and len(node.iter.args) == 1:
            raise Exception(
                "CUDA Quantum only supports `for VAR in range(UPPER):`")

        if self.verbose:
            print('[Visit For]')

        # Get the rangeOperand, this is the upper bound on the loop
        self.generic_visit(node.iter)
        rangeOperand = self.popValue()

        # Get the name of the variable
        varName = node.target.id

        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

        # create a scope
        scope = cc.ScopeOp([])
        scopeBlock = Block.create_at_start(scope.initRegion, [])
        with InsertionPoint(scopeBlock):
            alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, iTy),
                                 TypeAttr.get(iTy)).result
            cc.StoreOp(zero, alloca)
            self.symbolTable[varName] = alloca

            loop = cc.LoopOp([], [], BoolAttr.get(False))
            bodyBlock = Block.create_at_start(loop.bodyRegion, [])
            with InsertionPoint(bodyBlock):
                self.generic_visit(node.body[0])
                cc.ContinueOp([])

            whileBlock = Block.create_at_start(loop.whileRegion, [])
            with InsertionPoint(whileBlock):
                loaded = cc.LoadOp(alloca)
                c = arith.CmpIOp(IntegerAttr.get(iTy, 2),
                                 loaded, rangeOperand).result
                cc.ConditionOp(c, [])

            stepBlock = Block.create_at_start(loop.stepRegion, [])
            with InsertionPoint(stepBlock):
                loaded = cc.LoadOp(alloca)
                incr = arith.AddIOp(loaded, one).result
                cc.StoreOp(incr, alloca)
                cc.ContinueOp([])
            cc.ContinueOp([])

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
                raise Exception(
                    "unhandled BinOp.Add types: {}".format(ast.unparse(node)))

        if isinstance(node.op, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.SubIOp(left, right).result)
                return
            else:
                raise Exception(
                    "unhandled BinOp.Add types: {}".format(ast.unparse(node)))
        else:
            raise Exception(
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
        # elif node.id == 'cudaq':
        #     return
        # else:
        #     raise Exception("unhandled name node: {}".format(ast.unparse(node)))


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
