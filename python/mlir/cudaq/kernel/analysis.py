# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast, inspect
from .utils import globalAstRegistry, globalKernelRegistry


class MidCircuitMeasurementAnalyzer(ast.NodeVisitor):
    """The `MidCircuitMeasurementAnalyzer` is a utility class searches for 
       common measurement - conditional patterns to indicate to the runtime 
       that we have a circuit with mid-circuit measurement and subsequent conditional 
       quantum operation application."""

    def __init__(self):
        self.measureResultsVars = []
        self.hasMidCircuitMeasures = False

    def isMeasureCallOp(self, node):
        return isinstance(
            node, ast.Call) and node.__dict__['func'].id in ['mx', 'my', 'mz']

    def visit_Assign(self, node):
        target = node.targets[0]
        if not 'func' in node.value.__dict__:
            return
        creatorFunc = node.value.func
        if 'id' in creatorFunc.__dict__ and creatorFunc.id in [
                'mx', 'my', 'mz'
        ]:
            self.measureResultsVars.append(target.id)

    def visit_If(self, node):
        condition = node.test
        # catch `if mz(q)`
        if self.isMeasureCallOp(condition):
            self.hasMidCircuitMeasures = True
            return

        # Catch if val, where `val = mz(q)`
        if 'id' in condition.__dict__ and condition.id in self.measureResultsVars:
            self.hasMidCircuitMeasures = True
        # Catch `if UnaryOp mz(q)`
        elif isinstance(condition, ast.UnaryOp):
            self.hasMidCircuitMeasures = self.isMeasureCallOp(condition.operand)
        # Catch `if something BoolOp mz(q)`
        elif isinstance(condition,
                        ast.BoolOp) and 'values' in condition.__dict__:
            for node in condition.__dict__['values']:
                if self.isMeasureCallOp(node):
                    self.hasMidCircuitMeasures = True
                    break


class RewriteMeasures(ast.NodeTransformer):
    """
    This NodeTransformer will analyze the AST for measurement 
    nodes that do not provide a `register_name=` keyword. If found 
    it will replace that node with one that sets the register_name to 
    the variable name in the assignment. 
    """

    def visit_FunctionDef(self, node):
        node.decorator_list.clear()
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # We only care about nodes with a Name target (mz,my,mx)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # The value has to be a Call node
            if isinstance(node.value, ast.Call):
                # The function we are calling should be Named
                if not isinstance(node.value.func, ast.Name):
                    return node

                # Make sure we are only seeing measurements
                if node.value.func.id not in ['mx', 'my', 'mz']:
                    return node

                # If we already have a register_name keyword
                # then we don't have to do anything
                if len(node.value.keywords):
                    for keyword in node.value.keywords:
                        if keyword.arg == 'register_name':
                            return node

                # If here, we have a measurement with no register name
                # We'll add one here
                newConstant = ast.Constant(value=node.targets[0].id)
                newCall = ast.Call(func=node.value.func,
                                   value=node.targets[0].id,
                                   args=node.value.args,
                                   keywords=[
                                       ast.keyword(arg='register_name',
                                                   value=newConstant)
                                   ])
                ast.copy_location(newCall, node.value)
                ast.copy_location(newConstant, node.value)
                ast.fix_missing_locations(newCall)
                node.value = newCall
                return node

        return node


class MatrixToRowMajorList(ast.NodeTransformer):

    def visit_Call(self, node):

        self.generic_visit(node)

        if not isinstance(node.func, ast.Attribute):
            return node 
        
        # is an attribute
        if not node.func.value.id in ['numpy','np']:
            return node 
        
        if not node.func.attr == 'array':
            return node 
        
        # this is an np.array
        args = node.args

        if len(args) != 1 and not isinstance(args[0], ast.List):
            return node 
        

        # [[this], [this], [this], [this], ...]
        subLists = args[0].elts 
        # convert to [this, this, this, this, ...]
        newElts = []
        for l in subLists:
            for e in l.elts:
                newElts.append(e)

        newList = ast.List(elts=newElts, ctx=ast.Load())
        newCall = ast.Call(func=node.func, args=[newList], keywords=[])
        ast.copy_location(newCall, node)
        ast.fix_missing_locations(newCall)
        return newCall

class LambdaOrLambdaAssignToFunctionDef(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.args[0], ast.Lambda):
                n = ast.FunctionDef(name=node.targets[0].id, args=node.value.args[0].args, body=[ast.Return(value=node.value.args[0].body)], decorator_list=[])
                ast.copy_location(n, node.value)
                ast.fix_missing_locations(n)
                for a in n.args.args:
                    a.annotation = ast.Name(id='float')
                return n

def preprocessCustomOperationLambda(unitaryCallable):
    unitarySrc = inspect.getsource(unitaryCallable)
    leadingSpaces = len(unitarySrc) - len(unitarySrc.lstrip())
    unitarySrc = '\n'.join(
        [line[leadingSpaces:] for line in unitarySrc.split('\n')])
    unitaryModule = ast.parse(unitarySrc)
    MatrixToRowMajorList().visit(unitaryModule)
    LambdaOrLambdaAssignToFunctionDef().visit(unitaryModule)
    return unitaryModule 

class FindDepKernelsVisitor(ast.NodeVisitor):

    def __init__(self):
        self.depKernels = {}

    def visit_FunctionDef(self, node):
        """
        Here we will look at this Functions arguments, if 
        there is a Callable, we will add any seen kernel/AST with the same 
        signature to the dependent kernels map. This enables the creation 
        of ModuleOps that contain all the functions necessary to inline and 
        synthesize callable block arguments.
        """
        for arg in node.args.args:
            annotation = arg.annotation
            if annotation == None:
                raise RuntimeError(
                    'cudaq.kernel functions must have argument type annotations.'
                )
            if isinstance(annotation,
                          ast.Subscript) and annotation.value.id == 'Callable':
                if not hasattr(annotation, 'slice'):
                    raise RuntimeError(
                        'Callable type must have signature specified.')

                # This is callable, let's add all in scope kernels
                # FIXME only add those with the same signature
                self.depKernels = {k: v for k, v in globalAstRegistry.items()}

        self.generic_visit(node)

    def visit_Call(self, node):
        """
        Here we look for function calls within this kernel. We will 
        add these to dependent kernels dictionary. We will also look for 
        kernels that are passed to control and adjoint.
        """
        if hasattr(node, 'func'):
            if isinstance(node.func,
                          ast.Name) and node.func.id in globalAstRegistry:
                self.depKernels[node.func.id] = globalAstRegistry[node.func.id]
            elif isinstance(node.func, ast.Attribute):
                if node.func.value.id == 'cudaq' and node.func.attr in [
                        'control', 'adjoint'
                ] and node.args[0].id in globalAstRegistry:
                    self.depKernels[node.args[0].id] = globalAstRegistry[
                        node.args[0].id]
