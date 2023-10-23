# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
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
                    'cudaq.kernel functions must have argument type annotations.')
            if isinstance(annotation, ast.Subscript) and annotation.value.id == 'Callable':
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
            if isinstance(node.func, ast.Name) and node.func.id in globalAstRegistry:
                self.depKernels[node.func.id] = globalAstRegistry[node.func.id]
            elif isinstance(node.func, ast.Attribute):
                if node.func.value.id == 'cudaq' and node.func.attr in ['control', 'adjoint'] and node.args[0].id in globalAstRegistry:
                    self.depKernels[node.args[0].id] = globalAstRegistry[node.args[0].id]

