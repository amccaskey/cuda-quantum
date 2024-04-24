# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast, inspect

class CompilerPatternPDLContext(object):
    def __init__(self):
        self.pdl = '' 
    
    def append(self, code):
        self.pdl += code + '\n'
    
    def __str__(self):
        return self.pdl


class Value(object):
    def __init__(self, ctx, varName):
        # Use inspect to get name of this value
        self.name = varName
        ctx.append(f'let {self.name}: Value;')

class Op(object):
    def __init__(self, ctx, varName, *operands):
        # Use inspect to get the op name
        self.name = varName
        ctx.append("let {} = op<>({});".format(self.name, *[','.join([o.name for o in operands])]))

def rewrite (ctx, op, with_=None):
    ctx.append('rewrite {} with {}'.format(op.name, '{'))
    with_(ctx) 
    ctx.append('};')

def replace(ctx, op, with_=None):
    ctx.append('replace {} with {};'.format(op.name, with_.name))

def erase(ctx, op):
    ctx.append(f'erase {op.name};') 

def OpsCommute(ctx, *args):
    ctx.append('OpsCommute({});'.format(','.join([a.name for a in args])))
    return 

class InjectContext(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list=[]
        node.args = ast.arguments(args=[ast.arg(arg='ctx')], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        ast.fix_missing_locations(node)
        [self.visit(n) for n in node.body]
        return node 
    
    def visit_Assign(self, node):
        # retain name for call op
        self.currentAssignName = node.targets[0].id 
        self.visit(node.value) 
        self.currentAssignName = None
        return node

    def visit_Call(self, node):
        newArgs = [ast.Name(id='ctx', ctx=ast.Load())]
        if self.currentAssignName != None:
            newArgs += [ast.Constant(value=self.currentAssignName)]
        node.args = newArgs + node.args 
        ast.fix_missing_locations(node)
        return node

class CompilerPatternDecorator(object):
    def __init__(self, function):
        self.context = CompilerPatternPDLContext()
        self.function = function
        src = inspect.getsource(self.function)
         # Strip off the extra tabs
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])

        # Create the AST
        self.astModule = ast.parse(self.funcSrc)

        # astpretty.pprint(self.astModule.body[0])
        vis = InjectContext()
        newtree = vis.visit(self.astModule)
        s = ast.unparse(newtree)
        # s = 'import cudaq\nfrom cudaq import Value, Op, OpsCommute, replace, erase, rewrite\n'+s
        res = self.function.__globals__
        exec(
            compile(ast.fix_missing_locations(newtree),
                    filename='<ast>',
                    mode='exec'), res)
        self.function = res[self.function.__name__]

        self.context.append('''Constraint IsHermitian (op:Op<>);
Constraint IsQuakeOperation(op:Op<>);
Constraint IsSameName(op:Op<>, op1:Op<>);
Constraint OpsCommute(op0:Op<>, op1:Op<>) {
  IsQuakeOperation(op0);
  IsQuakeOperation(op1);
  IsHermitian(op0);
  IsSameName(op0, op1);
}''')
        self.context.append(f'Pattern {self.function.__name__} {{')
        self.function(self.context)
        self.context.append("}")

    def __call__ (self):
        return 

def compiler_pattern(function=None, **kwargs):
    if function:
        return CompilerPatternDecorator(function)

    def wrapper(function):
        return CompilerPatternDecorator(function, **kwargs)
    
    return wrapper 