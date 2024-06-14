import inspect
import importlib
import ast 
from .ast_bridge import CapturedDataStorage, PyASTBridge

class PyRegisterOpDecorator(object):

    def __init__(self, function, num_targets=None, num_params=None):
        self.kernelFunction = function

        src = inspect.getsource(self.kernelFunction)

        # Strip off the extra tabs
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])

        # Create the AST
        self.astModule = ast.parse(self.funcSrc)
        bridge = PyASTBridge(CapturedDataStorage(),knownResultType=list[complex], returnTypeIsFromPython=True)
        bridge.visit(self.astModule)
        print(bridge.module)


def register_operation(function=None, **kwargs):
    if function:
        return PyRegisterOpDecorator(function)
    else:

        def wrapper(function):
            return PyRegisterOpDecorator(function, **kwargs)

        return wrapper
