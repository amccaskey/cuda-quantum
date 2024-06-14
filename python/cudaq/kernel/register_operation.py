import inspect
import importlib
import ast 
from .ast_bridge import CapturedDataStorage, PyASTBridge

class PyRegisterOpDecorator(object):

    def __init__(self, function, *args, num_targets=None, num_params=None):
        self.kernelFunction = function
        print('args: ', *args)

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


def register_operation(function=None, *args, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA-Q language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA-Q kernel and should be compile and executed on 
    an available quantum coprocessor.

    Verbose logging can be enabled via `verbose=True`. 
    """
    if function:
        return PyRegisterOpDecorator(function, *args)
    else:

        def wrapper(function):
            return PyRegisterOpDecorator(function, *args, **kwargs)

        return wrapper
