#!/usr/bin/env python3

import ast, astpretty, tempfile, subprocess, os, shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=argparse.FileType('r'),
                    help='The file to process')
args = parser.parse_args()
fileContents = None
with args.file as file:
    fileContents = file.read()
if fileContents == None:
    raise RuntimeError("Invalid file, could not read contents.")

fileName = Path(args.file.name).stem
cudaq_install_path = Path(__file__).resolve().parent.parent
cudaq_extensions_path = str(cudaq_install_path) + os.path.sep + 'extensions'
cudaq_unitary_path = str(cudaq_extensions_path) + os.path.sep + 'unitaries'
cudaq_headers_path = str(cudaq_extensions_path) + os.path.sep + 'headers'
cudaq_irdl_path = str(cudaq_extensions_path) + os.path.sep + 'irdl'

# astpretty.pprint(ast.parse(fileContents))

if not os.path.exists(cudaq_extensions_path):
    os.makedirs(cudaq_extensions_path)

if not os.path.exists(cudaq_unitary_path):
    os.makedirs(cudaq_unitary_path)

if not os.path.exists(cudaq_headers_path):
    os.makedirs(cudaq_headers_path)

if not os.path.exists(cudaq_irdl_path):
    os.makedirs(cudaq_irdl_path)


class CustomOperation(object):

    def __init__(self):
        self.name = None
        self.num_parameters = 0
        self.cpp_generator = None


class CustomOperationVisitor(ast.NodeVisitor):

    def __init__(self):
        self.operations = []
        self.currentOperation = None
        self.traits = []

    def visit_ClassDef(self, node):
        # FIXME Screen out only quantum_operations
        self.currentOperation = CustomOperation()
        print(f'create new operation with name \'{node.name}\'')
        self.currentOperation.name = node.name
        [self.visit(n) for n in node.body]
        self.operations.append(self.currentOperation)
        self.currentOperation = None

    def visit_Assign(self, node: ast.Assign):
        if node.targets[0].id == 'num_parameters':
            print(
                f'{self.currentOperation.name} requires {node.value.value} parameters'
            )
            self.currentOperation.num_parameters = node.value.value
            return

        if node.targets[0].id == 'cpp_generator':
            print(
                f'{self.currentOperation.name} provides\n\n{node.value.value}\n\nas its unitary generator code'
            )
            self.currentOperation.cpp_generator = node.value.value

        if node.targets[0].id == 'traits':
            self.traits = [trait.id for trait in node.value.elts]
            print(f'traits applied = {self.traits}')


astModule = ast.parse(fileContents)
visitor = CustomOperationVisitor()
visitor.visit(astModule)

cppTemplate = '''
#include <vector>
struct CComplex {{
  double real = 0.0;
  double imag = 0.0;
  CComplex() = default;
  CComplex(double r) : real(r){{ }}
}};

using unitary = std::vector<CComplex>;
auto internal_call_lambda = {}

extern "C" void {}(const double *params, std::size_t numParams,
                           CComplex **output) {{
  std::vector<double> input(params, params + numParams);
  auto tmpOutput = internal_call_lambda(input);
  *output = new CComplex[tmpOutput.size()];
  std::copy(tmpOutput.begin(), tmpOutput.end(), *output);
  return;
}}
'''

noParamSingleTargetTemplateDefine = '''
#if CUDAQ_USE_STD20
#define TEMPLATE(SORT)                                                         \\
  template <typename mod = SORT, typename QubitRange>                          \\
    requires(std::ranges::range<QubitRange>)
#else
#define TEMPLATE(SORT)                                                         \\
  template <typename mod = SORT, typename QubitRange,                          \\
            typename = std::enable_if_t<!std::is_same_v<                       \\
                std::remove_reference_t<std::remove_cv_t<QubitRange>>,         \\
                cudaq::qubit>>>
#endif
'''
singleParamSingleTargetTemplateDefine = '''
#if CUDAQ_USE_STD20
#define TEMPLATE(SORT)                                                         \\
  template <typename mod = SORT, typename ScalarAngle, typename QubitRange>    \\
    requires(std::ranges::range<QubitRange>)
#else
#define TEMPLATE(SORT)                                                         \\
  template <typename mod = SORT, typename ScalarAngle, typename QubitRange,    \\
            typename = std::enable_if_t<!std::is_same_v<                       \\
                std::remove_reference_t<std::remove_cv_t<QubitRange>>,         \\
                cudaq::qubit>>>
#endif
'''

headerTemplate = '''namespace cudaq {{
namespace qubit_op {{
ConcreteQubitOp({})
}}

{}

{}

}}
'''

irdlTemplate = '''irdl.operation @{} {{
    %target = irdl.is !quake.ref
    {}
    irdl.operands({} %target)
}}
'''

for op in visitor.operations:
    cpp_code = cppTemplate.format(op.cpp_generator, op.name + "_generator")
    print(cpp_code)
    # Create a temporary file to save the C++ code
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as tmp_file:
        tmp_file_path = tmp_file.name
        tmp_file.write(cpp_code.encode())

    # Compile the C++ code
    compile_command = [
        "clang++-15", tmp_file_path, '-std=c++17', '-shared', '-fPIC', '-o',
        f'lib{fileName}.so'
    ]
    subprocess.run(compile_command, check=True)

    # Clean up
    os.remove(tmp_file_path)

    # MOVE to the correct install location
    shutil.move(f'lib{fileName}.so',
                cudaq_unitary_path + os.path.sep + f'lib{fileName}.so')

    if op.num_parameters == 1:
        header_code = headerTemplate.format(
            op.name, singleParamSingleTargetTemplateDefine,
            f'CUDAQ_QIS_PARAM_ONE_TARGET_({op.name})')
        print(header_code)
    else:
        header_code = headerTemplate.format(
            op.name, noParamSingleTargetTemplateDefine,
            f'CUDAQ_QIS_ONE_TARGET_QUBIT_({op.name})')

    # WRITE HEADER to install location
    with open(cudaq_headers_path+os.path.sep+f'{fileName}.h', 'w') as f:
        f.write(header_code)

    # Write the IRDL file
    if op.num_parameters == 0:
        irdl_code = irdlTemplate.format(op.name, '', '')
    if op.num_parameters == 1:
        irdl_code = irdlTemplate.format(op.name, '%param = irdl.is f64',
                                        '%param,')

    print(irdl_code)

    with open(cudaq_irdl_path+os.path.sep+f'{fileName}.irdl', 'w') as f:
        f.write(irdl_code)