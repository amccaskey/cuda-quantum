#!/usr/bin/env python3

import ast, astpretty, tempfile, subprocess, os, shutil, json
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
        self.num_targets = 0
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

        if node.targets[0].id == 'num_targets':
            print(
                f'{self.currentOperation.name} is applied to {node.value.value} target qubits'
            )
            self.currentOperation.num_targets = node.value.value
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

auto internal_call_lambda_{} = {}

extern "C" void {}(const double *params, std::size_t numParams,
                           std::complex<double> *output) {{
  std::vector<double> input(params, params + numParams);
  auto tmpOutput = internal_call_lambda_{}(input);
  for (int i = 0; i < tmpOutput.size(); i++) output[i] = tmpOutput[i];
  return;
}}
'''

headerTemplate = '''namespace cudaq {{

template<typename mod = base, typename... GeneralArgs>
void {}(GeneralArgs&&... args) {{
  applyQuakeExtOperation<mod>("{}", {}, std::forward<GeneralArgs>(args)...);
}}

}}
'''


irdlTemplate = '''irdl.operation @{} {{
    %target = irdl.is !quake.ref
    %controls = irdl.is !quake.ref
    {}
    irdl.operands({} variadic %controls, %target)
}}
'''

metadata = {}
cpp_code = '''#include <vector>
#include <complex>

using unitary = std::vector<std::complex<double>>;

'''
header_code = ''
irdl_code = ''
for op in visitor.operations:
    cpp_code += cppTemplate.format(op.name, op.cpp_generator, op.name + "_generator", op.name)
    print(cpp_code)
   
    # Write the header file for the operation
    header_code += headerTemplate.format(
        op.name, op.name, op.num_targets)
    print(header_code)

    # Write the IRDL file
    irdl_code += irdlTemplate.format(op.name, '%param = irdl.is f64',
                                    '%param,'*op.num_parameters)
    print(irdl_code)

    # Write the metadata file
    metadata[op.name] = {'num_targets':op.num_targets, 'num_parameters':op.num_parameters}
    with open(cudaq_irdl_path + os.path.sep + f'{fileName}.json', 'w') as f:
        json.dump(metadata, f)

    
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

# WRITE HEADER to install location
with open(cudaq_headers_path+os.path.sep+f'{fileName}.h', 'w') as f:
    f.write(header_code)

# Scan quake_ext.h, create if not there, update if there
toAdd = f'#include "{fileName}.h"'
quakeExtHeader = cudaq_headers_path + os.path.sep + "quake_ext.h"
if not os.path.exists(quakeExtHeader):
    with open (quakeExtHeader, 'w') as f:
        f.write('#pragma once\n')

# Check the file contents first 
update = True
with open(quakeExtHeader, 'r') as f:
    if toAdd in f.read(): update = False

if update:
    with open(quakeExtHeader, 'a') as f:
        f.write(toAdd + '\n')

with open(cudaq_irdl_path+os.path.sep+f'{fileName}.irdl', 'w') as f:
    f.write(irdl_code)

