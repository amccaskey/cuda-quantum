# CUDA-Q Runtime Refactor Notes 

## Intended Library Linkage Graph: 

Red line dependencies represent non-ideal linkages (would be good to figure out to break).

![Runtime UML Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/amccaskey/cuda-quantum/refactor_qpu_t/runtime/cudaq/platform/dependency_graph.puml)

## Breaking Changes: 

- Remote QPUs do not provide state data. 

## Remaining Task List:

- [X] Current QPU Mutability (remote rest is a qpu, but need to delegate to simulator qpu for emulation)
  - [X] I think qpu stack is the way to go here. 
- [ ] Validate the platform_config and target yaml files
- [ ] Runtime Tests (tests on the new API)
  - [X] Can get default qpu 
  - [X] Can create a qpu manually 
  - [X] Can push and pop a qpu from the stack 
  - [ ] More...
- [ ] Need to Port: 
  - [ ] cuQuantum simulator backends (including internal)
    - [X] cuStateVec
    - [X] cuDensityMat
    - [ ] cuTensorNet 
  - [X] Dynamics (cudaq::evolve)
  - [X] Simple Qudit Test 
  - [X] PhotonicsQPU and photonics_qis.h
  - [X] MQPU 
  - [ ] Remote QPUs
    - [ ] Remote REST 
    - [ ] Remote Simulator 
  - Orca 
  - Pasqal 
  - QuEra 
  - Fermioniq 
- [ ] Turn back on new estimate resources (rework it, connected components too tightly)
- [ ] Python Updates 

## Tests Passing 
From build directory, these are the tests I'm running that are passing 
```bash
$ ctest --test-dir unittests/ -E Tracer 

100% tests passed, 0 tests failed out of 670

Total Test time (real) = 770.51 sec
```

FileCheck tests in `test` folder pass as well. 
```bash
llvm-lit -j 24 test 
```

## Potential Existing Bugs Found: 

- U3 CustomOp Adj in QubitQISTester 
- QubitQISTester.cpp:26 - h(f, qq[2]) treated as control not broadcast 
- AST-Quake/ctrl_vector.cpp is control and not broadcast

## Future Explorations 

- [ ] CircuitSimulator refactor 
- [ ] Traits instead of ExecutionContexts 
- [ ] QPU* + PyQPU* duplication due to MLIR/LLVM static linking
- [ ] Python AST refactor for more user extensibility, better maintenance 

