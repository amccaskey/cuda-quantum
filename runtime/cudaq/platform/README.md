# CUDA-Q Runtime Refactor Notes 

## Intended Library Linkage Graph: 

Red line dependencies represent non-ideal linkages (would be good to figure out to break).

![Runtime UML Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/amccaskey/cuda-quantum/refactor_qpu_t/runtime/cudaq/platform/dependency_graph.puml)

## Breaking Changes: 

- Remote QPUs do not provide state data. 

## Remaining Task List:

- [ ] Current QPU Mutability (remote rest is a qpu, but need to delegate to simulator qpu for emulation)
- [ ] Validate the platform_config and target yaml files
- [ ] Runtime Tests (tests on the new API)
- [ ] Need to Port: 
  - [ ] cuQuantum simulator backends (including internal)
  - [ ] Dynamics (cudaq::evolve)
  - [ ] MQPU 
  - [ ] Remote QPUs
    - [ ] Remote REST 
    - [ ] Remote Simulator 
  - Orca 
  - Pasqal 
  - QuEra 
  - Fermioniq 
- [ ] Turn back on new estimate resources (rework it, connected components too tightly)
- [ ] Python Updates 

## Potential Existing Bugs Found: 

- [ ] U3 CustomOp Adj in QubitQISTester 
- [ ] QubitQISTester.cpp:26 - h(f, qq[2]) treated as control not broadcast 

## Future Explorations 

- [ ] CircuitSimulator refactor 
- [ ] Traits instead of ExecutionContexts 
- [ ] QPU* + PyQPU* duplication due to MLIR/LLVM static linking

