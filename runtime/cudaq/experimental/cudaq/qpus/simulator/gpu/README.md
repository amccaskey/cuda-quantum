# GPU State Vector Simulator

## Overview

This directory contains the experimental GPU-accelerated state vector simulator implementation using NVIDIA cuStateVec. It follows the new trait-based QPU design pattern used in the CUDA Quantum experimental runtime.

## Architecture

### Design Pattern

The simulator follows the CRTP (Curiously Recurring Template Pattern) with trait-based interfaces:

- **QPU Base**: `qpu<state_vector<T>, ...Traits>` 
- **Traits Implemented**:
  - `traits::simulator<state_vector<T>>` - Provides quantum simulation operations
  - `traits::local_trait<state_vector<T>>` - Enables local (in-process) execution

### File Structure

```
gpu/
├── state_vector.h              # Public API header (no CUDA headers leak)
├── state_vector_impl.h         # Internal implementation (CUDA/cuStateVec)
├── state_vector.cpp            # Main implementation
├── state_vector_state.h        # SimulationState wrapper interface
├── state_vector_state.cpp      # SimulationState implementation
├── state_vector_kernels.cu     # CUDA kernels for initialization
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

### Key Design Decisions

1. **Header Isolation**: CUDA and cuStateVec headers are kept in `*_impl.h` and `.cpp` files, preventing them from leaking into user code that includes `state_vector.h`.

2. **PIMPL Pattern**: The main `state_vector` class uses a pointer to `state_vector_impl` to hide CUDA-specific details.

3. **Template Support**: Both FP32 and FP64 precision are supported via template parameter `ScalarType`.

4. **Namespace**: `cudaq::simulator::gpu` to clearly indicate the implementation type.

## Components

### 1. state_vector (Main Simulator Class)

The main QPU class that implements:
- Qubit allocation and deallocation
- Gate application
- Measurement operations
- State management
- Pauli rotation operations

### 2. gpu_state_vector (Simulation State)

Implements `cudaq::SimulationState` interface for GPU device memory:
- Manages GPU device pointers
- Provides amplitude extraction
- Supports state overlap computation
- Handles device-to-host transfers

### 3. state_vector_impl (Internal Implementation)

Encapsulates all cuStateVec operations:
- cuStateVec handle management
- Gate matrix application
- Measurement and reset operations
- Workspace management

### 4. CUDA Kernels

Provides efficient GPU kernels for:
- State vector initialization to |0...0⟩

## Usage

### Basic Example

```cpp
#include <cudaq.h>
#include <cudaq/qpus/simulator/gpu/state_vector.h>

using namespace cudaq::simulator;

// Create simulator
gpu::state_vector<double> qpu;

// Allocate qubits
qpu.allocateQudits(2, 2);

// Apply gates
std::vector<std::complex<double>> h_gate = {/* ... */};
cudaq::traits::operation_metadata meta("h");
qpu.apply(h_gate, {}, {0}, meta);

// Measure
auto result = qpu.mz(0);

// Get state
auto state = qpu.get_state();
auto amplitude = state.amplitude({0, 1});
```

### With Launch API

```cpp
auto my_kernel = []() {
    cudaq::qubit q;
    h(q);
    x<cudaq::ctrl>(q);
};

gpu::state_vector<double> qpu;
auto results = cudaq::launch(qpu, cudaq::sample_policy{}, my_kernel);
```

## Building

### Prerequisites

- CUDA Toolkit (12.0 or later)
- cuStateVec library
- CMake 3.20+

### CMake Configuration

Set `CUSTATEVEC_ROOT` to the cuStateVec installation directory:

```bash
cmake -DCUSTATEVEC_ROOT=/path/to/custatevec ..
```

### Build Options

The simulator is only built if `CUSTATEVEC_ROOT` is set. If not available, the build system will skip this target gracefully.

## Testing

Comprehensive tests are provided in `experimental/tests/gpu_state_vector_test.cpp`:

- Basic construction and configuration
- Qubit allocation
- Gate application (X, H, CNOT)
- Measurement operations
- Reset operations
- State management
- Pauli rotations
- Error handling
- Float precision support
- Random seed determinism

Run tests with:
```bash
ctest -R gpu_state_vector_tester
```

## Implementation Status

### Completed Features

- ✅ Qubit allocation
- ✅ Single and controlled gate application
- ✅ Measurement in Z basis
- ✅ Reset operations
- ✅ Pauli rotation operations
- ✅ State extraction and management
- ✅ FP32 and FP64 precision support
- ✅ SimulationState interface
- ✅ Comprehensive test suite

### TODO/Future Work

- ⬜ Dynamic qubit allocation (expanding existing state)
- ⬜ Deallocation support
- ⬜ Control region implementation
- ⬜ Adjoint region implementation
- ⬜ Batch execution support
- ⬜ Sample result collection
- ⬜ Noise model support
- ⬜ Multi-GPU support
- ⬜ Allocation from user-provided state data

## Performance Considerations

1. **GPU Memory**: State vector size grows exponentially with qubit count (2^n complex numbers)
2. **Initialization**: CUDA initialization (`cudaFree(0)`) is done once in constructor
3. **Workspace**: cuStateVec operations may require additional GPU memory
4. **Synchronization**: Explicit synchronization is performed after kernel launches

## Integration with Experimental Runtime

The simulator integrates with:
- `cudaq::traits::simulator<T>` trait for quantum operations
- `cudaq::traits::local_trait<T>` for local execution
- `cudaq::kernel_simulator_api` for kernel callbacks
- `cudaq::SimulationState` for state representation
- `cudaq::sample_policy` for execution policies

## References

- [cuStateVec Documentation](https://docs.nvidia.com/cuda/cuquantum/custatevec/)
- [CUDA Quantum Documentation](https://nvidia.github.io/cuda-quantum/)
- Trait-based design: `runtime/cudaq/experimental/cudaq/traits/`

