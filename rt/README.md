# CUDA-Q Runtime Architecture Document

This document describes the architecture of a new runtime system for CUDA-Q that introduces a unified QPU (Quantum Processing Unit) abstraction with static trait-based types and configurable execution policies.

## Architecture Overview

The new runtime introduces a fundamental shift from dynamic QPU selection to compile-time static QPU types with varying capabilities expressed through C++20 concepts and trait-based inheritance. The system provides a unified launcher infrastructure that supports both local simulation and remote quantum hardware execution through user-defined execution policies.

### Core Components

The runtime consists of three primary architectural layers: trait-based QPU abstraction, execution policy system, and unified launcher infrastructure.

## QPU Trait System

The trait system forms the foundation of the new architecture, enabling compile-time QPU capability detection and dispatch through the Curiously Recurring Template Pattern (CRTP).

### Base QPU Class

```cpp
template <typename Derived, typename... Traits>
class qpu : public Traits... {
public:
  std::string name() const { 
    return crtp_cast<Derived>(this)->name(); 
  }
  ... Other base methods ...
};
```

### Trait Definitions

The system defines several core traits that QPUs can inherit to express their capabilities:

#### Sampling Trait

```cpp
namespace traits {
template <typename Derived>
class sample_trait {
public:
  sample_result sample(std::size_t num_shots, 
                       const std::string &kernel_name,
                       const std::function<void()> &wrapped_kernel) {
    return static_cast<Derived*>(this)->sample(num_shots, kernel_name, 
                                               wrapped_kernel);
  }
};
}

template <typename T>
concept SamplingQPU = requires {
  requires std::derived_from<std::decay_t<T>, 
                             traits::sample_trait<std::decay_t<T>>>;
};
```

#### Simulator Trait

```cpp
namespace traits {
template <typename Derived>
class simulator {
public:
  void dump_state(std::ostream &os) {
    return crtp_cast<Derived>(this)->dump_state(os);
  }
  
  cudaq::state get_state() { 
    return crtp_cast<Derived>(this)->get_state(); 
  }
  
  std::size_t allocateQudit(std::size_t numLevels = 2) {
    return crtp_cast<Derived>(this)->allocateQudit(numLevels);
  }
  
  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const operation_metadata &metadata) {
    return crtp_cast<Derived>(this)->apply(matrixRowMajor, controls, 
                                           targets, metadata);
  }
  
  std::size_t mz(std::size_t idx, const std::string regName = "") {
    return crtp_cast<Derived>(this)->mz(idx, regName);
  }
};
}

template <typename T>
concept SimulatorQPU = requires {
  requires std::derived_from<std::decay_t<T>, traits::simulator<T>>;
};
```

#### Remote Trait

```cpp
namespace traits {
class remote {};
}

template <typename T>
concept RemoteQPU = requires {
  requires std::derived_from<std::decay_t<T>, traits::remote>;
};

template <typename T>
bool is_remote(T &&t) {
  return std::is_base_of_v<traits::remote, std::decay_t<T>>;
}
```


## Concrete QPU Implementations

### GPU State Vector Simulator

The GPU state vector simulator demonstrates the trait system implementation:

```cpp
namespace cudaq::simulator::gpu {
class state_vector : public qpu<state_vector, 
                               traits::simulator<state_vector>,
                               traits::sample_trait<state_vector>> {
public:
  std::string name() const { return "gpu::state_vector"; }

  sample_result sample(std::size_t num_shots, 
                       const std::string &kernel_name,
                       const std::function<void()> &wrapped);

  void dump_state(std::ostream &os);
  cudaq::state get_state();
  std::size_t allocateQudit(std::size_t numLevels = 2);
  
  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const traits::operation_metadata &metadata);
             
  std::size_t mz(std::size_t idx, const std::string regName = "");

private:
  std::unique_ptr<Impl> pImpl;
};
}
```


### Remote Quantinuum QPU

The remote QPU implementation shows how hardware backends integrate:

```cpp
namespace cudaq::remote {
class quantinuum : public qpu<quantinuum, 
                             traits::remote, 
                             traits::sample_trait<quantinuum>> {
public:
  std::string name() const { return "remote::quantinuum"; }

  sample_result sample(std::size_t num_shots, 
                       const std::string &kernel_name,
                       const std::function<void()> &wrapped_kernel) {
    // Remote execution implementation
    return sample_result();
  }
};
}
```

## Execution Policy System

The execution policy system provides a declarative approach to specify quantum computation execution parameters and behavior.

### Policy Structure

```cpp
template <typename T>
concept ExecutionPolicy = requires {
  typename T::result_type;
} || requires {
  typename T::template result_type<int>;
} || requires {
  typename T::execution_policy_tag;
};
```

### Sample Policy

The sample policy demonstrates the policy pattern implementation:

```cpp
struct sample_policy {
  using result_type = sample_result;
  std::optional<std::size_t> shots = 1000;
  
  template <SamplingQPU QPU, typename QuantumKernel, typename... Args>
  friend auto launch_impl(QPU &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    auto kernelName = ""; // get kernel name
    
    if (cudaq::is_simulator(qpu))
      return qpu.sample(policy.shots.value_or(100), kernelName, [&]() {
        cudaq::set_kernel_api(qpu);
        kernel(args...);
      });
    
    return qpu.sample(policy.shots.value_or(100), kernelName,
                      [&]() { kernel(args...); });
  }
};
```

### Explicit Measurements Policy

```cpp
struct explicit_measurements_policy {
  using result_type = bit_table;
  using execution_policy_tag = void;
  
  std::optional<std::size_t> shots = 1000;
  
  template <typename QuantumKernel, typename... Args>
  friend auto launch_impl(const explicit_measurements_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    auto &current_qpu = cudaq::get_qpu();
    if (auto *sampler = current_qpu.as<sample_trait>()) {
      auto kernelName = cudaq::getKernelName(kernel);
      auto quakeCode = cudaq::get_quake_by_name(kernelName, false);
      return sampler->sample_explicit_measurements(
          policy.shots.value(), quakeCode,
          [&, ... args = std::forward<Args>(args)]() mutable {
            kernel(std::forward<Args>(args)...);
          });
    }
    throw std::runtime_error(
        "current target does not support the explicit measurement policy.");
    return bit_table();
  }
};
```

## Unified Launcher Infrastructure

The launcher provides a consistent interface for executing quantum kernels across different QPU types and execution policies.

### Core Launcher Functions

```cpp
// Main launcher with explicit QPU and policy
template <typename QPU, typename ExecutionPolicy, typename QuantumKernel,
          typename... Args>
auto launch(QPU &qpu, ExecutionPolicy &&policy, QuantumKernel &&kernel,
            Args &&...args) {
  return launch_impl(qpu, std::forward<ExecutionPolicy>(policy), kernel,
                     std::forward<Args>(args)...);
}

// QPU type specified as template parameter
template <typename QPU, typename ExecutionPolicy, typename QuantumKernel,
          typename... Args>
auto launch(ExecutionPolicy &&policy, QuantumKernel &&kernel, Args &&...args) {
  QPU qpu;
  return launch_impl(qpu, std::forward<ExecutionPolicy>(policy),
                     std::forward<QuantumKernel>(kernel),
                     std::forward<Args>(args)...);
}

// Default launcher using compiler-specified QPU
template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
auto launch(ExecutionPolicy &&policy, QuantumKernel &&kernel, Args &&...args) {
  return launch<config::default_qpu>(std::forward<ExecutionPolicy>(policy),
                                     std::forward<QuantumKernel>(kernel),
                                     std::forward<Args>(args)...);
}

// Fallback for direct kernel execution
template <typename QPU, typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel, Args...>
auto launch_impl(QPU &, QuantumKernel &&kernel, Args &&...args)
    -> std::invoke_result_t<QuantumKernel, Args...> {
  return kernel(std::forward<Args>(args)...);
}
```

### Default QPU Configuration

The system supports compile-time QPU selection through configuration:

```cpp
namespace cudaq::config {
#ifdef CUDAQ_TARGET_GPU_STATEVECTOR
using default_qpu = simulator::gpu::state_vector;
#elif defined(CUDAQ_TARGET_QUANTINUUM)
using default_qpu = remote::quantinuum;
#else
using default_qpu = simulator::gpu::state_vector;
#endif
}
```

## Multi-QPU Support

The CUDA-Q runtime architecture provides comprehensive support for multi-QPU execution through a trait-based parallel execution system. This design enables embarrassingly parallel quantum computations across multiple quantum processing units, particularly optimized for multi-GPU simulation scenarios where independent quantum tasks can be executed concurrently without state sharing.[1]

### Core Parallel Trait

The foundation of multi-QPU support is the `parallel_trait`, which transforms any QPU into a parallelized version capable of managing multiple underlying QPU instances:

```cpp
namespace cudaq::traits {
template <typename Derived>
class parallel_trait {
public:
  // Batch execution interface - the core parallel capability
  template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
  auto execute_batch(const std::vector<std::tuple<Args...>>& task_args,
                    ExecutionPolicy&& policy,
                    const std::string& kernel_name,
                    QuantumKernel&& kernel) -> std::vector<sample_result> {
    return crtp_cast<Derived>(this)->sample_batch(task_args, 
                                                  std::forward<ExecutionPolicy>(policy),
                                                  kernel_name,
                                                  std::forward<QuantumKernel>(kernel));
  }
  
  // Parameter sweep execution
  template <typename ExecutionPolicy, typename QuantumKernel, typename ParamType>
  auto execute_sweep(const std::vector<ParamType>& parameters,
                    ExecutionPolicy&& policy,
                    const std::string& kernel_name, 
                    QuantumKernel&& kernel) -> std::vector<sample_result> {
    return crtp_cast<Derived>(this)->sample_sweep(parameters,
                                                  std::forward<ExecutionPolicy>(policy),
                                                  kernel_name,
                                                  std::forward<QuantumKernel>(kernel));
  }
  
protected:
  std::size_t acquire_qpu() {
    std::unique_lock<std::mutex> lock(pool_mutex);
    qpu_available.wait(lock, [this] { return !available_indices.empty(); });
    
    auto idx = available_indices.front();
    available_indices.pop();
    return idx;
  }
  
  void release_qpu(std::size_t idx) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    available_indices.push(idx);
    qpu_available.notify_one();
  }

private:
  std::queue<std::size_t> available_indices;
  std::mutex pool_mutex;
  std::condition_variable qpu_available;
};
}
```

The trait provides compile-time detection of parallel capabilities through C++20 concepts:

```cpp
template <typename T>
concept ParallelQPU = requires {
  requires std::derived_from<std::decay_t<T>, 
                             traits::parallel_trait<std::decay_t<T>>>;
};
```

## Multi-GPU State Vector Implementation

### GPU-Affinity QPU Pool

The multi-GPU state vector simulator demonstrates the practical implementation of the parallel trait system, creating GPU-bound QPU instances for optimal resource utilization:

```cpp
namespace cudaq::simulator::gpu {
class multi_state_vector : public qpu<multi_state_vector,
                                     traits::simulator<multi_state_vector>,
                                     traits::sample_trait<multi_state_vector>,
                                     traits::parallel_trait<multi_state_vector>> {
public:
  using underlying_qpu_type = state_vector;
  
private:
  std::vector<std::unique_ptr<state_vector>> gpu_qpus;
  std::queue<std::size_t> available_qpus;
  std::mutex pool_mutex;
  std::condition_variable qpu_available;
  std::size_t num_gpus;
  
public:
  explicit multi_state_vector(std::size_t gpu_count = 0) 
    : num_gpus(gpu_count == 0 ? get_gpu_device_count() : gpu_count) {
    initialize_gpu_pool();
  }
  
  std::string name() const override { 
    return "gpu::multi_state_vector[" + std::to_string(num_gpus) + "]"; 
  }

private:
  void initialize_gpu_pool() {
    gpu_qpus.reserve(num_gpus);
    
    for (std::size_t i = 0; i < num_gpus; ++i) {
      // Each state_vector automatically binds to GPU i through CUDA context
      cudaSetDevice(static_cast<int>(i));
      gpu_qpus.push_back(std::make_unique<state_vector>());
      available_qpus.push(i);
    }
  }
};
}
```

## Configuration Integration

### Compile-Time Configuration

Multi-QPU systems integrate seamlessly with the compile-time configuration system, allowing `nvq++` to automatically configure parallel execution:

```cpp
void configure_from_registry() override {
  auto& registry = config::configuration_registry::get();
  
  num_qpus = registry.get_parameter<int>("parallel.num_qpus", get_default_qpu_count());
  
  // Initialize base QPUs with their own configuration
  qpu_instances.reserve(num_qpus);
  for (std::size_t i = 0; i < num_qpus; ++i) {
    auto qpu = std::make_unique<BaseQPU>();
    
    // Configure each QPU with device-specific parameters
    if constexpr (ConfigurableQPU<BaseQPU>) {
      // Override device ID for GPU QPUs
      if (registry.has_parameter("gpu.device_id")) {
        qpu->set_configuration_parameter("device_id", static_cast<int>(i));
      }
      qpu->configure_from_registry();
    }
    
    qpu_instances.push_back(std::move(qpu));
  }
}
```

Users can specify multi-GPU execution through `nvq++` compilation flags:

```bash
# Compile for 4 GPUs with single precision
nvq++ --target parallel-gpu --parallel-num-gpus 4 --gpu-precision single my_kernel.cpp

# Auto-detect available GPUs
nvq++ --target parallel-gpu --parallel-base-target gpu my_kernel.cpp
```
