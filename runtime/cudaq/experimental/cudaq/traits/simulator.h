/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/noise_model.h"
#include "cudaq/qis/state.h"
#include "cudaq/utils/operation_metadata.h"
#include "cudaq/utils/type_traits.h"

namespace cudaq::traits {

// ============================================================================
// Simulator Trait (CRTP)
// ============================================================================

/// Trait for QPUs that simulate quantum circuits
///
/// This trait defines the interface for quantum memory allocation, gate
/// application, and measurement operations. QPUs that implement this trait
/// can be used with kernel code that performs these operations.
///
/// Example:
/// ```cpp
/// class my_simulator : public qpu<my_simulator, simulator<my_simulator>> {
/// public:
///   std::size_t allocateQudit(std::size_t numLevels) { /* ... */ }
///   void deallocate(std::size_t idx) { /* ... */ }
///   void apply(const std::vector<std::complex<double>>& matrix,
///              const std::vector<std::size_t>& controls,
///              const std::vector<std::size_t>& targets) { /* ... */ }
///   std::size_t mz(std::size_t idx, const std::string& regName) { /* ... */ }
/// };
/// ```

template <typename Derived>
class simulator {
private:
  /// Stack of extra control qubit IDs for nested control regions
  std::vector<std::size_t> extraControlIds_;

  /// Type alias for a queued instruction
  using Instruction =
      std::tuple<std::string, std::vector<std::complex<double>>,
                 std::vector<std::size_t>, std::vector<std::size_t>, bool,
                 std::vector<double>>;

  /// Stack of instruction queues for nested adjoint regions
  std::vector<std::vector<Instruction>> adjointQueueStack_;

  /// Helper to compute adjoint (conjugate transpose) of a matrix
  static std::vector<std::complex<double>>
  getAdjoint(const std::vector<std::complex<double>> &input, int rows,
             int cols) {
    std::vector<std::complex<double>> output(input.size());
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        output[j * rows + i] = std::conj(input[i * cols + j]);
      }
    }
    return output;
  }

public:
  void dump_state(std::ostream &os) {
    return crtp_cast<Derived>(this)->dump_state(os);
  }

  cudaq::state get_state() { return crtp_cast<Derived>(this)->get_state(); }

  cudaq::state get_state(const state_data &data) {
    return crtp_cast<Derived>(this)->get_state(data);
  }

  std::unique_ptr<cudaq::SimulationState>
  get_internal_state(const state_data &data) {
    return crtp_cast<Derived>(this)->get_internal_state(data);
  }

  simulation_precision get_precision() const {
    return crtp_cast<const Derived>(this)->get_precision();
  }

  std::size_t allocateQudit(std::size_t numLevels = 2) {
    return crtp_cast<Derived>(this)->allocateQudit(numLevels);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const void *state,
                                          simulation_precision precision) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels, state,
                                                    precision);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const SimulationState *state) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels,
                                                    state);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels = 2) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels);
  }

  void deallocate(std::size_t idx) {
    return crtp_cast<Derived>(this)->deallocate(idx);
  }

  void deallocate(const std::vector<std::size_t> &idxs) {
    return crtp_cast<Derived>(this)->deallocate(idxs);
  }

  /// @brief Apply a quantum operation (handles control/adjoint regions)
  /// This is the public interface that handles region composition
  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const operation_metadata &metadata) {
    auto [operationName, opParams, isAdjoint] = metadata;

    // Compose all controls, including those from nested control regions
    std::vector<std::size_t> mutable_controls;
    for (auto &e : extraControlIds_)
      mutable_controls.push_back(e);
    for (auto &e : controls)
      mutable_controls.push_back(e);

    // Determine adjoint stack parity for correct adjoint logic
    bool evenAdjointStack = (adjointQueueStack_.size() % 2) == 0;

    if (!adjointQueueStack_.empty()) {
      // If in an adjoint region, queue the instruction for later reversal
      adjointQueueStack_.back().emplace_back(
          operationName, matrixRowMajor, mutable_controls, targets,
          isAdjoint != !evenAdjointStack, opParams);
      return;
    }

    // If adjoint, apply the adjoint of the operation matrix
    if (isAdjoint) {
      auto mutableName = operationName;
      auto mutableParams = opParams;
      
      // Handle special gate name transformations
      if (operationName == "t")
        mutableName = "tdg";
      else if (operationName == "s")
        mutableName = "sdg";

      // Handle special parameter transformations for certain gates
      if (operationName == "u3") {
        mutableParams[0] = -1.0 * opParams[0];
        mutableParams[1] = -1.0 * opParams[2];
        mutableParams[2] = -1.0 * opParams[1];
      } else if (operationName == "u2") {
        mutableParams[0] = -1.0 * opParams[1] - M_PI;
        mutableParams[1] = -1.0 * opParams[0] + M_PI;
      } else {
        // For most gates, just negate the parameters
        for (std::size_t i = 0; i < opParams.size(); i++)
          mutableParams[i] *= -1.0;
      }
      
      return crtp_cast<Derived>(this)->apply_impl(
          getAdjoint(matrixRowMajor, std::pow(2, targets.size()),
                     std::pow(2, targets.size())),
          mutable_controls, targets,
          operation_metadata{mutableName, mutableParams, false});
    }

    // Otherwise, apply the operation as-is
    return crtp_cast<Derived>(this)->apply_impl(
        matrixRowMajor, mutable_controls, targets, metadata);
  }

  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets) {
    operation_metadata metadata("custom_op");
    apply(matrixRowMajor, controls, targets, metadata);
  }

  /// @brief Apply a controlled region (implemented in base trait)
  /// Derived classes do not need to implement this
  void applyControlRegion(const std::vector<std::size_t> &controls,
                          const std::function<void()> &wrapped) {
    // Push controls for this region
    for (auto &c : controls)
      extraControlIds_.push_back(c);
    
    wrapped();
    
    // Pop controls after region
    extraControlIds_.resize(extraControlIds_.size() - controls.size());
  }

  /// @brief Apply an adjoint region (implemented in base trait)
  /// Derived classes do not need to implement this
  void applyAdjointRegion(const std::function<void()> &wrapped) {
    // Start a new adjoint instruction queue
    adjointQueueStack_.emplace_back();
    wrapped();

    // Pop and reverse the instruction queue to apply adjoint operations
    auto adjointQueue = std::move(adjointQueueStack_.back());
    adjointQueueStack_.pop_back();

    std::reverse(adjointQueue.begin(), adjointQueue.end());
    for (auto &[name, mat, controls, targets, isAdjoint, opParams] :
         adjointQueue) {

      if (isAdjoint) {
        // Handle special gate name transformations
        if (name == "t")
          name = "tdg";
        else if (name == "s")
          name = "sdg";

        // Handle special parameter transformations
        if (name == "u3") {
          opParams[0] = -1.0 * opParams[0];
          opParams[1] = -1.0 * opParams[2];
          opParams[2] = -1.0 * opParams[1];
        } else if (name == "u2") {
          opParams[0] = -1.0 * opParams[1] - M_PI;
          opParams[1] = -1.0 * opParams[0] + M_PI;
        } else {
          for (std::size_t i = 0; i < opParams.size(); i++)
            opParams[i] *= -1.0;
        }
        
        crtp_cast<Derived>(this)->apply_impl(
            getAdjoint(mat, std::pow(2, targets.size()),
                       std::pow(2, targets.size())),
            controls, targets, operation_metadata{name, opParams, false});
      } else {
        crtp_cast<Derived>(this)->apply_impl(
            mat, controls, targets, operation_metadata{name, opParams, false});
      }
    }
  }

  void reset(std::size_t qidx) { return crtp_cast<Derived>(this)->reset(qidx); }

  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term) {
    return crtp_cast<Derived>(this)->apply_exp_pauli(theta, controls, qubitIds,
                                                     term);
  }

  std::size_t mz(std::size_t idx, const std::string regName = "") {
    return crtp_cast<Derived>(this)->mz(idx, regName);
  }

  std::vector<std::size_t> mz(const std::vector<std::size_t> &qubits) {
    std::vector<std::size_t> ret;
    for (auto &q : qubits)
      ret.push_back(mz(q, ""));
    return ret;
  }

  // ============================================================================
  // Noise Model Support
  // ============================================================================

  /// @brief Set the noise model for this simulator
  void set_noise(const cudaq::noise_model *model) {
    return crtp_cast<Derived>(this)->set_noise(model);
  }

  /// @brief Get the current noise model (may be nullptr)
  const cudaq::noise_model *get_noise() const {
    return crtp_cast<const Derived>(this)->get_noise();
  }

  /// @brief Apply a noise channel to qubits
  void apply_noise(const std::vector<cudaq::kraus_op> &ops,
                   const std::vector<std::size_t> &qubits) {
    return crtp_cast<Derived>(this)->apply_noise(ops, qubits);
  }
};
} // namespace cudaq::traits

namespace cudaq {
#ifndef CUDAQ_NO_STD20

template <typename T>
concept SimulatorQPU = requires {
  // Type requirement: T must derive from sample_trait<T> (CRTP pattern)
  requires std::derived_from<std::decay_t<T>, traits::simulator<T>>;
};
#endif

template <typename T>
bool is_simulator(T &&t) {
  return std::is_base_of_v<traits::simulator<std::decay_t<T>>, std::decay_t<T>>;
}

struct kernel_simulator_api {
  // Basic qubit management
  std::function<std::size_t()> q_allocator;
  std::function<std::size_t(std::size_t)> q_qudit_allocator; // levels
  std::function<std::vector<std::size_t>(std::size_t, std::size_t)>
      q_multi_allocator; // count, levels
  std::function<void(std::size_t)> q_deallocator;
  std::function<void(const std::vector<std::size_t> &)> q_multi_deallocator;

  // Gate application
  std::function<void(const std::vector<std::complex<double>> &,
                     const std::vector<std::size_t> &,
                     const std::vector<std::size_t> &,
                     const traits::operation_metadata &)>
      q_applicator;

  // Measurement operations
  std::function<std::size_t(std::size_t, const std::string &)> q_measurer;
  std::function<std::vector<std::size_t>(const std::vector<std::size_t> &)>
      q_multi_measurer;

  // Reset operations
  std::function<void(std::size_t)> q_resetter;

  // Specialized operations
  std::function<void(double, const std::vector<std::size_t> &,
                     const std::vector<std::size_t> &,
                     const cudaq::spin_op_term &)>
      q_exp_pauli_applicator;

  // Control flow operations
  std::function<void(const std::vector<std::size_t> &,
                     const std::function<void()> &)>
      q_control_region;
  std::function<void(const std::function<void()> &)> q_adjoint_region;

  // State management
  std::function<cudaq::state()> q_state_getter;
  std::function<void(std::ostream &)> q_state_dumper;
  // std::function<simulation_precision()> q_precision_getter;

  // Noise model support
  std::function<void(const cudaq::noise_model *)> q_noise_setter;
  std::function<const cudaq::noise_model *()> q_noise_getter;
  std::function<void(const std::vector<cudaq::kraus_op> &,
                     const std::vector<std::size_t> &)>
      q_noise_applicator;

  // Lifecycle management
  std::shared_ptr<void> simulator_handle; // Keep simulator alive
};

// Thread-local storage for kernel API - each thread gets its own instance
thread_local static std::unique_ptr<kernel_simulator_api> m_kernel_api;

template <typename T>
void set_kernel_api(traits::simulator<T> &simulator) {
  // Create a shared pointer to manage the simulator's lifetime
  auto sim_ptr = std::shared_ptr<traits::simulator<T>>(
      &simulator, [](traits::simulator<T> *) {
        // Don't delete - we don't own the simulator
      });

  // Reset any existing API for this thread
  m_kernel_api = std::make_unique<kernel_simulator_api>();

  // Basic qubit management
  m_kernel_api->q_allocator = [sim_ptr]() -> std::size_t {
    return sim_ptr->allocateQudit(2);
  };

  m_kernel_api->q_qudit_allocator =
      [sim_ptr](std::size_t levels) -> std::size_t {
    return sim_ptr->allocateQudit(levels);
  };

  m_kernel_api->q_multi_allocator =
      [sim_ptr](std::size_t count,
                std::size_t levels) -> std::vector<std::size_t> {
    return sim_ptr->allocateQudits(count, levels);
  };

  m_kernel_api->q_deallocator = [sim_ptr](std::size_t idx) {
    sim_ptr->deallocate(idx);
  };

  m_kernel_api->q_multi_deallocator =
      [sim_ptr](const std::vector<std::size_t> &idxs) {
        sim_ptr->deallocate(idxs);
      };

  // Gate application
  m_kernel_api->q_applicator =
      [sim_ptr](const std::vector<std::complex<double>> &matrixRowMajor,
                const std::vector<std::size_t> &controls,
                const std::vector<std::size_t> &targets,
                const traits::operation_metadata &metadata) {
        sim_ptr->apply(matrixRowMajor, controls, targets, metadata);
      };

  // Measurement operations
  m_kernel_api->q_measurer =
      [sim_ptr](std::size_t idx, const std::string &regName) -> std::size_t {
    return sim_ptr->mz(idx, regName);
  };

  m_kernel_api->q_multi_measurer =
      [sim_ptr](
          const std::vector<std::size_t> &qubits) -> std::vector<std::size_t> {
    return sim_ptr->mz(qubits);
  };

  // Reset operations
  m_kernel_api->q_resetter = [sim_ptr](std::size_t idx) {
    sim_ptr->reset(idx);
  };

  // Specialized operations
  m_kernel_api->q_exp_pauli_applicator =
      [sim_ptr](double theta, const std::vector<std::size_t> &controls,
                const std::vector<std::size_t> &qubitIds,
                const cudaq::spin_op_term &term) {
        sim_ptr->apply_exp_pauli(theta, controls, qubitIds, term);
      };

  // Control flow operations
  m_kernel_api->q_control_region =
      [sim_ptr](const std::vector<std::size_t> &controls,
                const std::function<void()> &wrapped) {
        sim_ptr->applyControlRegion(controls, wrapped);
      };

  m_kernel_api->q_adjoint_region =
      [sim_ptr](const std::function<void()> &wrapped) {
        sim_ptr->applyAdjointRegion(wrapped);
      };

  // State management
  m_kernel_api->q_state_getter = [sim_ptr]() -> cudaq::state {
    return sim_ptr->get_state();
  };

  m_kernel_api->q_state_dumper = [sim_ptr](std::ostream &os) {
    sim_ptr->dump_state(os);
  };

  // Noise model support
  m_kernel_api->q_noise_setter = [sim_ptr](const cudaq::noise_model *model) {
    sim_ptr->set_noise(model);
  };

  m_kernel_api->q_noise_getter = [sim_ptr]() { return sim_ptr->get_noise(); };

  m_kernel_api->q_noise_applicator =
      [sim_ptr](const std::vector<cudaq::kraus_op> &ops,
                const std::vector<std::size_t> &qubits) {
        sim_ptr->apply_noise(ops, qubits);
      };

  // Store the simulator handle to keep it alive
  m_kernel_api->simulator_handle = sim_ptr;
}

// Thread-safe API access - no synchronization needed for thread_local storage
inline kernel_simulator_api *get_kernel_api() noexcept {
  return m_kernel_api.get();
}

// Clean up API for current thread only
inline void clear_kernel_api() noexcept { m_kernel_api.reset(); }

// Check if API is initialized for current thread
inline bool has_kernel_api() noexcept { return m_kernel_api != nullptr; }

// Optional: Initialize API if not already done for current thread
template <typename T>
inline kernel_simulator_api *
ensure_kernel_api(traits::simulator<T> &simulator) {
  if (!has_kernel_api()) {
    set_kernel_api(simulator);
  }
  return get_kernel_api();
}

} // namespace cudaq::traits
