/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "common/ThunkInterface.h"

#include <fstream>

///
/// Here we enumerate available QPU traits
///
/// To-date we have the following traits
///
/// - simulation (library-mode, local simulation)
/// - mlir_launch (compatibility with our MLIR trampoline functions)
/// - noise modeling
/// TBD add more
///
/// For each trait, we also provide the mixin template type.

namespace cudaq {
class gradient;
}

namespace cudaq {

enum class qpu_traits { simulation, mlir_launch, noise };

/// \brief Metadata describing a quantum operation.
struct operation_metadata {
  static inline const std::vector<double>
      empty{}; ///< Static empty parameter vector.

  const std::string &name;               ///< Name of the operation.
  const std::vector<double> &parameters; ///< Operation parameters.
  bool isAdjoint = false; ///< Whether this operation is an adjoint.

  /// \brief Construct metadata for a parameterless operation.
  /// \param n Name of the operation.
  operation_metadata(const std::string &n) : name(n), parameters(empty) {}

  /// \brief Construct metadata for a parameterized operation.
  /// \param n Name of the operation.
  /// \param p Operation parameters.
  operation_metadata(const std::string &n, const std::vector<double> &p)
      : name(n), parameters(p) {}

  /// \brief Construct metadata for a parameterized operation with adjoint
  /// flag. \param n Name of the operation. \param p Operation parameters.
  /// \param isadj True if this is an adjoint operation.
  operation_metadata(const std::string &n, const std::vector<double> &p,
                     bool isadj)
      : name(n), parameters(p), isAdjoint(isadj) {}
};

/// \brief Trait interface for noise modeling capability on QPUs.
class noise_trait {
public:
  virtual ~noise_trait() = default;

  /// \brief Set the noise model for the QPU.
  /// \param in_noise The noise model to set.
  virtual void set_noise(const noise_model &in_noise) = 0;

  /// \brief Get the current noise model.
  /// \return Pointer to the current noise model.
  virtual const noise_model *get_noise() = 0;

  /// \brief Reset the noise model to the default state.
  virtual void reset_noise() = 0;

  /// \brief Apply a noise channel to specific targets.
  /// \param channelName The Kraus channel to apply.
  /// \param targets Indices of the target qubits.
  virtual void apply_noise(const kraus_channel &channelName,
                           const std::vector<std::size_t> &targets) = 0;
};

/// \brief Trait interface for launching MLIR kernels on QPUs.
class mlir_launch_trait {
public:
  /// \brief Virtual destructor.
  virtual ~mlir_launch_trait() = default;

  /// \brief Launch a kernel with the given arguments and metadata.
  /// \param name Name of the kernel.
  /// \param kernelFunc Function pointer to the kernel thunk.
  /// \param args Pointer to kernel arguments.
  /// \param argSize Size of the argument buffer.
  /// \param resultOffset Offset for the result in the buffer.
  /// \param rawArgs Raw pointers to kernel arguments.
  /// \return Result of the kernel thunk.
  [[nodiscard]] virtual KernelThunkResultType
  launch_kernel(const std::string &name, KernelThunkType kernelFunc, void *args,
                std::uint64_t argSize, std::uint64_t resultOffset,
                const std::vector<void *> &rawArgs) = 0;

  /// \brief Launch a kernel with raw argument pointers.
  /// \param name Name of the kernel.
  /// \param rawArgs Raw pointers to kernel arguments.
  virtual void launch_kernel(const std::string &name,
                             const std::vector<void *> &rawArgs) = 0;

  virtual void launch_vqe(const std::string &name, const void *kernelArgs,
                          cudaq::gradient *gradient, const cudaq::spin_op &H,
                          cudaq::optimizer &optimizer, const int n_params,
                          const std::size_t shots) {
    throw std::runtime_error("remote launch_vqe not supported for this qpu.");
  }

  virtual std::optional<cudaq::state>
  local_state_from_data(const state_data &) {
    return std::nullopt;
  }
};

/// \brief Trait interface for simulation capability on QPUs.
class simulation_trait {
public:
  // FIXME This should not be how we do this for run()
  std::string outputLog = "";

  /// \brief Dump the current quantum state to the given output stream.
  /// \param os Output stream to write the state to.
  virtual void dump_state(std::ostream &os) = 0;

  /// \brief Get the current quantum state.
  /// \return The current state.
  virtual cudaq::state get_state() = 0;

  /// \brief Get the quantum state from provided state data.
  /// \param data State data to initialize from.
  /// \return The constructed state.
  virtual cudaq::state get_state(const state_data &data) = 0;

  /// \brief Get the internal quantum state from provided state data.
  virtual std::unique_ptr<cudaq::SimulationState>
  get_internal_state(const state_data &data) = 0;

  /// \brief Get the simulation precision.
  /// \return The current simulation precision.
  virtual simulation_precision get_precision() const = 0;

  /// \brief Allocate a single qudit (quantum digit) with the specified number
  /// of levels.
  /// \param numLevels Number of levels for the qudit (default 2).
  /// \return Index of the allocated qudit.
  virtual std::size_t allocateQudit(std::size_t numLevels = 2) = 0;

  /// \brief Allocate multiple qudits with specified levels and initial state.
  /// \param numQudits Number of qudits to allocate.
  /// \param numLevels Number of levels per qudit.
  /// \param state Pointer to initial state data.
  /// \param precision Simulation precision.
  /// \return Indices of the allocated qudits.
  virtual std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const void *state, simulation_precision precision) = 0;

  /// \brief Allocate multiple qudits with specified levels and simulation
  /// state.
  /// \param numQudits Number of qudits to allocate.
  /// \param numLevels Number of levels per qudit.
  /// \param state Pointer to simulation state.
  /// \return Indices of the allocated qudits.
  virtual std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const SimulationState *state) = 0;

  /// \brief Allocate multiple qudits with specified levels.
  /// \param numQudits Number of qudits to allocate.
  /// \param numLevels Number of levels per qudit (default 2).
  /// \return Indices of the allocated qudits.
  virtual std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels = 2) = 0;

  /// \brief Deallocate a single qudit.
  /// \param idx Index of the qudit to deallocate.
  virtual void deallocate(std::size_t idx) = 0;

  /// \brief Deallocate multiple qudits.
  /// \param idxs Indices of the qudits to deallocate.
  virtual void deallocate(const std::vector<std::size_t> &idxs) = 0;

  /// \brief Apply a custom operation to the quantum state.
  /// \param matrixRowMajor Row-major matrix representing the operation.
  /// \param controls Indices of control qudits.
  /// \param targets Indices of target qudits.
  /// \param metadata Operation metadata.
  virtual void apply(const std::vector<std::complex<double>> &matrixRowMajor,
                     const std::vector<std::size_t> &controls,
                     const std::vector<std::size_t> &targets,
                     const operation_metadata &metadata) = 0;

  /// \brief Apply a custom operation to the quantum state with default
  /// metadata.
  /// \param matrixRowMajor Row-major matrix representing the operation.
  /// \param controls Indices of control qudits.
  /// \param targets Indices of target qudits.
  virtual void apply(const std::vector<std::complex<double>> &matrixRowMajor,
                     const std::vector<std::size_t> &controls,
                     const std::vector<std::size_t> &targets) {
    operation_metadata metadata("custom_op");
    apply(matrixRowMajor, controls, targets, metadata);
  }

  /// \brief Apply a control region, wrapping the provided function.
  /// \param controls Indices of control qudits.
  /// \param wrapped Function to execute within the control region.
  virtual void applyControlRegion(const std::vector<std::size_t> &controls,
                                  const std::function<void()> &wrapped) = 0;

  /// \brief Apply an adjoint region, wrapping the provided function.
  /// \param wrapped Function to execute within the adjoint region.
  virtual void applyAdjointRegion(const std::function<void()> &wrapped) = 0;

  /// \brief Reset the specified qudit to its initial state.
  /// \param qidx Index of the qudit to reset.
  virtual void reset(std::size_t qidx) = 0;

  /// \brief Apply an exponential of a Pauli operator term.
  /// \param theta Rotation angle.
  /// \param controls Indices of control qudits.
  /// \param qubitIds Indices of target qubits.
  /// \param term Pauli operator term.
  virtual void apply_exp_pauli(double theta,
                               const std::vector<std::size_t> &controls,
                               const std::vector<std::size_t> &qubitIds,
                               const cudaq::spin_op_term &term) = 0;

  /// \brief Measure the specified qudit in the Z basis.
  /// \param idx Index of the qudit to measure.
  /// \param regName Optional register name for the measurement result.
  /// \return Measurement result.
  virtual std::size_t mz(std::size_t idx, const std::string regName = "") = 0;

  /// \brief Measure multiple qudits in the Z basis.
  /// \param qubits Indices of the qudits to measure.
  /// \return Measurement results.
  virtual std::vector<std::size_t> mz(const std::vector<std::size_t> &qubits) {
    std::vector<std::size_t> ret;
    for (auto &q : qubits)
      ret.push_back(mz(q, ""));
    return ret;
  }
};

} // namespace cudaq
