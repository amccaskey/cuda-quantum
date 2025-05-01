/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../../platform.h"
#include "../../qpu.h"

#include "cudaq/Support/TargetConfig.h"

#include "cudaq/utils/cudaq_utils.h"

#include "CircuitSimulator.h"

#include "common/EigenDense.h"

#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::CircuitSimulator)

namespace cudaq::v2 {

class simulator_config : public config::platform_config {
private:
  int getCudaGetDeviceCount() {
#ifdef CUDAQ_ENABLE_CUDA
    int nDevices{0};
    const auto status = cudaGetDeviceCount(&nDevices);
    return status != cudaSuccess ? 0 : nDevices;
#else
    return 0;
#endif
  }

public:
  void configure_qpus(std::vector<std::unique_ptr<qpu_handle>> &platform_qpus,
                      const platform_metadata &metadata) override {
    const auto &options = metadata.target_options;
    const auto &cfg = metadata.target_config;

    // Did the user request MQPU?
    bool isMqpu = false;
    for (std::size_t idx = 0; idx < options.size();) {
      const auto argsStr = options[idx + 1];
      if (argsStr == "mqpu")
        isMqpu = true;
      idx += 2;
    }

    // If not, just add a single simulated qpu
    if (!isMqpu) {
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
      cudaq::info("Added 1 qpu to the platform {}", platform_qpus.size());
      return;
    }

    // If so, get the number of available GPUs.
    // If none print a warning
    auto numGpus = getCudaGetDeviceCount();
    if (numGpus == 0) {
      cudaq::warn("mqpu platform requested, but could not find any GPUs.");
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
      return;
    }

    cudaq::info("multi-qpu has been requested - numGPUs is {}", numGpus);

    // Add the available simulators for each GPU.
    for (int i = 0; i < numGpus; i++)
      platform_qpus.emplace_back(
          qpu_handle::get("circuit_simulator", metadata));
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(config::platform_config, simulator_config)
};
CUDAQ_REGISTER_EXTENSION_TYPE(simulator_config)

/// \brief Compute the adjoint (conjugate transpose) of a matrix given in
/// row-major vector form.
/// \param input The input matrix as a row-major vector.
/// \param rows Number of rows in the matrix.
/// \param cols Number of columns in the matrix.
/// \return The adjoint matrix as a row-major vector.
std::vector<cudaq::complex> getAdjoint(const std::vector<cudaq::complex> &input,
                                       int rows, int cols) {
  using namespace Eigen;
  // Map input to row-major matrix
  const Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> original =
      Map<const Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(
          input.data(), rows, cols);

  // Compute adjoint and map back to row-major vector
  Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> adjoint =
      original.adjoint();
  std::vector<std::complex<double>> output(adjoint.size());
  Map<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(
      output.data(), adjoint.rows(), adjoint.cols()) = adjoint;

  return output;
}

class local_simulator : public qpu<
                            /*Implements LibraryMode Simulation*/
                            simulation_trait,
                            /*Implements MLIR Launch API*/
                            mlir_launch_trait,
                            /*Provides NoiseSim Support*/
                            noise_trait> {

protected:
  /// The core circuit simulator backend.
  std::unique_ptr<CircuitSimulator> simulator;

  /// Stack of extra control qubit IDs for nested control regions.
  std::vector<std::size_t> extraControlIds;

  /// \brief Type alias for a quantum instruction tuple.
  /// - string: operation name
  /// - vector<complex>: matrix (row-major)
  /// - vector<size_t>: controls
  /// - vector<size_t>: targets
  /// - bool: is adjoint
  /// - std::vector<double>: rotation parameters
  using Instruction =
      std::tuple<std::string, std::vector<cudaq::complex>,
                 std::vector<std::size_t>, std::vector<std::size_t>, bool,
                 std::vector<double>>;

  /// \brief Type alias for a queue of instructions (used for adjoint regions).
  using InstructionQueue = std::vector<Instruction>;

  /// Stack of instruction queues for nested adjoint regions.
  std::vector<InstructionQueue> adjointQueueStack;

  /// \brief Handle the observation (expectation value) task for an
  /// ExecutionContext.
  /// \param localContext The execution context, must be in "observe" mode.
  void handleObservation(ExecutionContext *localContext) {
    // Only execute if this is an observe context.
    bool execute = localContext && localContext->name == "observe";
    if (!execute)
      return;
    ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                           "QPU::handleObservation (after flush)");
    double sum = 0.0;
    if (!localContext->spin.has_value())
      throw std::runtime_error("[QPU] Observe ExecutionContext specified "
                               "without a cudaq::spin_op.");

    std::vector<cudaq::ExecutionResult> results;
    cudaq::spin_op &H = localContext->spin.value();
    assert(cudaq::spin_op::canonicalize(H) == H);

    // If the backend can handle the observe task natively, delegate to it.
    if (localContext->canHandleObserve) {
      simulator->measureSpinOp(H);
      return;
    }

    // Manually compute the expectation value by looping over spin op terms.
    for (const auto &term : H) {
      if (term.is_identity())
        sum += term.evaluate_coefficient().real();
      else {
        simulator->measureSpinOp(term);
        auto exp = localContext->expectationValue.value();
        results.emplace_back(localContext->result.to_map(), term.get_term_id(),
                             exp);
        sum += term.evaluate_coefficient().real() * exp;
      }
    };

    localContext->expectationValue = sum;
    localContext->result = cudaq::sample_result(sum, results);
  }

public:
  local_simulator(const platform_metadata &metadata) : qpu(metadata) {
    // Extract the simulation backend name from the config, default to "qpp".
    std::string simulatorName = "qpp";
    const auto &target_config = metadata.target_config;
    const auto &target_options = metadata.target_options;

    if (auto maybeBackendConfig = target_config.BackendConfig) {
      auto backendConfig = *maybeBackendConfig;
      auto simValues = backendConfig.SimulationBackend.values;
      if (!simValues.empty())
        simulatorName = simValues[0];
    }

    if (target_config.Name.find("nvidia") != std::string::npos) {
      bool isFp64 = [&]() {
        if (target_config.Name.find("-fp64") != std::string::npos)
          return true;

        for (std::size_t idx = 0; idx < target_options.size();) {
          const auto argsStr = target_options[idx + 1];
          if (argsStr == "fp64")
            return true;
          idx += 2;
        }
        return false;
      }();
      if (isFp64)
        simulatorName = "custatevec_fp64";
      else
        simulatorName = "custatevec_fp32";
    }

    // Final possible override for CircuitSimulator, user can
    // say e.g. --target-option simulator:stim
    auto maybeOverride = [&]() -> std::optional<std::string> {
      for (std::size_t idx = 0; idx < target_options.size();) {
        const auto argsStr = target_options[idx + 1];
        if (argsStr.find("simulator:") != std::string::npos)
          return cudaq::split(argsStr, ':')[1];
        idx += 2;
      }
      return std::nullopt;
    }();

    simulatorName = maybeOverride.value_or(simulatorName);

    cudaq::info("instantiating the {} simulator, uid={}", simulatorName,
                qpu_uid);
    simulator = CircuitSimulator::get(simulatorName);
  }
  bool supports_task_distribution() const override { return true; }
  void handle_async_task_launch_impl() const override {
#ifdef CUDAQ_ENABLE_CUDA
    cudaSetDevice(qpu_uid);
#endif
  }

  /// \brief Set the execution context for the simulator.
  /// \param context The execution context.
  void set_execution_context(ExecutionContext *context) override {
    simulator->set_execution_context(context);
  }

  const std::optional<std::string> get_current_context_name() override {
    auto *ctx = simulator->getExecutionContext();
    if (ctx)
      return ctx->name;
    return std::nullopt;
  }

  /// \brief Reset the execution context, handling observation if necessary.
  void reset_execution_context() override {
    handleObservation(simulator->getExecutionContext());
    simulator->reset_execution_context();
  }

  /// \brief Release simulator resources before MPI finalize.
  void tear_down() override { simulator->tearDownBeforeMPIFinalize(); }

  /// \brief Set the random seed for the simulator.
  /// \param seed The random seed value.
  void set_random_seed(std::size_t seed) override {
    simulator->setRandomSeed(seed);
  }

  /// \brief Dump the quantum state to an output stream.
  /// \param os The output stream.
  void dump_state(std::ostream &os) override { simulator->dump_state(os); }

  /// \brief Get the current quantum state from the simulator.
  /// \return The quantum state.
  state get_state() override {
    return state(simulator->getSimulationState().release());
  }

  /// \brief Create a quantum state from provided state data.
  /// \param data The state data.
  /// \return The quantum state.
  cudaq::state get_state(const state_data &data) override {
    return state(simulator->createStateFromData(data).release());
  }

  /// \brief Get the simulation precision (fp32 or fp64).
  /// \return The simulation precision.
  simulation_precision get_precision() const override {
    return simulator->isSinglePrecision() ? simulation_precision::fp32
                                          : simulation_precision::fp64;
  }

  /// \brief Allocate a single qudit (qubit) in the simulator.
  /// \return The index of the allocated qudit.
  std::size_t allocateQudit(std::size_t numLevels) override {
    return simulator->allocateQubit();
  }

  /// \brief Allocate multiple qudits (qubits).
  /// \param numQudits Number of qudits to allocate.
  /// \return Vector of allocated indices.
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels) override {
    return simulator->allocateQubits(numQudits);
  }

  /// \brief Allocate multiple qudits with initial state and precision.
  /// \param numQudits Number of qudits.
  /// \param state Pointer to initial state data.
  /// \param precision Simulation precision.
  /// \return Vector of allocated indices.
  std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const void *state, simulation_precision precision) override {
    return simulator->allocateQubits(numQudits, state, precision);
  }

  /// \brief Allocate multiple qudits with a simulation state.
  /// \param numQudits Number of qudits.
  /// \param state Simulation state pointer.
  /// \return Vector of allocated indices.
  std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const SimulationState *state) override {
    return simulator->allocateQubits(numQudits, state);
  }

  /// \brief Deallocate a single qudit by index.
  /// \param idx The index to deallocate.
  void deallocate(std::size_t idx) override { simulator->deallocate(idx); }

  /// \brief Deallocate multiple qudits by indices.
  /// \param idxs Vector of indices to deallocate.
  void deallocate(const std::vector<std::size_t> &idxs) override {
    simulator->deallocateQubits(idxs);
  }

  /// \brief Apply a quantum operation to the state.
  /// \param matrixRowMajor Operation matrix (row-major).
  /// \param controls Control qudit indices.
  /// \param targets Target qudit indices.
  /// \param metadata Operation metadata (name, parameters, adjoint flag).
  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const operation_metadata &metadata) override {

    auto [operationName, opParams, isAdjoint] = metadata;

    // Compose all controls, including those from nested control regions.
    std::vector<std::size_t> mutable_controls;
    for (auto &e : extraControlIds)
      mutable_controls.push_back(e);

    for (auto &e : controls)
      mutable_controls.push_back(e);

    // Determine adjoint stack parity for correct adjoint logic.
    bool evenAdjointStack = (adjointQueueStack.size() % 2) == 0;

    if (!adjointQueueStack.empty()) {
      // If in an adjoint region, queue the instruction for later reversal.
      adjointQueueStack.back().emplace_back(
          operationName, matrixRowMajor, mutable_controls, targets,
          isAdjoint != !evenAdjointStack, opParams);
      return;
    }

    // If adjoint, apply the adjoint of the operation matrix.
    if (isAdjoint) {
      auto mutableName = operationName;
      auto mutableParams = opParams;
      if (operationName == "t")
        mutableName = "tdg";
      else if (operationName == "s")
        mutableName = "sdg";

      if (operationName == "u3") {
        mutableParams[0] = -1.0 * opParams[0];
        mutableParams[1] = -1.0 * opParams[2];
        mutableParams[2] = -1.0 * opParams[1];
      } else if (operationName == "u2") {
        mutableParams[0] = -1.0 * opParams[1] - M_PI;
        mutableParams[1] = -1.0 * opParams[0] + M_PI;
      } else {
        for (std::size_t i = 0; i < opParams.size(); i++)
          mutableParams[i] *= -1.0;
      }
      return simulator->applyCustomOperation(
          getAdjoint(matrixRowMajor, std::pow(2, targets.size()),
                     std::pow(2, targets.size())),
          mutable_controls, targets, mutableName, mutableParams);
    }

    // Otherwise, apply the operation as-is.
    simulator->applyCustomOperation(matrixRowMajor, mutable_controls, targets,
                                    operationName, opParams);
  }

  /// \brief Apply an exponential Pauli operator to the quantum state.
  /// \param theta Rotation angle.
  /// \param controls Control qudit indices.
  /// \param qubitIds Target qudit indices.
  /// \param term The Pauli term.
  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term) override {
    std::vector<std::size_t> mutable_controls;
    for (auto &e : extraControlIds)
      mutable_controls.push_back(e);

    for (auto &e : controls)
      mutable_controls.push_back(e);
    simulator->applyExpPauli(theta, mutable_controls, qubitIds, term);
  }

  /// \brief Measure a single qudit in the Z basis.
  /// \param idx The qudit index.
  /// \param regName Optional register name.
  /// \return The measurement result.
  std::size_t mz(std::size_t idx, const std::string regName) override {
    return simulator->mz(idx, regName);
  }

  /// \brief Measure multiple qudits in the Z basis.
  /// \param qubits Vector of qudit indices.
  /// \return Vector of measurement results.
  std::vector<std::size_t> mz(const std::vector<std::size_t> &qubits) override {
    std::vector<std::size_t> ret;
    for (const auto &q : qubits)
      ret.push_back(mz(q, ""));
    return ret;
  }

  /// \brief Reset a qudit to the ground state.
  /// \param qidx The qudit index.
  void reset(std::size_t qidx) override { simulator->resetQubit(qidx); }

  // ----- Noise Modeling -----

  /// \brief Apply a noise channel to the specified targets.
  /// \param channel The Kraus channel.
  /// \param targets Target qudit indices.
  void apply_noise(const kraus_channel &channel,
                   const std::vector<std::size_t> &targets) override {
    simulator->applyNoise(channel, targets);
  }

  /// \brief Set the noise model for simulation.
  /// \param noise The noise model.
  void set_noise(const noise_model &noise) override {
    simulator->setNoiseModel(const_cast<noise_model &>(noise));
  }

  /// \brief Reset the noise model to default.
  void reset_noise() override { simulator->resetNoiseModel(); }

  /// \brief Get the current noise model.
  /// \return Pointer to the noise model.
  const noise_model *get_noise() override { return simulator->getNoiseModel(); }
  // --------------------------

  /// \brief Apply a controlled region to the quantum state.
  /// \param controls Control qudit indices.
  /// \param wrapped Function representing the controlled region.
  void applyControlRegion(const std::vector<std::size_t> &controls,
                          const std::function<void()> &wrapped) override {
    // Push controls for this region
    for (auto &c : controls)
      extraControlIds.push_back(c);
    wrapped();
    // Pop controls after region
    extraControlIds.resize(extraControlIds.size() - controls.size());
  }

  /// \brief Apply an adjoint region to the quantum state.
  /// \param wrapped Function representing the adjoint region.
  void applyAdjointRegion(const std::function<void()> &wrapped) override {
    // Start a new adjoint instruction queue
    adjointQueueStack.emplace_back();
    wrapped();

    // Pop and reverse the instruction queue to apply adjoint operations in
    // reverse order
    auto adjointQueue = std::move(adjointQueueStack.back());
    adjointQueueStack.pop_back();

    std::reverse(adjointQueue.begin(), adjointQueue.end());
    for (auto &[name, mat, controls, targets, isAdjoint, opParams] :
         adjointQueue) {

      if (isAdjoint) {
        if (name == "t")
          name = "tdg";
        else if (name == "s")
          name = "sdg";

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
        simulator->applyCustomOperation(getAdjoint(mat,
                                                   std::pow(2, targets.size()),
                                                   std::pow(2, targets.size())),
                                        controls, targets, name, opParams);
      } else
        simulator->applyCustomOperation(mat, controls, targets, name, opParams);
    }
  }

  /// \brief Launch a kernel by name with raw function pointer and arguments.
  /// \param name The kernel name.
  /// \param kernelFunc The raw function pointer.
  /// \param args Struct-packed arguments pointer.
  /// \param ... Additional arguments (unused).
  /// \param rawArgs Vector of raw argument pointers.
  /// \return The kernel thunk result.
  KernelThunkResultType
  launch_kernel(const std::string &name, KernelThunkType kernelFunc, void *args,
                std::uint64_t, std::uint64_t,
                const std::vector<void *> &rawArgs) override {
    // For simulation, simply invoke the thunk.
    return kernelFunc(args, false);
  }

  void launch_kernel(const std::string &name,
                     const std::vector<void *> &rawArgs) override {
    throw std::runtime_error("Wrong kernel launch point: Attempt to launch "
                             "kernel in streamlined for JIT mode on local "
                             "simulated QPU. This is not supported.");
  }

  static inline bool register_type() {
    auto &registry = get_registry();
    registry[local_simulator::class_identifier] = local_simulator::create;
    return true;
  }

  static const bool registered_;
  static inline const std::string class_identifier = "circuit_simulator";
  static std::unique_ptr<qpu_handle> create(const platform_metadata &m) {
    return std::make_unique<local_simulator>(m);
  }
};

const bool local_simulator::registered_ = local_simulator::register_type();

} // namespace cudaq::v2
