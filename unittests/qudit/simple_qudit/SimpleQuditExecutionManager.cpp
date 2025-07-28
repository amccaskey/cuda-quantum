/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "common/ExecutionContext.h"
#include "common/Logger.h"

#include "cudaq/operators.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/utils/cudaq_utils.h"
#include "qpp.h"
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

namespace cudaq {

class SimpleQuditSimulator : public cudaq::qpu_mixin<simulation_trait> {
private:
  qpp::ket state;
  std::vector<std::size_t> sampleQudits;
  std::size_t counter = 0;
  ExecutionContext *context = nullptr;

public:
  std::string name() const override { return class_identifier; }

  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term) override {
    throw std::runtime_error("Photonics apply_exp_pauli not implemented.");
  }

  simulation_precision get_precision() const override {
    return simulation_precision::fp64;
  }

  void dump_state(std::ostream &os) override { os << state << '\n'; }
  cudaq::state get_state() override { return cudaq::state(nullptr); }

  cudaq::state get_state(const state_data &data) override {
    return cudaq::state(nullptr);
  }

  std::unique_ptr<SimulationState>
  get_internal_state(const state_data &data) override {
    return nullptr;
  }
  std::size_t allocateQudit(std::size_t numLevels) override {
    cudaq::info("Allocate new qudit of numLevels = {}", numLevels);
    if (state.size() == 0) {
      // qubit will give [1,0], qutrit will give [1,0,0]
      state = qpp::ket::Zero(numLevels);
      state(0) = 1.0;
      return counter++;
    }

    qpp::ket zeroState = qpp::ket::Zero(numLevels);
    zeroState(0) = 1.0;
    state = qpp::kron(state, zeroState);
    return counter++;
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels) override {
    std::vector<std::size_t> idxs;
    for (std::size_t i = 0; i < numQudits; i++)
      idxs.push_back(allocateQudit(numLevels));

    return idxs;
  }
  std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const void *state, simulation_precision precision) override {
    throw std::runtime_error(
        "photonics qpu - allocation from state is not implemented");
    return {};
  }

  std::vector<std::size_t>
  allocateQudits(std::size_t numQudits, std::size_t numLevels,
                 const SimulationState *state) override {
    throw std::runtime_error(
        "photonics qpu - allocation from state is not implemented");
    return {};
  }

  void deallocate(std::size_t idx) override {}
  void deallocate(const std::vector<std::size_t> &idxs) override {}

  void set_execution_context(ExecutionContext *ctx) override { context = ctx; }

  void reset_execution_context() override {
    if (context && context->name == "sample") {
      auto sampleResult = qpp::sample(context->shots, state, sampleQudits, 3);
      sampleQudits.clear();

      ExecutionResult execResult;
      for (auto [result, count] : sampleResult) {
        std::cout << fmt::format("Sample {} : {}", result, count) << "\n";
        // Populate counts dictionary. FIXME - handle qudits with >= 10 levels
        // better.
        std::string resultStr;
        resultStr.reserve(result.size());
        for (auto x : result)
          resultStr += std::to_string(x);
        execResult.counts[resultStr] = count;
      }
      context->result.append(execResult);
    }
  }

  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const operation_metadata &metadata) override {
    using namespace Eigen;
    // Map input to row-major matrix
    const Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> u =
        Map<const Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(
            matrixRowMajor.data(), 3, 3);
    state = qpp::apply(state, u, targets, 3);
  }

  std::size_t mz(std::size_t idx, const std::string regName) override {
    if (context && context->name == "sample") {
      sampleQudits.push_back(idx);
      return 0;
    }

    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple =
        qpp::measure(state, qpp::cmat::Identity(3, 3), {idx},
                     /*qudit dimension=*/3, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                       collapsed_state.size());

    cudaq::info("Measured qubit {} -> {}", idx, measurement_result);
    return measurement_result;
  }

  void applyControlRegion(const std::vector<std::size_t> &controls,
                          const std::function<void()> &wrapped) override {
    throw std::runtime_error("qudit applyControlRegion not implemented.");
  }

  void applyAdjointRegion(const std::function<void()> &wrapped) override {
    throw std::runtime_error("qudit applyAdjointRegion not implemented.");
  }

  void reset(std::size_t qidx) override {
    state = qpp::reset(state, {qidx}, 3);
  }

  SimpleQuditSimulator(const platform_metadata &m) : qpu_mixin(m) {}

  virtual ~SimpleQuditSimulator() = default;

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      SimpleQuditSimulator,
      static std::unique_ptr<qpu> create(const platform_metadata &m) {
        return std::make_unique<SimpleQuditSimulator>(m);
      })
};

CUDAQ_REGISTER_EXTENSION(SimpleQuditSimulator)

} // namespace cudaq
