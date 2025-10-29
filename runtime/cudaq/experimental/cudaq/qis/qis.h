/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qudit.h"
#include "cudaq/qis/noise_model.h"
#include "cudaq/traits/simulator.h"
#include "gates.h"

#include <algorithm>
#include <functional>
#include <vector>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA-Q kernels using the type-erased simulator trait.

namespace cudaq {

// Operation modifiers
struct base {};
struct ctrl {};
struct adj {};

// ============================================================================
// Helper Functions for ID Extraction
// ============================================================================

/// @brief Extract qubit IDs from a single qubit
inline void extract_qubit_ids(std::vector<std::size_t> &ids, const qubit &q) {
  ids.push_back(q.id());
}

/// @brief Extract qubit IDs from a container (qvector, qarray, qview)
template <typename QubitContainer>
auto extract_qubit_ids(std::vector<std::size_t> &ids, QubitContainer &&qubits)
    -> decltype(qubits.begin(), void()) {
  for (auto &q : qubits) {
    ids.push_back(q.id());
  }
}

/// @brief Collect all qubit IDs from variadic arguments (qubits or containers)
template <typename... QubitArgs>
std::vector<std::size_t> collect_qubit_ids(QubitArgs &&...args) {
  std::vector<std::size_t> ids;
  (extract_qubit_ids(ids, std::forward<QubitArgs>(args)), ...);
  return ids;
}

// ============================================================================
// Generic Gate Application Template
// ============================================================================

/// @brief Generic single-qubit gate applicator with modifier support
/// Supports: base (broadcast), ctrl, adj
/// Handles both individual qubits and containers (qvector, qarray, qview)
template <typename GateType, typename mod = base, typename... QubitArgs>
void apply_gate(QubitArgs &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  GateType gate_obj;
  auto gate_matrix = gate_obj.getGate();
  std::string gate_name = gate_obj.name();
  
  // Collect all qubit IDs from arguments (handles both qubits and containers)
  auto qubit_ids = collect_qubit_ids(std::forward<QubitArgs>(args)...);
  
  // Base: broadcast to all qubits
  if constexpr (std::is_same_v<mod, base>) {
    for (auto id : qubit_ids) {
      api->q_applicator(gate_matrix, {}, {id},
                       traits::operation_metadata{gate_name});
    }
  }
  // Ctrl: last qubit is target, rest are controls
  else if constexpr (std::is_same_v<mod, ctrl>) {
    std::vector<std::size_t> controls(qubit_ids.begin(), qubit_ids.end() - 1);
    std::vector<std::size_t> targets = {qubit_ids.back()};
    api->q_applicator(gate_matrix, controls, targets,
                     traits::operation_metadata{gate_name});
  }
  // Adj: apply adjoint to single qubit
  else if constexpr (std::is_same_v<mod, adj>) {
    for (auto id : qubit_ids) {
      api->q_applicator(gate_matrix, {}, {id},
                       traits::operation_metadata{gate_name});
    }
  }
}

/// @brief Generic parametric single-qubit gate applicator with modifier support
/// Handles both individual qubits and containers
template <typename GateType, typename mod = base, typename... QubitArgs>
void apply_parametric_gate(double angle, QubitArgs &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  GateType gate_obj;
  auto gate_matrix = gate_obj.getGate({angle});
  std::string gate_name = gate_obj.name();
  
  // Collect all qubit IDs from arguments
  auto qubit_ids = collect_qubit_ids(std::forward<QubitArgs>(args)...);
  
  if constexpr (std::is_same_v<mod, base>) {
    // Broadcast to all qubits
    for (auto id : qubit_ids) {
      api->q_applicator(gate_matrix, {}, {id},
                       traits::operation_metadata{gate_name});
    }
  } else if constexpr (std::is_same_v<mod, ctrl>) {
    // Last qubit is target, rest are controls
    std::vector<std::size_t> controls(qubit_ids.begin(), qubit_ids.end() - 1);
    std::vector<std::size_t> targets = {qubit_ids.back()};
    api->q_applicator(gate_matrix, controls, targets,
                     traits::operation_metadata{gate_name});
  } else if constexpr (std::is_same_v<mod, adj>) {
    // Apply adjoint
    for (auto id : qubit_ids) {
      api->q_applicator(gate_matrix, {}, {id},
                       traits::operation_metadata{gate_name});
    }
  }
}

// ============================================================================
// Single Qubit Gates (No Parameters)
// ============================================================================

template <typename mod = base, typename... QubitArgs>
void h(QubitArgs &&...qubits) {
  apply_gate<gates::h<>, mod>(std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void x(QubitArgs &&...qubits) {
  apply_gate<gates::x<>, mod>(std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void y(QubitArgs &&...qubits) {
  apply_gate<gates::y<>, mod>(std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void z(QubitArgs &&...qubits) {
  apply_gate<gates::z<>, mod>(std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void s(QubitArgs &&...qubits) {
  apply_gate<gates::s<>, mod>(std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void t(QubitArgs &&...qubits) {
  apply_gate<gates::t<>, mod>(std::forward<QubitArgs>(qubits)...);
}

// Adjoint versions
inline void sdg(qubit &q) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  api->q_applicator(gates::sdg<>().getGate(), {}, {q.id()},
                    traits::operation_metadata{"sdg"});
}

inline void tdg(qubit &q) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  api->q_applicator(gates::tdg<>().getGate(), {}, {q.id()},
                    traits::operation_metadata{"tdg"});
}

// ============================================================================
// Single Qubit Parametric Gates
// ============================================================================

template <typename mod = base, typename... QubitArgs>
void rx(double angle, QubitArgs &&...qubits) {
  apply_parametric_gate<gates::rx<>, mod>(angle, std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void ry(double angle, QubitArgs &&...qubits) {
  apply_parametric_gate<gates::ry<>, mod>(angle, std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void rz(double angle, QubitArgs &&...qubits) {
  apply_parametric_gate<gates::rz<>, mod>(angle, std::forward<QubitArgs>(qubits)...);
}

template <typename mod = base, typename... QubitArgs>
void r1(double angle, QubitArgs &&...qubits) {
  apply_parametric_gate<gates::r1<>, mod>(angle, std::forward<QubitArgs>(qubits)...);
}

// ============================================================================
// Convenience Functions for Controlled Gates
// ============================================================================

inline void cnot(qubit &ctrl, qubit &target) { x<cudaq::ctrl>(ctrl, target); }
inline void cx(qubit &ctrl, qubit &target) { x<cudaq::ctrl>(ctrl, target); }
inline void cy(qubit &ctrl, qubit &target) { y<cudaq::ctrl>(ctrl, target); }
inline void cz(qubit &ctrl, qubit &target) { z<cudaq::ctrl>(ctrl, target); }
inline void ch(qubit &ctrl, qubit &target) { h<cudaq::ctrl>(ctrl, target); }
inline void cs(qubit &ctrl, qubit &target) { s<cudaq::ctrl>(ctrl, target); }
inline void ct(qubit &ctrl, qubit &target) { t<cudaq::ctrl>(ctrl, target); }

// Toffoli gate (CCX)
inline void ccx(qubit &ctrl1, qubit &ctrl2, qubit &target) {
  x<cudaq::ctrl>(ctrl1, ctrl2, target);
}

// Controlled parametric gates
inline void crx(double angle, qubit &ctrl, qubit &target) {
  rx<cudaq::ctrl>(angle, ctrl, target);
}

inline void cry(double angle, qubit &ctrl, qubit &target) {
  ry<cudaq::ctrl>(angle, ctrl, target);
}

inline void crz(double angle, qubit &ctrl, qubit &target) {
  rz<cudaq::ctrl>(angle, ctrl, target);
}

inline void cr1(double angle, qubit &ctrl, qubit &target) {
  r1<cudaq::ctrl>(angle, ctrl, target);
}

// ============================================================================
// Two-Qubit Gates
// ============================================================================

inline void swap(qubit &q1, qubit &q2) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  api->q_applicator(gates::swap<>().getGate(), {}, {q1.id(), q2.id()},
                    traits::operation_metadata{"swap"});
}

inline void cswap(qubit &ctrl, qubit &q1, qubit &q2) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  api->q_applicator(gates::swap<>().getGate(), {ctrl.id()},
                    {q1.id(), q2.id()}, traits::operation_metadata{"swap"});
}

// ============================================================================
// Measurement Operations  
// ============================================================================
// Note: Measurement also supports containers via the same dispatch mechanism

inline bool mz(qubit &q) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  return api->q_measurer(q.id(), "") != 0;
}

inline bool mx(qubit &q) {
  // Measure in X basis: apply H, measure in Z, apply H
  h(q);
  bool result = mz(q);
  h(q);
  return result;
}

inline bool my(qubit &q) {
  // Measure in Y basis
  r1(-M_PI_2, q);
  h(q);
  bool result = mz(q);
  return result;
}

/// @brief Measure multiple qubits (works with containers via dispatch)
template <typename... QubitArgs>
std::vector<bool> mz(QubitArgs &&...args) {
  auto qubit_ids = collect_qubit_ids(std::forward<QubitArgs>(args)...);
  std::vector<bool> results;
  for (auto id : qubit_ids) {
    auto *api = cudaq::get_kernel_api();
    if (!api)
      throw std::runtime_error("Kernel API not initialized");
    results.push_back(api->q_measurer(id, "") != 0);
  }
  return results;
}

// ============================================================================
// Reset Operation
// ============================================================================

inline void reset(qubit &q) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");
  api->q_resetter(q.id());
}

// ============================================================================
// Control and Adjoint Regions
// ============================================================================

template <typename QuantumKernel, typename... Args>
void control(QuantumKernel &&kernel, qubit &ctrl, Args &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  std::vector<std::size_t> ctrls{ctrl.id()};
  api->q_control_region(ctrls, [&]() {
    kernel(std::forward<Args>(args)...);
  });
}

template <typename QuantumKernel, typename QubitRange, typename... Args>
void control(QuantumKernel &&kernel, QubitRange &ctrl_qubits, Args &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  std::vector<std::size_t> ctrls;
  for (auto &q : ctrl_qubits) {
    ctrls.push_back(q.id());
  }

  api->q_control_region(ctrls, [&]() {
    kernel(std::forward<Args>(args)...);
  });
}

template <typename QuantumKernel, typename... Args>
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  api->q_adjoint_region([&]() {
    kernel(std::forward<Args>(args)...);
  });
}

// ============================================================================
// Compute-Action Patterns
// ============================================================================

template <typename ComputeFunction, typename ActionFunction>
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

template <typename ComputeFunction, typename ActionFunction>
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

// ============================================================================
// Noise Application
// ============================================================================

/// @brief Apply noise channel with runtime parameters
/// Supports both individual qubits and containers via dispatch mechanism
template <typename KrausChannel, typename... QubitArgs,
          typename = std::enable_if_t<std::is_base_of_v<kraus_channel, KrausChannel>>>
void apply_noise(const std::vector<double> &params, QubitArgs &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  // Get noise model from simulator
  const auto *noise_model = api->q_noise_getter();
  if (!noise_model) {
    // Per spec: no noise model provided, emit warning and skip
    return;
  }

  // Collect all qubit IDs using our dispatch mechanism
  auto qubit_ids = collect_qubit_ids(std::forward<QubitArgs>(args)...);

  // Get the Kraus operators for this channel
  auto channel = noise_model->template get_channel<KrausChannel>(params);
  if (channel.empty()) {
    // Per spec: channel not registered, skip application
    return;
  }

  // Apply the noise channel to the qubits
  api->q_noise_applicator(channel, qubit_ids);
}

/// @brief Apply noise channel with compile-time parameters
/// Extracts parameters from leading double arguments
template <typename KrausChannel, typename... Args,
          typename = std::enable_if_t<std::is_base_of_v<kraus_channel, KrausChannel>>>
void apply_noise(Args &&...args) {
  auto *api = cudaq::get_kernel_api();
  if (!api)
    throw std::runtime_error("Kernel API not initialized");

  const auto *noise_model = api->q_noise_getter();
  if (!noise_model)
    return;

  // Separate parameters (leading doubles) from qubits
  std::vector<double> params;
  std::vector<std::size_t> qubit_ids;

  // Helper to extract parameters and qubits
  ([&] {
    using T = std::decay_t<decltype(args)>;
    if constexpr (std::is_floating_point_v<T>) {
      params.push_back(static_cast<double>(args));
    } else {
      extract_qubit_ids(qubit_ids, std::forward<Args>(args));
    }
  }(), ...);

  // Get the Kraus operators for this channel
  auto channel = noise_model->template get_channel<KrausChannel>(params);
  if (channel.empty())
    return;

  // Apply the noise channel
  api->q_noise_applicator(channel, qubit_ids);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Convert measurement results to integer
inline std::int64_t to_integer(const std::vector<bool> &bits) {
  std::int64_t result = 0;
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (bits[i]) {
      result |= (1ULL << i);
    }
  }
  return result;
}

} // namespace cudaq
