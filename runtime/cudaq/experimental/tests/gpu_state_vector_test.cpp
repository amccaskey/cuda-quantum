/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qpu.h"
#include "cudaq/qpus/simulator/gpu/state_vector.h"

#include <gtest/gtest.h>
#include <cmath>
#include <complex>

using namespace cudaq::simulator;

// Helper to check if two complex numbers are approximately equal
bool approx_equal(std::complex<double> a, std::complex<double> b,
                  double tolerance = 1e-6) {
  return std::abs(a - b) < tolerance;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(GPUStateVectorTest, Construction) {
  gpu::state_vector<double> qpu;
  EXPECT_EQ(qpu.name(), "gpu::state_vector");
  EXPECT_EQ(qpu.get_precision(), cudaq::simulation_precision::fp64);
}

TEST(GPUStateVectorTest, ConstructionWithConfig) {
  cudaq::qpu_configuration config;
  config.insert({"random_seed", std::size_t(42)});

  gpu::state_vector<double> qpu(config);
  EXPECT_EQ(qpu.get_configuration().size(), 1);
  EXPECT_TRUE(qpu.get_configuration().contains("random_seed"));
}

TEST(GPUStateVectorTest, PrecisionFloat) {
  gpu::state_vector<float> qpu;
  EXPECT_EQ(qpu.get_precision(), cudaq::simulation_precision::fp32);
}

// ============================================================================
// Qubit Allocation Tests
// ============================================================================

TEST(GPUStateVectorTest, AllocateSingleQubit) {
  gpu::state_vector<double> qpu;
  auto idx = qpu.allocateQudit(2);
  EXPECT_EQ(idx, 0);
}

TEST(GPUStateVectorTest, AllocateMultipleQubits) {
  gpu::state_vector<double> qpu;
  auto indices = qpu.allocateQudits(3, 2);
  EXPECT_EQ(indices.size(), 3);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 1);
  EXPECT_EQ(indices[2], 2);
}

TEST(GPUStateVectorTest, AllocateQuditThrows) {
  gpu::state_vector<double> qpu;
  EXPECT_THROW(qpu.allocateQudit(3), std::runtime_error);
}

// ============================================================================
// Gate Application Tests
// ============================================================================

TEST(GPUStateVectorTest, ApplyPauliX) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(1, 2);

  // X gate matrix
  std::vector<std::complex<double>> x_matrix = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cudaq::traits::operation_metadata metadata("x");
  qpu.apply(x_matrix, {}, {0}, metadata);

  auto state = qpu.get_state();
  EXPECT_EQ(state.get_num_qubits(), 1);

  // After X, state should be |1⟩
  auto amp0 = state.amplitude({0});
  auto amp1 = state.amplitude({1});

  EXPECT_TRUE(approx_equal(amp0, {0.0, 0.0}));
  EXPECT_TRUE(approx_equal(amp1, {1.0, 0.0}));
}

TEST(GPUStateVectorTest, ApplyHadamard) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(1, 2);

  // Hadamard gate matrix
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  std::vector<std::complex<double>> h_matrix = {
      {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0}};

  cudaq::traits::operation_metadata metadata("h");
  qpu.apply(h_matrix, {}, {0}, metadata);

  auto state = qpu.get_state();

  // After H, state should be (|0⟩ + |1⟩)/√2
  auto amp0 = state.amplitude({0});
  auto amp1 = state.amplitude({1});

  EXPECT_TRUE(approx_equal(amp0, {inv_sqrt2, 0.0}));
  EXPECT_TRUE(approx_equal(amp1, {inv_sqrt2, 0.0}));
}

TEST(GPUStateVectorTest, ApplyCNOT) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(2, 2);

  // First apply H to qubit 0
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  std::vector<std::complex<double>> h_matrix = {
      {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0}};

  cudaq::traits::operation_metadata h_meta("h");
  qpu.apply(h_matrix, {}, {0}, h_meta);

  // Then apply CNOT (control=0, target=1)
  std::vector<std::complex<double>> x_matrix = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cudaq::traits::operation_metadata cnot_meta("cx");
  qpu.apply(x_matrix, {0}, {1}, cnot_meta);

  auto state = qpu.get_state();

  // After H-CNOT, we should have Bell state: (|00⟩ + |11⟩)/√2
  auto amp00 = state.amplitude({0, 0});
  auto amp01 = state.amplitude({0, 1});
  auto amp10 = state.amplitude({1, 0});
  auto amp11 = state.amplitude({1, 1});

  EXPECT_TRUE(approx_equal(amp00, {inv_sqrt2, 0.0}));
  EXPECT_TRUE(approx_equal(amp01, {0.0, 0.0}));
  EXPECT_TRUE(approx_equal(amp10, {0.0, 0.0}));
  EXPECT_TRUE(approx_equal(amp11, {inv_sqrt2, 0.0}));
}

// ============================================================================
// Measurement Tests
// ============================================================================

TEST(GPUStateVectorTest, MeasureZeroBasis) {
  gpu::state_vector<double> qpu;
  qpu.set_random_seed(42);
  qpu.allocateQudits(1, 2);

  // In |0⟩ state, should always measure 0
  auto result = qpu.mz(0);
  EXPECT_EQ(result, 0);
}

TEST(GPUStateVectorTest, MeasureAfterX) {
  gpu::state_vector<double> qpu;
  qpu.set_random_seed(42);
  qpu.allocateQudits(1, 2);

  // Apply X gate
  std::vector<std::complex<double>> x_matrix = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cudaq::traits::operation_metadata metadata("x");
  qpu.apply(x_matrix, {}, {0}, metadata);

  // In |1⟩ state, should always measure 1
  auto result = qpu.mz(0);
  EXPECT_EQ(result, 1);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST(GPUStateVectorTest, ResetQubit) {
  gpu::state_vector<double> qpu;
  qpu.set_random_seed(42);
  qpu.allocateQudits(1, 2);

  // Apply X to get |1⟩
  std::vector<std::complex<double>> x_matrix = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cudaq::traits::operation_metadata metadata("x");
  qpu.apply(x_matrix, {}, {0}, metadata);

  // Reset should bring back to |0⟩
  qpu.reset(0);

  auto state = qpu.get_state();
  auto amp0 = state.amplitude({0});
  auto amp1 = state.amplitude({1});

  EXPECT_TRUE(approx_equal(amp0, {1.0, 0.0}));
  EXPECT_TRUE(approx_equal(amp1, {0.0, 0.0}));
}

// ============================================================================
// State Management Tests
// ============================================================================

TEST(GPUStateVectorTest, GetState) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(2, 2);

  auto state = qpu.get_state();
  EXPECT_EQ(state.get_num_qubits(), 2);
  EXPECT_TRUE(state.is_on_gpu());
  EXPECT_EQ(state.get_precision(), cudaq::SimulationState::precision::fp64);
}

TEST(GPUStateVectorTest, StateAmplitudes) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(2, 2);

  auto state = qpu.get_state();

  // In |00⟩ state
  EXPECT_TRUE(approx_equal(state.amplitude({0, 0}), {1.0, 0.0}));
  EXPECT_TRUE(approx_equal(state.amplitude({0, 1}), {0.0, 0.0}));
  EXPECT_TRUE(approx_equal(state.amplitude({1, 0}), {0.0, 0.0}));
  EXPECT_TRUE(approx_equal(state.amplitude({1, 1}), {0.0, 0.0}));
}

TEST(GPUStateVectorTest, DumpState) {
  gpu::state_vector<double> qpu;
  qpu.allocateQudits(1, 2);

  std::ostringstream os;
  qpu.dump_state(os);

  std::string output = os.str();
  EXPECT_FALSE(output.empty());
}

// ============================================================================
// Pauli Rotation Tests
// ============================================================================

// TEST(GPUStateVectorTest, ApplyExpPauliZ) {
//   gpu::state_vector<double> qpu;
//   qpu.allocateQudits(1, 2);

//   // Apply Hadamard first to get superposition
//   const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
//   std::vector<std::complex<double>> h_matrix = {
//       {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0}};

//   cudaq::traits::operation_metadata h_meta("h");
//   qpu.apply(h_matrix, {}, {0}, h_meta);

//   // Apply exp(-i * π/4 * Z)
//   cudaq::spin_op z_term = cudaq::spin::z(0);
//   qpu.apply_exp_pauli(M_PI / 4.0, {}, {0}, z_term);

//   // State should still be a valid superposition
//   auto state = qpu.get_state();
//   auto amp0 = state.amplitude({0});
//   auto amp1 = state.amplitude({1});

//   // Check normalization
//   double norm =
//       std::norm(amp0) + std::norm(amp1);
//   EXPECT_NEAR(norm, 1.0, 1e-6);
// }

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(GPUStateVectorTest, ApplyGateWithoutAllocation) {
  gpu::state_vector<double> qpu;

  std::vector<std::complex<double>> x_matrix = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cudaq::traits::operation_metadata metadata("x");
  EXPECT_THROW(qpu.apply(x_matrix, {}, {0}, metadata), std::runtime_error);
}

TEST(GPUStateVectorTest, MeasureWithoutAllocation) {
  gpu::state_vector<double> qpu;
  EXPECT_THROW(qpu.mz(0), std::runtime_error);
}

TEST(GPUStateVectorTest, GetStateWithoutAllocation) {
  gpu::state_vector<double> qpu;
  EXPECT_THROW(qpu.get_state(), std::runtime_error);
}

// ============================================================================
// Float Precision Tests
// ============================================================================

TEST(GPUStateVectorTest, FloatPrecisionBasic) {
  gpu::state_vector<float> qpu;
  qpu.allocateQudits(2, 2);

  // Apply H to first qubit
  const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
  std::vector<std::complex<double>> h_matrix = {
      {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0}};

  cudaq::traits::operation_metadata h_meta("h");
  qpu.apply(h_matrix, {}, {0}, h_meta);

  auto state = qpu.get_state();
  EXPECT_EQ(state.get_precision(), cudaq::SimulationState::precision::fp32);

  // Check approximate equality with lower precision
  auto amp0 = state.amplitude({0, 0});
  EXPECT_NEAR(std::abs(amp0 - std::complex<double>(inv_sqrt2, 0.0)), 0.0, 1e-5);
}

// ============================================================================
// Random Seed Tests
// ============================================================================

TEST(GPUStateVectorTest, RandomSeedDeterminism) {
  // Create two QPUs with same seed
  gpu::state_vector<double> qpu1;
  qpu1.set_random_seed(12345);
  qpu1.allocateQudits(1, 2);

  gpu::state_vector<double> qpu2;
  qpu2.set_random_seed(12345);
  qpu2.allocateQudits(1, 2);

  // Apply H to both
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  std::vector<std::complex<double>> h_matrix = {
      {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0}};

  cudaq::traits::operation_metadata h_meta("h");
  qpu1.apply(h_matrix, {}, {0}, h_meta);
  qpu2.apply(h_matrix, {}, {0}, h_meta);

  // Measurements should be the same
  std::vector<int> results1, results2;
  for (int i = 0; i < 10; ++i) {
    // Note: measurement collapses state, so we need fresh instances
    gpu::state_vector<double> qpu_temp1;
    qpu_temp1.set_random_seed(12345 + i);
    qpu_temp1.allocateQudits(1, 2);
    qpu_temp1.apply(h_matrix, {}, {0}, h_meta);
    results1.push_back(qpu_temp1.mz(0));

    gpu::state_vector<double> qpu_temp2;
    qpu_temp2.set_random_seed(12345 + i);
    qpu_temp2.allocateQudits(1, 2);
    qpu_temp2.apply(h_matrix, {}, {0}, h_meta);
    results2.push_back(qpu_temp2.mz(0));
  }

  EXPECT_EQ(results1, results2);
}

