/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state_vector.h"
#include "state_vector_impl.h"

namespace cudaq::simulator::gpu {

state_vector::state_vector() : pImpl(std::make_unique<Impl>()) {}

state_vector::~state_vector() = default;

sample_result state_vector::sample(std::size_t num_shots,
                                   const std::string &kernel_name,
                                   const std::function<void()> &wrapped) {
  return pImpl->sample(num_shots, kernel_name, wrapped);
}

void state_vector::dump_state(std::ostream &os) { pImpl->dump_state(os); }

cudaq::state state_vector::get_state() {
  return state(pImpl->get_internal_state().release());
}

cudaq::state state_vector::get_state(const state_data &data) {
  return state(nullptr);
}

std::unique_ptr<cudaq::SimulationState>
state_vector::get_internal_state(const state_data &data) {
  return pImpl->get_internal_state();
}

simulation_precision state_vector::get_precision() const {
  return pImpl->get_precision();
}

std::size_t state_vector::allocateQudit(std::size_t numLevels) {
  return pImpl->allocateQudit(numLevels);
}

std::vector<std::size_t>
state_vector::allocateQudits(std::size_t numQudits, std::size_t numLevels,
                             const void *state,
                             simulation_precision precision) {
  return pImpl->allocateQudits(numQudits, numLevels, state, precision);
}

std::vector<std::size_t>
state_vector::allocateQudits(std::size_t numQudits, std::size_t numLevels,
                             const SimulationState *state) {
  return pImpl->allocateQudits(numQudits, numLevels, state);
}

std::vector<std::size_t> state_vector::allocateQudits(std::size_t numQudits,
                                                      std::size_t numLevels) {
  return pImpl->allocateQudits(numQudits, numLevels);
}

void state_vector::deallocate(std::size_t idx) { pImpl->deallocate(idx); }

void state_vector::deallocate(const std::vector<std::size_t> &idxs) {
  pImpl->deallocate(idxs);
}

void state_vector::apply(
    const std::vector<std::complex<double>> &matrixRowMajor,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const traits::operation_metadata &metadata) {
  pImpl->apply(matrixRowMajor, controls, targets, metadata);
}

void state_vector::applyControlRegion(const std::vector<std::size_t> &controls,
                                      const std::function<void()> &wrapped) {
  pImpl->applyControlRegion(controls, wrapped);
}

void state_vector::applyAdjointRegion(const std::function<void()> &wrapped) {
  pImpl->applyAdjointRegion(wrapped);
}

void state_vector::reset(std::size_t qidx) { pImpl->reset(qidx); }

void state_vector::apply_exp_pauli(double theta,
                                   const std::vector<std::size_t> &controls,
                                   const std::vector<std::size_t> &qubitIds,
                                   const cudaq::spin_op_term &term) {
  pImpl->apply_exp_pauli(theta, controls, qubitIds, term);
}

std::size_t state_vector::mz(std::size_t idx, const std::string regName) {
  return pImpl->mz(idx, regName);
}

} // namespace cudaq::simulator::gpu
