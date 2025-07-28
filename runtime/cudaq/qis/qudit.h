/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform/platform.h"

namespace cudaq {

template <std::size_t Levels>
class qvector;

/// The qudit models a general d-level quantum system.
/// This type is templated on the number of levels d.
template <std::size_t Levels>
class qudit {
  /// Every qudit has a logical index in the global qudit register,
  /// `idx` is this logical index, it must be
  /// provided at construction and is immutable.
  std::size_t idx = 0;

  // Bool to indicate if we are currently negated
  // as a control qudit.
  bool isNegativeControl = false;

  friend class qvector<Levels>;

public:
  /// Construct a qudit, will allocated a new unique index
  qudit()
      : idx(cudaq::get_qpu().as<cudaq::simulation_trait>()->allocateQudit(
            Levels)) {}
  qudit(bool) {}
  qudit(std::size_t uid) : idx(uid) {}

  qudit(const std::vector<complex> &state) {
    if (state.size() != Levels)
      throw std::runtime_error(
          "Invalid number of state vector elements for qudit allocation (" +
          std::to_string(state.size()) + ").");

    auto norm = std::inner_product(
                    state.begin(), state.end(), state.begin(), complex{0., 0.},
                    [](auto a, auto b) { return a + b; },
                    [](auto a, auto b) { return std::conj(a) * b; })
                    .real();
    if (std::fabs(1.0 - norm) > 1e-4)
      throw std::runtime_error("Invalid vector norm for qudit allocation.");

    // Perform the initialization
    auto precision = std::is_same_v<complex::value_type, float>
                         ? simulation_precision::fp32
                         : simulation_precision::fp64;
    idx = cudaq::get_qpu().as<cudaq::simulation_trait>()->allocateQudits(
        1, Levels, state.data(), precision)[0];
  }
  qudit(const std::initializer_list<complex> &list)
      : qudit({list.begin(), list.end()}) {}

  // Alex - might as well make this change now
  // Qudits cannot be copied
  // qudit(const qudit &q) = delete;
  // // qudits cannot be moved
  // qudit(qudit &&) = delete;

  // Return the unique id / index for this qudit
  std::size_t id() const { return idx; }

  // Return this qudit's dimension
  static constexpr std::size_t n_levels() { return Levels; }

  // Qudits used as controls can be negated, i.e
  // instead of applying a target op if the qudit is
  // in the vacuum state, you can apply a target op if the
  // state is in an excited state if this control qudit is negated
  qudit<Levels> &negate() {
    isNegativeControl = !isNegativeControl;
    return *this;
  }

  // Is this qudit negated?
  bool is_negative() { return isNegativeControl; }

  // Syntactic sugar for negating a control
  qudit<Levels> &operator!() { return negate(); }

  // Destructor, return the qudit so it can be reused
  ~qudit() { cudaq::get_qpu().as<cudaq::simulation_trait>()->deallocate(idx); }
};

// A qubit is a qudit with 2 levels.
using qubit = qudit<2>;


} // namespace cudaq
