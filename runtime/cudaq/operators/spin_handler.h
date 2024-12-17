/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operator.h"

namespace cudaq::experimental::spin {

class spin_handler : public details::operator_handler {
protected:
  void expandToNQubits(std::vector<std::vector<std::size_t>> &terms,
                       const std::size_t numQubits);

public:
  details::operator_data initialize() const override;

  std::size_t num_qubits(const details::operator_data &thisPtr) const override;

  std::vector<double>
  get_data_representation(const details::operator_data &) const override;

  operator_matrix to_matrix(const details::operator_data &,
                            const parameter_map &p) const override;

  std::vector<operator_matrix>
  get_support_matrices(const details::operator_data &,
                       const parameter_map &p) const override;

  /// @brief Add the given spin_op to this one and return *this
  void add_assign(details::operator_data &thisPtr,
                  const details::operator_data &v) override;
  /// @brief Multiply the given spin_op with this one and return *this
  void mult_assign(details::operator_data &thisPtr,
                   const details::operator_data &v) override;

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  bool check_equality(const details::operator_data &thisPtr,
                      const details::operator_data &v) const override;
  std::string to_string(const details::operator_data &thisPtr,
                        bool printCoeffs) const override;
};

using spin_op = operator_<spin_handler>;

spin_op i(std::size_t idx);
spin_op x(std::size_t idx);
spin_op y(std::size_t idx);
spin_op z(std::size_t idx);

spin_op from_word(const std::string &pauliWord);
} // namespace cudaq::experimental::spin
