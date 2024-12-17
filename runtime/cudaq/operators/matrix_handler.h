/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operator.h"

namespace cudaq::experimental::matrix {

class matrix_handler : public details::operator_handler {

public:
  details::operator_data initialize() const override;
  std::size_t num_qubits(const details::operator_data &thisPtr) const override;
  std::vector<double>
  get_data_representation(const details::operator_data &) const override {
    throw std::runtime_error("not available");
    return {};
  }

  operator_matrix to_matrix(const details::operator_data &data,
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

} // namespace cudaq::experimental::matrix

namespace cudaq::experimental {

using matrix_op = operator_<matrix::matrix_handler>;

matrix_op from_matrix(const operator_matrix &m,
                      const std::vector<std::size_t> &supports);

} // namespace cudaq::experimental
