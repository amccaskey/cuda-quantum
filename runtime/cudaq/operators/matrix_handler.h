/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operator.h"

namespace cudaq::experimental {

struct elementary_matrix_op {
  std::vector<std::size_t> supports;
  operator_matrix_generator generator;
};

using matrix_op_term_t = std::vector<elementary_matrix_op>;

using matrix_op_data = details::operator_data<matrix_op_term_t>;

class matrix_handler : public details::operator_handler<matrix_op_term_t> {
public:
  using term_t = matrix_op_term_t;

  matrix_op_data initialize() const override;
  std::size_t num_qubits(const matrix_op_data &thisPtr) const override;
  std::vector<double>
  get_data_representation(const matrix_op_data &) const override {
    throw std::runtime_error("not available");
    return {};
  }

  operator_matrix to_matrix(const matrix_op_data &data, const parameter_map &p,
                            const dimensions_map &dimensions) const override;

  std::vector<operator_matrix>
  get_support_matrices(const matrix_op_data &,
                       const dimensions_map &dimensions) const override;

  /// @brief Add the given spin_op to this one and return *this
  void add_assign(matrix_op_data &thisPtr, const matrix_op_data &v) override;

  /// @brief Multiply the given spin_op with this one and return *this
  void mult_assign(matrix_op_data &thisPtr, const matrix_op_data &v) override;

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  bool check_equality(const matrix_op_data &thisPtr,
                      const matrix_op_data &v) const override;

  bool is_template(const matrix_op_data &op,
                   const dimensions_map &) const override;
  std::set<std::size_t> get_supports(const matrix_op_data &) const override;

  std::string to_string(const matrix_op_data &thisPtr,
                        bool printCoeffs) const override;
};

using matrix_op = operator_<matrix_handler>;
matrix_op from_matrix(const operator_matrix &m, const std::size_t quditIdx);
matrix_op from_matrix(const operator_matrix_generator &m,
                      const std::size_t quditIdx);
matrix_op from_matrix(const operator_matrix_generator &m,
                      const std::vector<std::size_t> &quditIdxs);

operator_matrix exponentiateMatrix(const operator_matrix &m);

template <typename T>
matrix_op exp(const operator_<T> &op) {
  // Create a matrix generator that will compute exp(op) when evaluated
  auto generator = [op](const dimensions_map &dims,
                        const parameter_map &params) {
    // Get the matrix representation of the operator
    return exponentiateMatrix(op.to_matrix(dims, params));
  };

  // Create matrix_op with the generator
  // Use max qubit index from original operator
  auto supports = op.get_supports();
  std::vector<std::size_t> tmp(supports.begin(), supports.end());
  return from_matrix(generator, tmp);
}

template <typename LHSHandler, typename RHSHandler>
auto operator*(const operator_<LHSHandler> &lhs,
               const operator_<RHSHandler> &rhs) {

  matrix_op_data result;

  // Handle each term in the product
  for (const auto &lhs_term : lhs) {
    for (const auto &rhs_term : rhs) {
      matrix_op_term_t product_term;

      auto lhs_supports = lhs_term.get_supports();
      auto rhs_supports = rhs_term.get_supports();

      auto maxSite =
          std::max(*std::max_element(lhs_supports.begin(), lhs_supports.end()),
                   *std::max_element(rhs_supports.begin(), rhs_supports.end()));

      auto setContains = [](const std::set<std::size_t> &vec, std::size_t idx) {
        return std::find(vec.begin(), vec.end(), idx) != vec.end();
      };

      for (std::size_t lCount = 0, rCount = 0, j = 0; j <= maxSite; j++) {
        if (setContains(lhs_supports, j) && setContains(rhs_supports, j)) {
          auto gen = [lCount, rCount, lhs_term,
                      rhs_term](const dimensions_map &dimensions,
                                const parameter_map &params) {
            auto lhs_ops = lhs_term.get_elementary_operators(dimensions);
            auto rhs_ops = rhs_term.get_elementary_operators(dimensions);
            return lhs_ops[lCount] * rhs_ops[rCount];
          };

          product_term.emplace_back(std::vector<std::size_t>{j}, gen);

          lCount++;
          rCount++;
        } else if (setContains(lhs_supports, j)) {
          auto gen = [lCount, lhs_term](const dimensions_map &dimensions,
                                        const parameter_map &params) {
            auto lhs_ops = lhs_term.get_elementary_operators(dimensions);
            return lhs_ops[lCount];
          };
          product_term.emplace_back(std::vector<std::size_t>{j}, gen);
          lCount++;

        } else {
          auto gen = [rCount, rhs_term](const dimensions_map &dimensions,
                                        const parameter_map &params) {
            auto rhs_ops = rhs_term.get_elementary_operators(dimensions);
            return rhs_ops[rCount];
          };
          product_term.emplace_back(std::vector<std::size_t>{j}, gen);
          rCount++;
        }
      }

      result.productTerms.push_back(product_term);

      // Multiply coefficients
      result.coefficients.push_back(1.0);
    }
  }

  return matrix_op(std::move(result));
}

template <typename LHSHandler, typename RHSHandler>
auto operator+(const operator_<LHSHandler> &lhs,
               const operator_<RHSHandler> &rhs) {
  matrix_op_data result;

  // Process LHS terms
  for (const auto &lhs_term : lhs) {
    matrix_op_term_t matrix_term;
    auto lhs_supports = lhs_term.get_supports();

    // Convert each support operator to matrix form
    for (std::size_t i = 0; i < lhs_supports.size(); i++) {
      auto generator = [lhs_term, i](const dimensions_map &dims,
                                     const parameter_map &params) {
        return lhs_term.get_elementary_operators(dims)[i];
      };
      matrix_term.emplace_back(
          std::vector<std::size_t>{*std::next(lhs_supports.begin(), i)},
          generator);
    }

    result.productTerms.push_back(matrix_term);
    result.coefficients.push_back(lhs_term.get_coefficient());
  }

  // Process RHS terms
  for (const auto &rhs_term : rhs) {
    matrix_op_term_t matrix_term;
    auto rhs_supports = rhs_term.get_supports();

    // Convert each support operator to matrix form
    for (std::size_t i = 0; i < rhs_supports.size(); i++) {
      auto generator = [rhs_term, i](const dimensions_map &dims,
                                     const parameter_map &params) {
        return rhs_term.get_elementary_operators(dims)[i];
      };
      matrix_term.emplace_back(
          std::vector<std::size_t>{*std::next(rhs_supports.begin(), i)},
          generator);
    }

    result.productTerms.push_back(matrix_term);
    result.coefficients.push_back(rhs_term.get_coefficient());
  }

  return matrix_op(std::move(result));
}

template <typename LHSHandler, typename RHSHandler>
auto operator-(const operator_<LHSHandler> &lhs,
               const operator_<RHSHandler> &rhs) {
  return operator+(lhs, -1. * rhs);
}

} // namespace cudaq::experimental