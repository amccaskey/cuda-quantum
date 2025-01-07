/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "matrix_handler.h"
#include "common/EigenDense.h"
#include "common/FmtCore.h"
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cudaq::experimental {

matrix_op_data matrix_handler::initialize() const {
  matrix_op_data data;
  return data;
}

operator_matrix tensor_product(const std::vector<operator_matrix> &matrices) {
  if (matrices.empty())
    return operator_matrix({1.0, 0.0, 0.0, 1.0}); // 2x2 Identity

  operator_matrix result = matrices[0];
  for (std::size_t i = 1; i < matrices.size(); i++)
    result = kronecker(result, matrices[i]);
  return result;
}

operator_matrix
matrix_handler::to_matrix(const matrix_op_data &data, const parameter_map &p,
                          const dimensions_map &dimensions) const {

  if (data.productTerms.empty()) {
    return operator_matrix();
  }

  // Determine total system size
  std::size_t max_idx = 0;
  for (const auto &term : data.productTerms)
    for (const auto &op : term)
      max_idx = std::max(
          max_idx, *std::max_element(op.supports.begin(), op.supports.end()));

  const std::size_t n_qubits = max_idx + 1;

  auto eye = [](std::size_t dim) {
    std::vector<std::complex<double>> d(dim * dim, 0.0);
    operator_matrix ret(d, {dim, dim});
    for (std::size_t jj = 0; jj < ret.get_rows(); jj++)
      ret[{jj, jj}] = 1.;
    return ret;
  };

  // Process each term in the sum
  operator_matrix result;
  for (std::size_t term_idx = 0; term_idx < data.productTerms.size();
       term_idx++) {
    const auto &term = data.productTerms[term_idx];
    const auto &coeff = data.coefficients[term_idx];

    // Build tensor product for this term
    std::vector<operator_matrix> term_matrices(n_qubits);

    // Initialize with identity matrices
    for (std::size_t i = 0; i < n_qubits; i++) {
      const std::size_t dim = dimensions.empty()    ? 2
                              : dimensions.count(i) ? dimensions.at(i)
                                                    : 2;
      term_matrices[i] = eye(dim);
    }

    // Insert operators at their positions
    for (const auto &op : term) {
      auto m = op.generator(dimensions, p);
      auto supports = op.supports;
      auto numSupports = supports.size();
      if (numSupports > 1) {
        term_matrices[op.supports[0]] = m;
        // Don't need to keep the identities, we have m
        for (std::size_t k = numSupports - 1; k > 0; k--)
          term_matrices.erase(term_matrices.begin() + k);
      } else {

        // FIXME This is not good, better to check isIdentity
        if (term_matrices[op.supports[0]].get_size() < m.get_size())
          term_matrices[op.supports[0]] = m;
        else
          term_matrices[op.supports[0]] *= m;
      }
    }

    // Compute tensor product and add to result
    auto term_matrix = tensor_product(term_matrices);
    term_matrix = coeff(p) * term_matrix;

    if (term_idx == 0)
      result = term_matrix;
    else
      result += term_matrix;
  }

  return result;
}

std::set<std::size_t>
matrix_handler::get_supports(const matrix_op_data &data) const {
  std::set<std::size_t> supports;
  for (const auto &term : data.productTerms)
    // Extract indices from operator-generator pairs
    for (const auto &[idxs, generator] : term)
      for (auto idx : idxs)
        supports.insert(idx);

  return supports;
}

std::size_t matrix_handler::num_qubits(const matrix_op_data &data) const {
  if (data.productTerms.empty())
    return 0;

  std::size_t max_qubit = 0;
  for (const auto &term : data.productTerms)
    for (const auto &op : term)
      max_qubit = std::max(
          max_qubit, *std::max_element(op.supports.begin(), op.supports.end()));
  return max_qubit + 1;
}

void matrix_handler::add_assign(matrix_op_data &lhs,
                                const matrix_op_data &rhs) {
  lhs.productTerms.insert(lhs.productTerms.end(), rhs.productTerms.begin(),
                          rhs.productTerms.end());
  lhs.coefficients.insert(lhs.coefficients.end(), rhs.coefficients.begin(),
                          rhs.coefficients.end());
}

void matrix_handler::mult_assign(matrix_op_data &lhs,
                                 const matrix_op_data &rhs) {

  // Initialize result data
  matrix_op_data result = initialize();

  // For each term in lhs
  for (std::size_t i = 0; i < lhs.productTerms.size(); i++) {
    const auto &lhs_term = lhs.productTerms[i];
    const auto &lhs_coeff = lhs.coefficients[i];

    // For each term in rhs
    for (std::size_t j = 0; j < rhs.productTerms.size(); j++) {
      const auto &rhs_term = rhs.productTerms[j];
      const auto &rhs_coeff = rhs.coefficients[j];

      // Create new product term by concatenating operators
      matrix_op_term_t new_term;

      // First add all operators from lhs term
      new_term.insert(new_term.end(), lhs_term.begin(), lhs_term.end());

      // Then add all operators from rhs term
      new_term.insert(new_term.end(), rhs_term.begin(), rhs_term.end());

      // Add the new term with product of coefficients
      result.productTerms.push_back(new_term);
      result.coefficients.push_back(lhs_coeff * rhs_coeff);
    }
  }

  // Assign result back to lhs
  lhs = result;
}

bool matrix_handler::is_template(const matrix_op_data &op,
                                 const dimensions_map &d) const {
  for (auto &term : op.productTerms) {
    for (auto &op : term) {
      try {
        // This will throw if the generator requires parameters,
        // thus this matrix op is a template
        op.generator(d, {});
      } catch (const std::out_of_range &e) {
        return true;
      }
    }
  }

  return false;
}

std::vector<operator_matrix>
matrix_handler::get_support_matrices(const matrix_op_data &m_data,
                                     const dimensions_map &dimensions) const {

  if (m_data.productTerms.empty()) {
    return {};
  }

  auto eye = [](std::size_t dim) {
    std::vector<std::complex<double>> d(dim * dim, 0.0);
    operator_matrix ret(d, {dim, dim});
    for (std::size_t jj = 0; jj < ret.get_rows(); jj++)
      ret[{jj, jj}] = 1.;
    return ret;
  };

  // Get the total number of qubits in the system
  std::size_t max_idx = 0;
  for (const auto &term : m_data.productTerms)
    for (const auto &op : term)
      max_idx = std::max(
          max_idx, *std::max_element(op.supports.begin(), op.supports.end()));

  const std::size_t n_qubits = max_idx + 1;
  // Initialize return vector with identity matrices
  std::vector<operator_matrix> support_matrices(n_qubits);
  for (std::size_t i = 0; i < n_qubits; i++) {
    const std::size_t dim = dimensions.count(i) ? dimensions.at(i) : 2;
    support_matrices[i] = eye(dim);
  }

  // Replace identities with actual operators where they exist
  for (const auto &op : m_data.productTerms[0])
    support_matrices[op.supports[0]] = op.generator(dimensions, {});

  return support_matrices;
}

bool matrix_handler::check_equality(const matrix_op_data &thisPtr,
                                    const matrix_op_data &v) const {

  return true;
}

std::string matrix_handler::to_string(const matrix_op_data &data,
                                      bool printCoeffs) const {
  if (data.productTerms.empty()) {
    return "0";
  }

  std::stringstream ss;
  for (std::size_t i = 0; i < data.productTerms.size(); i++) {

    // Handle coefficient
    if (printCoeffs) {
      if (data.coefficients[i].has_value()) {
        auto coeff = data.coefficients[i].constant_value();
        ss << fmt::format("[{}{}{}j]", coeff.real(),
                          coeff.imag() < 0.0 ? "-" : "+",
                          std::fabs(coeff.imag()))
           << " ";
      } else
        ss << "f(params...) ";
    }

    // Print operator
    ss << "M[";
    std::set<std::size_t> qbits;
    for (auto &op : data.productTerms[i])
      for (auto q : op.supports)
        qbits.insert(q);
    for (std::size_t counter = 0; auto &q : qbits)
      ss << q << (counter++ >= qbits.size() - 1 ? "" : ",");
    ss << "]\n";
  }

  return ss.str();
}

operator_matrix exponentiateMatrix(const operator_matrix &mat) {
  // Convert to Eigen matrix for exponentiation
  Eigen::MatrixXcd eigenMat(mat.rows(), mat.cols());
  for (std::size_t i = 0; i < mat.rows(); i++)
    for (std::size_t j = 0; j < mat.cols(); j++)
      eigenMat(i, j) = mat[{i, j}];

  // Compute matrix exponential
  Eigen::MatrixXcd expMat = eigenMat.exp();

  // Convert back to operator_matrix
  std::vector<std::complex<double>> data;
  data.reserve(expMat.rows() * expMat.cols());
  for (std::size_t i = 0; i < expMat.rows(); i++)
    for (std::size_t j = 0; j < expMat.cols(); j++)
      data.push_back(expMat(i, j));

  return operator_matrix(data, {expMat.rows(), expMat.cols()});
}

matrix_op from_matrix(const operator_matrix &m, const std::size_t quditIdx) {
  auto generator = [=](const dimensions_map &, const parameter_map &) {
    return m;
  };
  auto term = elementary_matrix_op{{quditIdx}, generator};
  matrix_op_data data;
  data.productTerms.push_back({term});
  data.coefficients.push_back(1.0);
  return matrix_op(std::move(data));
}

matrix_op from_matrix(const operator_matrix_generator &generator,
                      const std::size_t quditIdx) {
  auto term = elementary_matrix_op{{quditIdx}, generator};
  matrix_op_data data;
  data.productTerms.push_back({term});
  data.coefficients.push_back(1.0);
  return matrix_op(std::move(data));
}

matrix_op from_matrix(const operator_matrix_generator &generator,
                      const std::vector<std::size_t> &quditIdxs) {
  auto term = elementary_matrix_op{quditIdxs, generator};
  matrix_op_data data;
  data.productTerms.push_back({term});
  data.coefficients.push_back(1.0);
  return matrix_op(std::move(data));
}

} // namespace cudaq::experimental
