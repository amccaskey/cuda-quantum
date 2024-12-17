/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/FmtCore.h"
#include "particle_handler.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cudaq::experimental::matrix {

details::operator_data matrix_handler::initialize() const {
  details::operator_data data;
  data.matrixProductTerms =
      std::make_optional<details::operator_data::matrix_product_sum>();
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

operator_matrix matrix_handler::to_matrix(const details::operator_data &data,
                                          const parameter_map &params) const {
  const std::size_t n_qubits = num_qubits(data);
  const std::size_t dim = 1 << n_qubits;

  if (!data.matrixProductTerms || data.matrixProductTerms->empty())
    return operator_matrix(std::vector<std::complex<double>>(dim * dim, 0.0),
                           {dim, dim});

  // Initialize result matrix
  operator_matrix result(std::vector<std::complex<double>>(dim * dim, 0.0),
                         {dim, dim});

  // 2x2 Identity matrix
  const operator_matrix I({1.0, 0.0, 0.0, 1.0}, {2, 2});

  for (std::size_t termIdx = 0; termIdx < data.matrixProductTerms->size();
       termIdx++) {
    const auto &term = (*data.matrixProductTerms)[termIdx];
    const auto &indices = data.productTerms[termIdx];

    // Build up the full operator matrix for this term
    std::vector<operator_matrix> term_matrices(n_qubits, I);

    // Place the actual operators at their specified indices
    for (std::size_t i = 0; i < indices.size(); i++) {
      auto idx = indices[i];
      term_matrices[idx] = term[i](params);
    }

    // Compute the tensor product of all matrices
    auto term_result = tensor_product(term_matrices);

    // Add to result with appropriate coefficient
    result += data.coefficients[termIdx](params) * term_result;
  }

  return result;
}

std::size_t
matrix_handler::num_qubits(const details::operator_data &data) const {
  if (data.productTerms.empty())
    return 0;

  std::size_t max_qubit = 0;
  for (const auto &term : data.productTerms)
    for (auto idx : term)
      max_qubit = std::max(max_qubit, idx);
  return max_qubit + 1;
}

void matrix_handler::add_assign(details::operator_data &lhs,
                                const details::operator_data &rhs) {
  if (!lhs.matrixProductTerms)
    lhs.matrixProductTerms =
        std::make_optional<details::operator_data::matrix_product_sum>();

  lhs.productTerms.insert(lhs.productTerms.end(), rhs.productTerms.begin(),
                          rhs.productTerms.end());
  lhs.coefficients.insert(lhs.coefficients.end(), rhs.coefficients.begin(),
                          rhs.coefficients.end());

  if (rhs.matrixProductTerms) {
    lhs.matrixProductTerms->insert(lhs.matrixProductTerms->end(),
                                   rhs.matrixProductTerms->begin(),
                                   rhs.matrixProductTerms->end());
  }
}
void matrix_handler::mult_assign(details::operator_data &lhs,
                                 const details::operator_data &rhs) {
  if (!lhs.matrixProductTerms || !rhs.matrixProductTerms)
    throw std::runtime_error("Matrix product terms not initialized");

  details::operator_data result = initialize();

  for (std::size_t i = 0; i < lhs.productTerms.size(); i++) {
    for (std::size_t j = 0; j < rhs.productTerms.size(); j++) {
      // Create new term with unique indices
      auto new_term = lhs.productTerms[i];
      auto lhs_gens = (*lhs.matrixProductTerms)[i];
      auto rhs_gens = (*rhs.matrixProductTerms)[j];

      // Track index mappings between lhs and rhs
      std::map<std::size_t, std::size_t> overlap_indices;

      // Find overlapping indices
      for (std::size_t rhs_idx = 0; rhs_idx < rhs.productTerms[j].size();
           rhs_idx++) {
        auto idx = rhs.productTerms[j][rhs_idx];
        auto it = std::find(new_term.begin(), new_term.end(), idx);

        if (it == new_term.end()) {
          new_term.push_back(idx);
        } else {
          // Store mapping between overlapping indices
          overlap_indices[std::distance(new_term.begin(), it)] = rhs_idx;
        }
      }

      // Create new matrix generators
      details::operator_data::matrix_product_term new_matrix_term;
      for (std::size_t k = 0; k < new_term.size(); k++) {
        auto overlap_it = overlap_indices.find(k);
        if (overlap_it != overlap_indices.end()) {
          // Create new generator that multiplies matrices for overlapping
          // indices
          auto lhs_gen = lhs_gens[k];
          auto rhs_gen = rhs_gens[overlap_it->second];
          new_matrix_term.push_back(
              [lhs_gen, rhs_gen](const parameter_map &params) {
                return lhs_gen(params) * rhs_gen(params);
              });
        } else if (k < lhs_gens.size()) {
          new_matrix_term.push_back(lhs_gens[k]);
        } else {
          new_matrix_term.push_back(rhs_gens[k - lhs_gens.size()]);
        }
      }

      result.productTerms.push_back(new_term);
      result.coefficients.push_back(lhs.coefficients[i] * rhs.coefficients[j]);
      result.matrixProductTerms->push_back(new_matrix_term);
    }
  }

  printf("After Mult: %lu \n", result.matrixProductTerms.value()[0].size());
  lhs = result;
}

std::vector<operator_matrix>
matrix_handler::get_support_matrices(const details::operator_data &m_data,
                                     const parameter_map &p) const {
  std::vector<operator_matrix> ret;
  for (auto &mgen : m_data.matrixProductTerms.value()[0])
    ret.push_back(mgen(p));

  return ret;
}

bool matrix_handler::check_equality(const details::operator_data &thisPtr,
                                    const details::operator_data &v) const {

  return true;
}

std::string matrix_handler::to_string(const details::operator_data &data,
                                      bool printCoeffs) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < data.productTerms.size(); i++) {
    if (printCoeffs) {
      if (data.coefficients[i].has_value())
        ss << data.coefficients[i].constant_value() << " * ";
      else
        ss << "[dynamic] * ";
    }
    ss << "Matrix[";
    for (auto idx : data.productTerms[i])
      ss << idx << ",";
    ss << "]\n";
  }
  return ss.str();
}

} // namespace cudaq::experimental::matrix

namespace cudaq::experimental {

matrix_op from_matrix(const operator_matrix &m,
                      const std::vector<std::size_t> &supports) {
  details::operator_data data{{supports},
                              {1.0},
                              details::operator_data::matrix_product_sum{
                                  {[=](const parameter_map &) { return m; }}}};
  return matrix_op(std::move(data));
}

} // namespace cudaq::experimental
