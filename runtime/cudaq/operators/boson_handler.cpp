/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "boson_handler.h"
#include "matrix_handler.h"

#include "common/FmtCore.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cudaq::experimental::boson {

void normal_order_term(std::vector<std::size_t> &term,
                       details::scalar_parameter &coeff, boson_data &data) {
  if (term.size() <= 2)
    return;

  std::vector<std::pair<std::size_t, std::size_t>> ops;
  for (std::size_t i = 0; i < term.size(); i += 2) {
    ops.emplace_back(term[i], term[i + 1]);
  }

  for (std::size_t i = 0; i < ops.size() - 1; i++) {
    for (std::size_t j = i + 1; j < ops.size(); j++) {
      auto &left = ops[i];
      auto &right = ops[j];

      // Handle annihilation-creation pair on same mode
      if (left.first == right.first && left.second == 0 && right.second == 1) {
        // For bosons: a a† = 1 + a†a
        data.productTerms.push_back({}); // Add identity term
        data.coefficients.push_back(coeff);
        std::swap(left, right); // Normal order the a†a term
      }
      // Normal ordering swap if needed
      else if ((left.second < right.second) ||
               (left.second == right.second && left.first > right.first)) {
        std::swap(left, right);
      }
    }
  }

  // Rebuild term in normal order
  term.clear();
  for (const auto &op : ops) {
    term.push_back(op.first);
    term.push_back(op.second);
  }
}

boson_data boson_handler::initialize() const {
  return details::operator_data<term_t>{};
}

std::size_t boson_handler::num_qubits(const boson_data &data) const {
  std::size_t max_qubit = 0;
  for (const auto &term : data.productTerms) {
    for (std::size_t i = 0; i < term.size(); i += 2) {
      max_qubit = std::max(max_qubit, term[i]);
    }
  }
  return max_qubit + 1;
}

void boson_handler::add_assign(boson_data &thisPtr, const boson_data &v) {

  for (std::size_t i = 0; i < v.productTerms.size(); i++) {
    bool found = false;
    // Search for matching term in existing terms
    for (std::size_t j = 0; j < thisPtr.productTerms.size(); j++) {
      if (thisPtr.productTerms[j] == v.productTerms[i]) {
        // Add coefficients for matching terms
        thisPtr.coefficients[j] = thisPtr.coefficients[j] + v.coefficients[i];
        found = true;
        break;
      }
    }

    // If term not found, append it
    if (!found) {
      thisPtr.productTerms.push_back(v.productTerms[i]);
      thisPtr.coefficients.push_back(v.coefficients[i]);
    }
  }

  // Remove terms with zero coefficients
  for (std::size_t i = 0; i < thisPtr.coefficients.size();) {
    if (thisPtr.coefficients[i].has_value() &&
        std::abs(thisPtr.coefficients[i].constant_value()) < 1e-12) {
      thisPtr.coefficients.erase(thisPtr.coefficients.begin() + i);
      thisPtr.productTerms.erase(thisPtr.productTerms.begin() + i);
    } else {
      i++;
    }
  }
}

void boson_handler::mult_assign(boson_data &thisPtr, const boson_data &v) {
  details::operator_data<term_t> result;

  for (std::size_t i = 0; i < thisPtr.productTerms.size(); i++) {
    for (std::size_t j = 0; j < v.productTerms.size(); j++) {
      std::vector<std::size_t> new_term;
      new_term.insert(new_term.end(), thisPtr.productTerms[i].begin(),
                      thisPtr.productTerms[i].end());
      new_term.insert(new_term.end(), v.productTerms[j].begin(),
                      v.productTerms[j].end());

      auto coeff = thisPtr.coefficients[i] * v.coefficients[j];
      normal_order_term(new_term, coeff, result);

      if (!new_term.empty()) {
        // Check if this term already exists
        bool found = false;
        for (size_t k = 0; k < result.productTerms.size(); k++) {
          if (result.productTerms[k] == new_term) {
            result.coefficients[k] = result.coefficients[k] + coeff;
            found = true;
            break;
          }
        }
        if (!found) {
          result.productTerms.push_back(new_term);
          result.coefficients.push_back(coeff);
        }
      }
    }
  }

  thisPtr = std::move(result);
}

bool boson_handler::check_equality(const boson_data &thisPtr,
                                   const boson_data &v) const {
  if (thisPtr.productTerms.size() != v.productTerms.size())
    return false;

  for (std::size_t i = 0; i < thisPtr.productTerms.size(); i++) {
    bool found = false;
    for (std::size_t j = 0; j < v.productTerms.size(); j++) {
      if (thisPtr.productTerms[i] == v.productTerms[j] &&
          std::abs(thisPtr.coefficients[i].constant_value() -
                   v.coefficients[j].constant_value()) < 1e-12) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

operator_matrix create_matrix(std::size_t dimension) {
  std::vector<std::complex<double>> data(dimension * dimension, 0.0);
  for (std::size_t i = 1; i < dimension; ++i) {
    data[i * dimension + (i - 1)] = std::sqrt(static_cast<double>(i));
  }
  return operator_matrix(data, {dimension, dimension});
}

operator_matrix annihilate_matrix(std::size_t dimension) {
  std::vector<std::complex<double>> data(dimension * dimension, 0.0);
  for (std::size_t i = 0; i < dimension - 1; ++i) {
    data[i * dimension + (i + 1)] = std::sqrt(static_cast<double>(i + 1));
  }
  return operator_matrix(data, {dimension, dimension});
}

std::vector<operator_matrix>
boson_handler::get_support_matrices(const boson_data &data,
                                    const dimensions_map &dimensions) const {
  std::vector<operator_matrix> support_matrices;
  auto &term = data.productTerms[0];
  for (std::size_t i = 0; i < term.size(); i += 2) {
    std::size_t dim = -1;
    try {
      dim = dimensions.at(term[i]);
    } catch (std::out_of_range &e) {
      throw std::runtime_error(
          "requested dimension not provided in dimensions map (" +
          std::to_string(term[i]) + ")");
    }

    operator_matrix m;
    if (term[i + 1])
      m = create_matrix(dim);
    else
      m = annihilate_matrix(dim);

    support_matrices.push_back(m);
  }

  // absorb the coefficient into the first matrix
  support_matrices.front() =
      data.coefficients[0].constant_value() * support_matrices.front();
  return support_matrices;
}

std::set<std::size_t>
boson_handler::get_supports(const boson_data &data) const {
  std::set<std::size_t> supports;
  for (const auto &term : data.productTerms)
    // Each pair of values represents (index, create/annihilate)
    for (std::size_t i = 0; i < term.size(); i += 2) {
      supports.insert(term[i]);
    }

  return supports;
}

operator_matrix kronecker_product(const operator_matrix &A,
                                  const operator_matrix &B) {
  const auto A_rows = A.rows();
  const auto A_cols = A.cols();
  const auto B_rows = B.rows();
  const auto B_cols = B.cols();

  // Result matrix dimensions
  const auto C_rows = A_rows * B_rows;
  const auto C_cols = A_cols * B_cols;

  // Initialize result matrix
  std::vector<std::complex<double>> data(C_rows * C_cols, 0.0);

  // Compute Kronecker product
  for (std::size_t i = 0; i < A_rows; i++) {
    for (std::size_t j = 0; j < A_cols; j++) {
      for (std::size_t k = 0; k < B_rows; k++) {
        for (std::size_t l = 0; l < B_cols; l++) {
          const auto row = i * B_rows + k;
          const auto col = j * B_cols + l;
          data[row * C_cols + col] = A[{i, j}] * B[{k, l}];
        }
      }
    }
  }

  return operator_matrix(data, {C_rows, C_cols});
}

operator_matrix
boson_handler::to_matrix(const boson_data &data, const parameter_map &params,
                         const dimensions_map &dimensions) const {

  if (data.productTerms.empty()) {
    return operator_matrix();
  }

  std::size_t n_sites = 0;
  for (const auto &[site, dim] : dimensions) {
    n_sites = std::max(n_sites, site + 1);
  }

  operator_matrix result;
  bool first_term = true;

  for (std::size_t term_idx = 0; term_idx < data.productTerms.size();
       term_idx++) {
    const auto &term = data.productTerms[term_idx];
    std::vector<operator_matrix> site_matrices(n_sites);

    // Initialize all sites with identity matrices first
    for (std::size_t site = 0; site < n_sites; site++) {
      std::size_t dim = dimensions.count(site) ? dimensions.at(site) : 2;
      std::vector<std::complex<double>> id_data(dim * dim, 0.0);
      for (std::size_t i = 0; i < dim; i++) {
        id_data[i * dim + i] = 1.0;
      }
      site_matrices[site] = operator_matrix(id_data, {dim, dim});
    }

    // Process operators and insert Z matrices for Jordan-Wigner string
    // Need to track parity for multi-hops.
    for (std::size_t i = 0; i < term.size(); i += 2) {
      std::size_t site = term[i];
      bool is_creation = term[i + 1];
      std::size_t dim = -1;
      try {
        dim = dimensions.at(term[i]);
      } catch (std::out_of_range &e) {
        throw std::runtime_error(
            "requested dimension not provided in dimensions map (" +
            std::to_string(term[i]) + ")");
      }

      // Insert Z matrices for all sites between the previous operator and this
      // one
      if (i > 0) {
        std::size_t prev_site = term[i - 2];
        for (std::size_t j = prev_site + 1; j < site; j++) {
          // Create Z matrix
          std::vector<std::complex<double>> z_data(4, 0.0);
          z_data[0] = 1.0;
          z_data[3] = -1.0;
          site_matrices[j] = operator_matrix(z_data, {2, 2});
        }
      }

      operator_matrix op_matrix =
          is_creation ? create_matrix(dim) : annihilate_matrix(dim);
      site_matrices[site] = site_matrices[site] * op_matrix;
    }

    // Build full operator matrix using kronecker products
    operator_matrix term_matrix = site_matrices[0];
    for (std::size_t i = 1; i < n_sites; i++) {
      term_matrix = kronecker_product(term_matrix, site_matrices[i]);
    }

    // Apply coefficient
    term_matrix = data.coefficients[term_idx](params) * term_matrix;

    if (first_term) {
      result = term_matrix;
      first_term = false;
    } else {
      std::vector<std::complex<double>> sum_data(result.get_size());
      for (std::size_t i = 0; i < sum_data.size(); i++) {
        sum_data[i] = result.get_data()[i] + term_matrix.get_data()[i];
      }
      result =
          operator_matrix(sum_data, {result.get_rows(), result.get_columns()});
    }
  }

  return result;
}

std::string boson_handler::to_string(const boson_data &thisPtr,
                                     bool printCoeffs) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < thisPtr.productTerms.size(); i++) {
    auto coeff = thisPtr.coefficients[i];

    if (i > 0)
      ss << " + ";
    if (printCoeffs)
      if (coeff.has_value()) {
        ss << fmt::format("[{}{}{}j]", coeff.constant_value().real(),
                          coeff.constant_value().imag() < 0.0 ? "-" : "+",
                          std::fabs(coeff.constant_value().imag()))
           << " ";
      } else {
        ss << "f(params...) ";
      }

    for (std::size_t j = 0; j < thisPtr.productTerms[i].size(); j += 2) {
      ss << thisPtr.productTerms[i][j]
         << (thisPtr.productTerms[i][j + 1] ? "^" : "") << " ";
    }
    ss << "\n";
  }

  return ss.str();
}

boson_op create(std::size_t idx) {
  boson_data data;
  data.productTerms.push_back({idx, 1});
  data.coefficients.push_back(1.0);
  return boson_op(std::move(data));
}

boson_op annihilate(std::size_t idx) {
  boson_data data;
  data.productTerms.push_back({idx, 0});
  data.coefficients.push_back(1.0);
  return boson_op(std::move(data));
}

matrix_op squeeze(std::size_t idx) {
  // Example with of a bosonic op whose matrix requires a unary op application
  auto squeezeFunctor = [](const parameter_map &m) {
    return m.at("squeezing");
  };
  auto squeezeFunctorConj = [](const parameter_map &m) {
    return std::conj(m.at("squeezing"));
  };
  auto diff = 0.5 * (squeezeFunctorConj * annihilate(idx) * annihilate(idx) -
                     squeezeFunctor * create(idx) * create(idx));
  return exp(diff);
}

matrix_op displace(std::size_t idx) {
  // Example with of a bosonic op whose matrix requires a unary op application
  auto displaceFunctor = [](const parameter_map &m) {
    return m.at("displacement");
  };
  auto displaceFunctorConj = [](const parameter_map &m) {
    return std::conj(m.at("displacement"));
  };
  auto diff =
      displaceFunctor * create(idx) - displaceFunctorConj * annihilate(idx);
  return exp(diff);
}

} // namespace cudaq::experimental::boson
