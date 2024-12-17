/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "particle_handler.h"
#include "common/FmtCore.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace cudaq::experimental::particle {

details::operator_data particle_handler::initialize() const {
  return details::operator_data{};
}

std::size_t
particle_handler::num_qubits(const details::operator_data &data) const {
  std::size_t max_qubit = 0;
  for (const auto &term : data.productTerms) {
    for (std::size_t i = 0; i < term.size(); i += 2) {
      max_qubit = std::max(max_qubit, term[i]);
    }
  }
  return max_qubit + 1;
}

void normal_order_term(std::vector<std::size_t> &term,
                       details::scalar_parameter &coeff) {
  std::vector<std::pair<std::size_t, std::size_t>> ops;
  for (std::size_t i = 0; i < term.size(); i += 2) {
    ops.emplace_back(term[i], term[i + 1]);
  }

  double sign = 1;
  for (std::size_t i = 0; i < ops.size(); i++) {
    for (std::size_t j = i + 1; j < ops.size(); j++) {
      if ((ops[i].second < ops[j].second) ||
          (ops[i].second == ops[j].second && ops[i].first < ops[j].first)) {
        std::swap(ops[i], ops[j]);
        sign *= -1; // particle anticommutation
      }
    }
  }

  // Apply sign change to coefficient
  coeff = coeff * sign;

  // Rebuild term in normal order
  for (std::size_t i = 0; i < ops.size(); i++) {
    term[2 * i] = ops[i].first;
    term[2 * i + 1] = ops[i].second;
  }
}

void particle_handler::add_assign(details::operator_data &thisPtr,
                                  const details::operator_data &v) {
  thisPtr.productTerms.insert(thisPtr.productTerms.end(),
                              v.productTerms.begin(), v.productTerms.end());
  thisPtr.coefficients.insert(thisPtr.coefficients.end(),
                              v.coefficients.begin(), v.coefficients.end());
}

void particle_handler::mult_assign(details::operator_data &thisPtr,
                                   const details::operator_data &v) {
  details::operator_data result;

  for (std::size_t i = 0; i < thisPtr.productTerms.size(); i++) {
    for (std::size_t j = 0; j < v.productTerms.size(); j++) {
      std::vector<std::size_t> new_term;
      new_term.insert(new_term.end(), thisPtr.productTerms[i].begin(),
                      thisPtr.productTerms[i].end());
      new_term.insert(new_term.end(), v.productTerms[j].begin(),
                      v.productTerms[j].end());

      auto coeff = thisPtr.coefficients[i] * v.coefficients[j];
      normal_order_term(new_term, coeff);

      result.productTerms.push_back(new_term);
      result.coefficients.push_back(coeff);
    }
  }

  thisPtr = std::move(result);
}

bool particle_handler::check_equality(const details::operator_data &thisPtr,
                                      const details::operator_data &v) const {
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
particle_handler::get_support_matrices(const details::operator_data &data,
                                       const parameter_map &p) const {
  std::vector<operator_matrix> support_matrices;
  auto &term = data.productTerms[0];
  for (std::size_t i = 0; i < term.size(); i += 2) {
    if (term[i + 1])
      support_matrices.push_back(create_matrix(i));
    else
      support_matrices.push_back(annihilate_matrix(i));
  }

  return support_matrices;
}

std::string particle_handler::to_string(const details::operator_data &thisPtr,
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

operator_matrix particle_handler::to_matrix(const details::operator_data &data,
                                            const parameter_map &p) const {
  // Get number of particleic modes
  throw std::runtime_error("particle to_matrix not implemented.");
  return {};
}

particle_op create(std::size_t idx) {
  details::operator_data data;
  data.productTerms.push_back({idx, 1});
  data.coefficients.push_back(1.0);
  return particle_op(std::move(data));
}

particle_op annihilate(std::size_t idx) {
  details::operator_data data;
  data.productTerms.push_back({idx, 0});
  data.coefficients.push_back(1.0);
  return particle_op(std::move(data));
}

particle_op position(std::size_t idx) {
  return std::complex<double>{0.5, 0.} * (create(idx) + annihilate(idx));
}

particle_op momentum(std::size_t idx) {
  return std::complex<double>{0.0, 0.5} * (create(idx) - annihilate(idx));
}

} // namespace cudaq::experimental::particle
