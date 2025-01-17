/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "operators/operator_data.h"

namespace cudaq::experimental {

/// @brief A templated class representing a quantum operator with a specific
/// handler type
/// @tparam HandlerTy The handler type that must inherit from operator_handler
template <typename HandlerTy>
class operator_ {

protected:
  /// @brief The handler instance for this operator
  HandlerTy handler;

  /// @brief The underlying operator data
  details::operator_data<typename HandlerTy::term_t> m_data;

public:
  /// @brief Default constructor that initializes operator data
  operator_() { m_data = handler.initialize(); }

  /// @brief Copy constructor
  operator_(const operator_ &op) = default;

  /// @brief Constructor from operator data
  /// @param data The operator data to initialize with
  operator_(const details::operator_data<typename HandlerTy::term_t> &data)
      : m_data(data) {}

  /// @brief Get mutable reference to operator data
  details::operator_data<typename HandlerTy::term_t> &data() { return m_data; }

  /// @brief Get const reference to operator data
  const details::operator_data<typename HandlerTy::term_t> &data() const {
    return m_data;
  }

  /// @brief Check if operator has no terms
  bool empty() const { return m_data.productTerms.empty(); }

  std::set<std::size_t> get_supports() const {
    return handler.get_supports(m_data);
  }

  /// @brief Evaluate operator with given parameters
  /// @param params Map of parameter names to values
  operator_<HandlerTy> operator()(const parameter_map &params) const {
    details::operator_data newData = m_data;
    for (std::size_t i = 0; i < newData.coefficients.size(); i++)
      newData.coefficients[i] = newData.coefficients[i](params);
    return operator_(newData);
  }

  /// @brief Convert operator to matrix representation
  /// @param parameters Parameter map for evaluation
  operator_matrix to_matrix(const parameter_map parameters = {}) const {
    auto concretized = operator()(parameters);
    return handler.to_matrix(concretized.m_data, parameters, {});
  }

  operator_matrix to_matrix(const dimensions_map &dimensions,
                            const parameter_map parameters = {}) const {
    auto concretized = operator()(parameters);
    return handler.to_matrix(concretized.m_data, parameters, dimensions);
  }

  /// @brief Return the matrices for each constituent operator in this product
  /// term. Throws for an operator that is a sum of product terms. Use
  /// for_each_term to get each product term's supporting operator matrices.
  std::vector<operator_matrix>
  get_elementary_operators(const dimensions_map dimensions = {}) const {
    if (is_template(dimensions))
      throw std::runtime_error("operator is a template, concretize before "
                               "calling get_elementary_operators");

    if (m_data.productTerms.size() > 1)
      throw std::runtime_error(
          "Can only retrieve support matrices for single-term operators. U");

    return handler.get_support_matrices(m_data, dimensions);
  }

  /// @brief Get number of qubits this operator acts on
  std::size_t num_qubits() const {
    if (empty())
      return 0;
    return handler.num_qubits(m_data);
  }

  /// @brief Get number of terms in this operator
  std::size_t num_terms() const { return m_data.productTerms.size(); }

  /// @brief Print operator to stdout
  void dump() const { printf("%s", handler.to_string(m_data, true).c_str()); }

  /// @brief Convert operator to string representation
  /// @param print Whether to print the string
  std::string to_string(bool print = true) const {
    return handler.to_string(m_data, print);
  }

  /// @brief Check if any terms in this operator have dynamic (parameterized)
  /// coefficients.
  bool is_template(const dimensions_map m = {}) const {
    for (auto &term_coeff : m_data.coefficients)
      if (!term_coeff.has_value())
        return true;

    if (handler.is_template(m_data, m))
      return true;

    return false;
  }

  /// @brief Apply function to each term in operator
  /// @param applicator Function to apply to each term
  void for_each_term(
      const std::function<void(operator_<HandlerTy> &)> &applicator) const {
    std::size_t i = 0;
    for (auto &term : m_data.productTerms) {
      operator_<HandlerTy> tmp({{term}, {m_data.coefficients[i++]}});
      applicator(tmp);
    }
  }

  /// @brief Get vector<double> representation of operator
  std::vector<double> get_data_representation() const {
    if (is_template())
      throw std::runtime_error("cannot map a template operator to a serialized "
                               "data representation.");
    return handler.get_data_representation(m_data);
  }

  /// @brief Distribute operator terms across chunks
  /// @param numChunks Number of chunks to distribute terms across
  std::vector<operator_<HandlerTy>> distribute_terms(std::size_t numChunks) {
    auto nTermsPerChunk = num_terms() / numChunks;
    auto leftover = num_terms() % numChunks;
    std::vector<operator_<HandlerTy>> chunks;
    std::size_t currentPos = 0;

    for (std::size_t chunkIx = 0; chunkIx < numChunks; chunkIx++) {
      auto count = nTermsPerChunk + (chunkIx < leftover ? 1 : 0);
      details::operator_data<typename HandlerTy::term_t> chunkData;
      for (std::size_t i = 0; i < count; i++) {
        chunkData.productTerms.push_back(m_data.productTerms[currentPos + i]);
        chunkData.coefficients.push_back(m_data.coefficients[currentPos + i]);
      }
      chunks.emplace_back(chunkData);
      currentPos += count;
    }
    return chunks;
  }

  /// @brief Get coefficient of single-term operator
  std::complex<double> get_coefficient() const {
    if (num_terms() == 0)
      return 0.0;
    if (num_terms() != 1 || !m_data.coefficients[0].has_value())
      throw std::runtime_error(
          "get_coefficient only supported for operators with 1 term.");
    return m_data.coefficients[0].constant_value();
  }

  // Operator overloads for arithmetic operations
  operator_<HandlerTy> &operator=(const operator_<HandlerTy> &other) {
    handler.assign(m_data, other.m_data);
    return *this;
  }

  operator_<HandlerTy> &operator+=(const operator_<HandlerTy> &other) {
    handler.add_assign(m_data, other.m_data);
    return *this;
  }

  operator_<HandlerTy> &operator+=(const details::scalar_parameter &scalar) {
    auto other = handler.initialize();
    // Create the constant op
    other.productTerms.emplace_back();
    other.coefficients.push_back(scalar);
    // Add it
    handler.add_assign(m_data, other);
    return *this;
  }

  operator_<HandlerTy> &operator-=(const operator_<HandlerTy> &other) {
    auto negated = other * -1.;
    handler.add_assign(m_data, negated.m_data);
    return *this;
  }

  operator_<HandlerTy> &operator-=(const details::scalar_parameter &other) {
    details::scalar_parameter p(-1);
    p = p * other;

    // If this is dynamic, just add it
    if (!other.has_value()) {
      m_data.productTerms.emplace_back();
      m_data.coefficients.push_back(p);
      return *this;
    }

    // If constant, see if we have other constants
    std::size_t i = 0;
    for (auto &term : m_data.productTerms) {
      if (term.empty() && m_data.coefficients[i].has_value()) {
        m_data.coefficients[i] = m_data.coefficients[i] + p;
        return *this;
      }
      i++;
    }

    // new constant, just add it
    m_data.productTerms.emplace_back();
    m_data.coefficients.push_back(p);
    return *this;
  }

  operator_<HandlerTy> &operator*=(const operator_<HandlerTy> &other) {
    handler.mult_assign(m_data, other.m_data);
    return *this;
  }

  bool operator==(const operator_<HandlerTy> &other) const noexcept {
    return handler.check_equality(m_data, other.m_data);
  }

  operator_<HandlerTy> &operator*=(const details::scalar_parameter &other) {
    handler.mult_scalar_assign(m_data, other);
    return *this;
  }

  operator_<HandlerTy> operator+(const operator_<HandlerTy> &other) const {
    operator_<HandlerTy> tmp = *this;
    tmp += other;
    return tmp;
  }

  operator_<HandlerTy> operator+(const details::scalar_parameter &other) const {
    operator_<HandlerTy> tmp = *this;
    tmp += other;
    return tmp;
  }

  operator_<HandlerTy> operator-(const details::scalar_parameter &other) const {
    operator_<HandlerTy> tmp;
    tmp -= other;
    return tmp;
  }

  operator_<HandlerTy> operator-(const operator_<HandlerTy> &other) const {
    operator_<HandlerTy> tmp = *this;
    tmp -= other;
    return tmp;
  }

  operator_<HandlerTy> operator*(const operator_<HandlerTy> &other) const {
    operator_<HandlerTy> tmp = *this;
    tmp *= other;
    return tmp;
  }

  operator_<HandlerTy> operator*(const details::scalar_parameter &other) const {
    operator_<HandlerTy> tmp = *this;
    tmp *= other;
    return tmp;
  }

  /// @brief Iterator class for quantum operator terms
  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = operator_<HandlerTy>;
    using pointer = value_type *;
    using reference = value_type &;

    iterator(const details::operator_data<typename HandlerTy::term_t> *data,
             std::size_t pos = 0)
        : op_data(data), current_pos(pos) {
      if (op_data && current_pos < op_data->productTerms.size()) {
        current_value.m_data.productTerms.clear();
        current_value.m_data.coefficients.clear();
        current_value.m_data.productTerms.push_back(op_data->productTerms[pos]);
        current_value.m_data.coefficients.push_back(op_data->coefficients[pos]);
      }
    }

    iterator &operator++() {
      if (op_data && current_pos < op_data->productTerms.size()) {
        ++current_pos;
        if (current_pos < op_data->productTerms.size()) {
          current_value.m_data.productTerms.clear();
          current_value.m_data.coefficients.clear();
          current_value.m_data.productTerms.push_back(
              op_data->productTerms[current_pos]);
          current_value.m_data.coefficients.push_back(
              op_data->coefficients[current_pos]);
        }
      }
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const iterator &other) const {
      return op_data == other.op_data && current_pos == other.current_pos;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }

    reference operator*() { return current_value; }
    pointer operator->() { return &current_value; }

  private:
    const details::operator_data<typename HandlerTy::term_t> *op_data;
    std::size_t current_pos;
    value_type current_value;
  };

  /// @brief Get iterator to beginning of operator terms
  iterator begin() { return iterator(&m_data, 0); }

  /// @brief Get iterator to end of operator terms
  iterator end() { return iterator(&m_data, m_data.productTerms.size()); }

  /// @brief Get const iterator to beginning of operator terms
  iterator begin() const { return iterator(&m_data, 0); }

  /// @brief Get const iterator to end of operator terms
  iterator end() const { return iterator(&m_data, m_data.productTerms.size()); }
};

// Non-member operator overloads
template <typename T>
operator_<T> operator-(const details::scalar_parameter &coeff,
                       const operator_<T> &rhs) {
  return operator_<T>() * coeff - rhs;
}

template <typename T>
operator_<T> operator-(const operator_<T> &rhs,
                       const details::scalar_parameter &coeff) {
  return rhs - operator_<T>() * coeff;
}

template <typename T>
operator_<T> operator+(const details::scalar_parameter &coeff,
                       const operator_<T> &rhs) {
  return operator_<T>() * coeff + rhs;
}

template <typename T>
operator_<T> operator+(const operator_<T> &rhs,
                       const details::scalar_parameter &coeff) {
  return rhs + operator_<T>() * coeff;
}

template <typename T>
operator_<T> operator*(const details::scalar_parameter &coeff,
                       const operator_<T> &rhs) {
  return rhs.operator*(coeff);
}

template <typename T>
operator_<T>
operator*(const details::scalar_parameter::dynamic_signature &coeff,
          const operator_<T> &rhs) {
  return rhs.operator*(coeff);
}

template <typename T>
operator_<T> operator==(const operator_<T> &lhs, const operator_<T> &rhs) {
  return lhs.operator==(rhs);
}
} // namespace cudaq::experimental

#include "cudaq/operators/boson_handler.h"
#include "cudaq/operators/fermion_handler.h"
#include "cudaq/operators/matrix_handler.h"
#include "cudaq/operators/spin_handler.h"
