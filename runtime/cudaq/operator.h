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

template <typename HandlerTy>
class quantum_operator {
protected:
  HandlerTy handler;

  details::operator_data m_data;

public:
  quantum_operator() = default;

  quantum_operator(const quantum_operator &op) = default;

  quantum_operator(const details::operator_data &data) : m_data(data) {}

  details::operator_data &data() { return m_data; }

  bool empty() const { return m_data.productTerms.empty(); }

  quantum_operator<HandlerTy> operator()(const parameter_map &params) {
    details::operator_data newData = m_data;
    for (std::size_t i = 0; i < newData.coefficients.size(); i++)
      newData.coefficients[i] = newData.coefficients[i](params);

    return quantum_operator(newData);
  }

  /// @brief Return the number of qubits this operator is on
  std::size_t num_qubits() const {
    if (empty())
      return 0;
    return handler.num_qubits(m_data);
  }

  /// @brief Return the number of terms in this operator
  std::size_t num_terms() const { return m_data.productTerms.size(); }

  void dump() const { printf("%s", handler.to_string(m_data, true).c_str()); }
  std::string to_string(bool print = true) const {
    return handler.to_string(m_data, print);
  }

  bool is_template() const {
    for (auto &term_coeff : m_data.coefficients)
      if (!term_coeff.has_value())
        return true;
    return false;
  }

  std::complex<double> get_coefficient() {
    if (num_terms() != 1 || !m_data.coefficients[0].has_value())
      throw std::runtime_error(
          "get_coefficient only supported for operators with 1 term.");

    return m_data.coefficients[0].constant_value();
  }

  /// @brief Set the provided spin_op equal to this one and return *this.
  quantum_operator<HandlerTy> &
  operator=(const quantum_operator<HandlerTy> &other) {
    handler.assign(m_data, other.m_data);
    return *this;
  }

  /// @brief Add the given spin_op to this one and return *this
  quantum_operator<HandlerTy> &
  operator+=(const quantum_operator<HandlerTy> &other) {
    handler.addAssign(m_data, other.m_data);
    return *this;
  }

  /// @brief Subtract the given spin_op from this one and return *this
  quantum_operator<HandlerTy> &
  operator-=(const quantum_operator<HandlerTy> &other) {
    auto negated = other * -1.;
    handler.addAssign(m_data, negated.m_data);
    return *this;
  }

  /// @brief Multiply the given spin_op with this one and return *this
  quantum_operator<HandlerTy> &
  operator*=(const quantum_operator<HandlerTy> &other) {
    handler.multAssign(m_data, other.m_data);
    return *this;
  }

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  bool operator==(const quantum_operator<HandlerTy> &other) const noexcept {
    return handler.checkEquality(m_data, other.m_data);
  }

  /// @brief Multiply this spin_op by the given double.
  quantum_operator<HandlerTy> &
  operator*=(const details::scalar_parameter &other) {
    handler.multScalarAssign(m_data, other);
    return *this;
  }

  quantum_operator<HandlerTy>
  operator+(const quantum_operator<HandlerTy> &other) {
    quantum_operator<HandlerTy> tmp = *this;
    tmp += other;
    return tmp;
  }

  quantum_operator<HandlerTy>
  operator-(const quantum_operator<HandlerTy> &other) {
    quantum_operator<HandlerTy> tmp = *this;
    tmp -= other;
    return tmp;
  }

  quantum_operator<HandlerTy>
  operator*(const quantum_operator<HandlerTy> &other) const {
    quantum_operator<HandlerTy> tmp = *this;
    tmp *= other;
    return tmp;
  }

  quantum_operator<HandlerTy>
  operator*(const details::scalar_parameter &other) const {
    quantum_operator<HandlerTy> tmp = *this;
    tmp *= other;
    return tmp;
  }

  // Iterator class definition
  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = quantum_operator<HandlerTy>;
    using pointer = value_type *;
    using reference = value_type &;

    iterator(details::operator_data *data, std::size_t pos = 0)
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
    details::operator_data *op_data;
    std::size_t current_pos;
    value_type current_value;
  };

  // Add iterator methods to quantum_operator
  iterator begin() { return iterator(&m_data, 0); }
  iterator end() { return iterator(&m_data, m_data.productTerms.size()); }
};

template <typename T>
quantum_operator<T>
operator*(const details::scalar_parameter::dynamic_signature &coeff,
          const quantum_operator<T> &rhs) {
  return rhs.operator*(coeff);
}

template <typename T>
quantum_operator<T> operator==(const quantum_operator<T> &lhs,
                               const quantum_operator<T> &rhs) {
  return lhs.operator==(rhs);
}

} // namespace cudaq::experimental

#include "cudaq/operators/spin_handler.h"