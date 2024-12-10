/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <complex>
#include <functional>
#include <optional>
#include <variant>
#include <vector>

namespace cudaq::experimental {
using parameter_map = std::unordered_map<std::string, std::complex<double>>;

namespace details {
class scalar_parameter {
public:
  using dynamic_signature =
      std::function<std::complex<double>(const parameter_map &)>;
  using type = std::variant<std::complex<double>, dynamic_signature>;

  type value;
  std::complex<double> operator()(const parameter_map &parameters) {
    if (value.index() == 0)
      return constant_value();

    return std::get<1>(value)(parameters);
  }

  scalar_parameter(double value) : value(value) {}
  scalar_parameter(std::complex<double> value) : value(value) {}
  scalar_parameter(const dynamic_signature &functor) : value(functor) {}
  scalar_parameter(const scalar_parameter &) = default;

  bool has_value() const { return value.index() == 0; }
  std::complex<double> constant_value() const { return std::get<0>(value); }

  scalar_parameter &operator=(const scalar_parameter &c) {
    value = c.value;
    return *this;
  }

  scalar_parameter &operator+(const scalar_parameter &c) {

    // cases:
    //   we are constant and incoming is constant
    if (has_value() && c.has_value()) {
      value = constant_value() + c.constant_value();
      return *this;
    }

    //   we are constant and incoming is dynamic
    if (has_value() && !c.has_value()) {
      auto f = std::get<1>(c.value);
      auto cvalue = constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue + f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is constant
    if (!has_value() && c.has_value()) {
      auto f = std::get<1>(value);
      auto cvalue = c.constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue + f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is dynamic
    auto f = std::get<1>(value);
    auto ff = std::get<1>(c.value);
    value = [f, ff](const parameter_map &params) {
      return f(params) + ff(params);
    };
    return *this;
  }

  scalar_parameter &operator-(const scalar_parameter &c) {

    // cases:
    //   we are constant and incoming is constant
    if (has_value() && c.has_value()) {
      value = constant_value() - c.constant_value();
      return *this;
    }

    //   we are constant and incoming is dynamic
    if (has_value() && !c.has_value()) {
      auto f = std::get<1>(c.value);
      auto cvalue = constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue - f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is constant
    if (!has_value() && c.has_value()) {
      auto f = std::get<1>(value);
      auto cvalue = c.constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue - f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is dynamic
    auto f = std::get<1>(value);
    auto ff = std::get<1>(c.value);
    value = [f, ff](const parameter_map &params) {
      return f(params) - ff(params);
    };
    return *this;
  }

  scalar_parameter &operator*(const scalar_parameter &c) {

    // cases:
    //   we are constant and incoming is constant
    if (has_value() && c.has_value()) {
      value = constant_value() * c.constant_value();
      return *this;
    }

    //   we are constant and incoming is dynamic
    if (has_value() && !c.has_value()) {
      auto f = std::get<1>(c.value);
      auto cvalue = constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue * f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is constant
    if (!has_value() && c.has_value()) {
      auto f = std::get<1>(value);
      auto cvalue = c.constant_value();
      value = [cvalue, f](const parameter_map &params) {
        return cvalue * f(params);
      };
      return *this;
    }

    //   we are dynamic and incoming is dynamic
    auto f = std::get<1>(value);
    auto ff = std::get<1>(c.value);
    value = [f, ff](const parameter_map &params) {
      return f(params) * ff(params);
    };
    return *this;
  }
};

struct operator_data {
  std::vector<std::vector<std::size_t>> productTerms;
  std::vector<scalar_parameter> coefficients;
};

class operator_handler {
public:
  virtual std::size_t num_qubits(const operator_data &thisPtr) const = 0;

  // Operations on operator...
  virtual void assign(operator_data &thisPtr, const operator_data &) = 0;

  /// @brief Add the given spin_op to this one and return *this
  virtual void addAssign(operator_data &thisPtr, const operator_data &v) = 0;

  /// @brief Multiply the given spin_op with this one and return *this
  virtual void multAssign(operator_data &thisPtr, const operator_data &v) = 0;

  virtual void multScalarAssign(operator_data &thisPtr,
                                const scalar_parameter &v) = 0;

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  virtual bool checkEquality(const operator_data &thisPtr,
                             const operator_data &v) const = 0;

  virtual std::string to_string(const operator_data &thisPtr,
                                bool print = true) const = 0;
  void dump(const operator_data &op) const;

  virtual ~operator_handler() = default;
};
} // namespace details
} // namespace cudaq::experimental