/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/operators/operator_data.h"

namespace cudaq::experimental::details {

scalar_parameter::type value;
std::complex<double>
scalar_parameter::operator()(const parameter_map &parameters) const {
  if (value.index() == 0)
    return constant_value();

  return std::get<1>(value)(parameters);
}

scalar_parameter::scalar_parameter(double value) : value(value) {}
scalar_parameter::scalar_parameter(std::complex<double> value) : value(value) {}
scalar_parameter::scalar_parameter(const dynamic_signature &functor)
    : value(functor) {}

bool scalar_parameter::has_value() const { return value.index() == 0; }
std::complex<double> scalar_parameter::constant_value() const {
  return std::get<0>(value);
}

scalar_parameter &scalar_parameter::operator=(const scalar_parameter &c) {
  value = c.value;
  return *this;
}

scalar_parameter &scalar_parameter::operator+(const scalar_parameter &c) {

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

scalar_parameter &scalar_parameter::operator-(const scalar_parameter &c) {

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

scalar_parameter &scalar_parameter::operator*(const scalar_parameter &c) {

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

// Operations on operator...
void operator_handler::assign(operator_data &thisPtr,
                              const operator_data &other) {
  thisPtr.productTerms = other.productTerms;
  thisPtr.coefficients = other.coefficients;
}

void operator_handler::mult_scalar_assign(operator_data &thisPtr,
                                          const scalar_parameter &v) {
  for (std::size_t i = 0; auto &term : thisPtr.productTerms) {
    thisPtr.coefficients[i] = thisPtr.coefficients[i] * v;
    i++;
  }

  return;
}

} // namespace cudaq::experimental::details
