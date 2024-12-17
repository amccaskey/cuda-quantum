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

#include "cudaq/utils/tensor.h"

namespace cudaq::experimental {

/// @brief A map of parameter names to complex values
using parameter_map = std::unordered_map<std::string, std::complex<double>>;

/// @brief A 2x2 matrix representing a quantum operator
using operator_matrix = cudaq::matrix_2;

/// @brief A function that generates an operator matrix given a parameter map
using operator_matrix_generator =
    std::function<operator_matrix(const parameter_map &)>;

namespace details {

/// @brief A class representing a scalar parameter that can be constant or
/// dynamic
class scalar_parameter {
public:
  /// @brief Function signature for dynamic parameters
  using dynamic_signature =
      std::function<std::complex<double>(const parameter_map &)>;

  /// @brief Variant type to hold either a constant value or a dynamic function
  using type = std::variant<std::complex<double>, dynamic_signature>;

  /// @brief The value of the scalar parameter
  type value;

  /// @brief Evaluate the scalar parameter given a parameter map
  /// @param parameters The parameter map
  /// @return The complex value of the scalar parameter
  std::complex<double> operator()(const parameter_map &parameters) const;

  /// @brief Constructor for real-valued constant parameters
  /// @param value The real value
  scalar_parameter(double value);

  /// @brief Constructor for complex-valued constant parameters
  /// @param value The complex value
  scalar_parameter(std::complex<double> value);

  /// @brief Constructor for dynamic parameters
  /// @param functor The dynamic function
  scalar_parameter(const dynamic_signature &functor);

  /// @brief Copy constructor
  scalar_parameter(const scalar_parameter &) = default;

  /// @brief Check if the parameter has a constant value
  /// @return True if the parameter is constant, false if dynamic
  bool has_value() const;

  /// @brief Get the constant value of the parameter
  /// @return The constant complex value
  std::complex<double> constant_value() const;

  /// @brief Assignment operator
  /// @param c The scalar parameter to assign from
  /// @return Reference to this scalar parameter
  scalar_parameter &operator=(const scalar_parameter &c);

  /// @brief Addition operator
  /// @param c The scalar parameter to add
  /// @return Reference to this scalar parameter
  scalar_parameter &operator+(const scalar_parameter &c);

  /// @brief Subtraction operator
  /// @param c The scalar parameter to subtract
  /// @return Reference to this scalar parameter
  scalar_parameter &operator-(const scalar_parameter &c);

  /// @brief Multiplication operator
  /// @param c The scalar parameter to multiply
  /// @return Reference to this scalar parameter
  scalar_parameter &operator*(const scalar_parameter &c);
};

/// @brief A struct containing data for quantum operators
struct operator_data {
  /// @brief A term in the matrix product representation
  using matrix_product_term = std::vector<operator_matrix_generator>;

  /// @brief A sum of matrix product terms
  using matrix_product_sum = std::vector<matrix_product_term>;

  /// @brief A term in the operator representation
  using operator_term = std::vector<std::size_t>;

  /// @brief A sum of operator terms
  using operator_sum = std::vector<operator_term>;

  /// @brief The sum of product terms for the operator
  operator_sum productTerms;

  /// @brief The coefficients for each term in the operator
  std::vector<scalar_parameter> coefficients;

  /// @brief Optional matrix product representation of the operator
  std::optional<matrix_product_sum> matrixProductTerms;
};

/// @brief An abstract base class for handling quantum operators
class operator_handler {
public:
  /// @brief Get the number of qubits for the operator
  /// @param thisPtr The operator data
  /// @return The number of qubits
  virtual std::size_t num_qubits(const operator_data &thisPtr) const = 0;

  /// @brief Initialize the operator data
  /// @return The initialized operator data
  virtual details::operator_data initialize() const = 0;

  /// @brief Get a vector representation of the operator data
  /// @param op The operator data
  /// @return A vector of doubles representing the operator
  virtual std::vector<double>
  get_data_representation(const details::operator_data &op) const = 0;

  /// @brief Convert the operator to a matrix representation
  /// @param op The operator data
  /// @param p The parameter map
  /// @return The matrix representation of the operator
  virtual operator_matrix to_matrix(const details::operator_data &op,
                                    const parameter_map &p) const = 0;

  /// @brief Get the support matrices for the operator
  /// @param op The operator data
  /// @param p The parameter map
  /// @return A vector of support matrices
  virtual std::vector<operator_matrix>
  get_support_matrices(const details::operator_data &op,
                       const parameter_map &p) const = 0;

  /// @brief Assign one operator to another
  /// @param thisPtr The operator data to assign to
  /// @param other The operator data to assign from
  virtual void assign(operator_data &thisPtr, const operator_data &other);

  /// @brief Add another operator to this one
  /// @param thisPtr The operator data to add to
  /// @param v The operator data to add
  virtual void add_assign(operator_data &thisPtr, const operator_data &v) = 0;

  /// @brief Multiply this operator by another
  /// @param thisPtr The operator data to multiply
  /// @param v The operator data to multiply by
  virtual void mult_assign(operator_data &thisPtr, const operator_data &v) = 0;

  /// @brief Multiply this operator by a scalar
  /// @param thisPtr The operator data to multiply
  /// @param v The scalar to multiply by
  virtual void mult_scalar_assign(operator_data &thisPtr,
                                  const scalar_parameter &v);

  /// @brief Check if two operators are equal
  /// @param thisPtr The first operator data
  /// @param v The second operator data
  /// @return True if the operators are equal, false otherwise
  virtual bool check_equality(const operator_data &thisPtr,
                              const operator_data &v) const = 0;

  /// @brief Convert the operator to a string representation
  /// @param thisPtr The operator data
  /// @param print Whether to print the string (default: true)
  /// @return The string representation of the operator
  virtual std::string to_string(const operator_data &thisPtr,
                                bool print = true) const = 0;

  /// @brief Dump the operator data for debugging
  /// @param op The operator data to dump
  void dump(const operator_data &op) const;

  /// @brief Virtual destructor
  virtual ~operator_handler() = default;
};
} // namespace details
} // namespace cudaq::experimental
