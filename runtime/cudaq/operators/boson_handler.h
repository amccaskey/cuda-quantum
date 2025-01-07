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
class matrix_handler;
using matrix_op = operator_<matrix_handler>;
} // namespace cudaq::experimental

namespace cudaq::experimental::boson {

/// @brief Namespace containing bosonic operator implementations for CUDA-Q
///
/// Provides classes and functions for working with bosonic operators in
/// quantum computing. The data encoding for boson operators uses pairs of
/// (index, type) where:
/// - index: qubit/mode index
/// - type: 1 for creation (a†), 0 for annihilation (a)
///
/// Implements canonical commutation relations:
/// \f[ \[a_i, a_j^\dagger\] = \delta_{ij}, \[a_i,a_i\] =
/// \[a_i^\dagger,a_j^\dagger\] = 0 \f]

/// @brief Type alias for storing bosonic operator terms as index-type pairs
using boson_term_t = std::vector<std::size_t>;

/// @brief Type alias for the complete bosonic operator data structure
using boson_data = details::operator_data<boson_term_t>;

/// @brief Handler class for bosonic operator algebra and matrix
/// representations
class boson_handler : public details::operator_handler<boson_term_t> {
public:
  /// @brief Required type definition for client code
  using term_t = boson_term_t;

  /// @brief Initialize an empty bosonic operator
  /// @return Empty boson_data structure
  boson_data initialize() const override;

  /// @brief Get vector representation of operator data
  /// @param data The bosonic operator data
  /// @return Empty vector (not implemented for bosons)
  std::vector<double>
  get_data_representation(const boson_data &data) const override {
    return {};
  }

  /// @brief Convert bosonic operator to matrix representation
  /// @param data The bosonic operator data
  /// @param p Parameter map for dynamic coefficients
  /// @param dimensions Map of site indices to local dimensions
  /// @return Matrix representation of the operator
  operator_matrix to_matrix(const boson_data &data, const parameter_map &p,
                            const dimensions_map &dimensions) const override;

  /// @brief Get number of qubits/modes operator acts on
  /// @param thisPtr The bosonic operator data
  /// @return Maximum site index plus one
  std::size_t num_qubits(const boson_data &thisPtr) const override;

  /// @brief Get elementary matrices for operator decomposition
  /// @param data The bosonic operator data
  /// @param dimensions Map of site indices to local dimensions
  /// @return Vector of constituent operator matrices
  std::vector<operator_matrix>
  get_support_matrices(const boson_data &data,
                       const dimensions_map &dimensions) const override;
  std::set<std::size_t> get_supports(const boson_data &data) const override;

  /// @brief Add another bosonic operator to this one
  /// @param thisPtr The operator to add to
  /// @param v The operator to add
  void add_assign(boson_data &thisPtr, const boson_data &v) override;

  /// @brief Multiply this operator by another
  /// @param thisPtr The operator to multiply
  /// @param v The operator to multiply by
  void mult_assign(boson_data &thisPtr, const boson_data &v) override;

  /// @brief Check if two operators are equal
  /// @param thisPtr First operator to compare
  /// @param v Second operator to compare
  /// @return True if operators are equal
  bool check_equality(const boson_data &thisPtr,
                      const boson_data &v) const override;

  /// @brief Convert operator to string representation
  /// @param thisPtr The operator to convert
  /// @param printCoeffs Whether to include coefficients
  /// @return String representation of the operator
  std::string to_string(const boson_data &thisPtr,
                        bool printCoeffs) const override;
};

/// @brief Type alias for bosonic operators using the boson_handler
using boson_op = operator_<boson_handler>;

/// @brief Create a bosonic creation operator
/// @param idx Site index for the creation operator
/// @return Creation operator a†_idx
boson_op create(std::size_t idx);

/// @brief Create a bosonic annihilation operator
/// @param idx Site index for the annihilation operator
/// @return Annihilation operator a_idx
boson_op annihilate(std::size_t idx);

matrix_op squeeze(std::size_t idx);
matrix_op displace(std::size_t idx);
} // namespace cudaq::experimental::boson
