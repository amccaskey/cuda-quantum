/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "registry.h"

#include <complex>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace cudaq {
enum class tensor_memory { host, cuda };

// Add slice type definition
struct slice {
  std::optional<std::size_t> start;
  std::optional<std::size_t> stop;
  std::optional<std::size_t> step;

  slice() = default;
  slice(std::size_t s) : start(s), stop(s + 1) {}
  slice(std::size_t start_, std::size_t stop_, std::size_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
};

} // namespace cudaq
namespace cudaq::details {

/// @brief Implementation class for tensor operations following the PIMPL idiom
template <typename Scalar = std::complex<double>>
class tensor_impl {
public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = Scalar;
 
  // Factory method to create tensor implementations
  static std::unique_ptr<tensor_impl<Scalar>>
  create(tensor_memory mem_type, Scalar *data,
         const std::vector<std::size_t> &shape);
  static std::unique_ptr<tensor_impl<Scalar>>
  create(tensor_memory mem_type, const std::vector<std::size_t> &shape);

  virtual void slice(const std::vector<slice> &slices,
                     std::vector<Scalar> &result_data) const = 0;

  /// @brief Get the rank of the tensor
  /// @return The rank of the tensor
  virtual std::size_t rank() const = 0;

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  virtual std::size_t size() const = 0;

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  virtual std::vector<std::size_t> shape() const = 0;

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  virtual scalar_type &at(const std::vector<size_t> &indices) = 0;

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  virtual const scalar_type &at(const std::vector<size_t> &indices) const = 0;

  virtual void fill_random() = 0;

  virtual scalar_type sum_all() const = 0;

  virtual bool any() const = 0;

  virtual void elementwise_add(const tensor_impl<Scalar> *other,
                               tensor_impl<Scalar> *result) const = 0;

  virtual void elementwise_multiply(const tensor_impl<Scalar> *other,
                                    tensor_impl<Scalar> *result) const = 0;

  virtual void elementwise_modulo(const tensor_impl<Scalar> *other,
                                  tensor_impl<Scalar> *result) const = 0;

  virtual void scalar_modulo(Scalar value,
                             tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_dot(const tensor_impl<Scalar> *other,
                          tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_vector_product(const tensor_impl<Scalar> *vec,
                                     tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_transpose(tensor_impl<Scalar> *result) const = 0;

  virtual void kron(const tensor_impl<Scalar> *other,
                    tensor_impl<Scalar> *result) const = 0;

  virtual Scalar minimal_eigenvalue() const = 0;
  virtual std::vector<Scalar> eigenvalues() const = 0;
  virtual void eigenvectors(tensor_impl<Scalar> *result) const = 0;

  /// @brief Get a pointer to the raw data of the tensor.
  /// This method provides direct access to the underlying data storage of the
  /// tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return scalar_type* A pointer to the mutable data of the tensor.
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid invalidating the tensor's internal state or violating its
  /// invariants.
  virtual scalar_type *data() = 0;

  /// @brief Get a const pointer to the raw data of the tensor.
  /// This method provides read-only access to the underlying data storage of
  /// the tensor. It returns a const pointer to the first element of the data
  /// array.
  ///
  /// @return const scalar_type * A const pointer to the immutable data of the
  /// tensor.
  /// @note This const version ensures that the tensor's data cannot be modified
  ///       through the returned pointer, preserving const correctness.
  virtual const scalar_type *data() const = 0;

  virtual void dump() const = 0;

  virtual ~tensor_impl() = default;
};

} // namespace cudaq::details
