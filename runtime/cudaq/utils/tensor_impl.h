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
#include <vector>

namespace cudaq {
/// @brief A template struct for mapping related types
/// @tparam T The base type
template <typename T>
struct RelatedTypesMap {
  using types = std::tuple<>;
};

/// @brief Specialization of RelatedTypesMap for int
template <>
struct RelatedTypesMap<int> {
  using types = std::tuple<std::size_t, long, short>;
};

/// @brief Specialization of RelatedTypesMap for std::size_t
template <>
struct RelatedTypesMap<std::size_t> {
  using types = std::tuple<int, long, short>;
};

/// @brief Specialization of RelatedTypesMap for long
template <>
struct RelatedTypesMap<long> {
  using types = std::tuple<int, std::size_t, short>;
};

/// @brief Specialization of RelatedTypesMap for short
template <>
struct RelatedTypesMap<short> {
  using types = std::tuple<int, long, std::size_t>;
};

/// @brief Specialization of RelatedTypesMap for double
template <>
struct RelatedTypesMap<double> {
  using types = std::tuple<float>;
};

/// @brief Specialization of RelatedTypesMap for float
template <>
struct RelatedTypesMap<float> {
  using types = std::tuple<double>;
};

/// @brief Specialization of RelatedTypesMap for std::string
template <>
struct RelatedTypesMap<std::string> {
  using types = std::tuple<const char *>;
};

/// @brief Type trait to check if a type is a bounded char array
template <class>
struct is_bounded_char_array : std::false_type {};

/// @brief Specialization for bounded char arrays
template <std::size_t N>
struct is_bounded_char_array<char[N]> : std::true_type {};

/// @brief Type trait to check if a type is a bounded array
template <class>
struct is_bounded_array : std::false_type {};

/// @brief Specialization for bounded arrays
template <class T, std::size_t N>
struct is_bounded_array<T[N]> : std::true_type {};

// Primary template (for unsupported types)
template <typename T>
constexpr std::string_view type_to_string() {
  return "unknown";
}

// Specializations for common scalar types
template <>
constexpr std::string_view type_to_string<int>() {
  return "int";
}

template <>
constexpr std::string_view type_to_string<double>() {
  return "double";
}

template <>
constexpr std::string_view type_to_string<float>() {
  return "float";
}

template <>
constexpr std::string_view type_to_string<long>() {
  return "long";
}
template <>
constexpr std::string_view type_to_string<std::size_t>() {
  return "stdsizet";
}
template <>
constexpr std::string_view type_to_string<std::complex<double>>() {
  return "complex<double>";
}
template <>
constexpr std::string_view type_to_string<std::complex<float>>() {
  return "complex<float>";
}

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
class tensor_impl : public extension_point<tensor_impl<Scalar>, const Scalar *,
                                           const std::vector<std::size_t>> {
public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = Scalar;
  using BaseExtensionPoint =
      extension_point<tensor_impl<Scalar>, const Scalar *,
                      const std::vector<std::size_t>>;

  virtual void
  slice(const std::vector<slice> &slices, std::vector<Scalar>& result_data) const = 0;

  /// @brief Create a tensor implementation with the given name and shape
  /// @param name The name of the tensor implementation
  /// @param shape The shape of the tensor
  /// @return A unique pointer to the created tensor implementation
  /// @throws std::runtime_error if the requested tensor implementation is
  /// invalid
  static std::unique_ptr<tensor_impl<Scalar>>
  get(const std::string &name, const std::vector<std::size_t> &shape) {
    auto &registry = BaseExtensionPoint::get_registry();
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("invalid tensor_impl requested: " + name);

    if (shape.empty())
      return iter->second(nullptr, {});

    std::size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<size_t>());
    scalar_type *data = new scalar_type[size]();
    return iter->second(data, shape);
  }

  /// @brief Create a tensor implementation with the given name, data, and shape
  /// @param name The name of the tensor implementation
  /// @param data Pointer to the tensor data
  /// @param shape The shape of the tensor
  /// @return A unique pointer to the created tensor implementation
  /// @throws std::runtime_error if the requested tensor implementation is
  /// invalid
  static std::unique_ptr<tensor_impl<Scalar>>
  get(const std::string &name, const scalar_type *data,
      const std::vector<std::size_t> &shape) {
    auto &registry = BaseExtensionPoint::get_registry();
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("invalid tensor_impl requested: " + name);
    return iter->second(data, shape);
  }

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
