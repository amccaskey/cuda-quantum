/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "utils/registry.h"
#include "utils/tensor_impl.h"

namespace cudaq {
inline constexpr std::size_t dynamic_rank =
    std::numeric_limits<std::size_t>::max();

namespace details {
void __deallocate_cuda(void *);

template <typename T>
using tensor_deleter_t = void (*)(T *);

template <typename T>
void cuda_deleter(T *ptr) {
  __deallocate_cuda(ptr);
}

void __allocate_cuda(void **, std::size_t);
} // namespace details
template <typename T>
std::unique_ptr<T[], details::tensor_deleter_t<T>>
make_unique_cuda(size_t size) {
  T *raw_ptr = nullptr;
  details::__allocate_cuda((void **)&raw_ptr, size * sizeof(T));
  return std::unique_ptr<T[], details::tensor_deleter_t<T>>(
      raw_ptr, details::cuda_deleter<T>);
}

namespace details {
const char *tensor_memory_to_string(tensor_memory tm) noexcept {
  switch (tm) {
  case tensor_memory::host:
    return "host_tensor";
  case tensor_memory::cuda:
    return "cuda_tensor";
  }
  return "Unknown";
}

// If data is on gpu device, return "cuda", else "host"
const char *__tensor_memory_or_pointer_to_string(void *ptr);
template <typename T>
const char *tensor_memory_or_pointer_to_string(T *ptr) {
  return __tensor_memory_or_pointer_to_string(ptr);
}

} // namespace details

/// @brief A basic_tensor class implementing the PIMPL idiom. The flattened data
/// is stored where the strides grow from right to left (similar to a
/// multi-dimensional C array).
template <std::size_t Rank = dynamic_rank,
          typename Scalar = std::complex<double>>
class basic_tensor {
private:
  tensor_memory memory;

  std::shared_ptr<details::tensor_impl<Scalar>> pimpl;

  basic_tensor<Rank, Scalar>
  matrix_vector(const basic_tensor<Rank, Scalar> &vec) const {
    if (get_rank() != 2 || vec.get_rank() != 1) {
      throw std::runtime_error(
          "Matrix-vector product requires rank-2 matrix and rank-1 vector");
    }
    if (shape()[1] != vec.shape()[0]) {
      throw std::runtime_error("Invalid dimensions for matrix-vector product");
    }

    basic_tensor<Rank, Scalar> result({shape()[0]});
    pimpl->matrix_vector_product(vec.pimpl.get(), result.pimpl.get());
    return result;
  }

  template <typename T>
  std::unique_ptr<T[]> vector_to_owned_pointer(const std::vector<T> &vec) {
    auto ptr = std::make_unique<T[]>(vec.size());
    std::copy(vec.begin(), vec.end(), ptr.get());
    return ptr;
  }

public:
  /// @brief Type alias for the scalar type used in the basic_tensor
  using scalar_type = typename details::tensor_impl<Scalar>::scalar_type;

  basic_tensor(tensor_memory tm = tensor_memory::host)
      : pimpl(details::tensor_impl<Scalar>::create(tm, {}).release()),
        memory(tm) {}

  /// @brief Construct a basic_tensor with the given shape
  /// @param shape The shape of the basic_tensor
  basic_tensor(const std::vector<std::size_t> &shape,
               tensor_memory tm = tensor_memory::host)
      : pimpl(details::tensor_impl<Scalar>::create(tm, shape).release()),
        memory(tm) {
    if (Rank != dynamic_rank && Rank != shape.size())
      throw std::runtime_error("Invalid shape for basic_tensor of Rank = " +
                               std::to_string(Rank));
  }

  basic_tensor(const std::vector<Scalar> &data,
               const std::vector<std::size_t> &shape,
               tensor_memory tm = tensor_memory::host)
      : pimpl(details::tensor_impl<Scalar>::create(
                  tm, vector_to_owned_pointer(data).release(), shape)
                  .release()),
        memory(tm) {
    if (Rank != dynamic_rank && Rank != shape.size())
      throw std::runtime_error("Invalid shape for basic_tensor of Rank = " +
                               std::to_string(Rank));
  }

  basic_tensor(std::size_t dim0, std::size_t dim1,
               tensor_memory tm = tensor_memory::host)
    requires(Rank == 2)
      : basic_tensor(std::vector<std::size_t>{dim0, dim1}, tm) {}

  basic_tensor(std::size_t dim0, tensor_memory tm = tensor_memory::host)
    requires(Rank == 1)
      : basic_tensor(std::vector<std::size_t>{dim0}, tm) {}

  basic_tensor(const std::vector<Scalar> &data, std::size_t dim0,
               std::size_t dim1, tensor_memory tm = tensor_memory::host)
    requires(Rank == 2)
      : basic_tensor(data, std::vector<std::size_t>{dim0, dim1}, tm) {}

  basic_tensor(const std::vector<Scalar> &data, std::size_t dim0,
               tensor_memory tm = tensor_memory::host)
    requires(Rank == 1)
      : basic_tensor(data, std::vector<std::size_t>{dim0}, tm) {}

  basic_tensor(
      std::unique_ptr<Scalar[], details::tensor_deleter_t<Scalar>> &&data,
      const std::vector<std::size_t> &shape)
      : pimpl(details::tensor_impl<Scalar>::create(
            std::string(details::tensor_memory_or_pointer_to_string(
                data.get())) == "cuda_tensor"
                ? tensor_memory::cuda
                : tensor_memory::host,
            data.release(), shape)) {
    if (Rank != dynamic_rank && Rank != shape.size())
      throw std::runtime_error("Invalid shape for basic_tensor of Rank = " +
                               std::to_string(Rank));
    memory = "cudaq_tensor" ==
                     details::tensor_memory_or_pointer_to_string(data.get())
                 ? tensor_memory::cuda
                 : tensor_memory::host;
  }

  basic_tensor(const basic_tensor<Rank, Scalar> &other)
      : pimpl(details::tensor_impl<Scalar>::create(other.memory, other.shape())
                  .release()),
        memory(other.memory) {
    std::copy(other.data(), other.data() + other.get_num_elements(),
              pimpl->data());
  }

  static basic_tensor<Rank, Scalar>
  random(const std::vector<std::size_t> &shape,
         tensor_memory tm = tensor_memory::host) {
    basic_tensor<Rank, Scalar> result(shape, tm);
    result.pimpl->fill_random();
    return result;
  }

  /// @brief Get the rank of the basic_tensor
  /// @return The rank of the basic_tensor
  std::size_t get_rank() const {
    return Rank == dynamic_rank ? pimpl->rank() : Rank;
  }

  /// @brief Get the total number of elements in the basic_tensor
  /// @return The total number of elements in the basic_tensor
  std::size_t get_num_elements() const { return pimpl->size(); }

  std::size_t element_size() const { return sizeof(Scalar); }

  /// @brief Get the shape of the basic_tensor
  /// @return A vector containing the dimensions of the basic_tensor
  std::vector<std::size_t> shape() const { return pimpl->shape(); }

  /// @brief Access a mutable element of the basic_tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  scalar_type &at(const std::vector<size_t> &indices) {
    if (indices.size() != get_rank())
      throw std::runtime_error(
          "Invalid indices provided to basic_tensor::at(), size "
          "must be equal to rank.");
    return pimpl->at(indices);
  }

  /// @brief Access a const element of the basic_tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  const scalar_type &at(const std::vector<size_t> &indices) const {
    return pimpl->at(indices);
  }

  basic_tensor<dynamic_rank, Scalar>
  operator[](const std::vector<slice> &slices) {
    if (slices.size() != get_rank()) {
      throw std::runtime_error("Number of slices must match tensor rank");
    }

    for (std::size_t i = 0; i < slices.size(); i++) {
      // for (auto &el : slices[i])
      if (slices[i].stop > shape()[i])
        throw std::runtime_error(
            "invalid slice, slice index greater than number of dimensions.");
    }
    std::vector<std::size_t> result_shape;
    result_shape.reserve(slices.size());

    for (size_t i = 0; i < slices.size(); i++) {
      const auto &s = slices[i];
      std::size_t start = s.start ? *s.start : 0;
      std::size_t stop = s.stop ? *s.stop : shape()[i];
      std::size_t step = s.step ? *s.step : 1;
      std::size_t dim_size = (stop - start + step - 1) / step;
      result_shape.push_back(dim_size);
    }

    // Create result tensor with correct shape
    basic_tensor<dynamic_rank, Scalar> result(result_shape, memory);

    // Get data from slice operation
    std::vector<Scalar> sliced_data;
    pimpl->slice(slices, sliced_data);

    // Copy data to result tensor
    std::copy(sliced_data.begin(), sliced_data.end(), result.data());

    return result;
  }

  template <typename... Args>
  typename std::enable_if_t<(std::is_integral_v<Args> && ...), scalar_type &> &
  operator()(const Args... indices) {
    if (sizeof...(Args) != shape().size())
      throw std::runtime_error("invalid element access, number of provided "
                               "extraction indices not equal to tensor rank.");

    return at({static_cast<std::size_t>(indices)...});
  }

  template <typename... Args>
  typename std::enable_if_t<(std::is_integral_v<Args> && ...),
                            const scalar_type &> &
  operator()(const Args... indices) const {
    if (sizeof...(Args) != shape().size())
      throw std::runtime_error("invalid element access, number of provided "
                               "extraction indices not equal to tensor rank.");

    return at({static_cast<std::size_t>(indices)...});
  }

  scalar_type &operator()(const std::vector<size_t> &indices) {
    if (indices.size() != get_rank())
      throw std::runtime_error(
          "Invalid indices provided to basic_tensor::at(), size "
          "must be equal to rank.");
    return pimpl->at(indices);
  }

  /// @brief Access a const element of the basic_tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  const scalar_type &operator()(const std::vector<size_t> &indices) const {
    return pimpl->at(indices);
  }

  // Scalar-resulting operations
  Scalar sum_all() const { return pimpl->sum_all(); }

  // Boolean-resulting operations
  bool any() const { return pimpl->any(); }

  // Elementwise operations
  basic_tensor<Rank, Scalar>
  operator+(const basic_tensor<Rank, Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("basic_tensor shapes must match for addition");
    }
    basic_tensor<Rank, Scalar> result(shape(), memory);
    pimpl->elementwise_add(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  basic_tensor<Rank, Scalar> operator*(Scalar other) const {
    auto tmp = *this;
    auto *d = tmp.pimpl->data();
    for (std::size_t i = 0; i < get_num_elements(); i++) {
      auto &el = d[i];
      el *= other;
    }
    return tmp;
  }

  basic_tensor<Rank, Scalar>
  operator*(const basic_tensor<Rank, Scalar> &other) const {

    // If matrices,
    if (get_rank() == 2 && other.get_rank() == 2 &&
        shape()[1] == other.shape()[0])
      return dot(other);

    if (shape() != other.shape()) {
      throw std::runtime_error(
          "basic_tensor shapes must match for multiplication");
    }

    basic_tensor<Rank, Scalar> result(shape(), memory);
    pimpl->elementwise_multiply(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  basic_tensor<Rank, Scalar>
  operator%(const basic_tensor<Rank, Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("basic_tensor shapes must match for modulo");
    }
    basic_tensor<Rank, Scalar> result(shape(), memory);
    pimpl->elementwise_modulo(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  // basic_tensor-Scalar operations
  basic_tensor<Rank, Scalar> operator%(Scalar value) const {
    basic_tensor<Rank, Scalar> result(shape(), memory);
    pimpl->scalar_modulo(value, result.pimpl.get());
    return result;
  }

  basic_tensor<Rank, Scalar> &
  operator=(const basic_tensor<Rank, Scalar> &other) {
    // Prevent self-assignment
    if (this != &other) {
      // Create a new implementation with copied data
      memory = other.memory;
      pimpl = std::shared_ptr<details::tensor_impl<Scalar>>(
          details::tensor_impl<Scalar>::create(other.memory, other.shape())
              .release());
      std::copy(other.data(), other.data() + other.get_num_elements(),
                pimpl->data());
    }
    return *this;
  }

  // Return the dot product of two tensors.
  // Want to reproduce NumPy semantics
  // https://numpy.org/doc/2.1/reference/generated/numpy.dot.html If both
  // tensors are rank-1, this returns the inner product of the vectors (without
  // complex conjugation). If both tensors are rank-2, this returns the result
  // of standard matrix multiplication. If either is rank-0 (scalar), return the
  // basic_tensor where all elements are scaled by the scalar. If this
  // basic_tensor is rank-N and the other is rank-1, return the sum product over
  // the last axis of this and other. If this is rank-N and other is rank-M,
  // return the sum product over the last axis of this and the second to last
  // axis of other.
  basic_tensor<Rank, Scalar>
  dot(const basic_tensor<Rank, Scalar> &other) const {

    if (get_rank() == 2 && other.get_rank() == 1)
      return matrix_vector(other);

    if (get_rank() != 2 || other.get_rank() != 2) {
      throw std::runtime_error("Dot product requires rank-2 tensors");
    }
    if (shape()[1] != other.shape()[0]) {
      throw std::runtime_error("Invalid matrix dimensions for dot product");
    }

    std::vector<std::size_t> result_shape = {shape()[0], other.shape()[1]};
    basic_tensor<Rank, Scalar> result(result_shape, memory);
    pimpl->matrix_dot(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  // Returns a basic_tensor with axes transposed.
  // Want to reproduce NumPy semantics
  // https://numpy.org/doc/2.1/reference/generated/numpy.transpose.html
  // For a rank-1 basic_tensor, this returns the same basic_tensor.
  // For a rank-2 basic_tensor, this returns the matrix transpose
  // For a rank-N basic_tensor, the axes must be provided. Their order indicates
  // how the axes are permuted.
  basic_tensor<Rank, Scalar>
  transpose(const std::vector<std::size_t> axes = {}) const {
    if (get_rank() != 2)
      throw std::runtime_error("Transpose only implemented for rank-2 tensors");

    std::vector<std::size_t> result_shape = {shape()[1], shape()[0]};
    basic_tensor<Rank, Scalar> result(result_shape, memory);
    pimpl->matrix_transpose(result.pimpl.get());
    return result;
  }

  Scalar minimal_eigenvalue() const {
    if (get_rank() != 2)
      throw std::runtime_error(
          "minimal_eigenvalue only supported for matrices.");
    return pimpl->minimal_eigenvalue();
  }

  std::vector<Scalar> eigenvalues() const {
    if (get_rank() != 2)
      throw std::runtime_error("eigenvalues only supported for matrices.");
    return pimpl->eigenvalues();
  }

  basic_tensor<2, Scalar> eigenvectors() const {
    if (get_rank() != 2)
      throw std::runtime_error("eigenvectors only supported for matrices.");
    basic_tensor<2, Scalar> result(shape(), memory);
    pimpl->eigenvectors(result.pimpl.get());
    return result;
  }

  // Kronecker product between two general tensor arrays. This function
  // assumes both ranks are equal.
  basic_tensor<Rank, Scalar> kron(const basic_tensor<Rank, Scalar> &other) {
    if (get_rank() != other.get_rank()) {
      throw std::runtime_error(
          "Kronecker product requires tensors of equal rank");
    }

    // Calculate result shape
    std::vector<std::size_t> result_shape;
    result_shape.reserve(get_rank());
    for (std::size_t i = 0; i < get_rank(); i++) {
      result_shape.push_back(shape()[i] * other.shape()[i]);
    }

    basic_tensor<Rank, Scalar> result(result_shape, memory);
    pimpl->kron(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  /// @brief Get a pointer to the raw data of the basic_tensor.
  ///
  /// This method provides direct access to the underlying data storage of the
  /// basic_tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return scalar_type* A pointer to the mutable data of the basic_tensor.
  ///
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid
  ///       invalidating the basic_tensor's internal state or violating its
  ///       invariants.
  scalar_type *data() { return pimpl->data(); }

  /// @brief Get a const pointer to the raw data of the basic_tensor.
  ///
  /// This method provides read-only access to the underlying data storage of
  /// the basic_tensor. It returns a const pointer to the first element of the
  /// data array.
  ///
  /// @return const scalar_type * A const pointer to the immutable data of the
  /// basic_tensor.
  ///
  /// @note This const version ensures that the basic_tensor's data cannot be
  /// modified
  ///       through the returned pointer, preserving const correctness.
  const scalar_type *data() const { return pimpl->data(); }

  void dump() const { pimpl->dump(); }
};

template <std::size_t Rank, typename Scalar>
basic_tensor<Rank, Scalar> operator*(Scalar value,
                                     const basic_tensor<Rank, Scalar> &rhs) {
  return rhs * value;
}

template <typename Scalar = std::complex<double>>
using tensor = basic_tensor<dynamic_rank, Scalar>;

template <std::size_t Rank, typename Scalar = std::complex<double>>
using fixed_tensor = basic_tensor<Rank, Scalar>;

// Create useful typedefs for matrices
template <typename Scalar = std::complex<double>>
using matrix = fixed_tensor<2, Scalar>;

// Create useful typedefs for vectors
template <typename Scalar = std::complex<double>>
using vector = fixed_tensor<1, Scalar>;

// Compute the Kronecker product of the two input matrices and return the
// result.
template <typename T>
matrix<T> kron(const matrix<T> &A, const matrix<T> &B) {
  return A.kron(B);
}

} // namespace cudaq