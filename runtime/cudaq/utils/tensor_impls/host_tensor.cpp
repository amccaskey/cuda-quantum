/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../tensor_impl.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrided_view.hpp>

#include <fmt/ranges.h>
#include <iostream>
namespace cudaq {

template <typename Scalar>
class host_tensor_reference : public tensor_reference<Scalar> {
private:
  Scalar &ref;

public:
  explicit host_tensor_reference(Scalar &value) : ref(value) {}

  host_tensor_reference &operator=(const Scalar &value) override {
    ref = value;
    return *this;
  }

  operator Scalar() const override { return ref; }
};

/// @brief An implementation of tensor_impl using xtensor library
template <typename Scalar>
class host_tensor : public cudaq::details::tensor_impl<Scalar> {
private:
  Scalar *m_data = nullptr;         ///< Pointer to the tensor data
  std::vector<std::size_t> m_shape; ///< Shape of the tensor

  /// @brief Check if the given indices are valid for this tensor
  /// @param idxs Vector of indices to check
  /// @return true if indices are valid, false otherwise
  bool validIndices(const std::vector<std::size_t> &idxs) const {
    if (idxs.size() != m_shape.size())
      return false;
    for (std::size_t dim = 0; auto idx : idxs)
      if (idx < 0 || idx >= m_shape[dim++])
        return false;
    return true;
  }
  // Pool of references indexed by position
  std::unordered_map<std::size_t,
                     std::unique_ptr<host_tensor_reference<Scalar>>>
      ref_pool;

  size_t compute_linear_index(const std::vector<size_t> &indices,
                              const std::vector<size_t> &shape) {
    if (indices.size() != shape.size()) {
      throw std::runtime_error("Number of indices must match tensor rank");
    }

    size_t linear_idx = 0;
    size_t stride = 1;

    // Compute row-major linear index (C-style ordering)
    // Iterate from rightmost dimension to leftmost
    for (int i = indices.size() - 1; i >= 0; i--) {
      if (indices[i] >= shape[i]) {
        throw std::runtime_error("Index out of bounds");
      }
      linear_idx += indices[i] * stride;
      stride *= shape[i];
    }

    return linear_idx;
  }

public:
  /// @brief Constructor for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  host_tensor(Scalar *d, const std::vector<std::size_t> &s)
      : m_data(const_cast<Scalar *>(d)), m_shape(s) {}

  host_tensor(const std::vector<std::size_t> &s) : m_shape(s) {
    m_data = new Scalar[size()]{};
  }

  details::tensor_reference<Scalar> &
  extract(const std::vector<size_t> &indices) override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid tensor indices");

    size_t linear_idx = compute_linear_index(indices, shape());

    // Create new reference if needed
    if (!ref_pool.contains(linear_idx)) {
      auto t = ref_pool.insert(
          {linear_idx,
           std::make_unique<host_tensor_reference<Scalar>>(xt::adapt(
               m_data, size(), xt::no_ownership(), m_shape)[indices])});
    }

    return *ref_pool[linear_idx];
  }

  /// @brief Get the rank of the tensor
  /// @return The rank (number of dimensions) of the tensor
  std::size_t rank() const override { return m_shape.size(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const override {
    if (rank() == 0)
      return 0;
    return std::accumulate(m_shape.begin(), m_shape.end(), 1,
                           std::multiplies<size_t>());
  }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const override { return m_shape; }

  void slice(const std::vector<slice> &slices,
             std::vector<Scalar> &result_data) const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);

    xt::xstrided_slice_vector xt_slices;
    xt_slices.reserve(slices.size());

    for (const auto &s : slices) {
      std::ptrdiff_t start = s.start ? *s.start : 0;
      std::ptrdiff_t stop = s.stop ? *s.stop : m_shape[xt_slices.size()];
      std::ptrdiff_t step = s.step ? *s.step : 1;
      xt_slices.push_back(xt::range(start, stop, step));
    }

    // Perform the slice
    auto sliced = xt::strided_view(x, xt_slices);

    // Resize result vector and copy data
    result_data.resize(sliced.size());
    std::copy(sliced.begin(), sliced.end(), result_data.begin());
  }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  Scalar &at(const std::vector<size_t> &indices) override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));

    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  const Scalar &at(const std::vector<size_t> &indices) const override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid constant tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));
    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Sum all elements of the tensor
  /// @return A scalar sum of all elements of the tensor
  Scalar sum_all() const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    return xt::sum(x)[0];
  }

  /// @brief Check if any values are non-zero
  /// @return Returns true if any value is truthy, false otherwise
  bool any() const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    bool result;
    // For non-complex types, use regular bool casting
    if constexpr (std::is_integral_v<Scalar>) {
      result = xt::any(x);
    }
    // For complex types, implement custom ny
    else {
      throw std::runtime_error("any() not supported on non-integral types.");
    }

    return result;
  }

  void elementwise_add(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const host_tensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = x + y;
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void
  elementwise_multiply(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const host_tensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = x * y;
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void elementwise_modulo(const details::tensor_impl<Scalar> *other,
                          details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const host_tensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    // For non-complex types, use regular modulo
    if constexpr (std::is_integral_v<Scalar>) {
      auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
      auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                         other_xt->shape());
      auto z = x % y;
      std::copy(z.begin(), z.end(), result_xt->data());
    }
    // For complex types, implement custom modulo
    else {
      throw std::runtime_error("modulo not supported on non-integral types.");
    }
  }

  void scalar_modulo(Scalar value,
                     details::tensor_impl<Scalar> *result) const override {
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    // For non-complex types, use regular modulo
    if constexpr (std::is_integral_v<Scalar>) {
      auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
      auto z = x % value;
      std::copy(z.begin(), z.end(), result_xt->data());
    }
    // For complex types, implement custom modulo
    else {
      throw std::runtime_error("modulo not supported on non-integral types.");
    }
  }

  void matrix_dot(const details::tensor_impl<Scalar> *other,
                  details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const host_tensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = xt::linalg::dot(x, y);
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void
  matrix_vector_product(const details::tensor_impl<Scalar> *vec,
                        details::tensor_impl<Scalar> *result) const override {
    auto *vec_xt = dynamic_cast<const host_tensor<Scalar> *>(vec);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!vec_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto v = xt::adapt(vec_xt->data(), vec_xt->size(), xt::no_ownership(),
                       vec_xt->shape());
    auto z = xt::linalg::dot(x, v);
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void matrix_transpose(details::tensor_impl<Scalar> *result) const override {
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto z = xt::transpose(x, {1, 0});
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  Scalar minimal_eigenvalue() const override {
    auto eigenvals = eigenvalues();
    return *std::min_element(eigenvals.begin(), eigenvals.end(),
                             [](const auto &a, const auto &b) {
                               return std::fabs(a) < std::fabs(b);
                             });
  }

  std::vector<Scalar> eigenvalues() const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);

    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      // For real types, use eigh which guarantees real eigenvalues
      auto eig_pair = xt::linalg::eigh(x);
      return std::vector<Scalar>(std::get<0>(eig_pair).begin(),
                                 std::get<0>(eig_pair).end());
    } else if constexpr (std::is_integral_v<Scalar>) {
      auto x_double = xt::cast<double>(x);
      auto eigenvals = xt::linalg::eigvals(x_double);

      // Convert back to integral type
      std::vector<Scalar> result;
      result.reserve(eigenvals.size());
      std::transform(eigenvals.begin(), eigenvals.end(),
                     std::back_inserter(result), [](const auto &val) {
                       return static_cast<Scalar>(std::round(std::real(val)));
                     });
      return result;
    } else {
      auto eigenvals = xt::linalg::eigvals(x);
      return std::vector<Scalar>(eigenvals.begin(), eigenvals.end());
    }
  }

  void eigenvectors(details::tensor_impl<Scalar> *result) const override {
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);
    if (!result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);

    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      // For real types, use eigh which guarantees real eigenvalues/vectors
      auto eig_pair = xt::linalg::eigh(x);
      auto eigenvecs = std::get<1>(eig_pair);
      std::copy(eigenvecs.begin(), eigenvecs.end(), result_xt->data());
    } else if constexpr (std::is_integral_v<Scalar>) {
      // For integral types, compute as double then convert back
      auto x_double = xt::cast<double>(x);
      auto eig_pair = xt::linalg::eigh(x_double);
      auto eigenvecs = xt::cast<Scalar>(std::get<1>(eig_pair));
      std::copy(eigenvecs.begin(), eigenvecs.end(), result_xt->data());
    } else {
      // For complex types, use regular eig
      auto eig_pair = xt::linalg::eig(x);
      auto eigenvecs = std::get<1>(eig_pair);
      std::copy(eigenvecs.begin(), eigenvecs.end(), result_xt->data());
    }
  }

  void kron(const details::tensor_impl<Scalar> *other,
            details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const host_tensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<host_tensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    if (other->rank() != 2)
      throw std::runtime_error("kron on host only supported for matrices.");
    if (rank() != 2)
      throw std::runtime_error("kron on host only supported for matrices.");

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());

    // Use xtensor's kron function
    auto z = xt::linalg::kron(x, y);
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  Scalar *data() override { return m_data; }
  const Scalar *data() const override { return m_data; }
  void dump() const override {
    std::cerr << xt::adapt(m_data, size(), xt::no_ownership(), m_shape) << '\n';
  }

  void fill_random() override {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Fill tensor with random values
    for (std::size_t i = 0; i < size(); i++) {
      if constexpr (std::is_same_v<Scalar, std::complex<double>> ||
                    std::is_same_v<Scalar, std::complex<float>>) {
        m_data[i] = Scalar(dist(gen), dist(gen));
      } else {
        m_data[i] = static_cast<Scalar>(dist(gen));
      }
    }
  }

  /// @brief Destructor for xtensor
  ~host_tensor() { delete m_data; }
};

template class host_tensor<std::complex<double>>;
template class host_tensor<std::complex<float>>;
template class host_tensor<int>;
template class host_tensor<uint8_t>;
template class host_tensor<double>;
template class host_tensor<float>;
template class host_tensor<std::size_t>;

} // namespace cudaq
