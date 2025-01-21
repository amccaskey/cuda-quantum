/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cstddef>

#include "cuComplex.h"
#include "cuda_kernels.h"

#include "cutensor.h"

#include "../tensor_impl.h"
#include <complex>
#include <tuple>

#include "common/FmtCore.h"

#include "cuda_runtime_api.h"

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(fmt::format("[tensor] %{} in {} (line {})",     \
                                           cudaGetErrorString(err),            \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  };

#define CUTENSOR_CHECK(x)                                                      \
  {                                                                            \
    cutensorStatus_t err = x;                                                  \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      throw std::runtime_error(std::string("cuTENSOR error: ") +               \
                               cutensorGetErrorString(err));                   \
    }                                                                          \
  }

namespace cudaq::details {
template <typename T, typename... Types>
inline constexpr bool is_any_of = (std::is_same_v<T, Types> || ...);

void __deallocate_cuda(void *ptr) { cudaFree(ptr); }
void __allocate_cuda(void **ptr, std::size_t size) {
  printf("Allocate: %lu\n", size);
  cudaMalloc((void **)ptr, size);
}

const char *__tensor_memory_or_pointer_to_string(void *ptr) {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.type > 1 ? "cuda_tensor" : "host_tensor";
}
} // namespace cudaq::details

namespace cudaq {
using namespace details;

template <typename Scalar>
class cuda_tensor_reference : public tensor_reference<Scalar> {
private:
  Scalar *device_ptr;
  size_t index;

public:
  cuda_tensor_reference(Scalar *ptr, size_t idx)
      : device_ptr(ptr), index(idx) {}

  cuda_tensor_reference &operator=(const Scalar &value) override {
    cudaMemcpy(device_ptr + index, &value, sizeof(Scalar),
               cudaMemcpyHostToDevice);
    return *this;
  }

  operator Scalar() const override {
    Scalar host_value;
    cudaMemcpy(&host_value, device_ptr + index, sizeof(Scalar),
               cudaMemcpyDeviceToHost);
    return host_value;
  }
};

template <typename Scalar>
class cuda_tensor : public cudaq::details::tensor_impl<Scalar> {
private:
  Scalar *d_data = nullptr; // Device pointer
  std::vector<std::size_t> m_shape;
  bool owns_data = true;
  mutable size_t last_accessed_index;
  mutable Scalar last_accessed_value;
  // Helper to compute total size
  std::size_t total_size() const {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1,
                           std::multiplies<std::size_t>());
  }
  std::tuple<cutensorDataType_t, cutensorComputeDescriptor_t>
  get_cutensor_types() const {
    if constexpr (std::is_same_v<Scalar, float>) {
      return {CUTENSOR_R_32F, CUTENSOR_COMPUTE_DESC_32F};
    } else if constexpr (std::is_same_v<Scalar, double>) {
      return {CUTENSOR_R_64F, CUTENSOR_COMPUTE_DESC_64F};
    } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
      return {CUTENSOR_C_64F, CUTENSOR_COMPUTE_DESC_64F};
    } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
      return {CUTENSOR_C_32F, CUTENSOR_COMPUTE_DESC_32F};
    } else if constexpr (std::is_same_v<Scalar, int>) {
      return {CUTENSOR_R_32I, CUTENSOR_COMPUTE_DESC_32F};
    } else {
      throw std::runtime_error("Unsupported data type for cuTENSOR reduction");
    }
  }

  // Pool of references indexed by position
  std::unordered_map<std::size_t,
                     std::unique_ptr<cuda_tensor_reference<Scalar>>>
      ref_pool;

public:
  using scalar_type = Scalar;

  // Constructor taking raw device pointer
  cuda_tensor(Scalar *device_ptr, const std::vector<std::size_t> &shape,
              bool take_ownership = true)
      : m_shape(shape), owns_data(take_ownership) {
    cudaPointerAttributes attributes;
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, d_data));
    if (attributes.type < 2) {
      cudaFree(d_data);
      cudaMalloc((void **)&d_data, size() * sizeof(Scalar));
      cudaMemcpy(d_data, device_ptr, size() * sizeof(Scalar),
                 cudaMemcpyHostToDevice);
    } else {
      d_data = device_ptr;
    }
  }

  // Constructor allocating new device memory
  cuda_tensor(const std::vector<std::size_t> &shape) : m_shape(shape) {
    auto size = total_size();
    cudaFree(d_data);
    cudaMalloc((void **)&d_data, size * sizeof(Scalar));
    cudaMemset(d_data, 0, size * sizeof(Scalar)); // Initialize to zeros
  }

  ~cuda_tensor() {
    if (owns_data && d_data) {
      cudaFree(d_data);
    }
  }

  tensor_reference<Scalar> &
  extract(const std::vector<size_t> &indices) override {
    size_t linear_idx = 0;
    size_t stride = 1;

    for (int i = indices.size() - 1; i >= 0; i--) {
      if (indices[i] >= m_shape[i])
        throw std::runtime_error("Index out of bounds");
      linear_idx += indices[i] * stride;
      stride *= m_shape[i];
    }

    // Create new reference if needed
    if (!ref_pool.contains(linear_idx)) {
      ref_pool.insert(
          {linear_idx, std::make_unique<cuda_tensor_reference<Scalar>>(
                           d_data, linear_idx)});
    }

    return *ref_pool[linear_idx];
  }

  std::size_t rank() const override { return m_shape.size(); }
  std::size_t size() const override { return total_size(); }
  std::vector<std::size_t> shape() const override { return m_shape; }

  Scalar *data() override { return d_data; }
  const Scalar *data() const override { return d_data; }

  void elementwise_add(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    // Dynamic cast to get cuda_tensor implementations
    const auto *cuda_other = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *cuda_result = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!cuda_other || !cuda_result) {
      throw std::runtime_error(
          "Invalid tensor implementation type for CUDA elementwise add");
    }
    // Create cuTENSOR handle
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorCreate(&handle));

    // Determine data type
    auto [dataType, computeDesc] = get_cutensor_types();

    // Convert shape to int64_t vector
    std::vector<int64_t> extent(this->shape().size());
    for (std::size_t i = 0; i < extent.size(); i++)
      extent[i] = shape()[i];

    // Create tensor descriptors
    cutensorTensorDescriptor_t descA, descB, descC;
    uint32_t const alignmentRequirement = 128; // Typical alignment requirement

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descA, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descB, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descC, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    // Create mode array (equivalent to axes)
    std::vector<int> modes(extent.size());
    std::iota(modes.begin(), modes.end(), 0);

    // Create operation descriptor
    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreateElementwiseBinary(
        handle, &desc, descA, modes.data(), CUTENSOR_OP_IDENTITY, descB,
        modes.data(), CUTENSOR_OP_IDENTITY, descC, modes.data(),
        CUTENSOR_OP_ADD, computeDesc));

    // Create plan preference
    cutensorPlanPreference_t planPref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(
        handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

    // Create plan
    cutensorPlan_t plan;
    CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, desc, planPref,
                                      0 // workspace limit
                                      ));

    // Set scalar constants
    Scalar alpha = 1.0;
    Scalar beta = 0.0;

    // Execute the operation
    CUTENSOR_CHECK(cutensorElementwiseBinaryExecute(
        handle, plan, &alpha, this->data(), &alpha, cuda_other->data(),
        cuda_result->data(),
        nullptr // stream
        ));

    // Cleanup
    CUTENSOR_CHECK(cutensorDestroyPlan(plan));
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(planPref));
    CUTENSOR_CHECK(cutensorDestroyOperationDescriptor(desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descB));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));
    CUTENSOR_CHECK(cutensorDestroy(handle));
  }

  void fill_random() override {
    // Allocate temporary host buffer
    std::vector<Scalar> host_data(size());

    // Generate random values on host
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (std::size_t i = 0; i < size(); i++) {
      if constexpr (std::is_same_v<Scalar, std::complex<double>> ||
                    std::is_same_v<Scalar, std::complex<float>>) {
        host_data[i] = Scalar(dist(gen), dist(gen));
      } else {
        host_data[i] = static_cast<Scalar>(dist(gen));
      }
    }

    cudaFree(d_data);
    cudaMalloc((void **)&d_data, size() * sizeof(Scalar));
    // Copy to device
    cudaMemcpy(d_data, host_data.data(), size() * sizeof(Scalar),
               cudaMemcpyHostToDevice);
  }

  void matrix_dot(const details::tensor_impl<Scalar> *other,
                  details::tensor_impl<Scalar> *result) const override {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    // Create cuTENSOR handle
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorCreate(&handle));

    // Define modes for the matrices
    const std::vector<int> modeA = {'i', 'k'};
    const std::vector<int> modeB = {'k', 'j'};
    const std::vector<int> modeC = {'i', 'j'};

    // Create tensor descriptors
    cutensorTensorDescriptor_t descA, descB, descC;
    const uint32_t alignment = 128;

    std::vector<int64_t> extent(this->shape().size());
    for (std::size_t i = 0; i < extent.size(); i++)
      extent[i] = shape()[i];

    std::vector<int64_t> extentOther(other->shape().size());
    for (std::size_t i = 0; i < extentOther.size(); i++)
      extentOther[i] = other->shape()[i];

    std::vector<int64_t> extentRes(result_cuda->shape().size());
    for (std::size_t i = 0; i < extentRes.size(); i++)
      extentRes[i] = result_cuda->shape()[i];

    // Get compute type based on scalar type
    // Determine data type
    auto [dataType, computeDesc] = get_cutensor_types();

    // we assume row major ordering
    std::vector<int64_t> stridesA(2);
    stridesA[1] = 1;                // Last dimension stride is 1
    stridesA[0] = this->shape()[1]; // First dimension stride is columns

    std::vector<int64_t> stridesB(2);
    stridesB[1] = 1;
    stridesB[0] = other_cuda->shape()[1];

    std::vector<int64_t> stridesC(2);
    stridesC[1] = 1;
    stridesC[0] = result_cuda->shape()[1];

    // Create descriptors
    CUTENSOR_CHECK(
        cutensorCreateTensorDescriptor(handle, &descA, 2, extent.data(),
                                       stridesA.data(), dataType, alignment));

    CUTENSOR_CHECK(
        cutensorCreateTensorDescriptor(handle, &descB, 2, extentOther.data(),
                                       stridesB.data(), dataType, alignment));

    CUTENSOR_CHECK(
        cutensorCreateTensorDescriptor(handle, &descC, 2, extentRes.data(),
                                       stridesC.data(), dataType, alignment));

    // Create contraction descriptor
    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreateContraction(
        handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descB,
        modeB.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(),
        CUTENSOR_OP_IDENTITY, descC, modeC.data(), computeDesc));

    // Create plan
    cutensorPlanPreference_t planPref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(
        handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

    cutensorPlan_t plan;
    CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, desc, planPref, 0));

    // Execute contraction
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    CUTENSOR_CHECK(cutensorContract(
        handle, plan, &alpha, this->data(), other_cuda->data(), &beta,
        result_cuda->data(), result_cuda->data(), nullptr, 0, 0));

    // Cleanup
    CUTENSOR_CHECK(cutensorDestroyPlan(plan));
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(planPref));
    CUTENSOR_CHECK(cutensorDestroyOperationDescriptor(desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descB));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));
    CUTENSOR_CHECK(cutensorDestroy(handle));
  }

  void slice(const std::vector<slice> &slices,
             std::vector<Scalar> &result_data) const override {}

  Scalar sum_all() const override {
    // Create cuTENSOR handle
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorCreate(&handle));

    // Determine data type and compute descriptor based on Scalar type
    auto [dataType, computeDesc] = get_cutensor_types();

    // Convert shape to int64_t vector
    std::vector<int64_t> extent(this->shape().size());
    for (std::size_t i = 0; i < extent.size(); i++)
      extent[i] = shape()[i];

    // Create tensor descriptor for input
    cutensorTensorDescriptor_t descA;
    uint32_t const alignmentRequirement = 128;
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descA, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    // Create descriptor for output (scalar)
    cutensorTensorDescriptor_t descC;
    int64_t scalarExtent = 1;
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descC, 0, &scalarExtent, nullptr, dataType,
        alignmentRequirement));

    // Create mode array for reduction
    std::vector<int> modes(extent.size());
    std::iota(modes.begin(), modes.end(), 0);
    // Create mode array for reduction
    std::vector<int32_t> modeA(extent.size());
    std::iota(modeA.begin(), modeA.end(), 0);

    // Create operation descriptor for reduction
    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreateReduction(handle, &desc, descA, modeA.data(),
                                           CUTENSOR_OP_IDENTITY, descC, nullptr,
                                           CUTENSOR_OP_IDENTITY, descC, nullptr,
                                           CUTENSOR_OP_ADD, computeDesc));

    // Create plan
    cutensorPlanPreference_t planPref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(
        handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSizeEstimate = 0;
    CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle, desc, planPref,
                                                 CUTENSOR_WORKSPACE_DEFAULT,
                                                 &workspaceSizeEstimate));

    cutensorPlan_t plan;
    CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, desc, planPref,
                                      workspaceSizeEstimate));

    // Allocate device memory for result
    Scalar *d_result;
    cudaMalloc(&d_result, sizeof(Scalar));
    void *workspace = nullptr;
    if (workspaceSizeEstimate > 0) {
      cudaMalloc(&workspace, workspaceSizeEstimate);
    }

    // Set scalar constants
    Scalar alpha = 1.0;
    Scalar beta = 0.0;

    CUTENSOR_CHECK(cutensorReduce(handle, plan, &alpha, d_data, &beta, d_result,
                                  d_result, workspace, workspaceSizeEstimate,
                                  nullptr)); // stream

    // Copy result back to host
    Scalar result;
    cudaMemcpy(&result, d_result, sizeof(Scalar), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_result);
    CUTENSOR_CHECK(cutensorDestroyPlan(plan));
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(planPref));
    CUTENSOR_CHECK(cutensorDestroyOperationDescriptor(desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));
    CUTENSOR_CHECK(cutensorDestroy(handle));

    return result;
  }

  bool any() const override {

    auto sum = sum_all();
    if (std::fabs(sum) < 1e-12)
      return false;

    return true;
  }

  void matrix_vector_product(const tensor_impl<Scalar> *vec,
                             tensor_impl<Scalar> *result) const override {
    auto *vec_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(vec);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!vec_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }
  }

  void matrix_transpose(tensor_impl<Scalar> *result) const override {
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);
    if (!result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }
  }

  Scalar minimal_eigenvalue() const override {
    std::vector<Scalar> eigs = eigenvalues();
    return Scalar();
  }

  std::vector<Scalar> eigenvalues() const override { return {}; }

  void eigenvectors(tensor_impl<Scalar> *result) const override {}

  Scalar &at(const std::vector<size_t> &indices) override {
    // Calculate linear index
    size_t linear_idx = 0;
    size_t stride = 1;

    for (int i = indices.size() - 1; i >= 0; i--) {
      if (indices[i] >= m_shape[i]) {
        throw std::runtime_error("Index out of bounds");
      }
      linear_idx += indices[i] * stride;
      stride *= m_shape[i];
    }

    // Allocate temporary host storage
    Scalar host_value;

    // Copy single element from device to host
    cudaMemcpy(&host_value, d_data + linear_idx, sizeof(Scalar),
               cudaMemcpyDeviceToHost);

    // Store the index for later writeback
    last_accessed_index = linear_idx;
    last_accessed_value = host_value;

    return last_accessed_value;
  }

  const Scalar &at(const std::vector<size_t> &indices) const override {
    // Calculate linear index
    size_t linear_idx = 0;
    size_t stride = 1;

    for (int i = indices.size() - 1; i >= 0; i--) {
      if (indices[i] >= m_shape[i]) {
        throw std::runtime_error("Index out of bounds");
      }
      linear_idx += indices[i] * stride;
      stride *= m_shape[i];
    }

    // Allocate temporary host storage
    Scalar host_value;

    // Copy single element from device to host
    cudaMemcpy(&host_value, d_data + linear_idx, sizeof(Scalar),
               cudaMemcpyDeviceToHost);

    last_accessed_value = host_value;
    return last_accessed_value;
  }

  void elementwise_multiply(const tensor_impl<Scalar> *other,
                            tensor_impl<Scalar> *result) const override {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    // Create cuTENSOR handle
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorCreate(&handle));

    // Determine data type
    auto [dataType, computeDesc] = get_cutensor_types();

    // Convert shape to int64_t vector
    std::vector<int64_t> extent(this->shape().size());
    for (std::size_t i = 0; i < extent.size(); i++)
      extent[i] = shape()[i];

    // Create tensor descriptors
    cutensorTensorDescriptor_t descA, descB, descC;
    uint32_t const alignmentRequirement = 128; // Typical alignment requirement

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descA, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descB, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
        handle, &descC, extent.size(), extent.data(),
        nullptr, // Use default strides
        dataType, alignmentRequirement));

    // Create mode array (equivalent to axes)
    std::vector<int> modes(extent.size());
    std::iota(modes.begin(), modes.end(), 0);

    // Create operation descriptor
    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreateElementwiseBinary(
        handle, &desc, descA, modes.data(), CUTENSOR_OP_IDENTITY, descB,
        modes.data(), CUTENSOR_OP_IDENTITY, descC, modes.data(),
        CUTENSOR_OP_MUL, computeDesc));

    // Create plan preference
    cutensorPlanPreference_t planPref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(
        handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

    // Create plan
    cutensorPlan_t plan;
    CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, desc, planPref,
                                      0 // workspace limit
                                      ));

    // Set scalar constants
    Scalar alpha = 1.0;
    Scalar beta = 0.0;

    // Execute the operation
    CUTENSOR_CHECK(cutensorElementwiseBinaryExecute(
        handle, plan, &alpha, this->data(), &alpha, other_cuda->data(),
        result_cuda->data(),
        nullptr // stream
        ));

    // Cleanup
    CUTENSOR_CHECK(cutensorDestroyPlan(plan));
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(planPref));
    CUTENSOR_CHECK(cutensorDestroyOperationDescriptor(desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descB));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));
    CUTENSOR_CHECK(cutensorDestroy(handle));
  }

  void scalar_modulo(Scalar value, tensor_impl<Scalar> *result) const override {
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;

    // kernels::scalar_modulo_kernel(threads, blocks, d_data, value,
    //                               result_cuda->data(), total_size());
  }

  void elementwise_modulo(const tensor_impl<Scalar> *other,
                          tensor_impl<Scalar> *result) const override {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
      auto *d = reinterpret_cast<cuDoubleComplex *>(d_data);
      auto *e = reinterpret_cast<cuDoubleComplex *>(
          const_cast<Scalar *>(other_cuda->data()));
      auto *f = reinterpret_cast<cuDoubleComplex *>(result_cuda->data());
      return kernels::elementwise_modulo_kernel<cuDoubleComplex>(
          threads, blocks, d, e, f, total_size());
    }

    if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
      auto *d = reinterpret_cast<cuFloatComplex *>(d_data);
      auto *e = reinterpret_cast<cuFloatComplex *>(
          const_cast<Scalar *>(other_cuda->data()));
      auto *f = reinterpret_cast<cuFloatComplex *>(result_cuda->data());
      return kernels::elementwise_modulo_kernel<cuFloatComplex>(
          threads, blocks, d, e, f, total_size());
    }

    kernels::elementwise_modulo_kernel(threads, blocks, d_data,
                                       other_cuda->data(), result_cuda->data(),
                                       total_size());
  }

  void kron(const tensor_impl<Scalar> *other,
            tensor_impl<Scalar> *result) const override {}

  void dump() const override {
    std::vector<Scalar> host_data(total_size());
    cudaMemcpy(host_data.data(), d_data, total_size() * sizeof(Scalar),
               cudaMemcpyDeviceToHost);

    std::cerr << "CUDA Tensor [" << fmt::format("{}", fmt::join(m_shape, ","))
              << "]:\n";
    // Print the data in a formatted way based on shape
    for (size_t i = 0; i < total_size(); ++i) {
      if constexpr (details::is_any_of<Scalar, cuDoubleComplex,
                                       cuFloatComplex>) {
        std::cerr << host_data[i].x << "+" << host_data[i].y << "j ";
      } else if constexpr (details::is_any_of<Scalar, std::complex<double>,
                                              std::complex<float>>) {
        std::cerr << host_data[i].real() << "+" << host_data[i].imag() << "j ";
      } else {
        std::cerr << host_data[i] << " ";
      }
      if ((i + 1) % m_shape.back() == 0)
        std::cerr << "\n";
    }
  }
};

template class cuda_tensor<std::complex<double>>;
template class cuda_tensor<float>;
template class cuda_tensor<std::complex<float>>;
template class cuda_tensor<int>;
template class cuda_tensor<uint8_t>;
template class cuda_tensor<double>;
template class cuda_tensor<std::size_t>;

} // namespace cudaq