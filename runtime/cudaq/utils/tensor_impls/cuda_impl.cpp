#include "cuda_kernels.h"
#include "../tensor_impl.h"
#include <complex>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
namespace cudaq {

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

public:
  using scalar_type = Scalar;

  // Constructor taking raw device pointer
  cuda_tensor(Scalar *device_ptr, const std::vector<std::size_t> &shape,
              bool take_ownership = true)
      : d_data(device_ptr), m_shape(shape), owns_data(take_ownership) {}

  // Constructor allocating new device memory
  cuda_tensor(const std::vector<std::size_t> &shape) : m_shape(shape) {
    auto size = total_size();
    cudaMalloc(&d_data, size * sizeof(Scalar));
  }

  ~cuda_tensor() {
    if (owns_data && d_data) {
      cudaFree(d_data);
    }
  }

  std::size_t rank() const override { return m_shape.size(); }
  std::size_t size() const override { return total_size(); }
  std::vector<std::size_t> shape() const override { return m_shape; }

  Scalar *data() override { return d_data; }
  const Scalar *data() const override { return d_data; }

  void elementwise_add(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    // Launch CUDA kernel for elementwise addition
    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;
    kernels::elementwise_add_kernel(threads, blocks, d_data, other_cuda->data(),
                                    result_cuda->data(), total_size());
  }

  void matrix_dot(const details::tensor_impl<Scalar> *other,
                  details::tensor_impl<Scalar> *result) const override {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    // Use cuBLAS for matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);

    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_shape[1],
                other_cuda->shape()[1], m_shape[0], &alpha, d_data, m_shape[1],
                other_cuda->data(), other_cuda->shape()[1], &beta,
                result_cuda->data(), result_cuda->shape()[1]);

    cublasDestroy(handle);
  }

  void slice(const std::vector<slice> &slices,
             std::vector<Scalar> &result_data) const override {
    // Allocate temporary device memory for result
    // Scalar *d_result;
    // auto slice_size = compute_slice_size(slices);
    // cudaMalloc(&d_result, slice_size * sizeof(Scalar));

    // // Launch kernel to perform slicing
    // dim3 block(256);
    // dim3 grid((slice_size + block.x - 1) / block.x);
    // slice_kernel<<<grid, block>>>(d_data, d_result, m_shape.data(),
    //                               slices.data(), m_shape.size());

    // // Copy result back to host
    // result_data.resize(slice_size);
    // cudaMemcpy(result_data.data(), d_result, slice_size * sizeof(Scalar),
    //            cudaMemcpyDeviceToHost);
    // cudaFree(d_result);
  }

  Scalar sum_all() const override {
    thrust::device_ptr<Scalar> dev_ptr(d_data);
    return thrust::reduce(dev_ptr, dev_ptr + total_size());
  }

  bool any() const override {
    thrust::device_ptr<Scalar> dev_ptr(d_data);
    return thrust::any_of(
        dev_ptr, dev_ptr + total_size(),
        [] __device__(const Scalar &x) { return x != Scalar{}; });
  }

  void matrix_vector_product(const tensor_impl<Scalar> *vec,
                             tensor_impl<Scalar> *result) const override {
    auto *vec_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(vec);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!vec_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;

    cublasSgemv(handle, CUBLAS_OP_N, m_shape[0], m_shape[1], &alpha, d_data,
                m_shape[1], vec_cuda->data(), 1, &beta, result_cuda->data(), 1);

    cublasDestroy(handle);
  }

  void matrix_transpose(tensor_impl<Scalar> *result) const override {
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);
    if (!result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;

    // Use geam for matrix transpose
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_shape[1], m_shape[0],
                &alpha, d_data, m_shape[0], &beta, nullptr, m_shape[1],
                result_cuda->data(), m_shape[1]);

    cublasDestroy(handle);
  }

  Scalar minimal_eigenvalue() const override {
    std::vector<Scalar> eigs = eigenvalues();
    return *std::min_element(
        eigs.begin(), eigs.end(),
        [](const auto &a, const auto &b) { return std::abs(a) < std::abs(b); });
  }

  std::vector<Scalar> eigenvalues() const override {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Allocate workspace
    int work_size = 0;
    cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                CUBLAS_FILL_MODE_LOWER, m_shape[0], d_data,
                                m_shape[1], &work_size);

    Scalar *d_work;
    cudaMalloc(&d_work, work_size * sizeof(Scalar));

    // Allocate eigenvalues array
    Scalar *d_eigenvals;
    cudaMalloc(&d_eigenvals, m_shape[0] * sizeof(Scalar));

    // Compute eigenvalues
    int *dev_info;
    cudaMalloc(&dev_info, sizeof(int));

    cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                     m_shape[0], d_data, m_shape[1], d_eigenvals, d_work,
                     work_size, dev_info);

    // Copy results back to host
    std::vector<Scalar> eigenvals(m_shape[0]);
    cudaMemcpy(eigenvals.data(), d_eigenvals, m_shape[0] * sizeof(Scalar),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_work);
    cudaFree(d_eigenvals);
    cudaFree(dev_info);
    cusolverDnDestroy(handle);

    return eigenvals;
  }

  void eigenvectors(tensor_impl<Scalar> *result) const override {
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);
    if (!result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Similar to eigenvalues() but with CUSOLVER_EIG_MODE_VECTOR
    // and copying the eigenvectors to result_cuda->data()

    cusolverDnDestroy(handle);
  }

  template <typename Scalar>
  Scalar &at(const std::vector<size_t> &indices) {
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

  template <typename Scalar>
  const Scalar &at(const std::vector<size_t> &indices) const {
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

  template <typename Scalar>
  void elementwise_multiply(const tensor_impl<Scalar> *other,
                            tensor_impl<Scalar> *result) const {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;

    kernels::elementwise_multiply_kernel(threads, blocks, d_data,
                                         other_cuda->data(),
                                         result_cuda->data(), total_size());
  }

  template <typename Scalar>
  void scalar_modulo(Scalar value, tensor_impl<Scalar> *result) const {
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;

    kernels::scalar_modulo_kernel(threads, blocks, d_data, value,
                                  result_cuda->data(), total_size());
  }

  template <typename Scalar>
  void elementwise_modulo(const tensor_impl<Scalar> *other,
                          tensor_impl<Scalar> *result) const {
    auto *other_cuda = dynamic_cast<const cuda_tensor<Scalar> *>(other);
    auto *result_cuda = dynamic_cast<cuda_tensor<Scalar> *>(result);

    if (!other_cuda || !result_cuda) {
      throw std::runtime_error("CUDA tensor operation requires CUDA tensors");
    }

    const int threads = 256;
    const int blocks = (total_size() + threads - 1) / threads;

    kernels::elementwise_modulo_kernel(threads, blocks, d_data,
                                       other_cuda->data(), result_cuda->data(),
                                       total_size());
  }

  void dump() const override {
    std::vector<Scalar> host_data(total_size());
    cudaMemcpy(host_data.data(), d_data, total_size() * sizeof(Scalar),
               cudaMemcpyDeviceToHost);

    std::cerr << "CUDA Tensor [" << fmt::join(m_shape, "x") << "]:\n";
    // Print the data in a formatted way based on shape
    for (size_t i = 0; i < total_size(); ++i) {
      std::cerr << host_data[i] << " ";
      if ((i + 1) % m_shape.back() == 0)
        std::cerr << "\n";
    }
  }

  static constexpr auto ScalarAsString = cudaq::type_to_string<Scalar>();

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      cuda_tensor<Scalar>, std::string("cuda") + std::string(ScalarAsString),
      static std::unique_ptr<cudaq::details::tensor_impl<Scalar>> create(
          const Scalar *d, const std::vector<std::size_t> s) {
        return std::make_unique<cuda_tensor<Scalar>>(d, s);
      })
};

#define INSTANTIATE_REGISTRY_TENSOR_IMPL(TYPE)                                 \
  INSTANTIATE_REGISTRY(cudaq::details::tensor_impl<TYPE>, const TYPE *,        \
                       const std::vector<std::size_t>)

// Register the CUDA tensor types
INSTANTIATE_REGISTRY_TENSOR_IMPL(std::complex<double>)
INSTANTIATE_REGISTRY_TENSOR_IMPL(std::complex<float>)
INSTANTIATE_REGISTRY_TENSOR_IMPL(int)
INSTANTIATE_REGISTRY_TENSOR_IMPL(uint8_t)
INSTANTIATE_REGISTRY_TENSOR_IMPL(double)
INSTANTIATE_REGISTRY_TENSOR_IMPL(float)
INSTANTIATE_REGISTRY_TENSOR_IMPL(std::size_t)

// Static registration for each type
template <>
const bool cuda_tensor<std::complex<double>>::registered_ =
    cuda_tensor<std::complex<double>>::register_type();
template <>
const bool cuda_tensor<std::complex<float>>::registered_ =
    cuda_tensor<std::complex<float>>::register_type();
template <>
const bool cuda_tensor<int>::registered_ = cuda_tensor<int>::register_type();
template <>
const bool cuda_tensor<uint8_t>::registered_ =
    cuda_tensor<uint8_t>::register_type();
template <>
const bool cuda_tensor<double>::registered_ =
    cuda_tensor<double>::register_type();
template <>
const bool cuda_tensor<float>::registered_ =
    cuda_tensor<float>::register_type();
template <>
const bool cuda_tensor<std::size_t>::registered_ =
    cuda_tensor<std::size_t>::register_type();

} // namespace cudaq