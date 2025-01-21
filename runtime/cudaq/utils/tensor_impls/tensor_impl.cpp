/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cuda_tensor.cpp"
#include "host_tensor.cpp"

namespace cudaq::details {
template <typename Scalar>
std::unique_ptr<tensor_impl<Scalar>>
tensor_impl<Scalar>::create(tensor_memory mem_type, Scalar *data,
                            const std::vector<std::size_t> &shape) {
  switch (mem_type) {
  case tensor_memory::host:
    return std::make_unique<host_tensor<Scalar>>(data, shape);
  case tensor_memory::cuda:
    return std::make_unique<cuda_tensor<Scalar>>(data, shape);
  default:
    throw std::runtime_error("Invalid tensor memory type");
  }
}

template <typename Scalar>
std::unique_ptr<tensor_impl<Scalar>>
tensor_impl<Scalar>::create(tensor_memory mem_type,
                            const std::vector<std::size_t> &shape) {
  switch (mem_type) {
  case tensor_memory::host:
    return std::make_unique<host_tensor<Scalar>>(shape);
  case tensor_memory::cuda:
    return std::make_unique<cuda_tensor<Scalar>>(shape);
  default:
    throw std::runtime_error("Invalid tensor memory type");
  }
}

template class tensor_impl<std::complex<double>>;
template class tensor_impl<std::complex<float>>;
template class tensor_impl<int>;
template class tensor_impl<uint8_t>;
template class tensor_impl<double>;
template class tensor_impl<float>;
template class tensor_impl<std::size_t>;

} // namespace cudaq::details