/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "simulator.h"

#include <condition_variable>
#include <mutex>
#include <queue>

namespace cudaq::traits {
/// QPUs that implement that batch_trait provide a way for 
/// execution policies to launch multiple kernel jobs at the 
/// same time. Implementations are free to parallelize this in a 
/// subtype specific way. 
template <typename Derived>
class batch_trait {
protected:
public:
  int get_num_qpus() { return crtp_cast<Derived>(this)->get_num_qpus(); }

  template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
  auto execute_batch(ExecutionPolicy &&policy, QuantumKernel &&kernel,
                     const std::vector<std::tuple<Args...>> &args)
       -> std::vector<typename ExecutionPolicy::result_type> {
    return crtp_cast<Derived>(this)->execute_batch(policy, kernel, args);
  }

  // Deduction helper that accepts initializer list
  template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
  auto execute_batch(ExecutionPolicy &&policy, QuantumKernel &&kernel,
                     std::initializer_list<std::tuple<Args...>> task_args)
      -> std::vector<typename ExecutionPolicy::result_type> {
    return execute_batch(std::forward<ExecutionPolicy>(policy),
                         std::forward<QuantumKernel>(kernel),
                         std::vector<std::tuple<Args...>>(task_args));
  }
};

} // namespace cudaq::traits

namespace cudaq {

template <typename T, typename = void>
struct has_batch_trait : std::false_type {};

// SFINAE-based detection for any batch_trait inheritance
template <typename T>
struct has_batch_trait<
    T, std::void_t<decltype(std::declval<typename T::parallel_trait_tag>())>>
    : std::true_type {};

#ifndef CUDAQ_NO_STD20
// Alternative: Check for specific methods that parallel traits provide
template <typename T>
constexpr bool has_batch_trait_v = requires(T t) {
  { t.get_num_qpus() } -> std::convertible_to<int>;
  // Add other parallel trait method requirements
};

template <typename T>
concept BatchQPU = requires { has_batch_trait_v<T>; };
#endif 

} // namespace cudaq
