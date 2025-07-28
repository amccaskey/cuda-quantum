/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <tuple>

namespace cudaq {

template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<std::size_t,
                           std::tuple_size<typename std::remove_reference<
                               TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each(TupleType &&t, FunctionType f,
                    std::integral_constant<std::size_t, I>) {
  f(std::get<I>(t));
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<std::size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<std::size_t, 0>());
}

template <typename TupleType, typename FunctionType>
void tuple_for_each_with_idx(
    TupleType &&, FunctionType,
    std::integral_constant<std::size_t,
                           std::tuple_size<typename std::remove_reference<
                               TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each_with_idx(TupleType &&t, FunctionType f,
                             std::integral_constant<std::size_t, I>) {
  f(std::get<I>(t), std::integral_constant<std::size_t, I>());
  tuple_for_each_with_idx(std::forward<TupleType>(t), f,
                          std::integral_constant<std::size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each_with_idx(TupleType &&t, FunctionType f) {
  tuple_for_each_with_idx(std::forward<TupleType>(t), f,
                          std::integral_constant<std::size_t, 0>());
}

} // namespace cudaq
