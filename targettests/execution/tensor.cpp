/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20

// clang-format off
// RUN: clang++ -std=c++20 \
// RUN:   -I %p/../../runtime -I %p/../../tpls/fmt/include \
// RUN:   -I %p/../../tpls/xtensor/include -I %p/../../tpls/xtl/include \
// RUN:   -c %p/../../runtime/cudaq/utils/details/impls/xtensor_impl.cpp -o %t.o
// RUN: nvq++ -std=c++20 --enable-mlir %s %t.o -o %t.x  && %t.x |& FileCheck %s
// clang-format on

#include <complex>
#include <cudaq.h>
#include <iostream>
#include <vector>

#include "cudaq.h"
#include "cudaq/utils/tensor.h"

void test_register() {
  auto registeredNames = cudaq::details::tensor_impl<>::get_registered();
  assert(registeredNames.size() == 1);
  assert(std::find(registeredNames.begin(), registeredNames.end(),
                   "xtensorcomplex<double>") != registeredNames.end());
  std::cout << "xtensorcomplex<double> is registered" << std::endl;
}

// CHECK: xtensorcomplex<double> is registered

void test_mult() {
  {
    std::complex<double> d[] = {1.0, 0.0, 0.0, 1.0};
    cudaq::tensor a(d, {2, 2});
    cudaq::tensor b(d, {2, 2});

    auto c = a * b;
    c.dump();
  }
}

// CHECK: {{\{*}}1.+0.i,  0.+0.i},
// CHECK: { 0.+0.i,  1.+0.i{{\}*}}

void test_constructors_and_access() {
  {
    std::vector<std::size_t> shape = {2, 2};
    std::complex<double> *data = new std::complex<double>[4];
    data[0] = {1.0, 0.0};
    data[1] = {0.0, 1.0};
    data[2] = {0.0, -1.0};
    data[3] = {1.0, 0.0};

    cudaq::tensor t(data, shape);

    std::cout << "rank: " << t.rank() << std::endl;
    std::cout << "size: " << t.size() << std::endl;
    std::cout << "shape: ";
    for (auto &e : t.shape())
      std::cout << e << ", ";
    std::cout << std::endl;

    std::cout << "t.at({0, 0}) = " << t.at({0, 0}) << std::endl;
    std::cout << "t.at({0, 1}) = " << t.at({0, 1}) << std::endl;
    std::cout << "t.at({1, 0}) = " << t.at({1, 0}) << std::endl;
    std::cout << "t.at({1, 1}) = " << t.at({1, 1}) << std::endl;
  }
}

// CHECK: rank: 2
// CHECK: size: 4
// CHECK: shape: 2, 2,
// CHECK: t.at({0, 0}) = (1,0)
// CHECK: t.at({0, 1}) = (0,1)
// CHECK: t.at({1, 0}) = (0,-1)
// CHECK: t.at({1, 1}) = (1,0)

void test_copy() {
  {
    std::vector<std::size_t> shape = {2, 2};
    std::vector<std::complex<double>> data = {
        {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
    cudaq::tensor t(shape);

    t.copy(data.data(), shape);

    std::cout << "t.at({0, 0}) = " << t.at({0, 0}) << std::endl;
    std::cout << "t.at({0, 1}) = " << t.at({0, 1}) << std::endl;
    std::cout << "t.at({1, 0}) = " << t.at({1, 0}) << std::endl;
    std::cout << "t.at({1, 1}) = " << t.at({1, 1}) << std::endl;
  }
}

// CHECK: t.at({0, 0}) = (1,0)
// CHECK: t.at({0, 1}) = (0,1)
// CHECK: t.at({1, 0}) = (0,-1)
// CHECK: t.at({1, 1}) = (1,0)

void test_borrow() {
  {
    std::vector<std::size_t> shape = {2, 2};
    std::vector<std::complex<double>> data = {
        {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
    cudaq::tensor t(shape);

    t.borrow(data.data(), shape);

    std::cout << "t.at({0, 0}) = " << t.at({0, 0}) << std::endl;
    std::cout << "t.at({0, 1}) = " << t.at({0, 1}) << std::endl;
    std::cout << "t.at({1, 0}) = " << t.at({1, 0}) << std::endl;
    std::cout << "t.at({1, 1}) = " << t.at({1, 1}) << std::endl;
  }
}

// CHECK: t.at({0, 0}) = (1,0)
// CHECK: t.at({0, 1}) = (0,1)
// CHECK: t.at({1, 0}) = (0,-1)
// CHECK: t.at({1, 1}) = (1,0)

void test_take() {
  {
    std::vector<std::size_t> shape = {2, 2};
    auto data = std::make_unique<std::complex<double>[]>(4);
    const std::vector<std::complex<double>> idata{
        {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
    std::copy(idata.begin(), idata.end(), data.get());
    cudaq::tensor t(shape);

    t.take(data, shape);

    std::cout << "t.at({0, 0}) = " << t.at({0, 0}) << std::endl;
    std::cout << "t.at({0, 1}) = " << t.at({0, 1}) << std::endl;
    std::cout << "t.at({1, 0}) = " << t.at({1, 0}) << std::endl;
    std::cout << "t.at({1, 1}) = " << t.at({1, 1}) << std::endl;
  }
}

// CHECK: t.at({0, 0}) = (1,0)
// CHECK: t.at({0, 1}) = (0,1)
// CHECK: t.at({1, 0}) = (0,-1)
// CHECK: t.at({1, 1}) = (1,0)

int main() {
  test_register();
  test_mult();
  test_constructors_and_access();
  test_copy();
  test_borrow();
  test_take();
  return 0;
}
