/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/operator.h"
#include "cudaq/utils/cudaq_utils.h"

using namespace cudaq::experimental;

TEST(ExperimentalSpinOpTester, checkConstruction) {
  auto op = spin::x(10);
  EXPECT_EQ(11, op.num_qubits());
  EXPECT_EQ(1, op.num_terms());
}

TEST(ExperimentalSpinOpTester, checkEquality) {
  auto xx = spin::x(5);
  EXPECT_EQ(xx, xx);
}

TEST(ExperimentalSpinOpTester, checkFromWord) {
  {
    auto s = spin::from_word("ZZZ");
    std::cout << s.to_string();
    EXPECT_EQ(spin::z(0) * spin::z(1) * spin::z(2), s);
  }
  {
    auto s = spin::from_word("XYX");
    std::cout << s.to_string();
    EXPECT_EQ(spin::x(0) * spin::y(1) * spin::x(2), s);
  }
  {
    auto s = spin::from_word("IZY");
    std::cout << s.to_string();
    EXPECT_EQ(spin::i(0) * spin::z(1) * spin::y(2), s);
  }
}

TEST(ExperimentalSpinOpTester, checkAddition) {
  auto op = spin::x(10);
  op.dump();

  auto added = op + op;
  EXPECT_EQ(11, added.num_qubits());
  EXPECT_EQ(1, added.num_terms());
  EXPECT_EQ(2.0, added.get_coefficient());

  //   op.dump();
  added.dump();

  auto added2 = spin::x(0) + spin::y(1) + spin::z(2);
  added2.dump();
  EXPECT_EQ(3, added2.num_terms());
  EXPECT_EQ(3, added2.num_qubits());

  for (auto &term : added2) {
    printf("term:\n");
    term.dump();
  }

  printf("subtracted\n");
  auto subtracted = spin::x(0) - spin::y(2);
  subtracted.dump();
}

TEST(ExperimentalSpinOpTester, checkFunctor) {
  auto f_t = [](const parameter_map &parameters) {
    return 2. * parameters.at("t") + 3. * parameters.at("omega");
  };

  auto H_t = f_t * spin::x(2);
  H_t.dump();

  EXPECT_TRUE(H_t.is_template());

  for (auto t : cudaq::linspace(-M_PI, M_PI, 10)) {
    auto concrete_H_t = H_t({{"t", t}, {"omega", t / 3.}});
    EXPECT_TRUE(!concrete_H_t.is_template());
    EXPECT_TRUE(H_t.is_template());

    concrete_H_t.dump();
    EXPECT_NEAR(2. * t + 3. * (t / 3.), concrete_H_t.get_coefficient().real(),
                1e-3);
  }
}