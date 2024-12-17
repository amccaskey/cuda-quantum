#include "cudaq/operator.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq::experimental;

TEST(MatrixOperatorTest, BasicConstruction) {
  operator_matrix m({1.0, 0.0, 0.0, 1.0}, {2, 2});
  auto op = from_matrix(m, {0});
  EXPECT_EQ(op.num_qubits(), 1);
  EXPECT_EQ(op.num_terms(), 1);
}

TEST(MatrixOperatorTest, Addition) {
  operator_matrix m1({1.0, 0.0, 0.0, 1.0}, {2, 2});
  operator_matrix m2({0.0, 1.0, 1.0, 0.0}, {2, 2});

  auto op1 = from_matrix(m1, {0});
  auto op2 = from_matrix(m2, {1});
  auto sum = op1 + op2;

  EXPECT_EQ(sum.num_terms(), 2);
  EXPECT_EQ(sum.num_qubits(), 2);
}

TEST(MatrixOperatorTest, Multiplication) {
  operator_matrix x({0.0, 1.0, 1.0, 0.0}, {2, 2});
  operator_matrix z({1.0, 0.0, 0.0, -1.0}, {2, 2});

  auto op1 = from_matrix(x, {0});
  auto op2 = from_matrix(z, {0});
  auto prod = op1 * op2;
  prod.dump();
  EXPECT_EQ(prod.num_terms(), 1);

  auto result = prod.to_matrix();
  operator_matrix expected({0.0, -1.0, 1.0, 0.0}, {2, 2});
  std::cout << result.dump() << "\n";
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_NEAR(std::abs(result[{i, j}].real() - expected[{i, j}].real()),
                  0.0, 1e-6);
      EXPECT_NEAR(std::abs(result[{i, j}].imag() - expected[{i, j}].imag()),
                  0.0, 1e-6);
    }
  }
}

TEST(MatrixOperatorTest, ScalarMultiplication) {
  operator_matrix m({1.0, 0.0, 0.0, 1.0}, {2, 2});
  auto op = from_matrix(m, {0});
  auto scaled = op * 2.0;

  auto result = scaled.to_matrix();
  operator_matrix expected({2.0, 0.0, 0.0, 2.0}, {2, 2});

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_NEAR(std::abs(result[{i, j}].real() - expected[{i, j}].real()),
                  0.0, 1e-6);
    }
  }
}

TEST(MatrixOperatorTest, TensorProduct) {
  operator_matrix x({0.0, 1.0, 1.0, 0.0}, {2, 2});
  auto op1 = from_matrix(x, {0});
  auto op2 = from_matrix(x, {1});
  auto tensor = op1 * op2;
  EXPECT_EQ(tensor.num_qubits(), 2);
  EXPECT_EQ(tensor.num_terms(), 1);

  auto result = tensor.to_matrix();
  std::cout << result.dump() << "\n";
  EXPECT_EQ(result.get_rows(), 4);
  EXPECT_EQ(result.get_columns(), 4);
}
