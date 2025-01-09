#include "cudaq/operator.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq::experimental;

TEST(MatrixOperatorTest, BasicConstruction) {
  {
    operator_matrix m({1.0, 0.0, 0.0, 1.0}, 2, 2);
    auto op = from_matrix(m, 0);
    EXPECT_EQ(op.num_qubits(), 1);
    EXPECT_EQ(op.num_terms(), 1);
    auto matrix = op.to_matrix();
    matrix.dump();
  }
  {
    auto op = from_matrix(
        [](const dimensions_map &, const parameter_map &p) {
          auto theta = p.at("theta");
          return operator_matrix({theta, 0, 0, theta}, 2, 2);
        },
        0);

    op.dump();
    auto matrix = op.to_matrix({{"theta", 1.0}});
    matrix.dump();
  }
}

TEST(MatrixOperatorTest, Addition) {
  operator_matrix m1({1.0, 0.0, 0.0, 1.0}, {2, 2});
  operator_matrix m2({0.0, 1.0, 1.0, 0.0}, {2, 2});

  auto op1 = from_matrix(m1, {0});
  auto op2 = from_matrix(m2, {1});
  auto sum = op1 + op2;
  sum.dump();
  EXPECT_EQ(sum.num_terms(), 2);
  EXPECT_EQ(sum.num_qubits(), 2);
}

TEST(MatrixOperatorTest, Multiplication) {
  operator_matrix x({0.0, 1.0, 1.0, 0.0}, {2, 2});
  operator_matrix y({0.0, {0., -1.0}, {0., 1.0}, 0.0}, {2, 2});
  operator_matrix z({1.0, 0.0, 0.0, -1.0}, {2, 2});

  auto op1 = from_matrix(x, {0});
  auto op2 = from_matrix(z, {0});
  auto op3 = from_matrix(y, {0});
  {
    auto prod = op1 * op2;
    prod.dump();
    EXPECT_EQ(prod.num_terms(), 1);

    auto m = prod.to_matrix();
    m.dump();
    // FIXME check elements
    std::vector<std::complex<double>> expected{0, -1, 1, 0};
    EXPECT_NEAR(std::abs(m[{0, 0}]), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(m[{0, 1}].real() + 1.), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(m[{1, 0}].real() - 1.), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(m[{1, 1}]), 0.0, 1e-6);
  }
  {
    auto prod = (op1 + op2) * (op3 + op1);
    prod.dump();
    auto m = prod.to_matrix();
    m.dump();
    std::vector<std::complex<double>> expected{
        {1, 1}, {1, -1}, {-1, -1}, {1, -1}};
    // FIXME check
  }
}

TEST(MatrixOperatorTest, IdentityOperator) {
  operator_matrix I({1.0, 0.0, 0.0, 1.0}, 2, 2);
  auto op = from_matrix(I, 0);
  auto matrix = op.to_matrix();

  EXPECT_EQ(op.num_qubits(), 1);
  EXPECT_EQ(op.num_terms(), 1);
  EXPECT_NEAR(std::abs(matrix[{0, 0}].real() - 1.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 1}].real() - 1.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{0, 1}].real()), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 0}].real()), 0.0, 1e-6);
}

TEST(MatrixOperatorTest, ParameterizedOperator) {
  auto op = from_matrix(
      [](const dimensions_map &, const parameter_map &p) {
        auto theta = p.at("theta").real();
        return operator_matrix({std::cos(theta),
                                {0, -std::sin(theta)},
                                {0, std::sin(theta)},
                                std::cos(theta)},
                               2, 2);
      },
      0);

  auto matrix = op.to_matrix({{"theta", M_PI / 4}});
  EXPECT_NEAR(std::abs(matrix[{0, 0}].real() - std::sqrt(2) / 2), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{0, 1}].imag() + std::sqrt(2) / 2), 0.0, 1e-6);
}

TEST(MatrixOperatorTest, TensorProduct) {
  operator_matrix X({0.0, 1.0, 1.0, 0.0}, 2, 2);
  operator_matrix Z({1.0, 0.0, 0.0, -1.0}, 2, 2);

  auto opX = from_matrix(X, 0);
  auto opZ = from_matrix(Z, 1);
  auto product = opX * opZ;
  product.dump();
  auto matrix = product.to_matrix();
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
  EXPECT_EQ(product.num_qubits(), 2);
}

TEST(MatrixOperatorTest, ScalarMultiplication) {
  operator_matrix X({0.0, 1.0, 1.0, 0.0}, 2, 2);
  auto op = from_matrix(X, 0);
  auto scaled = op * 2.0;

  auto matrix = scaled.to_matrix();
  EXPECT_NEAR(std::abs(matrix[{0, 1}].real() - 2.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 0}].real() - 2.0), 0.0, 1e-6);
}

TEST(MatrixOperatorTest, GetElementaryOperators) {
  // Single site operator
  {
    operator_matrix X({0.0, 1.0, 1.0, 0.0}, 2, 2);
    auto op = from_matrix(X, 0);
    auto matrices = op.get_elementary_operators();

    EXPECT_EQ(matrices.size(), 1);
    EXPECT_NEAR(std::abs(matrices[0][{0, 1}].real() - 1.0), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(matrices[0][{1, 0}].real() - 1.0), 0.0, 1e-6);
  }

  // Two-site operator with gap
  {
    operator_matrix X({0.0, 1.0, 1.0, 0.0}, 2, 2);
    operator_matrix Z({1.0, 0.0, 0.0, -1.0}, 2, 2);
    auto op = from_matrix(X, 0) * from_matrix(Z, 2);
    op.dump();
    auto matrices = op.get_elementary_operators();

    EXPECT_EQ(matrices.size(), 3);
    // Check X matrix
    EXPECT_NEAR(std::abs(matrices[0][{0, 1}].real() - 1.0), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(matrices[0][{1, 0}].real() - 1.0), 0.0, 1e-6);
    // Check identity matrix
    EXPECT_NEAR(std::abs(matrices[1][{0, 0}].real() - 1.0), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(matrices[1][{1, 1}].real() - 1.0), 0.0, 1e-6);
    // Check Z matrix
    EXPECT_NEAR(std::abs(matrices[2][{0, 0}].real() - 1.0), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(matrices[2][{1, 1}].real() + 1.0), 0.0, 1e-6);
  }

  // Should throw for sum of operators
  {
    operator_matrix X({0.0, 1.0, 1.0, 0.0}, 2, 2);
    operator_matrix Z({1.0, 0.0, 0.0, -1.0}, 2, 2);
    auto sum = from_matrix(X, 0) + from_matrix(Z, 1);
    EXPECT_THROW(sum.get_elementary_operators(), std::runtime_error);
  }

  // Test with different dimensions
  {
    operator_matrix qutrit({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0},
                           {3, 3}); // bug here if you dont specify dimensions
    auto op = from_matrix(qutrit, 0);
    auto matrices = op.get_elementary_operators({{0, 3}});
    matrices[0].dump();
    EXPECT_EQ(matrices.size(), 1);
    EXPECT_EQ(matrices[0].shape()[0], 3);
    EXPECT_EQ(matrices[0].shape()[1], 3);
    EXPECT_NEAR(std::abs(matrices[0][{2, 2}].real() + 1.0), 0.0, 1e-6);
  }

  // Test parameterized operator
  {
    auto op = from_matrix(
        [](const dimensions_map &, const parameter_map &p) {
          auto theta = p.at("theta");
          return operator_matrix({std::cos(theta), -std::sin(theta),
                                  std::sin(theta), std::cos(theta)},
                                 2, 2);
        },
        0);

    // Cannot get elementary operators from a template
    EXPECT_THROW(op.get_elementary_operators(), std::runtime_error);
  }
}

TEST(MixedOperatorTest, SpinFermionMultiplication) {
  // Create spin and fermion operators
  auto spin_op = spin::x(0);
  auto fermion_op = fermion::create(1);

  // Test multiplication results in matrix operator
  auto mixed = spin_op * fermion_op;
  EXPECT_EQ(mixed.num_qubits(), 2);

  // Check matrix representation
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = mixed.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(MixedOperatorTest, ElementaryOperators) {
  auto spin_op = spin::z(0);
  auto fermion_op = fermion::annihilate(1);
  auto mixed = spin_op * fermion_op;

  // Get elementary operators
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrices = mixed.get_elementary_operators(dims);

  // Should have two elementary operators
  EXPECT_EQ(matrices.size(), 2);

  // First matrix should be Z
  EXPECT_NEAR(std::abs(matrices[0][{0, 0}].real() - 1.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrices[0][{1, 1}].real() + 1.0), 0.0, 1e-6);

  // Second should be annihilation operator
  EXPECT_NEAR(std::abs(matrices[1][{0, 1}].real() - 1.0), 0.0, 1e-6);
}

TEST(MixedOperatorTest, MultiTermOperators) {
  // Create multi-term operators
  auto spin_sum = spin::x(0) + spin::z(0);
  auto fermion_sum = fermion::create(1) + fermion::annihilate(1);

  auto mixed = spin_sum * fermion_sum;
  EXPECT_EQ(mixed.num_terms(), 4);

  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = mixed.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(MixedOperatorTest, ParameterizedOperators) {
  // Create parameterized operators
  auto param_spin = spin::x(0) * std::complex<double>(0.5, 0);
  auto param_fermion = fermion::create(1) * std::complex<double>(0.0, 1.0);

  auto mixed = param_spin * param_fermion;

  parameter_map params;
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = mixed.to_matrix(dims, params);
  matrix.dump();

  // Check coefficient is multiplied correctly
  EXPECT_NEAR(std::abs(matrix[{3, 0}].real()), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{3, 0}].imag() - 0.5), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 2}].real()), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 2}].imag() - 0.5), 0.0, 1e-6);
}

TEST(MixedOperatorTest, DifferentDimensions) {
  // Test with different local dimensions
  auto spin_op = spin::x(0);
  auto fermion_op = fermion::create(1);
  auto mixed = spin_op * fermion_op;

  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 3}};
  auto matrix = mixed.to_matrix(dims);

  EXPECT_EQ(matrix.shape()[0], 6);
  EXPECT_EQ(matrix.shape()[1], 6);
}

TEST(OperatorAdditionTest, DifferentTypeAddition) {
  // Test adding spin and fermion operators
  auto spin_op = spin::x(0);
  auto fermion_op = fermion::create(0);
  auto result = spin_op + fermion_op;

  // Result should be matrix_op type
  EXPECT_EQ(result.num_terms(), 2);
  EXPECT_EQ(result.num_qubits(), 1);

  // Verify matrix representation

  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = result.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorAdditionTest, SpinFermionAddition) {
  auto spin_z = spin::z(0);
  auto fermion_n = fermion::create(0) * fermion::annihilate(0);
  auto sum = spin_z + fermion_n;

  // Check dimensions
  EXPECT_EQ(sum.num_qubits(), 1);

  // Verify matrix elements
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};

  auto matrix = sum.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorAdditionTest, MultiQubitAddition) {
  auto spin_op = spin::x(0) * spin::z(1);
  auto fermion_op = fermion::create(0) * fermion::annihilate(1);
  auto result = spin_op + fermion_op;

  EXPECT_EQ(result.num_qubits(), 2);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};

  auto matrix = result.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(OperatorAdditionTest, ParameterizedAddition) {
  // Create parameterized operators
  auto param_spin = 0.5 * spin::x(0);
  auto param_fermion = std::complex<double>(0.0, 1.0) * fermion::create(0);
  auto sum = param_spin + param_fermion;

  EXPECT_EQ(sum.num_terms(), 2);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};

  // Test matrix with parameters
  auto matrix = sum.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorAdditionTest, EmptyOperatorAddition) {
  matrix_op empty;
  auto spin_op = spin::x(0);
  auto result = empty + spin_op;

  EXPECT_EQ(result.num_terms(), 1);
  EXPECT_EQ(result.num_qubits(), 1);
}

TEST(OperatorAdditionTest, AdditionWithScalar) {
  auto spin_op = spin::x(0);
  auto fermion_op = fermion::create(0);
  auto result = (2.0 * spin_op) + (std::complex<double>(0.0, 1.0) * fermion_op);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};

  EXPECT_EQ(result.num_terms(), 2);
  auto matrix = result.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorSubtractionTest, DifferentTypeAddition) {
  // Test adding spin and fermion operators
  auto spin_op = spin::x(0);
  auto fermion_op = fermion::create(0);
  auto result = spin_op - fermion_op;

  // Result should be matrix_op type
  EXPECT_EQ(result.num_terms(), 2);
  EXPECT_EQ(result.num_qubits(), 1);

  // Verify matrix representation

  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = result.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorExpTest, SingleQubitExponential) {
  // Test exp(iX)
  auto x = spin::x(0);
  auto exp_ix = exp(std::complex<double>(0, 1) * x);

  // Check matrix dimensions
  auto matrix = exp_ix.to_matrix();
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);

  // Should be cos(1) I + i*sin(1) X
  EXPECT_NEAR(std::abs(matrix[{0, 0}].real() - std::cos(1)), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{0, 1}].imag() - std::sin(1)), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 0}].imag() - std::sin(1)), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 1}].real() - std::cos(1)), 0.0, 1e-6);
}

TEST(OperatorExpTest, NumberOperatorExponential) {
  // Test exp(n) where n is fermionic number operator
  auto n = fermion::create(0) * fermion::annihilate(0);
  auto exp_n = exp(n);

  dimensions_map dims{{0, 2}};
  auto matrix = exp_n.to_matrix(dims);
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);

  // Should be diagonal with [1, e]
  EXPECT_NEAR(std::abs(matrix[{0, 0}].real() - 1.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(matrix[{1, 1}].real() - std::exp(1.0)), 0.0, 1e-6);
}

TEST(OperatorExpTest, MultiQubitExponential) {
  // Test exp(XX) two-qubit interaction
  auto xx = spin::x(0) * spin::x(1);
  auto exp_xx = exp(xx);

  auto matrix = exp_xx.to_matrix();
  matrix.dump();

  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(OperatorExpTest, ParameterizedExponential) {
  // Test exp(θX)
  auto x = spin::x(0);
  auto param_op = [](const parameter_map &params) {
    return std::complex<double>(params.at("theta").real(), 0);
  };

  auto exp_theta_x = exp(param_op * x);

  parameter_map params{{"theta", std::complex<double>(M_PI / 2, 0)}};
  auto matrix = exp_theta_x.to_matrix({}, params);

  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
}

TEST(OperatorExpTest, HermitianOperatorExponential) {
  // Test that exp(iH) is unitary for Hermitian H
  auto h = spin::x(0) + spin::z(0);
  auto exp_ih = exp(std::complex<double>(0, 1) * h);

  auto matrix = exp_ih.to_matrix();

  // Check unitarity by computing U†U
  operator_matrix conj_matrix = matrix;
  for (std::size_t i = 0; i < matrix.shape()[0]; i++)
    for (std::size_t j = 0; j < matrix.shape()[1]; j++)
      conj_matrix[{i, j}] =
          std::conj(matrix[{i, j}]); // should be transpose in general

  auto prod = matrix * conj_matrix;
  for (std::size_t i = 0; i < matrix.shape()[0]; i++) {
    EXPECT_NEAR(std::abs(prod[{i, i}].real() - 1.0), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(prod[{i, i}].imag()), 0.0, 1e-6);
  }

  // Can access with variadic indices
  std::cout << "TEST: " << matrix(1, 1) << "\n";

  operator_matrix empty;

  empty = matrix;
  empty(0, 0) = 2.2;
  empty.dump();
  matrix.dump();
}
