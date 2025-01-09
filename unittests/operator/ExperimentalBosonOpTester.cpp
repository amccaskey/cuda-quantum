#include "cudaq/operator.h"
#include <gtest/gtest.h>

using namespace cudaq::experimental::boson;
TEST(BosonOperatorTester, canCreateEmpty) {
  boson_op op;
  EXPECT_TRUE(op.empty());
  EXPECT_EQ(0, op.num_terms());
  EXPECT_EQ(0, op.num_qubits());
}

TEST(BosonOperatorTester, canCreateSingleOperator) {
  auto c = create(1);
  EXPECT_EQ(1, c.num_terms());
  EXPECT_EQ(2, c.num_qubits());

  auto a = annihilate(1);
  EXPECT_EQ(1, a.num_terms());
  EXPECT_EQ(2, a.num_qubits());
}

TEST(BosonOperatorTester, maintainsCommutationRelations) {
  auto a = annihilate(0);
  auto adag = create(0);

  // Test [a, a†] = 1
  auto comm = a * adag - adag * a;
  EXPECT_EQ(comm.get_coefficient(), std::complex<double>(1.0, 0.0));

  // Test [a, a] = 0
  auto comm2 = a * a - a * a;
  EXPECT_EQ(comm2.get_coefficient(), std::complex<double>(0.0, 0.0));

  // Test [a†, a†] = 0
  auto comm3 = adag * adag - adag * adag;
  EXPECT_EQ(comm3.get_coefficient(), std::complex<double>(0.0, 0.0));
}

TEST(BosonOperatorTester, numberOperator) {
  auto a = annihilate(0);
  auto adag = create(0);

  auto n = adag * a;
  std::cout << "N:\n";
  n.dump();

  std::cout << "Ndag:\n";
  auto n_dag = a * adag;
  n_dag.dump();

  std::cout << "Expected:\n";
  auto expected = n + 1;
  expected.dump();

  EXPECT_EQ(n_dag, n + 1);
}

TEST(BosonOperatorTester, displacementOperator) {
  auto a = annihilate(0);
  auto adag = create(0);
  std::complex<double> alpha(0.5, 0.0);

  // D(α) = exp(αa† - α*a)
  auto D = alpha * adag - std::conj(alpha) * a;
  EXPECT_EQ(D.num_terms(), 2);
}

TEST(BosonOperatorTester, squeezingOperator) {
  auto a = annihilate(0);
  auto adag = create(0);
  std::complex<double> zeta(0.1, 0.0);

  // S(ζ) = exp(1/2(ζ*a² - ζ(a†)²))
  auto S = std::complex<double>(0.5, 0.0) *
           (std::conj(zeta) * a * a - zeta * adag * adag);
  EXPECT_EQ(S.num_terms(), 2);
}

TEST(BosonOperatorTester, multiModeOperations) {
  auto a0 = annihilate(0);
  auto a1 = annihilate(1);
  auto adag0 = create(0);
  auto adag1 = create(1);

  // Test [a_i, a_j†] = δ_ij
  auto comm01 = a0 * adag1 - adag1 * a0;
  EXPECT_EQ(comm01.get_coefficient(), std::complex<double>(0.0, 0.0));

  // Test beam splitter operation
  auto BS = adag0 * a1 + a0 * adag1;
  EXPECT_EQ(BS.num_terms(), 2);
}

TEST(BosonOperatorTester, canGetMatrixRepresentation) {
  auto a = annihilate(0);
  auto adag = create(0);

  // Test matrix representation with different truncation dimensions
  std::unordered_map<std::size_t, std::size_t> dims{{0, 3}};
  auto matrix = a.to_matrix(dims);
  matrix.dump() ;

  // Verify annihilation operator matrix elements for 3-level truncation
  EXPECT_NEAR(std::abs(matrix[{0, 1}]), std::sqrt(1.0), 1e-12);
  EXPECT_NEAR(std::abs(matrix[{1, 2}]), std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(std::abs(matrix[{0, 0}]), 0.0, 1e-6);

  // Test creation operator
  matrix = adag.to_matrix(dims);
  EXPECT_NEAR(std::abs(matrix[{1, 0}]), std::sqrt(1.0), 1e-12);
  EXPECT_NEAR(std::abs(matrix[{2, 1}]), std::sqrt(2.0), 1e-12);
}

TEST(BosonOperatorTester, canGetElementaryOperators) {
  auto a = annihilate(0);
  auto adag = create(0);
  auto n = adag * a;

  std::unordered_map<std::size_t, std::size_t> dims{{0, 4}};

  // Test elementary operators for number operator
  auto elementaryOps = n.get_elementary_operators(dims);
  EXPECT_EQ(elementaryOps.size(), 2);

  // First operator should be creation
  auto firstOp = elementaryOps[0];
  EXPECT_NEAR(std::abs(firstOp[{1, 0}]), std::sqrt(1.0), 1e-12);
  EXPECT_NEAR(std::abs(firstOp[{2, 1}]), std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(std::abs(firstOp[{3, 2}]), std::sqrt(3.0), 1e-12);

  // Second operator should be annihilation
  auto secondOp = elementaryOps[1];
  EXPECT_NEAR(std::abs(secondOp[{0, 1}]), std::sqrt(1.0), 1e-12);
  EXPECT_NEAR(std::abs(secondOp[{1, 2}]), std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(std::abs(secondOp[{2, 3}]), std::sqrt(3.0), 1e-12);
}

TEST(BosonOperatorTester, canHandleMultiModeMatrices) {
  auto a0 = annihilate(0);
  auto a1 = annihilate(1);
  auto adag0 = create(0);
  auto adag1 = create(1);

  // Test beam splitter operator
  auto bs = adag0 * a1 + a0 * adag1;

  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = bs.to_matrix(dims);

  // Verify matrix dimension is correct (2x2 for each mode -> 4x4 total)
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(BosonOperatorTester, canHandleParameterizedOperators) {
  auto a = annihilate(0);
  auto adag = create(0);

  // Create parameterized ln (displacement operator)
  std::complex<double> alpha(0.5, 0.0);
  auto D = alpha * adag - std::conj(alpha) * a;

  std::unordered_map<std::size_t, std::size_t> dims{{0, 3}};
  cudaq::experimental::parameter_map params;
  auto matrix = D.to_matrix(dims, params);

  // Verify matrix is not identity
  EXPECT_NE(std::abs(matrix[{0, 0}]), std::complex<double>(1.0, 0.0));
  EXPECT_NE(std::abs(matrix[{1, 1}]), std::complex<double>(1.0, 0.0));
}

TEST(BosonOperatorTester, canHandleHigherExcitationStates) {
  auto a = annihilate(0);
  auto adag = create(0);

  // Test double excitation operator
  auto doubleEx = adag * adag;

  std::unordered_map<std::size_t, std::size_t> dims{{0, 4}};
  auto matrix = doubleEx.to_matrix(dims);

  // Verify matrix elements for double excitation
  EXPECT_NEAR(std::abs(matrix[{2, 0}]), std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(std::abs(matrix[{3, 1}]), std::sqrt(6.0), 1e-12);
}
