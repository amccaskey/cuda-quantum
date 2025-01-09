#include "cudaq/operator.h"
#include <gtest/gtest.h>

using namespace cudaq::experimental::fermion;

TEST(FermionOperatorTester, canCreateEmpty) {
  fermion_op op;
  EXPECT_TRUE(op.empty());
  EXPECT_EQ(0, op.num_terms());
  EXPECT_EQ(0, op.num_qubits());
}

TEST(FermionOperatorTester, canCreateSingleOperator) {
  auto c = create(1);
  c.dump();
  EXPECT_EQ(1, c.num_terms());
  EXPECT_EQ(2, c.num_qubits());

  auto a = annihilate(1);
  a.dump();
  EXPECT_EQ(1, a.num_terms());
  EXPECT_EQ(2, a.num_qubits());
}

TEST(FermionOperatorTester, maintainsNormalOrdering) {
  auto c1 = create(1);
  auto a2 = annihilate(2);
  auto term = c1 * a2;
  term.dump();
  // Should maintain c1 a2 ordering
  std::string expected = "[1+0j] 1^ 2 \n";
  EXPECT_EQ(expected, term.to_string(true));

  // Test reverse order gets normalized
  auto term2 = a2 * c1;
  EXPECT_EQ("[-1+0j] 1^ 2 \n", term2.to_string(true));
}

TEST(FermionOperatorTester, canAddTerms) {
  auto c1 = create(1);
  auto c2 = create(2);
  auto sum = c1 + c2;

  EXPECT_EQ(2, sum.num_terms());
  EXPECT_EQ(3, sum.num_qubits());
}

TEST(FermionOperatorTester, canMultiplyByScalar) {
  auto c1 = create(1);
  auto scaled = c1 * 2.0;

  EXPECT_EQ(1, scaled.num_terms());
  std::string expected = "[2+0j] 1^ \n";
  EXPECT_EQ(expected, scaled.to_string(true));
}

TEST(FermionOperatorTester, canIterateTerms) {
  auto op = create(0) + create(1) + create(2);

  std::size_t count = 0;
  for (const auto &term : op) {
    EXPECT_EQ(1, term.num_terms());
    count++;
  }
  EXPECT_EQ(3, count);
}

TEST(FermionOperatorTester, checkEquality) {
  auto term1 = create(1) * annihilate(2);
  auto term2 = create(1) * annihilate(2);
  auto term3 = create(2) * annihilate(1);

  EXPECT_EQ(term1, term2);
  EXPECT_NE(term1, term3);
}

TEST(FermionOperatorTester, canDistributeTerms) {
  auto op = create(0) + create(1) + create(2) + create(3);
  auto chunks = op.distribute_terms(2);

  EXPECT_EQ(2, chunks.size());
  EXPECT_EQ(2, chunks[0].num_terms());
  EXPECT_EQ(2, chunks[1].num_terms());
}

TEST(FermionOperatorTester, complexOperatorConstruction) {
  // Build number operator n_i = a†_i a_i
  auto n1 = create(1) * annihilate(1);
  EXPECT_EQ(1, n1.num_terms());

  // Build hopping term a†_i a_j
  auto hop = create(1) * annihilate(2);
  EXPECT_EQ(1, hop.num_terms());

  // Combine terms
  auto hamiltonian = n1 + hop;
  EXPECT_EQ(2, hamiltonian.num_terms());
}

TEST(FermionOperatorTester, checkAnticommutation) {
  // Test fermionic anticommutation relations {a_i, a_j†} = δ_ij
  // [a0, c0] = 1 -> ac + ca = 1
  auto c0 = create(0);
  auto a0 = annihilate(0);
  auto c1 = create(1);
  auto a1 = annihilate(1);
  (a0 * c0).dump();
  // Test {a_i, a_j†} = δ_ij
  auto anticomm_00 = a0 * c0 + c0 * a0;
  anticomm_00.dump();
  EXPECT_EQ(anticomm_00.num_terms(), 1);
  EXPECT_EQ(anticomm_00.get_coefficient(), std::complex<double>(1.0, 0.0));

  auto anticomm_01 = a0 * c1 + c1 * a0;
  anticomm_01.dump();
  EXPECT_EQ(anticomm_01.get_coefficient(), std::complex<double>(0.0, 0.0));

  // Test {a_i, a_j} = 0
  auto anticomm_aa = a0 * a1 + a1 * a0;
  EXPECT_EQ(anticomm_aa.get_coefficient(), std::complex<double>(0.0, 0.0));

  // Test {a_i†, a_j†} = 0
  auto anticomm_cc = c0 * c1 + c1 * c0;
  EXPECT_EQ(anticomm_cc.get_coefficient(), std::complex<double>(0.0, 0.0));
}

TEST(FermionOperatorTester, checkNumberOperator) {
  // Test number operator n_i = a_i† a_i
  auto c0 = create(0);
  auto a0 = annihilate(0);
  auto n0 = c0 * a0;
  n0.dump();
  // Number operator should be Hermitian
  auto n0_dag = a0 * c0 - 1.;
  n0_dag.dump();
  EXPECT_EQ(n0.num_terms(), n0_dag.num_terms());

  // n_i^2 = n_i for fermions
  auto n0_squared = n0 * n0;
  n0_squared.dump();
  EXPECT_EQ(n0_squared.get_coefficient(), n0.get_coefficient());
}

TEST(FermionOperatorTester, checkHoppingTerm) {
  // Test hopping term t(a_i† a_j + a_j† a_i)
  auto c0 = create(0);
  auto a0 = annihilate(0);
  auto c1 = create(1);
  auto a1 = annihilate(1);

  auto hopping = c0 * a1 + c1 * a0;
  EXPECT_EQ(hopping.num_terms(), 2);

  // Should be Hermitian
  auto hopping_dag = a1 * c0 + a0 * c1;
  hopping.dump();
  printf("dag\n");
  hopping_dag.dump();
  EXPECT_EQ(hopping, -1. * hopping_dag);
}

TEST(FermionOperatorTester, checkPauliPrinciple) {
  // Test Pauli exclusion principle
  auto c0 = create(0);

  // c_i† c_i† = 0 for fermions
  auto double_create = c0 * c0;
  EXPECT_EQ(double_create.get_coefficient(), std::complex<double>(0.0, 0.0));

  auto a0 = annihilate(0);
  // a_i a_i = 0 for fermions
  auto double_annihilate = a0 * a0;
  EXPECT_EQ(double_annihilate.get_coefficient(),
            std::complex<double>(0.0, 0.0));
}

TEST(FermionOperatorTester, checkJordanWignerString) {
  // Test Jordan-Wigner string when operators act on different sites
  auto c0 = create(0);
  auto c2 = create(2);

  // Should pick up a minus sign due to anticommutation through site 1
  auto term = c2 * c0;
  EXPECT_EQ(term.get_coefficient(), std::complex<double>(-1.0, 0.0));

  auto term2 = c0 * c2;
  EXPECT_EQ(term2.get_coefficient(), std::complex<double>(1.0, 0.0));
}

TEST(ParticleMatrixTester, EmptyOperatorReturnsEmptyMatrix) {
  fermion_op empty;
  auto matrix = empty.to_matrix();
  EXPECT_TRUE(matrix.get_num_elements() == 0);
}

TEST(ParticleMatrixTester, SingleFermionCreation) {
  auto c = create(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};
  auto matrix = c.to_matrix(dims);

  std::cout << "Single\n";
  matrix.dump();

  // For fermions, creation operator should be:
  // [0 0]
  // [1 0]
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
  EXPECT_EQ((matrix[{0, 0}]), std::complex<double>(0, 0));
  EXPECT_EQ((matrix[{0, 1}]), std::complex<double>(0, 0));
  EXPECT_EQ((matrix[{1, 0}]), std::complex<double>(1, 0));
  EXPECT_EQ((matrix[{1, 1}]), std::complex<double>(0, 0));
}

TEST(ParticleMatrixTester, NumberOperatorFermion) {
  auto n = create(0) * annihilate(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};
  auto matrix = n.to_matrix(dims);
  std::cout << "NOp\n";
  matrix.dump();

  // Number operator for fermions should be:
  // [0 0]
  // [0 1]
  EXPECT_EQ(matrix.shape()[0], 2);
  EXPECT_EQ(matrix.shape()[1], 2);
  EXPECT_EQ((matrix[{0, 0}]), std::complex<double>(0, 0));
  EXPECT_EQ((matrix[{1, 1}]), std::complex<double>(1, 0));
}

TEST(ParticleMatrixTester, MultiSiteFermionHopping) {
  auto hop = create(0) * annihilate(1);
  hop.dump();
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = hop.to_matrix(dims);
  std::cout << "TEST:\n";
  matrix.dump();
  // Hopping term should have proper dimensions
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(ParticleMatrixTester, ParameterizedOperator) {
  auto paramOp = [](const cudaq::experimental::parameter_map &params) {
    return std::complex<double>(params.at("t").real(), 0);
  };
  auto hop = paramOp * (create(0) * annihilate(1));

  cudaq::experimental::parameter_map params{
      {"t", std::complex<double>(0.5, 0)}};
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};

  auto matrix = hop.to_matrix(dims, params);
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(FermionMatrixTester, CheckGapOperatorMatrix) {
  // Test operator with gaps like 0^3^
  auto op = create(0) * create(3);
  std::unordered_map<std::size_t, std::size_t> dims{
      {0, 2}, {1, 2}, {2, 2}, {3, 2}};
  auto matrix = op.to_matrix(dims);

  // Should be 16x16 matrix (2^4 dimensions)
  EXPECT_EQ(matrix.shape()[0], 16);
  EXPECT_EQ(matrix.shape()[1], 16);
}

TEST(FermionMatrixTester, CheckMultiSiteProduct) {
  // Test product of operators on same site
  auto op = create(0) * annihilate(0) * create(2);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}, {2, 2}};
  auto matrix = op.to_matrix(dims);

  // Should be 8x8 matrix (2^3 dimensions)
  EXPECT_EQ(matrix.shape()[0], 8);
  EXPECT_EQ(matrix.shape()[1], 8);
}

TEST(FermionMatrixTester, CheckMissingDimensions) {
  auto op = create(0) * create(2);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};

  // Should throw when dimensions are missing
  EXPECT_THROW(op.to_matrix(dims), std::runtime_error);
}

TEST(FermionMatrixTester, CheckHigherDimensionalModes) {
  // Test with modes having dimension > 2
  auto op = create(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 3}};
  auto matrix = op.to_matrix(dims);

  EXPECT_EQ(matrix.shape()[0], 3);
  EXPECT_EQ(matrix.shape()[1], 3);
  EXPECT_EQ((matrix[{1, 0}]), std::complex<double>(1, 0));
  EXPECT_EQ((matrix[{2, 1}]), std::complex<double>(std::sqrt(2), 0));
}

TEST(FermionMatrixTester, CheckSumOperatorMatrix) {
  // Test sum of operators
  auto op = create(0) + create(1);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = op.to_matrix(dims);
  matrix.dump();
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(FermionMatrixTester, CheckParameterizedSum) {
  auto param_op = [](const cudaq::experimental::parameter_map &params) {
    return std::complex<double>(params.at("g").real(), 0);
  };

  auto op = param_op * (create(0) + create(1));
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  cudaq::experimental::parameter_map params{
      {"g", std::complex<double>(0.5, 0)}};

  auto matrix = op.to_matrix(dims, params);
  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
}

TEST(ParticleMatrixTester, MultiSiteFermionHoppingElements) {
  auto hop = create(0) * annihilate(1);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = hop.to_matrix(dims);

  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
  // |00> -> 0, |01> -> 1, |10> -> 2, |11> -> 3
  EXPECT_EQ((matrix[{2, 1}]), std::complex<double>(1, 0)); // <10|hop|01>
  EXPECT_EQ((matrix[{0, 0}]), std::complex<double>(0, 0)); // <00|hop|00>
  EXPECT_EQ((matrix[{3, 3}]), std::complex<double>(0, 0)); // <11|hop|11>
}

TEST(ParticleMatrixTester, GapOperatorElements) {
  auto op = create(0) * create(2);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}, {2, 2}};
  auto matrix = op.to_matrix(dims);
   matrix.dump();
  EXPECT_EQ(matrix.shape()[0], 8);
  EXPECT_EQ(matrix.shape()[1], 8);
  EXPECT_EQ((matrix[{5, 0}]), std::complex<double>(1, 0));
  EXPECT_EQ((matrix[{7, 2}]), std::complex<double>(-1, 0));
}

TEST(ParticleMatrixTester, NumberOperatorElements) {
  auto n = create(1) * annihilate(1);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};
  auto matrix = n.to_matrix(dims);

  EXPECT_EQ(matrix.shape()[0], 4);
  EXPECT_EQ(matrix.shape()[1], 4);
  // Check diagonal elements
  EXPECT_EQ((matrix[{0, 0}]), std::complex<double>(0, 0)); // <00|n|00>
  EXPECT_EQ((matrix[{1, 1}]), std::complex<double>(1, 0)); // <01|n|01>
  EXPECT_EQ((matrix[{2, 2}]), std::complex<double>(0, 0)); // <10|n|10>
  EXPECT_EQ((matrix[{3, 3}]), std::complex<double>(1, 0)); // <11|n|11>
}

TEST(ParticleMatrixTester, HigherDimensionalModeElements) {
  auto c = create(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 3}};
  auto matrix = c.to_matrix(dims);

  EXPECT_EQ(matrix.shape()[0], 3);
  EXPECT_EQ(matrix.shape()[1], 3);
  EXPECT_EQ((matrix[{1, 0}]), std::complex<double>(1, 0));
  EXPECT_EQ((matrix[{2, 1}]), std::complex<double>(std::sqrt(2), 0));
  EXPECT_EQ((matrix[{0, 0}]), std::complex<double>(0, 0));
  EXPECT_EQ((matrix[{2, 0}]), std::complex<double>(0, 0));
}

TEST(FermionOperatorTester, ElementaryOperatorsSingleSite) {
  auto c0 = create(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}};

  auto matrices = c0.get_elementary_operators(dims);
  EXPECT_EQ(matrices.size(), 1);

  // Verify creation operator matrix structure
  EXPECT_EQ(matrices[0].shape()[0], 2);
  EXPECT_EQ(matrices[0].shape()[1], 2);
  EXPECT_EQ((matrices[0][{1, 0}]), std::complex<double>(1, 0));
}

TEST(FermionOperatorTester, ElementaryOperatorsProductTerm) {
  auto op = create(0) * annihilate(1);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};

  auto matrices = op.get_elementary_operators(dims);
  EXPECT_EQ(matrices.size(), 2);

  // First matrix should be creation on site 0
  EXPECT_EQ(matrices[0].shape()[0], 2);
  EXPECT_EQ(matrices[0].shape()[1], 2);
  EXPECT_EQ((matrices[0][{1, 0}]), std::complex<double>(1, 0));

  // Second matrix should be annihilation on site 1
  EXPECT_EQ(matrices[1].shape()[0], 2);
  EXPECT_EQ(matrices[1].shape()[1], 2);
  EXPECT_EQ((matrices[1][{0, 1}]), std::complex<double>(1, 0));
}

TEST(FermionOperatorTester, ElementaryOperatorsThrowsOnSum) {
  auto op = create(0) + create(1);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 2}, {1, 2}};

  EXPECT_THROW(op.get_elementary_operators(dims), std::runtime_error);
}

TEST(FermionOperatorTester, ElementaryOperatorsWithDimensions) {
  auto op = create(0);
  std::unordered_map<std::size_t, std::size_t> dims{{0, 3}};

  auto matrices = op.get_elementary_operators(dims);
  EXPECT_EQ(matrices.size(), 1);

  // Verify 3-level creation operator
  EXPECT_EQ(matrices[0].shape()[0], 3);
  EXPECT_EQ(matrices[0].shape()[1], 3);
  EXPECT_EQ((matrices[0][{1, 0}]), std::complex<double>(1, 0));
  EXPECT_EQ((matrices[0][{2, 1}]), std::complex<double>(std::sqrt(2), 0));
}
