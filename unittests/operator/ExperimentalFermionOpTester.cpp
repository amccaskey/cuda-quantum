#include "cudaq/operator.h"
#include <gtest/gtest.h>

using namespace cudaq::experimental::particle;

TEST(FermionOperatorTester, canCreateEmpty) {
  particle_op op;
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
