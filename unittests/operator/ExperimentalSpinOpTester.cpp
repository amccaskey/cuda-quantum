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

enum Pauli : int8_t { I = 0, X, Y, Z };
constexpr Pauli paulis[4] = {Pauli::I, Pauli::X, Pauli::Y, Pauli::Z};

// Function to multiply two single-qubit Pauli operators
static std::pair<std::complex<double>, Pauli> multiply_paulis(Pauli a,
                                                              Pauli b) {
  using namespace std::complex_literals;
  // I    X    Y    Z
  constexpr std::complex<double> table[4][4] = {
      {1., 1., 1, 1},    // I
      {1., 1., 1i, -1i}, // X
      {1., -1i, 1, 1i},  // Y
      {1., 1i, -1i, 1}   // Z
  };
  if (a == b)
    return {1.0, Pauli::I};
  if (a == Pauli::I)
    return {1.0, b};
  if (b == Pauli::I)
    return {1.0, a};
  return {table[a][b], paulis[a ^ b]};
}

// Function to multiply two multi-qubit Pauli words
static std::pair<std::complex<double>, std::vector<Pauli>>
multiply_pauli_words(const std::vector<Pauli> &a, const std::vector<Pauli> &b,
                     bool verbose = false) {
  std::complex<double> phase = 1.0;
  std::string info;
  std::vector<Pauli> result(a.size(), Pauli::I);
  for (size_t i = 0; i < a.size(); ++i) {
    auto [p, r] = multiply_paulis(a[i], b[i]);
    phase *= p;
    result[i] = r;
  }
  return {phase, result};
}

// Generates a pauli word out of a binary representation of it.
static std::vector<Pauli> generate_pauli_word(int64_t id, int64_t num_qubits) {
  constexpr int64_t mask = 0x3;
  std::vector<Pauli> word(num_qubits, Pauli::I);
  for (int64_t i = 0; i < num_qubits; ++i) {
    assert((id & mask) < 4);
    word[i] = paulis[id & mask];
    id >>= 2;
  }
  return word;
}

static std::string generate_pauli_string(const std::vector<Pauli> &word) {
  constexpr char paulis_name[4] = {'I', 'X', 'Y', 'Z'};
  std::string result(word.size(), 'I');
  for (int64_t i = 0; i < word.size(); ++i)
    result[i] = paulis_name[word[i]];
  return result;
}

static auto generate_cudaq_spin(int64_t id, int64_t num_qubits,
                                bool addI = true) {
  constexpr int64_t mask = 0x3;
  cudaq::experimental::spin::spin_op result;
  for (int64_t i = 0; i < num_qubits; ++i) {
    switch (paulis[id & mask]) {
    case Pauli::I:
      if (addI)
        result *= spin::i(i);
      break;
    case Pauli::X:
      result *= spin::x(i);
      break;
    case Pauli::Y:
      result *= spin::y(i);
      break;
    case Pauli::Z:
      result *= spin::z(i);
      break;
    }
    id >>= 2;
  }
  return result;
}

TEST(ExperimentalSpinOpTester, checkConstruction) {
  auto op = spin::x(10);
  EXPECT_EQ(11, op.num_qubits());
  EXPECT_EQ(1, op.num_terms());

  spin::spin_op op2;
  op2.dump();

  op2 *= spin::x(2);
  op2.dump();

  EXPECT_EQ(spin::x(2), op2);
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

TEST(SpinOpTester, checkMultiplication) {
  for (int num_qubits = 1; num_qubits <= 4; ++num_qubits) {
    int64_t num_words = std::pow(4, num_qubits);
    for (int64_t i = 0; i < num_words; ++i) {
      for (int64_t j = 0; j < num_words; ++j) {
        // Expected result:
        std::vector<Pauli> a_word = generate_pauli_word(i, num_qubits);
        std::vector<Pauli> b_word = generate_pauli_word(j, num_qubits);
        auto [phase, result] = multiply_pauli_words(a_word, b_word);

        // Result:
        auto a_spin = generate_cudaq_spin(i, num_qubits);
        auto b_spin = generate_cudaq_spin(j, num_qubits, false);
        auto result_spin = a_spin * b_spin;

        // Check result
        EXPECT_EQ(generate_pauli_string(result), result_spin.to_string(false));
        EXPECT_EQ(phase, result_spin.get_coefficient());
      }
    }
  }
}

TEST(SpinOpTester, canBuildDeuteron) {
  auto H = 5.907 - 2.1433 * spin::x(0) * spin::x(1) -
           2.1433 * spin::y(0) * spin::y(1) + .21829 * spin::z(0) -
           6.125 * spin::z(1);

  H.dump();

  EXPECT_EQ(5, H.num_terms());
  EXPECT_EQ(2, H.num_qubits());
}

// TEST(SpinOpTester, checkGetSparseMatrix) {
//   auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 *
//   z(0) -
//            6.125 * z(1);
//   auto matrix = H.to_matrix();
//   matrix.dump();
//   auto [values, rows, cols] = H.to_sparse_matrix();
//   for (std::size_t i = 0; auto &el : values) {
//     std::cout << rows[i] << ", " << cols[i] << ", " << el << "\n";
//     EXPECT_NEAR(matrix(rows[i], cols[i]).real(), el.real(), 1e-3);
//     i++;
//   }
// }

TEST(SpinOpTester, checkGetMatrix) {
  auto H = 5.907 - 2.1433 * spin::x(0) * spin::x(1) -
           2.1433 * spin::y(0) * spin::y(1) + .21829 * spin::z(0) -
           6.125 * spin::z(1);
  auto matrix = H.to_matrix();
  printf("TEST: \n%s\n", matrix.dump().c_str());
  //   auto groundEnergy = matrix.minimal_eigenvalue();

  //   {
  //     EXPECT_NEAR(groundEnergy.real(), -1.74, 1e-2);

  //     std::vector<double> expected{.00029,  0, 0, 0,       0,       -.43619,
  //                                  -4.2866, 0, 0, -4.2866, 12.2503, 0,
  //                                  0,       0, 0, 11.8137};
  //     for (std::size_t i = 0; i < 16; ++i)
  //       EXPECT_NEAR(matrix.data()[i].real(), expected[i], 1e-3);
  //   }
  //   {
  //     // Create the G=ground state for the above hamiltonian
  //     cudaq::complex_matrix vec(4, 1);
  //     vec.set_zero();
  //     vec(2, 0) = .292786;
  //     vec(1, 0) = .956178;

  //     // Compute H |psi_g>
  //     auto tmp = matrix * vec;

  //     // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
  //     for (std::size_t i = 0; i < 4; i++)
  //       EXPECT_NEAR(tmp(i, 0).real(), groundEnergy.real() * vec(i, 0).real(),
  //                   1e-2);
  //   }
  //   {
  //     // do the same thing, but use std vector instead
  //     std::vector<std::complex<double>> vec{0.0, .956178, .292786, 0.0};

  //     // Compute H |psi_g>
  //     auto tmp = matrix * vec;

  //     // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
  //     for (std::size_t i = 0; i < 4; i++)
  //       EXPECT_NEAR(tmp(i, 0).real(), groundEnergy.real() * vec[i].real(),
  //       1e-2);
  //   }

  //   {
  //     // test <psi_g | H | psi_g>
  //     cudaq::complex_matrix psig(4, 1);
  //     psig.set_zero();
  //     psig(2, 0) = .292786;
  //     psig(1, 0) = .956178;
  //     auto prod = matrix * psig;
  //     double sum = 0.0;
  //     for (std::size_t i = 0; i < 4; i++) {
  //       sum += (std::conj(psig(i, 0)) * prod(i, 0)).real();
  //     }
  //     EXPECT_NEAR(sum, -1.74, 1e-2);
  //   }
}

TEST(SpinOpTester, checkIterator) {
  auto H = 5.907 - 2.1433 * spin::x(0) * spin::x(1) -
           2.1433 * spin::y(0) * spin::y(1) + .21829 * spin::z(0) -
           6.125 * spin::z(1);

  std::size_t count = 0;
  for (auto term : H) {
    std::cout << "TEST: " << term.to_string();
    count++;
  }

  EXPECT_EQ(count, H.num_terms());

  H.for_each_term([](const spin::spin_op &term) {
    printf("HELLO ");
    term.dump();
  });
}

TEST(SpinOpTester, checkDistributeTerms) {
  auto H = 5.907 - 2.1433 * spin::x(0) * spin::x(1) -
           2.1433 * spin::y(0) * spin::y(1) + .21829 * spin::z(0) -
           6.125 * spin::z(1);

  auto distributed = H.distribute_terms(2);

  EXPECT_EQ(distributed.size(), 2);
  EXPECT_EQ(distributed[0].num_terms(), 3);
  EXPECT_EQ(distributed[1].num_terms(), 2);
}
