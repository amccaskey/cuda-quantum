/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "common/EigenSparse.h"
#include "common/FmtCore.h"
#include <cudaq/spin_op.h>
#include <stdint.h>
#include <unsupported/Eigen/KroneckerProduct>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace cudaq {

namespace details {

/// @brief Helper function to convert ordering of matrix elements
/// to match internal simulator state ordering.
std::size_t convertOrdering(std::size_t numQubits, std::size_t idx) {
  std::size_t newIdx = 0;
  for (std::size_t i = 0; i < numQubits; ++i)
    if (idx & (1ULL << i))
      newIdx |= (1ULL << ((numQubits - 1) - i));
  return newIdx;
}

/// @brief Compute the action
std::pair<std::string, std::complex<double>>
actionOnBra(spin_op &term, const std::string &bitConfiguration) {
  auto coeff = term.get_coefficient();
  auto newConfiguration = bitConfiguration;
  std::complex<double> i(0, 1);

  term.for_each_pauli([&](pauli p, std::size_t idx) {
    if (p == pauli::Z) {
      coeff *= (newConfiguration[idx] == '1' ? -1 : 1);
    } else if (p == pauli::X) {
      newConfiguration[idx] = newConfiguration[idx] == '1' ? '0' : '1';
    } else if (p == pauli::Y) {
      coeff *= (newConfiguration[idx] == '1' ? i : -i);
      newConfiguration[idx] = (newConfiguration[idx] == '1' ? '0' : '1');
    }
  });

  return std::make_pair(newConfiguration, coeff);
}

std::pair<std::vector<bool>, std::complex<double>>
mult(const std::vector<bool> &p1, const std::vector<bool> &p2,
     const std::complex<double> &p1Coeff, const std::complex<double> &p2Coeff) {
  auto [minSize, maxSize] = std::minmax({p1.size(), p2.size()});
  std::size_t minNumQubits = minSize / 2;
  std::size_t maxNumQubits = maxSize / 2;
  std::vector<bool> result(maxSize, false);
  int yCount = 0;
  int cPhase = 0;

  for (std::size_t i = 0; i < minNumQubits; ++i) {
    bool p1_x = p1[i];
    bool p1_z = p1[i + (p1.size() / 2)];
    bool p2_x = p2[i];
    bool p2_z = p2[i + (p2.size() / 2)];

    // Compute the resulting Pauli operator
    result[i] = p1_x ^ p2_x;
    result[i + maxNumQubits] = p1_z ^ p2_z;

    yCount +=
        (p1_x & p1_z) + (p2_x & p2_z) - (result[i] & result[i + maxNumQubits]);
    cPhase += p1_x & p2_z;
  }

  const std::vector<bool> &big = p1.size() < p2.size() ? p2 : p1;
  for (std::size_t i = minNumQubits; i < maxNumQubits; ++i) {
    result[i] = big[i];
    result[i + maxNumQubits] = big[i + maxNumQubits];
  }

  // Normalize the phase to a value in the range [0, 3]
  int phaseFactor = (2 * cPhase + yCount) % 4;
  if (phaseFactor < 0)
    phaseFactor += 4;

  // Phase correction factors based on the total phase
  using namespace std::complex_literals;
  std::array<std::complex<double>, 4> phase{1.0, -1i, -1.0, 1i};
  std::complex<double> resultCoeff = p1Coeff * phase[phaseFactor] * p2Coeff;

  // Handle the "-0" issue
  if (std::abs(resultCoeff.real()) < 1e-12)
    resultCoeff.real(0);

  return {result, resultCoeff};
}
} // namespace details

spin_op::spin_op() {
  std::vector<bool> init(2);
  terms.emplace(init, 1.0);
}

spin_op::spin_op(
    const std::unordered_map<spin_op_term, std::complex<double>> &_terms)
    : terms(_terms) {}

spin_op::spin_op(std::size_t numQubits) {
  std::vector<bool> init(2 * numQubits);
  terms.emplace(init, 1.0);
}

spin_op::spin_op(const spin_op_term &term, const std::complex<double> &coeff) {
  terms.emplace(term, coeff);
}

spin_op::spin_op(const std::vector<spin_op_term> &bsf,
                 const std::vector<std::complex<double>> &coeffs) {
  for (std::size_t i = 0; auto &t : bsf)
    terms.emplace(t, coeffs[i++]);
}

spin_op::spin_op(pauli type, const std::size_t idx,
                 std::complex<double> coeff) {
  auto numQubits = idx + 1;
  std::vector<bool> d(2 * numQubits);

  if (type == pauli::X)
    d[idx] = 1;
  else if (type == pauli::Y) {
    d[idx] = 1;
    d[idx + numQubits] = 1;
  } else if (type == pauli::Z)
    d[idx + numQubits] = 1;

  terms.emplace(d, coeff);
}

spin_op::spin_op(const spin_op &o) : terms(o.terms) {}

spin_op::spin_op(
    std::pair<const spin_op_term, std::complex<double>> &termData) {
  terms.insert(termData);
}
spin_op::spin_op(
    const std::pair<const spin_op_term, std::complex<double>> &termData) {
  terms.insert(termData);
}

spin_op::iterator<spin_op> spin_op::begin() {
  auto startIter = terms.begin();
  return iterator<spin_op>(startIter);
}

spin_op::iterator<spin_op> spin_op::end() {
  auto endIter = terms.end();
  return iterator<spin_op>(endIter);
}

spin_op::iterator<const spin_op> spin_op::begin() const {
  auto startIter = terms.cbegin();
  return iterator<const spin_op>(startIter);
}

spin_op::iterator<const spin_op> spin_op::end() const {
  auto endIter = terms.cend();
  return iterator<const spin_op>(endIter);
}

complex_matrix spin_op::to_matrix() const {
  auto n = num_qubits();
  auto dim = 1UL << n;
  auto getBitStrForIdx = [&](std::size_t i) {
    std::stringstream s;
    for (int k = n - 1; k >= 0; k--)
      s << ((i >> k) & 1);
    return s.str();
  };

  // To construct the matrix, we are looping over every
  // row, computing the binary representation for that index,
  // e.g <100110|, and then we will compute the action of
  // each pauli term on that binary configuration, returning a new
  // product state and coefficient. Call this new state <colState|,
  // we then compute <rowState | Paulis | colState> and set it in the matrix
  // data.

  complex_matrix A(dim, dim);
  auto rawData = A.data;
#if defined(_OPENMP)
#pragma omp parallel for shared(rawData)
#endif
  for (std::size_t rowIdx = 0; rowIdx < dim; rowIdx++) {
    auto rowBitStr = getBitStrForIdx(rowIdx);
    for_each_term([&](spin_op &term) {
      auto [res, coeff] = details::actionOnBra(term, rowBitStr);
      auto colIdx = std::stol(res, nullptr, 2);
      rawData[details::convertOrdering(n, rowIdx) * dim +
              details::convertOrdering(n, colIdx)] += coeff;
    });
  }
  return A;
}

spin_op::csr_spmatrix spin_op::to_sparse_matrix() const {
  auto n = num_qubits();
  auto dim = 1UL << n;
  using Triplet = Eigen::Triplet<std::complex<double>>;
  using SpMat = Eigen::SparseMatrix<std::complex<double>>;
  std::vector<Triplet> xT{Triplet{0, 1, 1}, Triplet{1, 0, 1}},
      iT{Triplet{0, 0, 1}, Triplet{1, 1, 1}},
      yT{{0, 1, std::complex<double>{0, -1}},
         {1, 0, std::complex<double>{0, 1}}},
      zT{Triplet{0, 0, 1}, Triplet{1, 1, -1}};
  SpMat x(2, 2), y(2, 2), z(2, 2), i(2, 2), mat(dim, dim);
  x.setFromTriplets(xT.begin(), xT.end());
  y.setFromTriplets(yT.begin(), yT.end());
  z.setFromTriplets(zT.begin(), zT.end());
  i.setFromTriplets(iT.begin(), iT.end());

  auto kronProd = [](const std::vector<SpMat> &ops) -> SpMat {
    SpMat ret = ops[0];
    for (std::size_t k = 1; k < ops.size(); ++k)
      ret = Eigen::kroneckerProduct(ret, ops[k]).eval();
    return ret;
  };

  for_each_term([&](spin_op &term) {
    auto termStr = term.to_string(false);
    std::vector<SpMat> operations;
    for (std::size_t k = 0; k < termStr.length(); ++k) {
      auto pauli = termStr[k];
      if (pauli == 'X')
        operations.emplace_back(x);
      else if (pauli == 'Y')
        operations.emplace_back(y);
      else if (pauli == 'Z')
        operations.emplace_back(z);
      else
        operations.emplace_back(i);
    }

    mat += term.get_coefficient() * kronProd(operations);
  });

  std::vector<std::complex<double>> values;
  std::vector<std::size_t> rows, cols;
  for (int k = 0; k < mat.outerSize(); ++k)
    for (SpMat::InnerIterator it(mat, k); it; ++it) {
      values.emplace_back(it.value());
      rows.emplace_back(details::convertOrdering(n, it.row()));
      cols.emplace_back(details::convertOrdering(n, it.col()));
    }

  return std::make_tuple(values, rows, cols);
}

std::complex<double> spin_op::get_coefficient() const {
  if (terms.size() != 1)
    throw std::runtime_error(
        "spin_op::get_coefficient called on spin_op with > 1 terms.");
  return terms.begin()->second;
}

std::tuple<std::vector<double>, std::size_t> spin_op::getDataTuple() const {
  return std::tuple(getDataRepresentation(), num_qubits());
}

void spin_op::for_each_term(std::function<void(spin_op &)> &&functor) const {
  for (auto iter = terms.begin(), e = terms.end(); iter != e; ++iter) {
    const auto &pair = *iter;
    spin_op tmp(pair);
    functor(tmp);
  }
}
void spin_op::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const {
  if (num_terms() != 1)
    throw std::runtime_error(
        "spin_op::for_each_pauli on valid for spin_op with n_terms == 1.");

  auto nQ = num_qubits();
  auto bsf = terms.begin()->first;
  for (std::size_t i = 0; i < nQ; i++) {
    if (bsf[i] && bsf[i + nQ]) {
      functor(pauli::Y, i);
    } else if (bsf[i]) {
      functor(pauli::X, i);
    } else if (bsf[i + nQ]) {
      functor(pauli::Z, i);
    } else {
      functor(pauli::I, i);
    }
  }
}

spin_op spin_op::random(std::size_t nQubits, std::size_t nTerms,
                        unsigned int seed) {
  std::mt19937 gen(seed);
  std::vector<std::complex<double>> coeffs(nTerms, 1.0);
  std::vector<spin_op_term> randomTerms;
  // Make sure we don't put duplicates into randomTerms by using dupCheckSet
  std::set<std::vector<bool>> dupCheckSet;
  if (nQubits <= 30) {
    // For the given algorithm below that sets bool=true for 1/2 of the the
    // termData, the maximum number of unique terms is n choose k, where n =
    // 2*nQubits, and k=nQubits. For up to 30 qubits, we can calculate n choose
    // k without overflows (i.e. 60 choose 30 = 118264581564861424) to validate
    // that nTerms is reasonable. For anything larger, the user can't set nTerms
    // large enough to run into actual problems because they would encounter
    // memory limitations long before anything else.
    // Note: use the multiplicative formula to evaluate n-choose-k. The
    // arrangement of multiplications and divisions do not truncate any division
    // remainders.
    std::size_t maxTerms = 1;
    for (std::size_t i = 1; i <= nQubits; i++) {
      maxTerms *= 2 * nQubits + 1 - i;
      maxTerms /= i;
    }
    if (nTerms > maxTerms)
      throw std::runtime_error(
          fmt::format("Unable to produce {} unique random terms for {} qubits",
                      nTerms, nQubits));
  }
  for (std::size_t i = 0; i < nTerms; i++) {
    std::vector<bool> termData(2 * nQubits);
    while (true) {
      std::fill_n(termData.begin(), nQubits, true);
      std::shuffle(termData.begin(), termData.end(), gen);
      if (dupCheckSet.contains(termData)) {
        // Prepare to loop again
        std::fill(termData.begin(), termData.end(), false);
      } else {
        dupCheckSet.insert(termData);
        break;
      }
    }
    randomTerms.push_back(std::move(termData));
  }

  return spin_op(randomTerms, coeffs);
}

spin_op spin_op::from_word(const std::string &word) {
  auto numQubits = word.length();
  spin_op_term term(2 * numQubits);
  for (std::size_t i = 0; i < numQubits; i++) {
    auto letter = word[i];
    if (std::islower(letter))
      letter = std::toupper(letter);

    if (letter == 'Y') {
      term[i] = true;
      term[i + numQubits] = true;
    } else if (letter == 'X') {
      term[i] = true;
    } else if (letter == 'Z') {
      term[i + numQubits] = true;
    } else {

      if (letter != 'I')
        throw std::runtime_error(
            "Invalid Pauli for spin_op::from_word, must be X, Y, Z, or I.");
    }
  }

  return spin_op(term, 1.0);
}

void spin_op::expandToNQubits(const std::size_t numQubits) {
  auto iter = terms.begin();
  while (iter != terms.end()) {
    auto coeff = iter->second;
    std::vector<bool> tmp = iter->first;
    if (tmp.size() == numQubits * 2) {
      iter++;
      continue;
    }

    auto newSize = numQubits * 2 - tmp.size();
    for (std::size_t i = 0; i < newSize / 2; i++) {
      tmp.insert(tmp.begin() + tmp.size() / 2, 0);
      tmp.insert(tmp.begin() + tmp.size(), 0);
    }

    terms.erase(iter++);
    terms.emplace(tmp, coeff);
  }
}

spin_op &spin_op::operator+=(const spin_op &v) noexcept {
  auto otherNumQubits = v.num_qubits();

  spin_op tmpv = v;
  if (otherNumQubits > num_qubits())
    expandToNQubits(otherNumQubits);
  else if (otherNumQubits < num_qubits())
    tmpv.expandToNQubits(num_qubits());

  for (auto [term, coeff] : tmpv.terms) {
    auto iter = terms.find(term);
    if (iter != terms.end())
      iter->second += coeff;
    else
      terms.emplace(term, coeff);
  }

  return *this;
}

spin_op &spin_op::operator-=(const spin_op &v) noexcept {
  return operator+=(-1.0 * v);
}

spin_op &spin_op::operator*=(const spin_op &v) noexcept {
  using term_and_coeff = std::pair<std::vector<bool>, std::complex<double>>;
  std::size_t numTerms = num_terms() * v.num_terms();
  std::vector<term_and_coeff> result(numTerms);
  std::size_t min = std::min(num_terms(), v.num_terms());

  // Put the `unordered_map` iterators into vectors to minimize pointer chasing
  // when doing the cartesian product of the spin operators' terms.
  using Iter =
      std::unordered_map<spin_op_term, std::complex<double>>::const_iterator;
  std::vector<Iter> thisTermIt;
  std::vector<Iter> otherTermIt;
  thisTermIt.reserve(terms.size());
  otherTermIt.reserve(v.terms.size());
  for (auto it = terms.begin(); it != terms.end(); ++it)
    thisTermIt.push_back(it);
  for (auto it = v.terms.begin(); it != v.terms.end(); ++it)
    otherTermIt.push_back(it);

#if defined(_OPENMP)
  // Threshold to start OpenMP parallelization.
  // 16 ~ 4-term * 4-term
  constexpr std::size_t spin_op_omp_threshold = 16;
#pragma omp parallel for shared(result) if (numTerms > spin_op_omp_threshold)
#endif
  for (std::size_t i = 0; i < numTerms; ++i) {
    Iter s = thisTermIt[i % min];
    Iter t = otherTermIt[i / min];
    if (terms.size() > v.terms.size()) {
      s = thisTermIt[i / min];
      t = otherTermIt[i % min];
    }
    result[i] = details::mult(s->first, t->first, s->second, t->second);
  }

  terms.clear();
  terms.reserve(numTerms);
  for (auto &&[term, coeff] : result) {
    auto [it, created] = terms.emplace(term, coeff);
    if (!created)
      it->second += coeff;
  }
  return *this;
}

bool spin_op::is_identity() const {
  for (auto &[row, c] : terms)
    for (auto e : row)
      if (e)
        return false;

  return true;
}

bool spin_op::operator==(const spin_op &v) const noexcept {
  // Could be that the term is identity with all zeros
  bool isId1 = true, isId2 = true;
  for (auto &[row, c] : terms)
    for (auto e : row)
      if (e) {
        isId1 = false;
        break;
      }

  for (auto &[row, c] : v.terms)
    for (auto e : row)
      if (e) {
        isId2 = false;
        break;
      }

  if (isId1 && isId2)
    return true;

  for (auto &[k, c] : terms) {
    if (v.terms.find(k) == v.terms.end())
      return false;
  }
  return true;
}

spin_op &spin_op::operator*=(const double v) noexcept {
  for (auto &[term, coeff] : terms)
    coeff *= v;

  return *this;
}

spin_op &spin_op::operator*=(const std::complex<double> v) noexcept {
  for (auto &[term, coeff] : terms)
    coeff *= v;

  return *this;
}

std::size_t spin_op::num_qubits() const {
  if (terms.empty())
    return 0;
  return terms.begin()->first.size() / 2;
}

std::size_t spin_op::num_terms() const { return terms.size(); }

std::vector<spin_op> spin_op::distribute_terms(std::size_t numChunks) const {
  // Calculate how many terms we can equally divide amongst the chunks
  auto nTermsPerChunk = num_terms() / numChunks;
  auto leftover = num_terms() % numChunks;

  // Slice the given spin_op into subsets for each chunk
  std::vector<spin_op> spins;
  auto termIt = terms.begin();
  for (std::size_t chunkIx = 0; chunkIx < numChunks; chunkIx++) {
    // Evenly distribute any leftovers across the early chunks
    auto count = nTermsPerChunk + (chunkIx < leftover ? 1 : 0);

    // Get the chunk from the terms list.
    std::unordered_map<spin_op_term, std::complex<double>> sliced;
    std::copy_n(termIt, count, std::inserter(sliced, sliced.end()));

    // Add to the return vector
    spins.emplace_back(sliced);

    // Get ready for the next loop
    std::advance(termIt, count);
  }

  // return the terms.
  return spins;
}

std::string spin_op::to_string(bool printCoeffs) const {
  std::stringstream ss;
  const auto termToStr = [](const std::vector<bool> &term) {
    std::string printOut;
    printOut.reserve(term.size() / 2);
    for (std::size_t i = 0; i < term.size() / 2; i++) {
      if (term[i] && term[i + term.size() / 2])
        printOut.push_back('Y');
      else if (term[i])
        printOut.push_back('X');
      else if (term[i + term.size() / 2])
        printOut.push_back('Z');
      else
        printOut.push_back('I');
    }
    return printOut;
  };

  if (!printCoeffs) {
    std::vector<std::string> printOut;
    printOut.reserve(terms.size());
    for (auto &[term, coeff] : terms)
      printOut.emplace_back(termToStr(term));
    // IMPORTANT: For a printing without coefficients, we want to
    // sort the terms to get a consistent order for printing.
    // This is necessary because unordered_map does not maintain order and our
    // code relies on the full string representation as the key to look up full
    // expectation of the whole `spin_op`.
    // FIXME: Make the logic to look up whole expectation value from
    // `sample_result` more robust.
    std::sort(printOut.begin(), printOut.end());
    ss << fmt::format("{}", fmt::join(printOut, ""));
  } else {
    for (auto &[term, coeff] : terms) {
      ss << fmt::format("[{}{}{}j]", coeff.real(),
                        coeff.imag() < 0.0 ? "-" : "+", std::fabs(coeff.imag()))
         << " ";
      ss << termToStr(term);
      ss << "\n";
    }
  }

  return ss.str();
}

void spin_op::dump() const {
  auto str = to_string();
  std::cout << str;
}

spin_op::spin_op(const std::vector<double> &input_vec, std::size_t nQubits) {
  auto n_terms = (int)input_vec.back();
  if (nQubits != (((input_vec.size() - 1) - 2 * n_terms) / n_terms))
    throw std::runtime_error("Invalid data representation for construction "
                             "spin_op. Number of data elements is incorrect.");

  for (std::size_t i = 0; i < input_vec.size() - 1; i += nQubits + 2) {
    std::vector<bool> tmpv(2 * nQubits);
    for (std::size_t j = 0; j < nQubits; j++) {
      double intPart;
      if (std::modf(input_vec[j + i], &intPart) != 0.0)
        throw std::runtime_error(
            "Invalid pauli data element, must be integer value.");

      int val = (int)input_vec[j + i];
      if (val == 1) { // X
        tmpv[j] = 1;
      } else if (val == 2) { // Z
        tmpv[j + nQubits] = 1;
      } else if (val == 3) { // Y
        tmpv[j + nQubits] = 1;
        tmpv[j] = 1;
      }
    }
    auto el_real = input_vec[i + nQubits];
    auto el_imag = input_vec[i + nQubits + 1];
    terms.emplace(tmpv, std::complex<double>{el_real, el_imag});
  }
}

std::pair<std::vector<spin_op::spin_op_term>, std::vector<std::complex<double>>>
spin_op::get_raw_data() const {
  std::vector<spin_op_term> data;
  std::vector<std::complex<double>> coeffs;
  for (auto &[term, c] : terms) {
    data.push_back(term);
    coeffs.push_back(c);
  }

  return std::make_pair(data, coeffs);
}

spin_op &spin_op::operator=(const spin_op &other) {
  terms = other.terms;
  return *this;
}

spin_op operator+(double coeff, spin_op op) {
  return spin_op(op.num_qubits()) * coeff + op;
}
spin_op operator+(spin_op op, double coeff) {
  return op + spin_op(op.num_qubits()) * coeff;
}
spin_op operator-(double coeff, spin_op op) {
  return spin_op(op.num_qubits()) * coeff - op;
}
spin_op operator-(spin_op op, double coeff) {
  return op - spin_op(op.num_qubits()) * coeff;
}

namespace spin {
spin_op i(const std::size_t idx) { return spin_op(pauli::I, idx); }
spin_op x(const std::size_t idx) { return spin_op(pauli::X, idx); }
spin_op y(const std::size_t idx) { return spin_op(pauli::Y, idx); }
spin_op z(const std::size_t idx) { return spin_op(pauli::Z, idx); }
} // namespace spin

std::vector<double> spin_op::getDataRepresentation() const {
  std::vector<double> dataVec;
  for (auto &[term, coeff] : terms) {
    auto nq = term.size() / 2;
    for (std::size_t i = 0; i < nq; i++) {
      if (term[i] && term[i + nq]) {
        dataVec.push_back(3.);
      } else if (term[i]) {
        dataVec.push_back(1.);
      } else if (term[i + nq]) {
        dataVec.push_back(2.);
      } else {
        dataVec.push_back(0.);
      }
    }
    dataVec.push_back(coeff.real());
    dataVec.push_back(coeff.imag());
  }
  dataVec.push_back(num_terms());
  return dataVec;
}

spin_op binary_spin_op_reader::read(const std::string &data_filename) {
  std::ifstream input(data_filename, std::ios::binary);
  if (input.fail())
    throw std::runtime_error(data_filename + " does not exist.");

  input.seekg(0, std::ios_base::end);
  std::size_t size = input.tellg();
  input.seekg(0, std::ios_base::beg);
  std::vector<double> input_vec(size / sizeof(double));
  input.read((char *)&input_vec[0], size);
  auto n_terms = (int)input_vec.back();
  auto nQubits = (input_vec.size() - 1 - 2 * n_terms) / n_terms;
  spin_op s(input_vec, nQubits);
  return s;
}
} // namespace cudaq
