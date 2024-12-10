/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "spin_handler.h"
#include "common/FmtCore.h"

namespace cudaq::experimental::spin {

std::pair<std::vector<std::size_t>, std::complex<double>>
mult(const std::vector<std::size_t> &p1, const std::vector<std::size_t> &p2,
     const std::complex<double> &p1Coeff, const std::complex<double> &p2Coeff) {
  auto [minSize, maxSize] = std::minmax({p1.size(), p2.size()});
  std::size_t minNumQubits = minSize / 2;
  std::size_t maxNumQubits = maxSize / 2;
  std::vector<std::size_t> result(maxSize, false);
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

  const std::vector<std::size_t> &big = p1.size() < p2.size() ? p2 : p1;
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

void spin_handler::expandToNQubits(std::vector<std::vector<std::size_t>> &terms,
                                   const std::size_t numQubits) {
  std::vector<std::vector<std::size_t>> reDone;
  for (std::size_t i = 0; i < terms.size(); i++) {
    std::vector<std::size_t> tmp = terms[i];
    if (tmp.size() == numQubits * 2) {
      reDone.push_back(tmp);
    } else {
      auto newSize = numQubits * 2 - tmp.size();
      for (std::size_t i = 0; i < newSize / 2; i++) {
        tmp.insert(tmp.begin() + tmp.size() / 2, 0);
        tmp.insert(tmp.begin() + tmp.size(), 0);
      }
      reDone.push_back(tmp);
    }
  }

  terms = reDone;
}

std::size_t
spin_handler::num_qubits(const details::operator_data &thisPtr) const {
  if (thisPtr.productTerms.empty())
    return 0;
  auto [data, coeffs] = thisPtr;
  return data[0].size() / 2;
}

/// @brief Add the given spin_op to this one and return *this
void spin_handler::addAssign(details::operator_data &thisPtr,
                             const details::operator_data &v) {
  auto &[data, coeffs] = thisPtr;
  auto [otherData, otherCoeffs] = v;
  auto otherNumQubits = v.productTerms[0].size() / 2;
  auto thisNumQubits = data[0].size() / 2;
  auto tmpv = otherData;
  if (otherNumQubits > thisNumQubits)
    expandToNQubits(data, otherNumQubits);
  else if (otherNumQubits < thisNumQubits)
    expandToNQubits(tmpv, thisNumQubits);

  for (std::size_t i = 0; auto &term : tmpv) {
    auto coeff = otherCoeffs[i++];
    auto iter = std::find(data.begin(), data.end(), term);
    auto idx = std::distance(data.begin(), iter);
    if (iter != data.end()) {
      coeffs[idx] = coeffs[idx] + coeff;
    } else {
      data.emplace_back(term);
      coeffs.push_back(coeff);
    }
  }

  return;
}

/// @brief Multiply the given spin_op with this one and return *this
void spin_handler::multAssign(details::operator_data &thisPtr,
                              const details::operator_data &v) {
  using term_and_coeff =
      std::pair<std::vector<std::size_t>, std::complex<double>>;
  auto numT = thisPtr.productTerms.size();
  auto otherNumT = v.productTerms.size();
  std::size_t numTerms = numT * otherNumT;
  std::vector<term_and_coeff> result(numTerms);
  std::size_t min = std::min(numT, otherNumT);

  // Put the `unordered_map` iterators into vectors to minimize pointer chasing
  // when doing the cartesian product of the spin operators' terms.
  using Iter = std::pair<std::vector<std::size_t>, std::complex<double>>;
  std::vector<Iter> thisTermIt;
  std::vector<Iter> otherTermIt;
  thisTermIt.reserve(thisPtr.productTerms.size());
  otherTermIt.reserve(v.productTerms.size());
  std::size_t i = 0;
  for (auto it = thisPtr.productTerms.begin(); it != thisPtr.productTerms.end();
       ++it)
    thisTermIt.push_back({*it, thisPtr.coefficients[i++].constant_value()});

  i = 0;
  for (auto it = v.productTerms.begin(); it != v.productTerms.end(); ++it)
    otherTermIt.push_back({*it, v.coefficients[i++].constant_value()});

#if defined(_OPENMP)
  // Threshold to start OpenMP parallelization.
  // 16 ~ 4-term * 4-term
  constexpr std::size_t spin_op_omp_threshold = 16;
#pragma omp parallel for shared(result) if (numTerms > spin_op_omp_threshold)
#endif
  for (std::size_t i = 0; i < numTerms; ++i) {
    Iter s = thisTermIt[i % min];
    Iter t = otherTermIt[i / min];
    if (numT > otherNumT) {
      s = thisTermIt[i / min];
      t = otherTermIt[i % min];
    }
    result[i] = mult(s.first, t.first, s.second, t.second);
  }

  // terms.clear();
  thisPtr.productTerms.clear();
  thisPtr.coefficients.clear();
  thisPtr.productTerms.reserve(numTerms);
  thisPtr.coefficients.reserve(numTerms);
  for (auto &&[term, coeff] : result) {
    auto iter = std::find(thisPtr.productTerms.begin(),
                          thisPtr.productTerms.end(), term);
    if (iter != thisPtr.productTerms.end()) {
      auto idx = std::distance(thisPtr.productTerms.begin(), iter);
      thisPtr.coefficients[idx] = thisPtr.coefficients[idx] + coeff;
    } else {
      thisPtr.productTerms.push_back(term);
      thisPtr.coefficients.push_back(coeff);
    }
  }
  return;
}

/// @brief Return true if this spin_op is equal to the given one. Equality
/// here does not consider the coefficients.
bool spin_handler::checkEquality(const details::operator_data &thisPtr,
                                 const details::operator_data &v) const {
  // Could be that the term is identity with all zeros
  bool isId1 = true, isId2 = true;
  for (auto &row : thisPtr.productTerms)
    for (auto e : row)
      if (e) {
        isId1 = false;
        break;
      }

  for (auto &row : v.productTerms)
    for (auto e : row)
      if (e) {
        isId2 = false;
        break;
      }

  if (isId1 && isId2)
    return true;

  for (auto &k : thisPtr.productTerms) {
    if (std::find(v.productTerms.begin(), v.productTerms.end(), k) ==
        v.productTerms.end())
      return false;
  }
  return true;
}

std::string spin_handler::to_string(const details::operator_data &thisPtr,
                                    bool printCoeffs) const {
  auto &[m_data, m_coefficients] = thisPtr;

  std::stringstream ss;
  const auto termToStr = [](const std::vector<std::size_t> &term) {
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
    printOut.reserve(m_data.size());
    for (auto &term : m_data)
      printOut.emplace_back(termToStr(term));
    // IMPORTANT: For a printing without coefficients, we want to
    // sort the terms to get a consistent order for printing.
    // This is necessary because unordered_map does not maintain order and our
    // code relies on the full string representation as the key to look up
    // full expectation of the whole `spin_op`.
    // FIXME: Make the logic to look up whole expectation value from
    // `sample_result` more robust.
    std::sort(printOut.begin(), printOut.end());
    ss << fmt::format("{}", fmt::join(printOut, ""));
  } else {
    for (std::size_t i = 0; auto &term : m_data) {
      auto coeff = m_coefficients[i++];
      if (coeff.has_value()) {
        ss << fmt::format("[{}{}{}j]", coeff.constant_value().real(),
                          coeff.constant_value().imag() < 0.0 ? "-" : "+",
                          std::fabs(coeff.constant_value().imag()))
           << " ";
      } else {
        ss << "f(params...) ";
      }
      ss << termToStr(term);
      ss << "\n";
    }
  }

  return ss.str();
}

spin_op i(std::size_t idx) {
  auto numQubits = idx + 1;
  std::vector<std::size_t> d(2 * numQubits);
  std::vector<std::vector<std::size_t>> data{d};
  return spin_op({data, {1.0}});
}

spin_op x(std::size_t idx) {
  auto numQubits = idx + 1;
  std::vector<std::size_t> d(2 * numQubits);
  d[idx] = 1;
  std::vector<std::vector<std::size_t>> data{d};
  return spin_op({data, {1.0}});
}

spin_op y(std::size_t idx) {
  auto numQubits = idx + 1;
  std::vector<std::size_t> d(2 * numQubits);
  d[idx] = 1;
  d[idx + numQubits] = 1;
  std::vector<std::vector<std::size_t>> data{d};
  return spin_op({data, {1.0}});
}

spin_op z(std::size_t idx) {
  auto numQubits = idx + 1;
  std::vector<std::size_t> d(2 * numQubits);
  d[idx + numQubits] = 1;
  std::vector<std::vector<std::size_t>> data{d};
  return spin_op({data, {1.0}});
}

spin_op from_word(const std::string &pauliWord) {
  auto numQubits = pauliWord.length();
  std::vector<std::size_t> term(2 * numQubits);
  for (std::size_t i = 0; i < numQubits; i++) {
    auto letter = pauliWord[i];
    if (std::islower(letter))
      letter = std::toupper(letter);

    if (letter == 'Y') {
      term[i] = 1;
      term[i + numQubits] = 1;
    } else if (letter == 'X') {
      term[i] = 1;
    } else if (letter == 'Z') {
      term[i + numQubits] = 1;
    } else {
      if (letter != 'I')
        throw std::runtime_error(
            "Invalid Pauli for spin_op::from_word, must be X, Y, Z, or I.");
    }
  }
  std::vector<std::vector<std::size_t>> data{term};
  return spin_op({data, {1.0}});
}

} // namespace cudaq::experimental::spin