
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qvector.h"
#include <vector>

namespace cudaq {
namespace details {
std::complex<double> &element(std::vector<std::complex<double>> &mat,
                              std::size_t Levels, std::size_t i,
                              std::size_t j) {
  return mat[i * Levels + j];
}

double _fast_factorial(int n) {
  static std::vector<double> FACTORIAL_TABLE = {
      1.,
      1.,
      2.,
      6.,
      24.,
      120.,
      720.,
      5040.,
      40320.,
      362880.,
      3628800.,
      39916800.,
      479001600.,
      6227020800.,
      87178291200.,
      1307674368000.,
      20922789888000.,
      355687428096000.,
      6402373705728000.,
      121645100408832000.,
      2432902008176640000.,
      51090942171709440000.,
      1124000727777607680000.,
      25852016738884976640000.,
      620448401733239439360000.,
      15511210043330985984000000.,
      403291461126605635584000000.,
      10888869450418352160768000000.,
      304888344611713860501504000000.,
      8841761993739701954543616000000.,
      265252859812191058636308480000000.,
  };
  if (n >
      30) { // We do not expect to get 30 photons in the loop at the same time
    throw std::invalid_argument("received invalid value, n <= 30");
  }
  return FACTORIAL_TABLE[n];
}

/// @brief Computes a single element in the matrix representing a beam
/// splitter gate
double _calc_beam_splitter_elem(int N1, int N2, int n1, int n2, double theta) {

  const double t = cos(theta); // transmission coefficient
  const double r = sin(theta); // reflection coefficient
  double sum = 0;
  for (int k = 0; k <= n1; ++k) {
    int l = N1 - k;
    if (l >= 0 && l <= n2) {
      double term1 = pow(r, (n1 - k + l)) * pow(t, (n2 + k - l));
      if (term1 == 0) {
        continue;
      }
      double term2 = pow((-1), (l)) *
                     (sqrt(_fast_factorial(n1)) * sqrt(_fast_factorial(n2)) *
                      sqrt(_fast_factorial(N1)) * sqrt(_fast_factorial(N2)));
      double term3 = (_fast_factorial(k) * _fast_factorial(n1 - k) *
                      _fast_factorial(l) * _fast_factorial(n2 - l));
      double term = term1 * term2 / term3;
      sum += term;
    } else {
      continue;
    }
  }

  return sum;
}
void beam_splitter(const double theta, std::size_t Levels,
                   std::vector<std::complex<double>> &BS) {
  int d = sqrt(Levels * Levels);
  //     """Returns a matrix representing a beam splitter
  for (int n1 = 0; n1 < d; ++n1) {
    for (int n2 = 0; n2 < d; ++n2) {
      int nxx = n1 + n2;
      int nxd = std::min(nxx + 1, d);
      for (int N1 = 0; N1 < nxd; ++N1) {
        int N2 = nxx - N1;
        if (N2 >= nxd) {
          continue;
        } else {
          details::element(BS, Levels * Levels, n1 * d + n2, N1 * d + N2) =
              // BS(n1 * d + n2, N1 * d + N2) =
              _calc_beam_splitter_elem(N1, N2, n1, n2, theta);
        }
      }
    }
  }
}
} // namespace details
/// @brief The `create` gate
// U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |d>
template <std::size_t Levels>
void create(qudit<Levels> &q) {
  std::vector<std::complex<double>> mat(Levels * Levels);
  details::element(mat, Levels, Levels - 1, Levels - 1) = 1;
  for (int i = 1; i < Levels; i++)
    details::element(mat, Levels, i, i - 1) = 1;

  v2::get_qpu().as<v2::simulation_trait>()->apply(mat, {}, {q.id()});
}

/// @brief The `annihilate` gate
// U|0> -> |0>, U|1> -> |0>, ..., and U|d> -> |d-1>
template <std::size_t Levels>
void annihilate(qudit<Levels> &q) {

  std::vector<std::complex<double>> mat(Levels * Levels);
  details::element(mat, Levels, 0, 0) = 1;
  for (int i = 0; i < Levels - 1; i++)
    details::element(mat, Levels, i, i + 1) = 1;

  v2::get_qpu().as<v2::simulation_trait>()->apply(mat, {}, {q.id()});
}

/// @brief The `plus` gate
// U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |0>
template <std::size_t Levels>
void plus(cudaq::qudit<Levels> &q) {
  std::vector<std::complex<double>> mat(Levels * Levels);
  details::element(mat, Levels, 0, Levels - 1) = 1;
  for (int i = 1; i < Levels; i++)
    details::element(mat, Levels, i, i - 1) = 1;

  v2::get_qpu().as<v2::simulation_trait>()->apply(mat, {}, {q.id()});
}

/// @brief The `phase shift` gate
template <std::size_t Levels>
void phase_shift(cudaq::qudit<Levels> &q, const double &phi) {
  std::vector<std::complex<double>> mat(Levels * Levels);
  for (int i = 0; i < Levels; i++)
    details::element(mat, Levels, i, i) =
        std::exp(i * phi * std::complex<double>(0, 1.));

  v2::get_qpu().as<v2::simulation_trait>()->apply(mat, {}, {q.id()});
}

/// @brief The `beam splitter` gate
template <std::size_t Levels>
void beam_splitter(cudaq::qudit<Levels> &q, cudaq::qudit<Levels> &r,
                   const double &theta) {
  std::vector<std::complex<double>> mat(Levels * Levels * Levels * Levels);
  details::beam_splitter(theta, Levels, mat);
  v2::get_qpu().as<v2::simulation_trait>()->apply(mat, {}, {q.id(), r.id()});
}

/// @brief Measure a qudit
template <std::size_t Levels>
int mz(cudaq::qudit<Levels> &q) {
  return v2::get_qpu().as<v2::simulation_trait>()->mz(q.id(), "");
}

/// @brief Measure a vector of qudits
template <std::size_t Levels>
std::vector<int> mz(cudaq::qvector<Levels> &q) {
  std::vector<int> ret;
  for (auto &qq : q)
    ret.emplace_back(mz(qq));
  return ret;
}
} // namespace cudaq
