/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include <functional>
#include <vector>

namespace cudaq::quera {

struct atom {
  const double coordinates[3];
};

struct qvector {
  // made up of qubits and positions
  std::vector<atom> atoms;
  qvector(const std::initializer_list<atom> &l) : atoms(l.begin(), l.end()) {}
};

struct analog_hamiltonian {
  std::function<double(double)> amplitude, detuning, phase;
  analog_hamiltonian(const std::function<double(double)> &a,
                     const std::function<double(double)> &d)
      : amplitude(a), detuning(d), phase([](double) { return 0.0; }) {}
  analog_hamiltonian &apply(quera::qvector &q) { return *this; }
};

struct KernelArguments {
  double T;
  double step;
  quera::analog_hamiltonian hamiltonian;
};

sample_result sample(const std::function<quera::analog_hamiltonian()> &kernel,
                     double T, double step) {
  auto H = kernel();
  KernelArguments args{T, step, H};
  cudaq::altLaunchKernel("quera_launch", nullptr, &args,
                         sizeof(KernelArguments), 0);

  return sample_result();
}
}; // namespace cudaq::quera
