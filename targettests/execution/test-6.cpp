/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ -fkernel-exec-kind=2 --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void cccx_measure_cleanup() {
  cudaq::qvector qubits(4);
  // Initialize
  x(qubits[0]);
  x(qubits[1]);
  x(qubits[2]);

  // Compute AND-operation
  cudaq::qubit ancilla;
  h(ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  x<cudaq::ctrl>(qubits[0], ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  h(ancilla);
  s<cudaq::adj>(ancilla);

  // Compute output
  x<cudaq::ctrl>(qubits[2], ancilla, qubits[3]);

  // AND's measurement based cleanup.
  bool result = mx(ancilla);
  if (result)
    z<cudaq::ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

int main() {
  auto result = cudaq::sample(10, cccx_measure_cleanup);
  std::cout << result.most_probable() << '\n';
  result.dump();
  return 0;
}

// CHECK: 1111
