/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

struct test_allocation {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    cudaq::qubit r;

    cnot(q, r);
    mz(q);
    mz(r);
  }
};

struct test_resizing {
  void operator()() __qpu__ {
    // Start with an initial allocation of 2 qubits.
    cudaq::qvector q(2);
    cudaq::x(q);
    auto result = mz(q[0]);
    auto result1 = mz(q[1]);
    if (result && result1) {
      // Allocate two more qubits mid-circuit.
      cudaq::qvector q2(2);
      auto result2 = mz(q2);
    }
  }
};

CUDAQ_TEST(AllocationTester, checkSimple) {
  test_allocation{}();

  auto counts = cudaq::sample(test_allocation{});
  EXPECT_EQ(2, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
  EXPECT_EQ(c, 1000);
}

#ifndef CUDAQ_BACKEND_TENSORNET
// Tests for a previous bug in the density simulator, where
// the qubit ordering flipped after resizing the density matrix
// with new qubits.
CUDAQ_TEST(AllocationTester, checkDensityOrderingBug) {
  test_resizing{}();

  auto counts = cudaq::sample(test_resizing{});
  EXPECT_EQ(1, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "1100");
  }
  EXPECT_EQ(c, 1000);
}
#endif

CUDAQ_TEST(TestNewQubitFunc, checkSimple) {
  {
    cudaq::qubit q;
    cudaq::applyQuakeExtOperation("h", 1, q);
    mz(q);
  }

  {
    cudaq::qubit q, r;
    cudaq::applyQuakeExtOperation("h", 1, q, r);
    mz(q);
  }

  {
    cudaq::qvector q(3);
    cudaq::applyQuakeExtOperation("h", 1, q);
    mz(q);
  }

  {
    cudaq::qubit q, r;
    cudaq::applyQuakeExtOperation<cudaq::ctrl>("x", 1, q, r);
    mz(q);
  }

  {
    cudaq::qubit q, r, s;
    cudaq::applyQuakeExtOperation<cudaq::ctrl>("x", 1, q, r, s);
    mz(q);
  }

  {
    cudaq::qubit q;
    cudaq::applyQuakeExtOperation("rx", 1, M_PI_2, q);
    mz(q);
  }

  {
    cudaq::qvector q(3);
    cudaq::applyQuakeExtOperation("rx", 1, M_PI_2, q);
    mz(q);
  }

  {
    cudaq::qubit q, r;
    cudaq::applyQuakeExtOperation<cudaq::ctrl>("rx", 1, M_PI_2, q, r);
    mz(q);
  }

  {
    cudaq::qubit q;
    cudaq::applyQuakeExtOperation<cudaq::adj>("rx", 1, M_PI_2, q);
    mz(q);
  }
}