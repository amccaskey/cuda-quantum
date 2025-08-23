/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/platform.h"

/// This following functions form a primitive default
/// instruction set for this simple qudit execution manager.
/// You could imagine these in their own header file, like we do
/// for qubit_qis.h

using namespace cudaq;

// Plus Gate : U|0> -> |1>, U|1> -> |2>, and U|2> -> |0>
void plusGate(cudaq::qudit<3> &q) {
  get_qpu().as<simulation_trait>()->apply({0, 0, 1, 1, 0, 0, 0, 1, 0},
                                                  {}, {q.id()});
}

int mz(cudaq::qudit<3> &q) {
  return get_qpu().as<simulation_trait>()->mz(q.id());
}

std::vector<int> mz(cudaq::qvector<3> &q) {
  std::vector<int> ret;
  for (auto &qq : q)
    ret.emplace_back(mz(qq));
  return ret;
}

TEST(SimpleQuditTester, checkSimple) {

  struct test {
    auto operator()() __qpu__ {
      cudaq::qvector<3> qutrits(2);
      plusGate(qutrits[0]);
      plusGate(qutrits[1]);
      plusGate(qutrits[1]);
      return mz(qutrits);
    }
  };

  struct test2 {
    void operator()() __qpu__ {
      cudaq::qvector<3> qutrits(2);
      plusGate(qutrits[0]);
      plusGate(qutrits[1]);
      plusGate(qutrits[1]);
      mz(qutrits);
    }
  };

  auto res = test{}();
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], 2);

  cudaq::sample(test2{});
}