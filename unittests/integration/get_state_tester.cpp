/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include <cudaq/algorithm.h>
#include <numeric>

using namespace cudaq;

CUDAQ_TEST(GetStateTester, checkSimple) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    cx(q, r);
  };

  auto state = cudaq::get_state(kernel);
  state.dump();
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(0.5, state(0, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(0, 3).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 3).real(), 1e-3);
#else
  EXPECT_NEAR(1. / std::sqrt(2.), state[0].real(), 1e-3);
  EXPECT_NEAR(0., state[1].real(), 1e-3);
  EXPECT_NEAR(0., state[2].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state[3].real(), 1e-3);
#endif

  EXPECT_NEAR(state.overlap(state), 1.0, 1e-3);
}
