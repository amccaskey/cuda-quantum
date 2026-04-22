/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/qis/measure_result.h"
#include <type_traits>
#include <vector>

// These tests pin down the source-level contract of `cudaq::measure_result`
// per the approved design. The class must be:
//   * exactly 16 bytes  — `{int64_t value, int64_t unique_id}` layout
//   * trivially copyable — so it can flow through QIR `Result*` handles and
//     be memcpy'd freely
//   * implicitly convertible to `bool` (for conditional feedback) and
//     explicitly convertible to `int` / `double`
//   * equality-comparable with both other `measure_result` values and with
//     plain `bool`

CUDAQ_TEST(MeasureResultTester, checkLayout) {
  EXPECT_EQ(sizeof(cudaq::measure_result), 16u);
  EXPECT_TRUE(std::is_trivially_copyable_v<cudaq::measure_result>);
  EXPECT_TRUE(std::is_default_constructible_v<cudaq::measure_result>);
  EXPECT_TRUE(std::is_copy_constructible_v<cudaq::measure_result>);
  EXPECT_TRUE(std::is_move_constructible_v<cudaq::measure_result>);
  EXPECT_TRUE(std::is_copy_assignable_v<cudaq::measure_result>);
}

CUDAQ_TEST(MeasureResultTester, checkConversions) {
  cudaq::measure_result one{1};
  cudaq::measure_result zero{0};
  EXPECT_TRUE(static_cast<bool>(one));
  EXPECT_FALSE(static_cast<bool>(zero));
  EXPECT_EQ(static_cast<int>(one), 1);
  EXPECT_EQ(static_cast<int>(zero), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(one), 1.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(zero), 0.0);
}

CUDAQ_TEST(MeasureResultTester, checkEquality) {
  cudaq::measure_result a{1};
  cudaq::measure_result b{1};
  cudaq::measure_result c{0};
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  // Symmetric comparison against plain bool.
  EXPECT_TRUE(a == true);
  EXPECT_TRUE(true == a);
  EXPECT_TRUE(c == false);
  EXPECT_TRUE(false == c);
  EXPECT_TRUE(a != false);
  EXPECT_TRUE(false != a);
}

CUDAQ_TEST(MeasureResultTester, checkUniqueIdRoundTrip) {
  cudaq::measure_result m{1, /*unique_id=*/42};
  EXPECT_EQ(m.getValue(), 1);
  EXPECT_EQ(m.getUniqueId(), 42);

  cudaq::measure_result copy = m;
  EXPECT_EQ(copy.getValue(), 1);
  EXPECT_EQ(copy.getUniqueId(), 42);
}

CUDAQ_TEST(MeasureResultTester, checkToBoolVector) {
  std::vector<cudaq::measure_result> results;
  results.emplace_back(cudaq::measure_result{1, 0});
  results.emplace_back(cudaq::measure_result{0, 1});
  results.emplace_back(cudaq::measure_result{1, 2});
  auto bits = cudaq::to_bool_vector(results);
  ASSERT_EQ(bits.size(), 3u);
  EXPECT_TRUE(bits[0]);
  EXPECT_FALSE(bits[1]);
  EXPECT_TRUE(bits[2]);
}
