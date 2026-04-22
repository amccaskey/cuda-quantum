/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace cudaq {

extern "C" {
bool __nvqpp__MeasureResultBoolConversion(std::int64_t);
}

class measure_result {
private:
  std::int64_t value = 0;
  std::int64_t unique_id = std::numeric_limits<std::int64_t>::max();

public:
  measure_result() = default;
  explicit measure_result(std::int64_t val) : value(val) {}
  explicit measure_result(std::int64_t val, std::int64_t id)
      : value(val), unique_id(id) {}

  measure_result(const measure_result &) = default;
  measure_result(measure_result &&) = default;
  measure_result &operator=(const measure_result &) = default;
  measure_result &operator=(measure_result &&) = default;

  operator bool() const { return __nvqpp__MeasureResultBoolConversion(value); }
  explicit operator int() const { return static_cast<int>(value); }
  explicit operator double() const { return static_cast<double>(value); }

  std::int64_t getValue() const { return value; }
  std::int64_t getUniqueId() const { return unique_id; }

  friend bool operator==(const measure_result &a, const measure_result &b) {
    return static_cast<bool>(a) == static_cast<bool>(b);
  }
  friend bool operator==(const measure_result &a, bool b) {
    return static_cast<bool>(a) == b;
  }
  friend bool operator==(bool b, const measure_result &a) {
    return static_cast<bool>(a) == b;
  }
  friend bool operator!=(const measure_result &a, const measure_result &b) {
    return !(a == b);
  }
  friend bool operator!=(const measure_result &a, bool b) { return !(a == b); }
  friend bool operator!=(bool b, const measure_result &a) { return !(a == b); }
};

static_assert(sizeof(measure_result) == 16,
              "measure_result must be a 16-byte {i64,i64} layout");
static_assert(std::is_trivially_copyable_v<measure_result>,
              "measure_result must be trivially copyable");

inline std::vector<bool>
to_bool_vector(const std::vector<measure_result> &results) {
  std::vector<bool> out;
  out.reserve(results.size());
  for (const auto &m : results)
    out.push_back(static_cast<bool>(m));
  return out;
}

} // namespace cudaq
