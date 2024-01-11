/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/FmtCore.h"
#include <cudaq.h>
#include <iostream>

/// Helper to printout vector<measure_result>
template <>
struct fmt::formatter<cudaq::measure_result> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const cudaq::measure_result &result, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{0}", static_cast<int>(result));
  }
};

__qpu__ void test0() {
  cudaq::qubit q;
  x(q);
  // implicit measurements
}

__qpu__ void test1() {
  cudaq::qvector q(3);
  x(q);
  // implicit measurements
}

__qpu__ void test2() {
  cudaq::qvector q(3);
  x(q);
  // measures specified, no registers
  mz(q);
}

__qpu__ void test3() {
  cudaq::qubit q;
  x(q);
  // measures specified, no registers
  mz(q);
}

__qpu__ void test4() {
  cudaq::qubit q;
  x(q);
  auto a = mz(q);
}

__qpu__ void test5() {
  cudaq::qvector q(3);
  x(q);
  auto a = mz(q);
}

__qpu__ void test6() {
  cudaq::qubit q, r;
  h(q);
  auto a = mz(q);
  if (a)
    x(r);
}

__qpu__ void test7(const int n_iter) {
  cudaq::qubit q0;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    auto q0result = mz(q0);
    if (q0result)
      break; // loop until it lands heads
  }
}

__qpu__ auto test8() {
  cudaq::qvector q(3);
  h(q);
  return mz(q);
}

__qpu__ void test9(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    if (mz(q0))
      x(q1); // toggle q1 on every q0 coin toss that lands heads
  }
  auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                          // the q0 measurements
}

__qpu__ void test10(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    auto q0result = mz(q0);
    if (q0result)
      x(q1); // toggle q1 on every q0 coin toss that lands heads
  }
  auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                          // the q0 measurements
}

__qpu__ void test11(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  std::vector<cudaq::measure_result> resultVector(n_iter);
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    resultVector[i] = mz(q0);
    if (resultVector[i])
      x(q1); // toggle q1 on every q0 coin toss that lands heads
  }
  auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                          // the q0 measurements
}

int main() {
  auto nShots = 20;

  auto vectorContains = [](const std::vector<std::string> &vector,
                           const std::string &element) {
    return std::find(vector.begin(), vector.end(), element) != vector.end();
  };

  auto vectorToString = [](const auto &vector, const std::string &delim = ",") {
    return fmt::format("{}", fmt::join(vector, delim));
  };

  // {
  //   std::cout << "Test 0 (implicit qubit measure):\n";
  //   auto counts = cudaq::sample(test0);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "__global__") &&
  //          "results had invalid register names");
  //   assert(counts.register_names().size() == 1);
  // }

  // {
  //   std::cout << "Test 1 (implicit qvector measure):\n";
  //   auto counts = cudaq::sample(test1);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "__global__") &&
  //          "results had invalid register names");
  //   assert(counts.register_names().size() == 1);
  // }

  // {
  //   std::cout << "Test 2 (measure qubit, no register name):\n";
  //   auto counts = cudaq::sample(test2);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "__global__") &&
  //          "results had invalid register names");
  //   assert(counts.register_names().size() == 1);
  // }

  // {
  //   std::cout << "Test 3 (measure qvector, no register name):\n";
  //   auto counts = cudaq::sample(test3);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "__global__") &&
  //          "results had invalid register names");
  //   assert(counts.register_names().size() == 1);
  // }

  // {
  //   std::cout << "Test 4 (measure qubit to register):\n";
  //   auto counts = cudaq::sample(test4);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "a") &&
  //          "results had invalid register names");
  //   assert(counts.register_names().size() == 1);
  // }

  // {
  //   std::cout << "\nTest 5 (measure qvector to register):\n";
  //   auto counts = cudaq::sample(test5);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "a") &&
  //          "results had invalid register names");
  //   assert(counts.to_map("a").size() == 1);
  //   assert(counts.register_names().size() == 1);
  // }

  {
    std::cout << "\nTest 6 (measure qubit to register, adaptive):\n";
    auto counts = cudaq::sample(20, test6);
    counts.dump();
    // assert(vectorContains(counts.register_names(), "__global__") &&
    //        "results had invalid register names");
    // assert(vectorContains(counts.register_names(), "a") &&
    //        "results had invalid register names");
    // assert(counts.adaptive_results("a").size() == nShots);
  }

  // {
  //   std::cout << "\nTest 3 (measure qubit to register, adaptive complex
  //   control "
  //                "flow):\n";
  //   auto nIters = 20;
  //   auto counts = cudaq::sample(nShots, test3, nIters);
  //   counts.dump();
  //   assert(vectorContains(counts.register_names(), "__global__") &&
  //          "results had invalid register names");
  //   assert(vectorContains(counts.register_names(), "q0result") &&
  //          "results had invalid register names");
  //   assert(counts.adaptive_results("q0result").size() == nShots);

  //   for (std::size_t i = 0; i < nShots; i++) {
  //     std::cout << "shot " << i << ": "
  //               << vectorToString(counts.adaptive_results("q0result")[i])
  //               << "\n";

  //     auto measurementsAtShot = counts.adaptive_results_at_shot(i,
  //     "q0result"); auto numOnes =
  //         std::count(measurementsAtShot.begin(), measurementsAtShot.end(),
  //         "1");
  //     assert(
  //         numOnes == 1 &&
  //         "there should be a single 1 in the measurements due to the
  //         break.");
  //   }
  // }

  // {
  //   std::cout << "\nTest 4 (sample with return type):\n";
  //   auto results = cudaq::sample(nShots, test4);
  //   for (std::size_t shot = 0; auto r : results)
  //     std::cout << "shot " << shot++ << ": " << vectorToString(r, "") <<
  //     "\n";

  //   assert(results.size() == nShots);
  // }

  // {
  //   std::cout
  //       << "\nTest 5 (adaptive, control flow ignore inner loop measures):\n";
  //   auto counts = cudaq::sample(nShots, test5, 20);
  //   counts.dump();
  //   auto q1result_0 = counts.count("0", "q1result");
  //   auto q1result_1 = counts.count("1", "q1result");
  //   assert(q1result_0 + q1result_1 == nShots && "invalid number of
  //   results.");
  // }

  // {
  //   std::cout
  //       << "\nTest 6 (adaptive, control flow retain inner loop measures):\n";
  //   auto counts = cudaq::sample(nShots, test6, 20);
  //   counts.dump();
  //   for (auto &[shot, result] : counts.adaptive_results("q0result"))
  //     assert(result.size() == 20);

  //   for (auto &[shot, result] : counts.adaptive_results("q1result"))
  //     assert(result.size() == 1);
  // }

  // {
  //   std::cout << "\nTest 7 (adaptive, control flow save to allocated vector "
  //                "register):\n";
  //   auto nIter = 15;
  //   auto counts = cudaq::sample(nShots, test7, nIter);
  //   counts.dump();
  //   for (auto shot : cudaq::range(nIter)) {
  //     auto resultsVecAtShot =
  //         counts.adaptive_results(fmt::format("resultVector", shot));
  //     assert(resultsVecAtShot.size() == nShots);
  //   }
  // }
}

__qpu__ void testAlloc(cudaq::complex * state) {
  // Initialize from amplitudes
  cudaq::qubit r = {1.0, 0.0};
  // Initialize from builtins
  cudaq::qubit q = cudaq::ket::one;
  // Initialize default
  cudaq::qubit defaultIsZero;
  // Initialize from state
  cudaq::qvector qv = state; 
}

int main () {

  auto kernel = cudaq::make_kernel();
  auto q = kernel.qalloc();
}