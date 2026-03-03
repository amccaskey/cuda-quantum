/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// Test 1: Multiple static kernel methods get unique names and distinct
// func.func definitions. Each operates on an inner quantum struct (block).
struct test {
  struct block {
    cudaq::qarray<3> q;
  };
  __qpu__ static void h(block &b) { cudaq::h(b.q); }
  __qpu__ static void x(block &b) { cudaq::x(b.q); }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test_h(
// CHECK-SAME:      %[[BLK:.*]]: !quake.struq<!quake.veq<3>>
// CHECK:           %[[Q:.*]] = quake.get_member %[[BLK]][0]
// CHECK:           quake.h
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test_x(
// CHECK-SAME:      %[[BLK:.*]]: !quake.struq<!quake.veq<3>>
// CHECK:           %[[Q:.*]] = quake.get_member %[[BLK]][0]
// CHECK:           quake.x
// CHECK:           return
// clang-format on

// Test 2: Free function calling static kernel methods produces call ops
// using the mlirgen names, not C++ mangled names.
__qpu__ void call_it() {
  test::block b;
  test::h(b);
  test::x(b);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_call_it
// CHECK:           %[[B:.*]] = quake.alloca !quake.struq<!quake.veq<3>>
// CHECK:           call @__nvqpp__mlirgen__test_h(%[[B]])
// CHECK:           call @__nvqpp__mlirgen__test_x(%[[B]])
// CHECK:           return
// CHECK-NOT:       call @_ZN4test
// clang-format on

// Test 3: Enum class parameters are lowered to their underlying integer type.
enum class color { red, green, blue };

struct with_enum {
  struct block {
    cudaq::qarray<2> q;
  };
  __qpu__ static void apply(block &b, color c) { cudaq::h(b.q); }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__with_enum_apply(
// CHECK-SAME:      %{{.*}}: !quake.struq<!quake.veq<2>>
// CHECK-SAME:      %{{.*}}: i32
// CHECK:           return
// clang-format on

// Test 4: Traditional operator() kernel with its own static method,
// calling static kernel methods from another struct.
struct caller {
  __qpu__ static void call_this(cudaq::qubit& q) {
    cudaq::h(q);
  }
  void operator()() __qpu__ {
    test::block b;
    test::h(b);
    test::x(b);
    call_this(b.q[0]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__caller_call_this(
// CHECK-SAME:      %[[Q:.*]]: !quake.ref
// CHECK:           quake.h %[[Q]]
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__caller()
// CHECK:           %[[B:.*]] = quake.alloca !quake.struq<!quake.veq<3>>
// CHECK:           call @__nvqpp__mlirgen__test_h(%[[B]])
// CHECK:           call @__nvqpp__mlirgen__test_x(%[[B]])
// CHECK:           call @__nvqpp__mlirgen__caller_call_this(
// CHECK:           return
// clang-format on
