/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %cpp_std %s -verify

#include "cudaq.h"

struct test {
  cudaq::qubit &r;
  cudaq::qview<> q;
};

__qpu__ test kernel(cudaq::qubit &q, cudaq::qview<> qq) { return test(q, qq); } // expected-error {{kernel result type not supported}}
