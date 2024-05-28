/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct lower_ctrl_as_qreg {
  void operator()() __qpu__ {
    cudaq::qvector reg1(4); // group of controls
    cudaq::qvector reg2(2); // some targets

    h<cudaq::ctrl>(reg1, reg2[0]);
    x<cudaq::ctrl>(reg1, reg2[1]);

    mz(reg2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__lower_ctrl_as_qreg() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h [%[[VAL_0]]] %[[VAL_2]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_0]]] %[[VAL_3]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_1]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
