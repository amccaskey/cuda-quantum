/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/measure_result.h"
#include <vector>

namespace cudaq {

// Declares a parity constraint over the referenced measurement handles. Under
// noise-free execution the XOR of the referenced measurements is
// deterministic. In MLIR mode the compiler front-end intercepts calls to
// these functions and emits `qec.*` ops; in library mode these are no-ops
// today.
template <typename... Rs>
inline void detector(const Rs &...) {}

inline void detector(const std::vector<measure_result> &) {}

inline void logical_observable(const std::vector<measure_result> &) {}

} // namespace cudaq
