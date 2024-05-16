/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "LinkedLibraryHolder.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq {
void bindAltLaunchKernel(LinkedLibraryHolder& holder, py::module &mod);
} // namespace cudaq