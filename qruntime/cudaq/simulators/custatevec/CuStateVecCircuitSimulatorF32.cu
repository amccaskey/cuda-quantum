/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_CUSTATEVEC_TOGGLE_CREATE
#include "CuStateVecCircuitSimulator.cu"
CUDAQ_SIMULATOR_INITIALIZER(CuStateVecCircuitSimulator<float>)
#undef __NVQIR_CUSTATEVEC_TOGGLE_CREATE
