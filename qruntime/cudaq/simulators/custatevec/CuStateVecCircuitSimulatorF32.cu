/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_CUSTATEVEC_TOGGLE_CREATE
#include "CuStateVecCircuitSimulator.cu"
template <>
const bool cudaq::CircuitSimulatorBase<
    float, CuStateVecCircuitSimulator<float>>::registered_ =
    cudaq::CircuitSimulatorBase<
        float, CuStateVecCircuitSimulator<float>>::register_type();
#undef __NVQIR_CUSTATEVEC_TOGGLE_CREATE
