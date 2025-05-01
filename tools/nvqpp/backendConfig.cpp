/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef NVQPP_TARGET_OPTIONS
#define NVQPP_TARGET_OPTIONS ""
#endif

#ifndef NVQPP_TARGET_BACKEND_CONFIG
#define NVQPP_TARGET_BACKEND_CONFIG "qpp-cpu"
#endif

/// This file is meant to be used by the nvq++ driver script, the
/// NVQPP_TARGET_BACKEND_CONFIG string must be replaced (e.g. with sed)
/// with the actual target backend string.
#include <stdio.h>
// TODO: Replace this file with a compiler generated constant string and cleanup
// the driver.
namespace cudaq {
void set_target_backend(const char *, const char *);
}

static constexpr const char targetBackendName[] = NVQPP_TARGET_BACKEND_CONFIG;
static constexpr const char targetOptions[] = NVQPP_TARGET_OPTIONS;

__attribute__((constructor)) void setTargetBackend() {
  cudaq::set_target_backend(targetBackendName, targetOptions);
}
