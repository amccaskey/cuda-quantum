# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(simulated)
add_subdirectory(photonics)
add_subdirectory(remote)
add_subdirectory(pasqal)
add_subdirectory(fermioniq)
add_subdirectory(orca)
add_subdirectory(quera)

add_target_config(opt-test)
