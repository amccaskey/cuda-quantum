/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform.h"
#include <gtest/gtest.h>

TEST(RuntimeTester, checkSimple) {
  {
    // We get a default qpu configured by command line compile flags.
    // Here we know its the qpp simulator behind the circuit_simulator qpu type
    auto &simulator = cudaq::get_qpu();
    EXPECT_EQ(cudaq::get_num_qpus(), 1);
    EXPECT_EQ(simulator.get_uid(), 0);
    EXPECT_FALSE(simulator.is_remote());
    EXPECT_FALSE(simulator.is_emulator());
    EXPECT_TRUE(simulator.is_simulator());
    EXPECT_TRUE(simulator.isa<cudaq::simulation_trait>());
    auto specs = simulator.get_specs();
    EXPECT_EQ(specs.name, "circuit_simulator");
  }
  {
    // One can create qpus manually if they'd like
    auto &simulator = cudaq::create_qpu(
        "circuit_simulator", cudaq::heterogeneous_map{{"simulator", "stim"}});

    // default sim is still there
    auto &defaultSim = cudaq::get_qpu();
    EXPECT_EQ(cudaq::get_num_qpus(), 1);
    EXPECT_EQ(defaultSim.get_uid(), 0);
    EXPECT_FALSE(defaultSim.is_remote());
    EXPECT_FALSE(defaultSim.is_emulator());
    EXPECT_TRUE(defaultSim.is_simulator());
    EXPECT_TRUE(defaultSim.isa<cudaq::simulation_trait>());

    // But ours is different
    EXPECT_EQ(simulator.get_uid(), 1);
    EXPECT_FALSE(simulator.is_remote());
    EXPECT_FALSE(simulator.is_emulator());
    EXPECT_TRUE(simulator.is_simulator());
    EXPECT_TRUE(simulator.isa<cudaq::simulation_trait>());

    // this newly created one can be pushed on the stack and used
    // by the main API
    cudaq::push_qpu(&simulator);
    auto &sameQpuFromAPI = cudaq::get_qpu();
    EXPECT_EQ(sameQpuFromAPI.get_uid(), 1);
    EXPECT_FALSE(sameQpuFromAPI.is_remote());
    EXPECT_FALSE(sameQpuFromAPI.is_emulator());
    EXPECT_TRUE(sameQpuFromAPI.is_simulator());
    EXPECT_TRUE(sameQpuFromAPI.isa<cudaq::simulation_trait>());
    cudaq::pop_qpu();

    // Should be back on the initial default qpu, id 0
    EXPECT_EQ(cudaq::get_qpu().get_uid(), 0);
  }
}
