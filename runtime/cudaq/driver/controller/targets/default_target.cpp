/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/controller/target.h"

#define CUDAQ_RTTI_DISABLED
// these headers use typeid, but we've turned it off
#include "common/PluginUtils.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/CircuitSimulator.h"

#undef CUDAQ_RTTI_DISABLED

#include "llvm/ADT/StringSwitch.h"

#include <filesystem>
#include <iostream>

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::target)

namespace cudaq::driver {

class default_target : public target {
protected:
  nvqir::CircuitSimulator *simulator;

public:
  void initialize(const config::TargetConfig &config) override {
    std::string nameToLookFor = "qpp";
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto libPath = cudaqLibPath.parent_path().parent_path() / "lib";

    // Pattern to look for: libnvqir-qpp.so
    std::string targetPattern = "libnvqir-" + nameToLookFor;
    std::filesystem::path libraryPath;

    // Iterate through all files in the directory
    for (const auto &entry : std::filesystem::directory_iterator(libPath)) {
      if (!entry.is_regular_file())
        continue;
      // Get the filename without path
      std::filesystem::path filePath = entry.path();
      std::string filename = filePath.filename().string();

      // Check if it's a shared library
      if (filePath.extension() != ".so")
        continue;

      // Get the stem (filename without extension)
      std::string stem = filePath.stem().string();

      // Check if it matches our pattern
      if (stem.find(targetPattern) != std::string::npos) {
        libraryPath = filePath;
        break;
      }
    }
    // FIXME Error Handling

    // Found the library, use it
    // Additional initialization with the found library
    dlopen(libraryPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW);
    simulator = getUniquePluginInstance<nvqir::CircuitSimulator>(
        std::string("getCircuitSimulator_") + nameToLookFor);
  }

  void allocate(std::size_t num) override {
    cudaq::info("Default Target is allocating {} qubits.", num);
    simulator->allocateQubits(num);
  }

  void deallocate(std::size_t num) override {
    for (std::size_t i = 0; i < num; i++) 
      simulator->deallocate(i);
  }

  void apply_opcode(const std::string &opCode,
                    const std::vector<double> &parameters,
                    const std::vector<std::size_t> &qudits) override {
    std::vector<std::size_t> localC, localT;
    localT.push_back(qudits.back());
    if (qudits.size() > 1)
      for (std::size_t i = 0; i < qudits.size() - 1; i++)
        localC.push_back(qudits[i]);

    llvm::StringSwitch<std::function<void()>>(opCode)
        .Case("h", [&]() { simulator->h(localC, localT[0]); })
        .Case("x", [&]() { simulator->x(localC, localT[0]); })
        .Case("y", [&]() { simulator->y(localC, localT[0]); })
        .Case("z", [&]() { simulator->z(localC, localT[0]); })
        .Case("rx", [&]() { simulator->rx(parameters[0], localC, localT[0]); })
        .Case("ry", [&]() { simulator->ry(parameters[0], localC, localT[0]); })
        .Case("rz", [&]() { simulator->rz(parameters[0], localC, localT[0]); })
        .Case("s", [&]() { simulator->s(localC, localT[0]); })
        .Case("t", [&]() { simulator->t(localC, localT[0]); })
        .Case("sdg", [&]() { simulator->sdg(localC, localT[0]); })
        .Case("tdg", [&]() { simulator->tdg(localC, localT[0]); })
        .Case("r1", [&]() { simulator->r1(parameters[0], localC, localT[0]); })
        .Case("u1", [&]() { simulator->u1(parameters[0], localC, localT[0]); })
        .Case("u3",
              [&]() {
                simulator->u3(parameters[0], parameters[1], parameters[2],
                              localC, localT[0]);
              })
        .Case("swap", [&]() { simulator->swap(localC, localT[0], localT[1]); })
        .Default([&]() {
          throw std::runtime_error("[Default Target] invalid gate "
                                   "application requested " +
                                   opCode + ".");
        })();
  }

  std::size_t measure_z(std::size_t qudit) override {
    return simulator->mz(qudit);
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(target, default_target);
};
CUDAQ_REGISTER_EXTENSION_TYPE(default_target)

} // namespace cudaq::driver