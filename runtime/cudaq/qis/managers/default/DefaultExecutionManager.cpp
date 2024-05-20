/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/ADT/StringSwitch.h"

#include "common/Logger.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

#include "nlohmann/json.hpp"

#include "nvqir/CircuitSimulator.h"

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

namespace {

using GenQuakeExtUnitaryFunctor = void (*)(double *, std::size_t,
                                           std::complex<double> *);

/// The DefaultExecutionManager will implement allocation, deallocation, and
/// quantum instruction application via calls to the current CircuitSimulator
class DefaultExecutionManager : public cudaq::BasicExecutionManager {

private:
  nvqir::CircuitSimulator *simulator() {
    return nvqir::getCircuitSimulatorInternal();
  }
  /// @brief To improve `qudit` allocation, we defer
  /// single `qudit` allocation requests until the first
  /// encountered `apply` call.
  std::vector<cudaq::QuditInfo> requestedAllocations;

  /// @brief Allocate all requested `qudits`.
  void flushRequestedAllocations() {
    if (requestedAllocations.empty())
      return;

    allocateQudits(requestedAllocations);
    requestedAllocations.clear();
  }

  std::map<std::string, GenQuakeExtUnitaryFunctor> quakeExtUnitaryFunctors;

protected:
  void allocateQudit(const cudaq::QuditInfo &q) override {
    requestedAllocations.emplace_back(2, q.id);
  }

  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    simulator()->allocateQubits(qudits.size());
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {

    // Before trying to deallocate, make sure the qudit hasn't
    // been requested but not allocated.
    auto iter =
        std::find(requestedAllocations.begin(), requestedAllocations.end(), q);
    if (iter != requestedAllocations.end()) {
      requestedAllocations.erase(iter);
      return;
    }

    simulator()->deallocate(q.id);
  }

  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    std::vector<std::size_t> local;
    for (auto &q : qudits) {
      auto iter = std::find(requestedAllocations.begin(),
                            requestedAllocations.end(), q);
      if (iter != requestedAllocations.end()) {
        requestedAllocations.erase(iter);
      } else {
        local.push_back(q.id);
      }
    }

    simulator()->deallocateQubits(local);
  }

  void handleExecutionContextChanged() override {
    requestedAllocations.clear();
    simulator()->setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    simulator()->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

    // Get the data, create the Qubit* targets
    auto [gateName, parameters, controls, targets, op] = instruction;

    // Map the Qudits to Qubits
    std::vector<std::size_t> localT;
    std::transform(targets.begin(), targets.end(), std::back_inserter(localT),
                   [](auto &&el) { return el.id; });
    std::vector<std::size_t> localC;
    std::transform(controls.begin(), controls.end(), std::back_inserter(localC),
                   [](auto &&el) { return el.id; });

    // Apply the gate
    llvm::StringSwitch<std::function<void()>>(gateName)
        .Case("h", [&]() { simulator()->h(localC, localT[0]); })
        .Case("x", [&]() { simulator()->x(localC, localT[0]); })
        .Case("y", [&]() { simulator()->y(localC, localT[0]); })
        .Case("z", [&]() { simulator()->z(localC, localT[0]); })
        .Case("rx",
              [&]() { simulator()->rx(parameters[0], localC, localT[0]); })
        .Case("ry",
              [&]() { simulator()->ry(parameters[0], localC, localT[0]); })
        .Case("rz",
              [&]() { simulator()->rz(parameters[0], localC, localT[0]); })
        .Case("s", [&]() { simulator()->s(localC, localT[0]); })
        .Case("t", [&]() { simulator()->t(localC, localT[0]); })
        .Case("sdg", [&]() { simulator()->sdg(localC, localT[0]); })
        .Case("tdg", [&]() { simulator()->tdg(localC, localT[0]); })
        .Case("r1",
              [&]() { simulator()->r1(parameters[0], localC, localT[0]); })
        .Case("u1",
              [&]() { simulator()->u1(parameters[0], localC, localT[0]); })
        .Case("u3",
              [&]() {
                simulator()->u3(parameters[0], parameters[1], parameters[2],
                                localC, localT[0]);
              })
        .Case("swap",
              [&]() { simulator()->swap(localC, localT[0], localT[1]); })
        .Case("exp_pauli",
              [&]() {
                simulator()->applyExpPauli(parameters[0], localC, localT, op);
              })
        .Default([&]() {
          if (registeredOperations.count(gateName)) {
            auto data = registeredOperations[gateName]->unitary(parameters);
            simulator()->applyCustomOperation(data, localC, localT);
            return;
          }

          auto iter = quakeExtUnitaryFunctors.find(gateName);
          if (iter != quakeExtUnitaryFunctors.end()) {
            std::size_t size = (1UL << targets.size());
            std::vector<std::complex<double>> data(size * size);
            iter->second(parameters.data(), parameters.size(), data.data());
            if (cudaq::details::should_log(cudaq::details::LogLevel::info)) {
              std::string dataStr = "";
              for (auto &el : data)
                dataStr += "(" + std::to_string(el.real()) + ", " +
                           std::to_string(el.imag()) + ") ";

              cudaq::info("Found quake_ext op {} with {} elements", gateName,
                          dataStr);
            }
            simulator()->applyCustomOperation(data, localC, localT);
            return;
          }

          throw std::runtime_error("[DefaultExecutionManager] invalid gate "
                                   "application requested " +
                                   gateName + ".");
        })();
  }

  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    flushRequestedAllocations();
    return simulator()->mz(q.id, registerName);
  }

  void flushGateQueue() override {
    synchronize();
    flushRequestedAllocations();
    simulator()->flushGateQueue();
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    flushRequestedAllocations();
    simulator()->flushGateQueue();

    if (executionContext->canHandleObserve) {
      auto result = simulator()->observe(*executionContext->spin.value());
      executionContext->expectationValue = result.expectation();
      executionContext->result = result.raw_data();
      return;
    }

    assert(op.num_terms() == 1 && "Number of terms is not 1.");

    cudaq::info("Measure {}", op.to_string(false));
    std::vector<std::size_t> qubitsToMeasure;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitsToMeasure.push_back(qubitIdx);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          simulator()->rx(!reverse ? M_PI_2 : -M_PI_2, qubitIdx);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { simulator()->h(qubitIdx); });
    });

    // Change basis, flush the queue
    if (!basisChange.empty()) {
      for (auto &basis : basisChange)
        basis(false);

      simulator()->flushGateQueue();
    }

    // Get whether this is shots-based
    int shots = 0;
    if (executionContext->shots > 0)
      shots = executionContext->shots;

    // Sample and give the data to the context
    cudaq::ExecutionResult result = simulator()->sample(qubitsToMeasure, shots);
    executionContext->expectationValue = result.expectationValue;
    executionContext->result = cudaq::sample_result(result);

    // Restore the state.
    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);

      simulator()->flushGateQueue();
    }
  }

public:
  DefaultExecutionManager() {
    cudaq::info("[DefaultExecutionManager] Creating the {} backend.",
                simulator()->name());

    // Need to know of any quake_ext ops.
    std::string libSuffix;
    cudaq::__internal__::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
    libSuffix = "dylib";
    cudaq::__internal__::getCUDAQLibraryPath(&data);
#else
    libSuffix = "so";
    dl_iterate_phdr(cudaq::__internal__::getCUDAQLibraryPath, &data);
#endif

    std::filesystem::path nvqirLibPath{data.path};
    auto cudaqLibPath = nvqirLibPath.parent_path();
    auto cudaqInstallPath = cudaqLibPath.parent_path();
    auto cudaqIRDLPath = cudaqInstallPath / "extensions" / "irdl";
    for (const auto &entry :
         std::filesystem::directory_iterator(cudaqIRDLPath)) {
      if (entry.path().extension().string() == ".json") {
        std::ifstream t(entry.path());
        std::string str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
        auto json = nlohmann::json::parse(str);
        for (auto it = json.begin(); it != json.end(); ++it) {

          auto libName = "lib" + entry.path().stem().string() + "." + libSuffix;
          auto libLoc = cudaqInstallPath / "extensions" / "unitaries" / libName;
          auto handle = dlopen(libLoc.string().c_str(), RTLD_LAZY);
          auto generatorName = it.key() + "_generator";
          GenQuakeExtUnitaryFunctor fcn =
              (GenQuakeExtUnitaryFunctor)(intptr_t)dlsym(handle,
                                                         generatorName.c_str());
          quakeExtUnitaryFunctors.insert({it.key(), fcn});
        }
      }
    }
  }

  virtual ~DefaultExecutionManager() = default;

  void resetQudit(const cudaq::QuditInfo &q) override {
    flushRequestedAllocations();
    simulator()->resetQubit(q.id);
  }
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(DefaultExecutionManager)
