/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"

#include <pybind11/complex.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "runtime/common/py_ExecutionContext.h"
#include "runtime/common/py_NoiseModel.h"
#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/algorithms/py_observe_async.h"
#include "runtime/cudaq/algorithms/py_optimizer.h"
#include "runtime/cudaq/algorithms/py_sample_async.h"
#include "runtime/cudaq/kernels/py_common_kernels.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "runtime/cudaq/qis/py_execution_manager.h"
#include "runtime/cudaq/qis/py_qubit_qis.h"
#include "runtime/cudaq/spin/py_matrix.h"
#include "runtime/cudaq/spin/py_spin_op.h"
#include "runtime/cudaq/target/py_runtime_target.h"
#include "runtime/mlir/py_register_dialects.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

static std::unique_ptr<cudaq::LinkedLibraryHolder> holder;

PYBIND11_MODULE(_quakeDialects, m) {
  holder =
      std::make_unique<cudaq::LinkedLibraryHolder>(/*override_rest_qpu*/ true);

  cudaq::bindRegisterDialects(m);

  auto cudaqRuntime = m.def_submodule("cudaq_runtime");
  cudaq::bindRuntimeTarget(cudaqRuntime, *holder.get());
  cudaq::bindMeasureCounts(cudaqRuntime);
  cudaq::bindObserveResult(cudaqRuntime);
  cudaq::bindComplexMatrix(cudaqRuntime);
  cudaq::bindSpinWrapper(cudaqRuntime);
  cudaq::bindQIS(cudaqRuntime);
  cudaq::bindOptimizerWrapper(cudaqRuntime);
  cudaq::bindCommonKernels(cudaqRuntime);
  cudaq::bindNoise(cudaqRuntime);
  cudaq::bindExecutionContext(cudaqRuntime);
  cudaq::bindExecutionManager(cudaqRuntime);
  cudaq::bindSampleAsync(cudaqRuntime);
  cudaq::bindObserveAsync(cudaqRuntime);
  cudaq::bindAltLaunchKernel(cudaqRuntime);
  cudaqRuntime.def("set_random_seed", &cudaq::set_random_seed,
                   "Provide the seed for backend quantum kernel simulation.");
  cudaqRuntime.def("set_noise", &cudaq::set_noise, "");
  cudaqRuntime.def("unset_noise", &cudaq::unset_noise, "");
  cudaqRuntime.def("cloneModule", [](MlirModule mod) {return wrap(unwrap(mod).clone());});
  cudaqRuntime.def("isTerminator", [](MlirOperation op) {
    return unwrap(op)->hasTrait<OpTrait::IsTerminator>();
  });
}