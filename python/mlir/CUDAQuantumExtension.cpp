/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllPasses.h"

#include <pybind11/stl.h>

#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/qis/py_qubit_qis.h"
#include "runtime/cudaq/spin/py_matrix.h"
#include "runtime/cudaq/spin/py_spin_op.h"
#include "runtime/cudaq/target/py_runtime_target.h"
#include "utils/LinkedLibraryHolder.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

static bool registered = false;

// This is a custom LinkedLibraryHolder that does not
// automatically load the Remote REST QPU, we will
// need a different Remote REST QPU to avoid the LLVM startup issues
static cudaq::LinkedLibraryHolder holder;

void registerQuakeDialectAndTypes(py::module &m) {
  auto quakeMod = m.def_submodule("quake");

  quakeMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }

        if (!registered) {
          cudaq::opt::registerOptCodeGenPasses();
          cudaq::opt::registerOptTransformsPasses();
          cudaq::opt::registerAggressiveEarlyInlining();
          cudaq::opt::registerUnrollingPipeline();
          cudaq::opt::registerTargetPipelines();
          registered = true;
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(quakeMod, "RefType", [](MlirType type) {
    return unwrap(type).isa<quake::RefType>();
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::RefType::get(unwrap(ctx)));
  });

  mlir_type_subclass(
      quakeMod, "VeqType",
      [](MlirType type) { return unwrap(type).isa<quake::VeqType>(); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, std::size_t size) {
            return wrap(quake::VeqType::get(unwrap(ctx), size));
          },
          py::arg("cls"), py::arg("context"), py::arg("size") = 0);
}

void registerCCDialectAndTypes(py::module &m) {

  auto ccMod = m.def_submodule("cc");

  ccMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle ccHandle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(ccHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ccHandle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(
      ccMod, "PointerType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::PointerType>(); })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::PointerType::get(unwrap(ctx), unwrap(elementType)));
          });

  mlir_type_subclass(
      ccMod, "ArrayType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::ArrayType>(); })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::StdvecType::get(unwrap(ctx), unwrap(elementType)));
          });

  mlir_type_subclass(
      ccMod, "StdvecType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::StdvecType>(); })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::StdvecType::get(unwrap(ctx), unwrap(elementType)));
          });
}

PYBIND11_MODULE(_quakeDialects, m) {
  registerQuakeDialectAndTypes(m);
  registerCCDialectAndTypes(m);

  auto cudaqRuntime = m.def_submodule("cudaq_runtime");

  cudaq::bindRuntimeTarget(cudaqRuntime, holder);
  cudaq::bindMeasureCounts(cudaqRuntime);
  cudaq::bindObserveResult(cudaqRuntime);
  cudaq::bindComplexMatrix(cudaqRuntime);
  cudaq::bindSpinWrapper(cudaqRuntime);
  cudaq::bindQIS(cudaqRuntime);

  py::class_<cudaq::ExecutionContext>(cudaqRuntime, "ExecutionContext")
      .def(py::init<std::string>())
      .def(py::init<std::string, std::size_t>())
      .def_readonly("result", &cudaq::ExecutionContext::result)
      .def("setSpinOperator", [](cudaq::ExecutionContext &ctx,
                                 cudaq::spin_op &spin) { ctx.spin = &spin; })
      .def("getExpectationValue", [](cudaq::ExecutionContext &ctx) {
        return ctx.expectationValue.value();
      });
  cudaqRuntime.def(
      "setExecutionContext",
      [](cudaq::ExecutionContext &ctx) {
        auto &self = cudaq::get_platform();
        self.set_exec_ctx(&ctx);
      },
      "");
  cudaqRuntime.def(
      "resetExecutionContext",
      []() {
        auto &self = cudaq::get_platform();
        self.reset_exec_ctx();
      },
      "");

  cudaqRuntime.def(
      "applyQuantumOperation",
      [](const std::string &name, std::vector<double> &params,
         std::vector<std::size_t> &controls, std::vector<std::size_t> &targets,
         bool isAdjoint, cudaq::spin_op &op) {
        std::vector<cudaq::QuditInfo> c, t;
        std::transform(controls.begin(), controls.end(), std::back_inserter(c),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        std::transform(targets.begin(), targets.end(), std::back_inserter(t),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        cudaq::getExecutionManager()->apply(name, params, c, t, isAdjoint, op);
      },
      py::arg("name"), py::arg("params"), py::arg("controls"),
      py::arg("targets"), py::arg("isAdjoint") = false,
      py::arg("op") = cudaq::spin_op());
}