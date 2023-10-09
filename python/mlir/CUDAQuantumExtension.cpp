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
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <pybind11/complex.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/algorithms/py_optimizer.h"
#include "runtime/cudaq/qis/py_qubit_qis.h"
#include "runtime/cudaq/spin/py_matrix.h"
#include "runtime/cudaq/spin/py_spin_op.h"
#include "runtime/cudaq/target/py_runtime_target.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

static bool registered = false;

// This is a custom LinkedLibraryHolder that does not
// automatically load the Remote REST QPU, we will
// need a different Remote REST QPU to avoid the LLVM startup issues
static std::unique_ptr<cudaq::LinkedLibraryHolder> holder;

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
          "get",
          [](py::object cls, MlirContext ctx, MlirType elementType,
             std::int64_t size) {
            return wrap(cudaq::cc::ArrayType::get(unwrap(ctx),
                                                  unwrap(elementType), size));
          },
          py::arg("cls"), py::arg("ctx"), py::arg("elementType"),
          py::arg("size") = std::numeric_limits<std::int64_t>::min());

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
  holder =
      std::make_unique<cudaq::LinkedLibraryHolder>(/*override_rest_qpu*/ true);
  registerQuakeDialectAndTypes(m);
  registerCCDialectAndTypes(m);

  auto cudaqRuntime = m.def_submodule("cudaq_runtime");

  cudaq::bindRuntimeTarget(cudaqRuntime, *holder.get());
  cudaq::bindMeasureCounts(cudaqRuntime);
  cudaq::bindObserveResult(cudaqRuntime);
  cudaq::bindComplexMatrix(cudaqRuntime);
  cudaq::bindSpinWrapper(cudaqRuntime);
  cudaq::bindQIS(cudaqRuntime);
  cudaq::bindOptimizerWrapper(cudaqRuntime);

  py::class_<cudaq::ExecutionContext>(cudaqRuntime, "ExecutionContext")
      .def(py::init<std::string>())
      .def(py::init<std::string, std::size_t>())
      .def_readonly("result", &cudaq::ExecutionContext::result)
      .def_readonly("simulationData", &cudaq::ExecutionContext::simulationData)
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

  cudaqRuntime.def("startAdjointRegion", []() {
    cudaq::getExecutionManager()->startAdjointRegion();
  });
  cudaqRuntime.def("endAdjointRegion",
                   []() { cudaq::getExecutionManager()->endAdjointRegion(); });

  cudaqRuntime.def("startCtrlRegion", [](std::vector<std::size_t> &controls) {
    cudaq::getExecutionManager()->startCtrlRegion(controls);
  });
  cudaqRuntime.def("endCtrlRegion", [](std::size_t nControls) {
    cudaq::getExecutionManager()->endCtrlRegion(nControls);
  });

  cudaqRuntime.def("measure", [](std::size_t id) {
    cudaq::getExecutionManager()->measure(cudaq::QuditInfo(2, id));
  });

  cudaqRuntime.def("pyAltLaunchKernel", [](const std::string &name,
                                           MlirModule module,
                                           py::args runtimeArgs) {
    auto mod = unwrap(module);
    auto cloned = mod.clone();
    auto context = cloned.getContext();
    registerLLVMDialectTranslation(*context);

    // FIXME, this should be dependent on the target
    PassManager pm(context);
    OpPassManager &optPM = pm.nest<func::FuncOp>();
    cudaq::opt::addAggressiveEarlyInlining(pm);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
    pm.addPass(createCanonicalizerPass());
    optPM.addPass(cudaq::opt::createQuakeAddDeallocs());
    optPM.addPass(cudaq::opt::createQuakeAddMetadata());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
    pm.addPass(cudaq::opt::createGenerateKernelExecution());
    optPM.addPass(cudaq::opt::createLowerToCFGPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(cudaq::opt::createConvertToQIRPass());

    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "cudaq::builder failed to JIT compile the Quake representation.");

    ExecutionEngineOptions opts;
    opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
    SmallVector<StringRef, 4> sharedLibs;
    opts.llvmModuleBuilder =
        [](Operation *module,
           llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
      llvmContext.setOpaquePointers(false);
      auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return nullptr;
      }
      ExecutionEngine::setupTargetTriple(llvmModule.get());
      return llvmModule;
    };

    auto jitOrError = ExecutionEngine::create(cloned, opts);
    assert(!!jitOrError);
    auto uniqueJit = std::move(jitOrError.get());
    auto *jit = uniqueJit.release();

    cudaq::OpaqueArguments argData;
    cudaq::packArgs(argData, runtimeArgs);
    auto expectedPtr = jit->lookup(name + ".argsCreator");
    if (!expectedPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get argsCreator function.");
    }
    auto argsCreator =
        reinterpret_cast<std::size_t (*)(void **, void **)>(*expectedPtr);
    void *rawArgs = nullptr;
    [[maybe_unused]] auto size = argsCreator(argData.data(), &rawArgs);

    auto thunkName = name + ".thunk";
    auto thunkPtr = jit->lookup(thunkName);
    if (!thunkPtr) {
      throw std::runtime_error("cudaq::builder failed to get thunk function");
    }

    // Invoke and free the args memory.
    auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

    struct ArgWrapper {
      ModuleOp mod;
      void *rawArgs = nullptr;
    };

    auto *wrapper = new ArgWrapper{mod, rawArgs};
    cudaq::altLaunchKernel(name.c_str(), thunk,
                           reinterpret_cast<void *>(wrapper), size, 0);
    delete wrapper;
    std::free(rawArgs);
    delete jit;
  });

  cudaqRuntime.def("cloneModuleOp",
                   [](MlirModule mod) { return wrap(unwrap(mod).clone()); });
}