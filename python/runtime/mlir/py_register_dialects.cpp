/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_register_dialects.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#ifdef __APPLE__
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#endif
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/Support.h"
#include "mlir/InitAllDialects.h"
#include <fmt/core.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using namespace mlir;

static bool registered = false;

static void registerQuakeDialectAndTypes(nanobind::module_ &m) {
  using namespace mlir::python::nanobind_adaptors;
  auto quakeMod = m.def_submodule("quake");

  quakeMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);

        if (!registered) {
#ifdef __APPLE__
          cudaq_internal::compiler::initializeMLIR();
#else
          cudaqRegisterAllPassesAndPipelines();
#endif
          registered = true;
        }
      },
      nanobind::arg("load") = true,
      nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(quakeMod, "RefType",
                     [](MlirType type) {
                       return cudaqTypeIsAQuakeRefType(type);
                     },
                     cudaqQuakeRefTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirContext context) {
            return cudaqQuakeRefTypeGet(context);
          },
          nanobind::arg("cls"), nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(quakeMod, "MeasureType",
                     [](MlirType type) {
                       return cudaqTypeIsAQuakeMeasureType(type);
                     },
                     cudaqQuakeMeasureTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirContext context) {
            return cudaqQuakeMeasureTypeGet(context);
          },
          nanobind::arg("cls"), nanobind::arg("context") = nanobind::none());

  mlir::python::nanobind_adaptors::mlir_type_subclass(
      quakeMod, "VeqType",
      [](MlirType type) {
        return cudaqTypeIsAQuakeVeqType(type);
      },
      cudaqQuakeVeqTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, std::size_t size, MlirContext context) {
            return cudaqQuakeVeqTypeGet(context, size);
          },
          nanobind::arg("cls"),
          nanobind::arg("size") = std::numeric_limits<std::size_t>::max(),
          nanobind::arg("context") = nanobind::none())
      .def_staticmethod(
          "hasSpecifiedSize",
          [](MlirType type) {
            if (!cudaqTypeIsAQuakeVeqType(type))
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return cudaqQuakeVeqTypeHasSpecifiedSize(type);
          },
          nanobind::arg("veqTypeInstance"))
      .def_staticmethod(
          "getSize",
          [](MlirType type) {
            if (!cudaqTypeIsAQuakeVeqType(type))
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return cudaqQuakeVeqTypeGetSize(type);
          },
          nanobind::arg("veqTypeInstance"));

  quakeMod.def(
      "isConstantQuantumRefType",
      [](MlirType type) {
        return cudaq::quake::isConstantQuantumRefType(unwrap(type));
      },
      nanobind::arg("type"));

  quakeMod.def(
      "getAllocationSize",
      [](MlirType type) {
        return cudaq::quake::getAllocationSize(unwrap(type));
      },
      nanobind::arg("type"));

  mlir_type_subclass(quakeMod, "StruqType",
                     [](MlirType type) {
                       return cudaqTypeIsAQuakeStruqType(type);
                     },
                     cudaqQuakeStruqTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, nanobind::list aggregateTypes,
             MlirContext context) {
            SmallVector<MlirType> inTys;
            for (nanobind::handle t : aggregateTypes)
              inTys.push_back(nanobind::cast<MlirType>(t));

            return cudaqQuakeStruqTypeGet(context, inTys.size(), inTys.data());
          },
          nanobind::arg("cls"), nanobind::arg("aggregateTypes"),
          nanobind::arg("context") = nanobind::none())
      .def_classmethod(
          "getNamed",
          [](nanobind::object cls, const std::string &name,
             nanobind::list aggregateTypes, MlirContext context) {
            SmallVector<MlirType> inTys;
            for (nanobind::handle t : aggregateTypes)
              inTys.push_back(nanobind::cast<MlirType>(t));

            return cudaqQuakeStruqTypeGetNamed(
                context, mlirStringRefCreate(name.data(), name.size()),
                inTys.size(), inTys.data());
          },
          nanobind::arg("cls"), nanobind::arg("name"),
          nanobind::arg("aggregateTypes"),
          nanobind::arg("context") = nanobind::none())
      .def_classmethod(
          "getTypes",
          [](nanobind::object cls, MlirType structTy) {
            if (!cudaqTypeIsAQuakeStruqType(structTy))
              throw std::runtime_error(
                  "invalid type passed to StruqType.getTypes(), must be a "
                  "quake.struq");
            std::vector<MlirType> ret;
            auto numMembers = cudaqQuakeStruqTypeGetNumMembers(structTy);
            ret.reserve(numMembers);
            for (intptr_t i = 0; i < numMembers; ++i)
              ret.push_back(cudaqQuakeStruqTypeGetMember(structTy, i));
            return ret;
          })
      .def_classmethod("getName", [](nanobind::object cls, MlirType structTy) {
        if (!cudaqTypeIsAQuakeStruqType(structTy))
          throw std::runtime_error(
              "invalid type passed to StruqType.getName(), must be a "
              "quake.struq");
        auto name = cudaqQuakeStruqTypeGetName(structTy);
        return std::string{name.data, name.length};
      });
}

static void registerCCDialectAndTypes(nanobind::module_ &m) {
  using namespace mlir::python::nanobind_adaptors;
  auto ccMod = m.def_submodule("cc");

  ccMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle ccHandle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(ccHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ccHandle, context);
        }
      },
      nanobind::arg("load") = true,
      nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "CharspanType",
                     [](MlirType type) {
                       return cudaqTypeIsACCCharspanType(type);
                     },
                     cudaqCCCharspanTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirContext context) {
            return cudaqCCCharspanTypeGet(context);
          },
          nanobind::arg("cls"), nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "MeasureHandleType",
                     [](MlirType type) {
                       return cudaqTypeIsACCMeasureHandleType(type);
                     },
                     cudaqCCMeasureHandleTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirContext context) {
            return cudaqCCMeasureHandleTypeGet(context);
          },
          nanobind::arg("cls"), nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "StateType",
                     [](MlirType type) {
                       return cudaqTypeIsAQuakeStateType(type);
                     },
                     cudaqQuakeStateTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirContext context) {
            return cudaqQuakeStateTypeGet(context);
          },
          nanobind::arg("cls"), nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "PointerType",
                     [](MlirType type) {
                       return cudaqTypeIsACCPointerType(type);
                     },
                     cudaqCCPointerTypeGetTypeID)
      .def_classmethod(
          "getElementType",
          [](nanobind::object cls, MlirType type) {
            if (!cudaqTypeIsACCPointerType(type))
              throw std::runtime_error(
                  "invalid type passed to PointerType.getElementType(), must "
                  "be cc.ptr type.");
            return cudaqCCPointerTypeGetElementType(type);
          })
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirType elementType, MlirContext context) {
            return cudaqCCPointerTypeGet(context, elementType);
          },
          nanobind::arg("cls"), nanobind::arg("elementType"),
          nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "ArrayType",
                     [](MlirType type) {
                       return cudaqTypeIsACCArrayType(type);
                     },
                     cudaqCCArrayTypeGetTypeID)
      .def_classmethod(
          "getElementType",
          [](nanobind::object cls, MlirType type) {
            if (!cudaqTypeIsACCArrayType(type))
              throw std::runtime_error(
                  "invalid type passed to ArrayType.getElementType(), must "
                  "be cc.array type.");
            return cudaqCCArrayTypeGetElementType(type);
          })
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirType elementType, std::int64_t size,
             MlirContext context) {
            return cudaqCCArrayTypeGet(context, elementType, size);
          },
          nanobind::arg("cls"), nanobind::arg("elementType"),
          nanobind::arg("size") = std::numeric_limits<std::int64_t>::min(),
          nanobind::arg("context") = nanobind::none());

  mlir_type_subclass(ccMod, "StructType",
                     [](MlirType type) {
                       return cudaqTypeIsACCStructType(type);
                     },
                     cudaqCCStructTypeGetTypeID)
      .def_classmethod(
          "get",
          [](nanobind::object cls, nanobind::list aggregateTypes,
             MlirContext context) {
            SmallVector<MlirType> inTys;
            for (nanobind::handle t : aggregateTypes)
              inTys.push_back(nanobind::cast<MlirType>(t));

            return cudaqCCStructTypeGet(context, inTys.size(), inTys.data());
          },
          nanobind::arg("cls"), nanobind::arg("aggregateTypes"),
          nanobind::arg("context") = nanobind::none())
      .def_classmethod(
          "getNamed",
          [](nanobind::object cls, const std::string &name,
             nanobind::list aggregateTypes, MlirContext context) {
            SmallVector<MlirType> inTys;
            for (nanobind::handle t : aggregateTypes)
              inTys.push_back(nanobind::cast<MlirType>(t));

            return cudaqCCStructTypeGetNamed(
                context, mlirStringRefCreate(name.data(), name.size()),
                inTys.size(), inTys.data());
          },
          nanobind::arg("cls"), nanobind::arg("name"),
          nanobind::arg("aggregateTypes"),
          nanobind::arg("context") = nanobind::none())
      .def_classmethod(
          "getTypes",
          [](nanobind::object cls, MlirType structTy) {
            if (!cudaqTypeIsACCStructType(structTy))
              throw std::runtime_error(
                  "invalid type passed to StructType.getTypes(), must be a "
                  "cc.struct");
            std::vector<MlirType> ret;
            auto numMembers = cudaqCCStructTypeGetNumMembers(structTy);
            ret.reserve(numMembers);
            for (intptr_t i = 0; i < numMembers; ++i)
              ret.push_back(cudaqCCStructTypeGetMember(structTy, i));
            return ret;
          })
      .def_classmethod("getName", [](nanobind::object cls, MlirType structTy) {
        if (!cudaqTypeIsACCStructType(structTy))
          throw std::runtime_error(
              "invalid type passed to StructType.getName(), must be a "
              "cc.struct");
        auto name = cudaqCCStructTypeGetName(structTy);
        return std::string{name.data, name.length};
      });

  mlir_type_subclass(ccMod, "CallableType",
                     [](MlirType type) {
                       return cudaqTypeIsACCCallableType(type);
                     },
                     cudaqCCCallableTypeGetTypeID)
      .def_classmethod("get",
                       [](nanobind::object cls, MlirContext context,
                          nanobind::list inTypes, nanobind::list resTypes) {
                         // Nanobind builder: make the builder for this type
                         // look like that of a FunctionType.
                         SmallVector<Type> inTys;
                         for (nanobind::handle t : inTypes)
                           inTys.push_back(unwrap(nanobind::cast<MlirType>(t)));
                         SmallVector<Type> resTys;
                         for (nanobind::handle t : resTypes)
                           resTys.push_back(
                               unwrap(nanobind::cast<MlirType>(t)));

                         auto functionType = wrap(FunctionType::get(
                             unwrap(context), inTys, resTys));
                         return cudaqCCCallableTypeGet(context, functionType);
                       })
      .def_classmethod(
          "getFunctionType", [](nanobind::object cls, MlirType type) {
            if (!cudaqTypeIsACCCallableType(type))
              throw std::runtime_error("must be a cc.callable type!");
            return cudaqCCCallableTypeGetFunctionType(type);
          });

  mlir_type_subclass(ccMod, "StdvecType",
                     [](MlirType type) {
                       return cudaqTypeIsACCStdvecType(type);
                     },
                     cudaqCCStdvecTypeGetTypeID)
      .def_classmethod(
          "getElementType",
          [](nanobind::object cls, MlirType type) {
            if (!cudaqTypeIsACCStdvecType(type))
              throw std::runtime_error(
                  "invalid type passed to StdvecType.getElementType(), must "
                  "be cc.array type.");
            return cudaqCCStdvecTypeGetElementType(type);
          })
      .def_classmethod(
          "get",
          [](nanobind::object cls, MlirType elementType, MlirContext context) {
            return cudaqCCStdvecTypeGet(context, elementType);
          },
          nanobind::arg("cls"), nanobind::arg("elementType"),
          nanobind::arg("context") = nanobind::none());
}

void cudaq::bindRegisterDialects(nanobind::module_ &mod) {
  registerQuakeDialectAndTypes(mod);
  registerCCDialectAndTypes(mod);

  mod.def("load_intrinsic", [](MlirModule module, std::string name) {
    auto unwrapped = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(unwrapped.getBody());
    if (failed(builder.loadIntrinsic(unwrapped, name)))
      unwrapped.emitError("failed to load intrinsic " + name);
  });

  mod.def("register_all_dialects", [](MlirContext context) {
    ::cudaqRegisterAllDialects(context);
  });

  mod.def("gen_vector_of_complex_constant",
          [](MlirLocation loc, MlirModule module, std::string name,
             const std::vector<std::complex<double>> &values) {
            ModuleOp modOp = unwrap(module);
            cudaq::IRBuilder builder = IRBuilder::atBlockEnd(modOp.getBody());
            SmallVector<std::complex<double>> newValues{values.begin(),
                                                        values.end()};
            builder.genVectorOfConstants(unwrap(loc), modOp, name, newValues);
          });
}
