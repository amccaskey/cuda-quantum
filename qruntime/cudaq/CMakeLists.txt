# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


set(LIBRARY_NAME cudaq2)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
add_library(target_config_empty OBJECT target_config.cpp)
set (CMAKE_ENABLE_EXPORTS TRUE)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(CUDAQ_MLIR_LIBS ${dialect_libs}
                    ${conversion_libs}
                    CCDialect
                    QuakeDialect
                    OptCodeGen
                    OptTransforms
                    MLIRTranslateLib
                    MLIRIR
                    MLIRParser
                    MLIRPass
                    MLIRTranslateLib
                    MLIRSupport
                    MLIROptLib
                    MLIRExecutionEngine
                    MLIRTransforms
                    MLIRTargetLLVMIRExport
                    MLIRLLVMCommonConversion
                    MLIRLLVMToLLVMIRTranslation)

add_link_options(-Wl,--no-as-needed)
# Create the CUDA-Q Library
add_library(${LIBRARY_NAME} 
         SHARED cudaq.cpp 
                builder/kernel_builder.cpp
                builder/QuakeValue.cpp 
                builder/kernels.cpp
                distributed/mpi_plugin.cpp
                simulators/qpp/QppCircuitSimulator.cpp
                spin/spin_op.cpp
                spin/matrix.cpp
                qis/execution_manager.cpp 
                qis/managers/default/DefaultExecutionManager.cpp
                qis/state.cpp
                utils/cudaq_utils.cpp 
                utils/Logger.cpp
                utils/MeasureCounts.cpp 
                utils/NoiseModel.cpp
                utils/Trace.cpp
                utils/Future.cpp
                utils/RuntimeMLIR.cpp 
                utils/ServerHelper.cpp
                utils/Executor.cpp
                platform/quantum_platform.cpp
                platform/default/DefaultQuantumPlatform.cpp
                platform/common/QuantumExecutionQueue.cpp
                platform/default/rest/RemoteRESTQPU.cpp
                platform/default/rest/helpers/quantinuum/QuantinuumServerHelper.cpp
                )

# We can only build the RestClient support if we have OpenSSL
if(OPENSSL_FOUND)
  message(STATUS "OpenSSL Found, building REST Client.")

  target_sources(${LIBRARY_NAME} PRIVATE utils/RestClient.cpp)
  target_link_libraries(${LIBRARY_NAME} PRIVATE cpr::cpr -Wl,--start-group ZLIB::ZLIB)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE -DCUDAQ_RESTCLIENT_AVAILABLE)
endif()

if(EXISTS "$ENV{CURL_INSTALL_PREFIX}/cacert.pem")
  # Copy to build directory so that developers can run from the build directory
  file(COPY "$ENV{CURL_INSTALL_PREFIX}/cacert.pem" DESTINATION "${CMAKE_BINARY_DIR}")
  # Also include it as an install target
  install(FILES "$ENV{CURL_INSTALL_PREFIX}/cacert.pem" DESTINATION .)
endif()

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

if (CUDA_FOUND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(${LIBRARY_NAME} PRIVATE 
     simulators/custatevec/CuStateVecCircuitSimulator.cu
     simulators/custatevec/CuStateVecCircuitSimulatorF32.cu
     )

  string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET CUDA_VERSION_LIST 0 CUDA_MAJOR_VERSION)

  find_library(CUSTATEVEC_LIB
      NAMES custatevec
      HINTS
      ${CUSTATEVEC_ROOT}/lib64
      ${CUSTATEVEC_ROOT}/lib
      ${CUSTATEVEC_ROOT}/lib64/${CUDA_MAJOR_VERSION}
      ${CUSTATEVEC_ROOT}/lib/${CUDA_MAJOR_VERSION})

  get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}")

  target_include_directories(${LIBRARY_NAME}
    PUBLIC $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/qruntime>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
            $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
            $<BUILD_INTERFACE:${CUSTATEVEC_ROOT}/include>)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC dl 
    PRIVATE fmt::fmt-header-only 
            CUDA::cudart_static 
            spdlog::spdlog 
            libqpp 
            ${CUSTATEVEC_LIB}
            ${CUDA_LIBRARIES} 
            cublas
            ${CUDAQ_MLIR_LIBS}
            target_config_empty)

  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_CUDA)
else()
  target_include_directories(${LIBRARY_NAME}
    PUBLIC $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/qruntime>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
    PRIVATE .)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC dl
    PRIVATE fmt::fmt-header-only 
            spdlog::spdlog 
            libqpp 
            ${CUDAQ_MLIR_LIBS}
            target_config_empty)
endif()

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq2-targets DESTINATION lib)

install(EXPORT cudaq2-targets
        FILE CUDAQTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq2)

add_subdirectory(simulators/targets)
# Install CUDAQ MPI interface header, reference implementation, and build script
# for manual activation on a local system if needed.
install(
  FILES
      distributed/distributed_capi.h
      distributed/builtin/mpi_comm_impl.cpp
      distributed/builtin/activate_custom_mpi.sh
  DESTINATION distributed_interfaces
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
